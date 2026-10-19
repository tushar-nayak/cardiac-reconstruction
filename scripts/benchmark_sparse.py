"""Compare INR and Gaussian fields using exactly the same annotated planes.

This is subject-specific reconstruction from sparse binary masks. It does not
train an echo-image encoder or use unobserved labels during optimization.

With ``--manifest`` subjects come from the frozen cohort split
(``benchmarks/protocol_v1.md``). Held-out test subjects cannot be selected.
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_common.cohort import enumerate_volumes, load_label
from cardiac_common.interpolation import interpolate_sections, plane_sections
from cardiac_common.sparse import (
    evaluate_grid, plane_samples, reconstruction_metrics, select_planes,
)
from code_gaussian.sparse_model import SparseGaussianField
from code_gaussian.src.data_loaders.mitea_loader import MITEADataset
from code_inr.inr_model import ImplicitOccupancyNetwork


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_field(model, points, labels, args, device, seed):
    """Identical minibatch sequence, BCE loss, and update budget for each method."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    sync(device)
    started = time.perf_counter()
    for step in range(args.steps):
        indices = torch.randint(len(points), (min(args.batch_size, len(points)),), generator=generator)
        prediction = model(points[indices].to(device)).reshape(-1)
        loss = F.binary_cross_entropy(prediction, labels[indices].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"  step {step + 1}/{args.steps}: BCE={loss.item():.6f}", flush=True)
    sync(device)
    return time.perf_counter() - started, float(loss.detach().cpu())


def summarize(results):
    summary = {}
    failed = [row for row in results if row.get("status") == "failed"]
    results = [row for row in results if row.get("status") != "failed"]
    for method in sorted({row["method"] for row in results}):
        rows = [row for row in results if row["method"] == method]
        scores = {
            "dice": [row["metrics"]["whole_grid"]["dice"] for row in rows],
            "iou": [row["metrics"]["whole_grid"]["iou"] for row in rows],
            "fit_seconds": [row["fit_seconds"] for row in rows],
        }
        unseen = [row["metrics"]["unobserved_voxels"] for row in rows]
        if all(value is not None for value in unseen):
            scores["unobserved_dice"] = [value["dice"] for value in unseen]
        delta_k = [row["metrics"].get("geometry", {}).get("absolute_delta_integrated_gaussian_curvature")
                   for row in rows]
        valid_delta_k = [value for value in delta_k if value is not None]
        if valid_delta_k:
            scores["absolute_delta_integrated_gaussian_curvature"] = valid_delta_k
        summary[method] = {
            "runs": len(rows),
            "subjects": len({row["subject"] for row in rows}),
            "delta_k_valid_runs": len(valid_delta_k),
            "delta_k_unavailable_runs": len(rows) - len(valid_delta_k),
            "failed_runs": sum(row["method"] == method for row in failed),
            "statistics_over_subject_seed_runs": {
                name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                for name, values in scores.items()
            },
        }
    if failed:
        summary["failed"] = [{key: row.get(key) for key in ("method", "subject", "seed", "error")}
                             for row in failed]
    return summary


def manifest_subjects(manifest, subset):
    """Development subjects for ``subset``; the test split is never returned."""
    folds = manifest["splits"]["folds"]
    if subset == "confirmatory":
        chosen = [(s, index) for index, fold in enumerate(folds) if index != 0 for s in fold]
    else:
        index = int(subset.removeprefix("fold"))
        chosen = [(s, index) for s in folds[index]]
    held_out = set(manifest["splits"]["test"])
    if any(subject in held_out for subject, _ in chosen):
        raise RuntimeError("manifest fold overlaps the held-out test split")
    return sorted(chosen)


def manifest_samples(args):
    """Yield protocol samples: scan 1 ED, foreground = any nonzero label."""
    manifest = json.loads(Path(args.manifest).read_text())
    chosen = manifest_subjects(manifest, args.subset)
    if args.subjects:
        available = {subject for subject, _ in chosen}
        missing = sorted(set(args.subjects) - available)
        if missing:
            raise ValueError(f"subjects not in subset {args.subset}: {missing}")
        chosen = [(s, fold) for s, fold in chosen if s in set(args.subjects)]
    paths = {(v.subject, v.scan, v.phase): v.label_path for v in enumerate_volumes(args.data_dir)}
    info = {"manifest": args.manifest, "manifest_sha256": manifest["splits_sha256"],
            "subset": args.subset, "subjects": [s for s, _ in chosen]}

    def samples():
        for position, (subject, fold) in enumerate(chosen):
            label, affine = load_label(paths[(subject, 1, "ED")])
            yield {"subject": subject, "index": position, "fold": fold,
                   "category": manifest["strata"][subject],
                   "occupancy_ed": torch.from_numpy((label > 0).astype(np.float32)),
                   "affine": torch.from_numpy(affine).float()}
    return info, samples()


def legacy_samples(args):
    datasets = {split: MITEADataset(args.data_dir, split=split, split_seed=args.split_seed)
                for split in ("train", "val", "test")}
    dataset = datasets[args.split]
    indices = list(range(len(dataset))) if args.all_subjects else (args.sample_indices or [0])
    if not indices or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f"sample indices must be in [0, {len(dataset) - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("sample indices must not contain duplicates")
    info = {"split_seed": args.split_seed,
            "subjects": {split: data.subject_ids for split, data in datasets.items()}}

    def samples():
        for index in indices:
            sample = dataset[index]
            sample["index"] = index
            yield sample
    return info, samples()


def build_model(method, args, points, labels):
    if method == "inr":
        return ImplicitOccupancyNetwork(args.hidden_dim, args.layers, args.frequencies), "network_default"
    model = SparseGaussianField(args.num_gaussians)
    if args.gaussian_init == "observed":
        has_foreground = model.initialize_from_observations(points, labels)
        return model, "observed_foreground" if has_foreground else "uniform_no_observed_foreground"
    return model, "uniform"


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.cpu_threads)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must not contain duplicates")
    split_info, samples = manifest_samples(args) if args.manifest else legacy_samples(args)

    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "splits.json", split_info)
    write_json(output / "config.json", vars(args))
    protocol = "mitea_sparse_planes_protocol_v1" if args.manifest else "fixed_sparse_binary_label_planes_v1"
    results = []
    for sample in samples:
        target = sample["occupancy_ed"].float()
        shape = tuple(target.shape)
        planes = select_planes(shape, args.num_planes, args.plane_layout, args.axial_fractions)
        points, labels, observed = plane_samples(target, planes)
        digest = hashlib.sha256(observed.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
        save_dense = args.save_occupancy or sample["subject"] in set(args.save_occupancy_subjects or ())
        for method in args.methods:
            # The nonlearned baseline is deterministic, so it runs once per subject.
            for seed in ([None] if method == "interpolation" else args.seeds):
                row = {
                    "protocol": protocol, "method": method,
                    "subject": sample["subject"], "fold": sample.get("fold"),
                    "category": sample.get("category"), "scan": 1, "phase": "ED",
                    "split": args.subset if args.manifest else args.split,
                    "sample_index": sample["index"], "seed": seed,
                    "plane_layout": args.plane_layout,
                    "volume_shape_zyx": shape, "planes_zyx_axis_index": planes,
                    "observed_voxels": int(observed.sum()), "supervision_sha256": digest,
                    "threshold": args.threshold, "eval_stride": args.eval_stride,
                    "device": str(device), "torch_version": str(torch.__version__),
                }
                destination = output / method / sample["subject"] / ("deterministic" if seed is None
                                                                     else f"seed_{seed}")
                destination.mkdir(parents=True)
                try:
                    probability, model = evaluate_method(method, row, sample, planes, points, labels,
                                                         observed, args, device, seed)
                except Exception as error:  # Failures are recorded, never silently dropped.
                    row.update(status="failed", error=f"{type(error).__name__}: {error}",
                               traceback=traceback.format_exc())
                    write_json(destination / "metrics.json", row)
                    results.append(row)
                    write_json(output / "results.json", {"runs": results, "summary": summarize(results)})
                    print(f"  FAILED {method} {sample['subject']} seed={seed}: {error}", flush=True)
                    continue
                row["status"] = "ok"
                write_json(destination / "metrics.json", row)
                if model is not None:
                    torch.save({"model_state_dict": {key: value.detach().cpu()
                                                    for key, value in model.state_dict().items()},
                                "config": vars(args), "result": row, "affine": sample["affine"]},
                               destination / "checkpoint.pth")
                if save_dense:
                    torch.save(probability, destination / "predicted_occupancy.pt")
                results.append(row)
                write_json(output / "results.json", {"runs": results, "summary": summarize(results)})
                print(f"  Dice={row['metrics']['whole_grid']['dice']:.4f}, fit={row['fit_seconds']:.2f}s",
                      flush=True)
                del model
    print(f"Results saved to {output}", flush=True)


def evaluate_method(method, row, sample, planes, points, labels, observed, args, device, seed):
    """Fit (or interpolate) one method and attach timing and metrics to ``row``."""
    target = sample["occupancy_ed"].float()
    shape = tuple(target.shape)
    stride = args.eval_stride
    print(f"{sample['subject']} {method} seed={seed} planes={planes}", flush=True)
    if method == "interpolation":
        started = time.perf_counter()
        mask, variant = interpolate_sections(plane_sections(target.numpy(), planes), shape,
                                             args.plane_layout)
        probability = torch.from_numpy(mask[::stride, ::stride, ::stride].astype(np.float32))
        model, seconds, final_loss = None, time.perf_counter() - started, None
        row.update(initialization=variant, learning_rate=None, optimizer_steps=0, parameter_count=0)
        eval_seconds, peak = 0.0, None
    else:
        torch.manual_seed(seed)
        model, initialization = build_model(method, args, points, labels)
        model = model.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seconds, final_loss = fit_field(model, points, labels, args, device, seed)
        sync(device)
        started = time.perf_counter()
        probability = evaluate_grid(model, shape, device, args.eval_chunk_size, stride)
        sync(device)
        eval_seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        row.update(initialization=initialization, learning_rate=args.lr,
                   optimizer_steps=args.steps,
                   parameter_count=sum(p.numel() for p in model.parameters()))
    if not torch.isfinite(probability).all():
        raise RuntimeError("nonfinite predicted occupancy")
    affine = np.asarray(sample["affine"]) @ np.diag([stride, stride, stride, 1])
    metric_started = time.perf_counter()
    metrics = reconstruction_metrics(
        probability, target[::stride, ::stride, ::stride],
        observed[::stride, ::stride, ::stride], affine, args.threshold,
    )
    row.update(fit_seconds=seconds, evaluation_seconds=eval_seconds,
               metric_evaluation_seconds=time.perf_counter() - metric_started,
               final_training_bce=final_loss, peak_cuda_allocated_mb=peak,
               empty_prediction=bool(not (np.asarray(probability) >= args.threshold).any()),
               metrics=metrics)
    return probability, model



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True, help="New output directory; never overwrites a run")
    parser.add_argument("--methods", nargs="+", choices=("inr", "gaussian", "interpolation"),
                        default=["inr", "gaussian"])
    parser.add_argument("--manifest", default=None,
                        help="Frozen cohort manifest; selects subjects with --subset instead of --split")
    parser.add_argument("--subset", choices=("fold0", "fold1", "fold2", "fold3", "fold4", "confirmatory"),
                        help="Development fold, or folds 1-4 for the confirmatory report")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Restrict the manifest subset to these subjects (smoke checks)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val",
                        help="Legacy loader split, used only without --manifest")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-indices", nargs="+", type=int, default=None)
    selection.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--plane-layout", choices=("axial", "orthogonal"), default="axial")
    parser.add_argument("--num-planes", type=int, default=3)
    parser.add_argument("--axial-fractions", nargs="+", type=float, default=None,
                        help="Axial plane depths as fractions; protocol v1 amendment uses 0.375 0.5 0.625")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--frequencies", type=int, default=4)
    parser.add_argument("--num-gaussians", type=int, default=256)
    parser.add_argument("--gaussian-init", choices=("uniform", "observed"), default="uniform",
                        help="Optionally seed Gaussian centers using observed foreground coordinates only")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-stride", type=int, default=1, help="Use >1 only for coarse smoke checks")
    parser.add_argument("--save-occupancy", action="store_true")
    parser.add_argument("--save-occupancy-subjects", nargs="+", default=None,
                        help="Save dense predictions only for these subjects")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    positive = (args.steps, args.batch_size, args.num_planes, args.num_gaussians,
                args.hidden_dim, args.layers, args.frequencies, args.eval_chunk_size,
                args.eval_stride, args.cpu_threads, args.lr)
    if min(positive) <= 0 or not 0 < args.threshold < 1:
        parser.error("counts and learning rate must be positive; threshold must be between 0 and 1")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must not contain duplicates")
    if args.manifest:
        if not args.subset:
            parser.error("--manifest requires --subset")
        if args.sample_indices is not None or args.all_subjects:
            parser.error("--sample-indices/--all-subjects apply only to the legacy split")
    elif args.subset or args.subjects:
        parser.error("--subset and --subjects require --manifest")
    if args.axial_fractions is not None and args.plane_layout != "axial":
        parser.error("--axial-fractions applies only to the axial layout")
    run(args)


if __name__ == "__main__":
    main()
