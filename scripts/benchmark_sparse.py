"""Compare INR and Gaussian fields using exactly the same annotated planes.

This is subject-specific reconstruction from sparse binary masks. It does not
train an echo-image encoder or use unobserved labels during optimization.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        summary[method] = {
            "runs": len(rows),
            "subjects": len({row["subject"] for row in rows}),
            "statistics_over_subject_seed_runs": {
                name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                for name, values in scores.items()
            },
        }
    return summary


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.cpu_threads)
    datasets = {split: MITEADataset(args.data_dir, split=split, split_seed=args.split_seed)
                for split in ("train", "val", "test")}
    dataset = datasets[args.split]
    indices = list(range(len(dataset))) if args.all_subjects else args.sample_indices
    if not indices or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f"sample indices must be in [0, {len(dataset) - 1}]")
    if len(set(indices)) != len(indices) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("sample indices and seeds must not contain duplicates")

    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {split: data.subject_ids for split, data in datasets.items()}
    write_json(output / "splits.json", {"split_seed": args.split_seed, "subjects": manifest})
    write_json(output / "config.json", vars(args))
    results = []
    for index in indices:
        sample = dataset[index]
        target = sample["occupancy_ed"].float()
        shape = tuple(target.shape)
        planes = select_planes(shape, args.num_planes, args.plane_layout)
        points, labels, observed = plane_samples(target, planes)
        digest = hashlib.sha256(observed.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
        for seed in args.seeds:
            for method in args.methods:
                torch.manual_seed(seed)
                model = (ImplicitOccupancyNetwork(args.hidden_dim, args.layers, args.frequencies)
                         if method == "inr" else SparseGaussianField(args.num_gaussians)).to(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                print(f"{sample['subject']} {method} seed={seed} planes={planes}", flush=True)
                seconds, final_loss = fit_field(model, points, labels, args, device, seed)
                sync(device)
                started = time.perf_counter()
                probability = evaluate_grid(model, shape, device, args.eval_chunk_size, args.eval_stride)
                sync(device)
                eval_seconds = time.perf_counter() - started
                stride = args.eval_stride
                affine = np.asarray(sample["affine"]) @ np.diag([stride, stride, stride, 1])
                metrics = reconstruction_metrics(
                    probability, target[::stride, ::stride, ::stride],
                    observed[::stride, ::stride, ::stride], affine, args.threshold,
                )
                row = {
                    "protocol": "fixed_sparse_binary_label_planes_v1", "method": method,
                    "subject": sample["subject"], "scan": 1, "phase": "ED",
                    "split": args.split, "sample_index": index, "seed": seed,
                    "volume_shape_zyx": shape, "planes_zyx_axis_index": planes,
                    "observed_voxels": int(observed.sum()), "supervision_sha256": digest,
                    "threshold": args.threshold, "eval_stride": stride,
                    "device": str(device), "torch_version": str(torch.__version__),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "fit_seconds": seconds, "evaluation_seconds": eval_seconds,
                    "final_training_bce": final_loss,
                    "peak_cuda_allocated_mb": (torch.cuda.max_memory_allocated(device) / 1024**2
                                               if device.type == "cuda" else None),
                    "metrics": metrics,
                }
                destination = output / method / sample["subject"] / f"seed_{seed}"
                destination.mkdir(parents=True)
                write_json(destination / "metrics.json", row)
                torch.save({"model_state_dict": {key: value.detach().cpu()
                                                for key, value in model.state_dict().items()},
                            "config": vars(args), "result": row, "affine": sample["affine"]},
                           destination / "checkpoint.pth")
                if args.save_occupancy:
                    torch.save(probability, destination / "predicted_occupancy.pt")
                results.append(row)
                write_json(output / "results.json", {"runs": results, "summary": summarize(results)})
                print(f"  Dice={metrics['whole_grid']['dice']:.4f}, fit={seconds:.2f}s", flush=True)
                del model
    print(f"Results saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True, help="New output directory; never overwrites a run")
    parser.add_argument("--methods", nargs="+", choices=("inr", "gaussian"), default=["inr", "gaussian"])
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-indices", nargs="+", type=int, default=[0])
    selection.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--plane-layout", choices=("axial", "orthogonal"), default="axial")
    parser.add_argument("--num-planes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--frequencies", type=int, default=4)
    parser.add_argument("--num-gaussians", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-stride", type=int, default=1, help="Use >1 only for coarse smoke checks")
    parser.add_argument("--save-occupancy", action="store_true")
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
    run(args)


if __name__ == "__main__":
    main()
