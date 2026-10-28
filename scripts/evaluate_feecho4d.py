"""Frozen external radial-view completion evaluation on FeEcho4D.

This evaluates withheld annotated views in (x, y, angle) index coordinates.
It is not Cartesian 3D reconstruction and uses no ultrasound intensities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_gaussian.sparse_model import SparseGaussianField
from code_inr.inr_model import ImplicitOccupancyNetwork


OBSERVED_VIEW_IDS = (7, 19, 31)  # one-based: 30, 90, 150 degrees
SEEDS = (17, 23, 42)
N_VIEWS = 37
BATCH_SIZE = 4096
STEPS = 1000
LR = 1e-3
THRESHOLD = 0.5


def parse_cfg(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    return values


def patient_data(read_file, names: set[str], patient: str):
    root = f"FeEcho4D/FeEcho4D_Annotated/{patient}/"
    cfg_name = f"{root}{patient}_info.cfg"
    if cfg_name not in names:
        raise FileNotFoundError(cfg_name)
    metadata = parse_cfg(read_file(cfg_name).decode("utf-8", "replace"))
    ed_time = int(metadata["ed_time"])
    masks = []
    observed_digest = hashlib.sha256()
    for view_id in range(1, N_VIEWS + 1):
        name = f"{root}mask/{patient}_slice{view_id:03d}time{ed_time:03d}.png"
        raw = read_file(name)
        if view_id in OBSERVED_VIEW_IDS:
            observed_digest.update(raw)
        image = np.asarray(Image.open(__import__("io").BytesIO(raw)))
        values = set(np.unique(image).tolist())
        if not values <= {0, 127, 255}:
            raise ValueError(f"Unexpected mask values {values} in {name}")
        masks.append(image > 0)
    mask = np.stack(masks)
    if mask.shape != (N_VIEWS, 485, 464):
        # Actual dimensions are taken from the archive and must be consistent
        # across all views of a subject.
        if mask.ndim != 3 or mask.shape[0] != N_VIEWS:
            raise ValueError(f"Unexpected mask stack shape for {patient}: {mask.shape}")
    diagnosis = metadata.get("diagnosis", "unknown")
    return mask, diagnosis, ed_time, observed_digest.hexdigest()


def all_grid_coordinates(shape: tuple[int, int, int], view_ids: list[int], device):
    views, height, width = shape
    rows = torch.linspace(-1, 1, height, dtype=torch.float32)
    cols = torch.linspace(-1, 1, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    result = []
    for view_id in view_ids:
        angle = (view_id - 1) * 5.0
        a = angle / 90.0 - 1.0
        third = torch.full((len(xy), 1), a)
        result.append(torch.cat((xy, third), dim=1))
    return torch.cat(result).to(device)


def predict_gaussian(model, points: torch.Tensor, chunk_size: int = 16384):
    """Batch the same Gaussian occupancy equation with bounded temporary memory."""
    model.eval()
    means = model.means
    scales = F.softplus(model.raw_scales) + 1e-4
    opacities = torch.sigmoid(model.raw_opacities)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            p = points[start : start + chunk_size]
            diff = (p[:, None, :] - means[0, None, :, :]) / scales[0, None, :, :]
            weights = torch.exp(-0.5 * diff.square().sum(dim=-1))
            density = (weights * opacities[0, :, 0]).sum(dim=-1)
            outputs.append(-torch.expm1(-density).cpu())
    return torch.cat(outputs)


def dice_iou(pred: np.ndarray, truth: np.ndarray):
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    ps, ts = int(pred.sum()), int(truth.sum())
    dice = 1.0 if ps + ts == 0 else 2 * tp / (ps + ts)
    union = int(np.logical_or(pred, truth).sum())
    iou = 1.0 if union == 0 else tp / union
    accuracy = float(np.mean(pred == truth))
    return {"dice": float(dice), "iou": float(iou), "pixel_accuracy": accuracy}


def fit_one(method, points, labels, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = (ImplicitOccupancyNetwork(64, 4, 4) if method == "inr"
             else SparseGaussianField(256)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator().manual_seed(seed)
    model.train()
    start = time.perf_counter()
    last_loss = None
    for _ in range(STEPS):
        idx = torch.randint(len(points), (BATCH_SIZE,), generator=rng)
        pred = model(points[idx].to(device)).reshape(-1)
        loss = F.binary_cross_entropy(pred, labels[idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    model.eval()
    return model, elapsed, last_loss


def evaluate_model(model, method, mask, device):
    height, width = mask.shape[1:]
    held_ids = [i for i in range(1, N_VIEWS + 1) if i not in OBSERVED_VIEW_IDS]
    coords = all_grid_coordinates((N_VIEWS, height, width), held_ids, device)
    if method == "gaussian":
        probability = predict_gaussian(model, coords)
    else:
        output = []
        with torch.no_grad():
            for start in range(0, len(coords), 32768):
                output.append(model(coords[start:start + 32768]).reshape(-1).cpu())
        probability = torch.cat(output)
    predicted = (probability.numpy() >= THRESHOLD).reshape(len(held_ids), height, width)
    per_view = [dice_iou(p, mask[i - 1]) for p, i in zip(predicted, held_ids)]
    pooled = dice_iou(predicted, mask[np.array(held_ids) - 1])
    return {"heldout_views": held_ids, "pooled": pooled,
            "mean_view": {key: float(np.mean([x[key] for x in per_view]))
                          for key in ("dice", "iou", "pixel_accuracy")}}


def summarize(rows, output_dir):
    summary = {"subjects": len({r["subject"] for r in rows}), "methods": {}}
    good = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    for method in ("inr", "gaussian"):
        mr = [r for r in good if r["method"] == method]
        by_sub = defaultdict(list)
        for r in mr:
            by_sub[r["subject"]].append(r["metrics"]["pooled"]["dice"])
        subject_scores = {s: float(np.mean(v)) for s, v in by_sub.items()}
        vals = np.array(list(subject_scores.values()), dtype=float)
        summary["methods"][method] = {
            "successful_runs": len(mr), "subjects_scored": len(vals),
            "pooled_dice_subject_mean": float(vals.mean()) if len(vals) else None,
            "pooled_dice_subject_sd": float(vals.std()) if len(vals) else None,
            "pooled_dice_subject_median": float(np.median(vals)) if len(vals) else None,
            "pooled_dice_subject_iqr": [float(np.quantile(vals, .25)), float(np.quantile(vals, .75))] if len(vals) else None,
            "subject_scores": subject_scores,
        }
    inr = summary["methods"]["inr"]["subject_scores"]
    gau = summary["methods"]["gaussian"]["subject_scores"]
    common = sorted(set(inr) & set(gau))
    if len(common) >= 2:
        diff = np.array([gau[s] - inr[s] for s in common])
        summary["gaussian_minus_inr"] = {
            "n_subjects": len(common), "mean_difference": float(diff.mean()),
            "median_difference": float(np.median(diff)),
            "wilcoxon_two_sided": float(stats.wilcoxon(diff).pvalue),
        }
        rng = np.random.default_rng(2027)
        boots = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
        summary["gaussian_minus_inr"]["bootstrap_95_ci"] = [float(x) for x in np.quantile(boots, [.025, .975])]
    summary["failed_runs"] = len(failed)
    summary["failures"] = [{k: r.get(k) for k in ("subject", "method", "seed", "error")} for r in failed]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="dataset/FeEcho4D.zip",
                        help="Original ZIP or extracted directory containing FeEcho4D/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--subjects", nargs="*", help="Optional fixed patient IDs; omit for all annotated subjects")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(4)
    run_config = {"protocol": "benchmarks/feecho4d_external_protocol.md",
                  "archive": str(Path(args.archive).resolve()), "device": str(device),
                  "observed_view_ids": list(OBSERVED_VIEW_IDS), "seeds": list(SEEDS),
                  "steps": STEPS, "batch_size": BATCH_SIZE, "learning_rate": LR,
                  "threshold": THRESHOLD, "methods": ["inr", "gaussian"],
                  "target": "union of mask values 127 and 255",
                  "coordinate_system": "normalized image x/y and radial angle index; not physical XYZ"}
    rows = []
    archive_path = Path(args.archive)
    archive = None
    if archive_path.is_dir():
        data_root = archive_path
        file_names = [str(p.relative_to(data_root)).replace("\\", "/") for p in data_root.rglob("*") if p.is_file()]
        read_file = lambda name: (data_root / name).read_bytes()
        names = set(file_names)
    else:
        archive = zipfile.ZipFile(archive_path)
        names = set(archive.namelist())
        read_file = archive.read
    try:
        patients = sorted({m.group(1) for n in names
                           if (m := re.search(r"FeEcho4D_Annotated/(Patient\d{3})/", n))})
        if args.subjects:
            unknown = sorted(set(args.subjects) - set(patients))
            if unknown:
                raise ValueError(f"Unknown annotated subjects: {unknown}")
            patients = [p for p in patients if p in set(args.subjects)]
        run_config["subjects"] = patients
        source_zip = Path("dataset/FeEcho4D.zip")
        run_config["source_archive"] = str(source_zip.resolve()) if source_zip.is_file() else None
        run_config["source_archive_bytes"] = source_zip.stat().st_size if source_zip.is_file() else None
        (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        for i, patient in enumerate(patients, 1):
            print(f"[{i}/{len(patients)}] {patient}", flush=True)
            try:
                mask, diagnosis, ed_time, fingerprint = patient_data(read_file, names, patient)
                height, width = mask.shape[1:]
                observed = all_grid_coordinates(mask.shape, list(OBSERVED_VIEW_IDS), device)
                labels = torch.from_numpy(mask[np.array(OBSERVED_VIEW_IDS) - 1].astype(np.float32).reshape(-1))
                # Coordinates and observed labels are the only mask values passed into fitting.
                for method in ("inr", "gaussian"):
                    for seed in SEEDS:
                        row = {"subject": patient, "diagnosis": diagnosis, "ed_time": ed_time,
                               "method": method, "seed": seed,
                               "observation_sha256": fingerprint, "status": "failed"}
                        try:
                            model, seconds, last_loss = fit_one(method, observed, labels, args, device, seed)
                            metrics = evaluate_model(model, method, mask, device)
                            row.update(status="ok", fit_seconds=seconds, final_bce=last_loss, metrics=metrics)
                            del model
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            print(f"  FAIL {method} seed={seed}: {row['error']}", flush=True)
                        rows.append(row)
                        (output / "runs.jsonl").open("a").write(json.dumps(row, allow_nan=False) + "\n")
            except Exception as exc:
                print(f"  DATA FAIL: {type(exc).__name__}: {exc}", flush=True)
                for method in ("inr", "gaussian"):
                    for seed in SEEDS:
                        row = {"subject": patient, "method": method, "seed": seed,
                               "status": "failed", "error": f"data: {type(exc).__name__}: {exc}"}
                        rows.append(row)
                        (output / "runs.jsonl").open("a").write(json.dumps(row, allow_nan=False) + "\n")
    finally:
        if archive is not None:
            archive.close()
    summary = summarize(rows, output)
    print(json.dumps({k: v for k, v in summary.items() if k != "methods"}, indent=2))


if __name__ == "__main__":
    main()
