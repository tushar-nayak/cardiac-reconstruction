"""Run the frozen synthetic-view NeRF ablation on MITEA development subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_nerf_mitea import RadianceField, make_rays, render_reference, volume_render

PROTOCOL = ROOT / "benchmarks/nerf_mitea_ablation_v1.md"
SEEDS = (17, 23, 42)
SCREEN_SUBJECTS = ("MITEA_007", "MITEA_008", "MITEA_009", "MITEA_011",
                   "MITEA_016", "MITEA_030", "MITEA_042", "MITEA_045")
CONFIRM_SUBJECTS = ("MITEA_048", "MITEA_051", "MITEA_064", "MITEA_074",
                    "MITEA_083", "MITEA_091", "MITEA_094", "MITEA_096",
                    "MITEA_116", "MITEA_123")
BASE = {"name": "baseline", "views": [2, 1, 0], "position_frequencies": 8,
        "direction_frequencies": 4, "use_view_directions": True, "width": 128,
        "depth": 4, "samples": 64, "opacity_weight": 0.25, "lr": 5e-4,
        "steps": 1200, "rays_per_step": 256}


def variants():
    result = [dict(BASE)]
    changes = [
        ("views1_z", {"views": [2]}), ("views2_xz", {"views": [0, 2]}),
        ("position_pe4", {"position_frequencies": 4}),
        ("position_pe12", {"position_frequencies": 12}),
        ("viewdir_off", {"use_view_directions": False}),
        ("viewdir_pe2", {"direction_frequencies": 2}),
        ("capacity_w64_d4", {"width": 64, "depth": 4}),
        ("capacity_w192_d6", {"width": 192, "depth": 6}),
        ("samples32", {"samples": 32}), ("samples128", {"samples": 128}),
        ("opacity_weight0", {"opacity_weight": 0.0}),
        ("opacity_weight1", {"opacity_weight": 1.0}),
        ("lr2p5e4", {"lr": 2.5e-4}), ("lr1e3", {"lr": 1e-3}),
        ("steps600", {"steps": 600}), ("steps2400", {"steps": 2400}),
    ]
    for name, change in changes:
        result.append({**BASE, **change, "name": name})
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_subject(data_dir, subject, device):
    image_path = data_dir / "images" / f"{subject}_scan1_ED.nii.gz"
    label_path = data_dir / "labels" / f"{subject}_scan1_ED.nii.gz"
    image_nii, label_nii = nib.load(str(image_path)), nib.load(str(label_path))
    if image_nii.shape != label_nii.shape or not np.allclose(image_nii.affine, label_nii.affine):
        raise ValueError(f"Image/label geometry mismatch for {subject}")
    image = image_nii.get_fdata(dtype=np.float32)
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)
    label = (label_nii.get_fdata(dtype=np.float32) > 0).astype(np.float32)
    image = torch.from_numpy(np.transpose(image, (2, 1, 0)).copy())[None, None].to(device)
    label = torch.from_numpy(np.transpose(label, (2, 1, 0)).copy())[None, None].to(device)
    shape = tuple(image.shape[-3:])
    views = {}
    for axis, name in enumerate("xyz"):
        for sign in (+1, -1):
            key = f"{'+' if sign > 0 else '-'}{name}"
            origins, directions = make_rays(shape, axis, sign, device)
            rgb, opacity = render_reference(image, label, origins, directions, 128)
            views[key] = (origins, directions, rgb, opacity)
    return views


def binary_dice(pred, target):
    pred, target = pred.bool(), target.bool()
    denom = int(pred.sum() + target.sum())
    if denom == 0:
        return 1.0
    return float(2 * (pred & target).sum().item() / denom)


def train_one(config, views, subject, seed, phase, device, output_dir):
    run_dir = output_dir / phase / config["name"] / subject / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        saved = json.loads(metrics_path.read_text())
        expected = {"config": config, "subject": subject, "seed": seed, "phase": phase}
        if any(saved.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Existing run configuration mismatch: {metrics_path}")
        return saved

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RadianceField(
        position_frequencies=config["position_frequencies"],
        direction_frequencies=config["direction_frequencies"], width=config["width"],
        depth=config["depth"], use_view_directions=config["use_view_directions"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_keys = [f"+{'xyz'[axis]}" for axis in config["views"]]
    train_sets = [views[key] for key in train_keys]
    model.train()
    loss_trace = []
    started = time.perf_counter()
    for step in range(config["steps"]):
        chosen_view = torch.randint(len(train_sets), (config["rays_per_step"],), device=device)
        chunks = [[], [], []]
        for view_index, data in enumerate(train_sets):
            mask = chosen_view == view_index
            count = int(mask.sum())
            if count == 0:
                continue
            origins, directions, rgb, opacity = data
            ray_ids = torch.randint(len(origins), (count,), device=device)
            chunks[0].append((mask, origins[ray_ids]))
            chunks[1].append((mask, directions[ray_ids]))
            chunks[2].append((mask, torch.stack((rgb[ray_ids], opacity[ray_ids]), dim=-1)))
        batch_o = torch.empty((config["rays_per_step"], 3), device=device)
        batch_d = torch.empty_like(batch_o)
        batch_t = torch.empty((config["rays_per_step"], 2), device=device)
        for mask, values in chunks[0]: batch_o[mask] = values
        for mask, values in chunks[1]: batch_d[mask] = values
        for mask, values in chunks[2]: batch_t[mask] = values
        pred_rgb, pred_opacity = volume_render(
            model, batch_o, batch_d, config["samples"], chunk=64)
        rgb_loss = F.mse_loss(pred_rgb, batch_t[:, 0])
        opacity_loss = F.mse_loss(pred_opacity, batch_t[:, 1])
        loss = rgb_loss + config["opacity_weight"] * opacity_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss {loss.item()} at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_trace.append(float(loss.detach()))
        if step == 0 or (step + 1) % 400 == 0:
            print(f"{phase} {config['name']} {subject} seed={seed} "
                  f"step={step + 1}/{config['steps']} loss={loss.item():.6f}", flush=True)
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    model.eval()
    heldout = {}
    with torch.no_grad():
        for key in ("-x", "-y", "-z"):
            origins, directions, target_rgb, target_opacity = views[key]
            pred_rgb, pred_opacity = volume_render(
                model, origins, directions, config["samples"], chunk=128)
            mse = float(F.mse_loss(pred_rgb, target_rgb))
            heldout[key] = {
                "opacity_dice": binary_dice(pred_opacity >= 0.5, target_opacity >= 0.5),
                "opacity_mse": float(F.mse_loss(pred_opacity, target_opacity)),
                "rgb_mse": mse, "rgb_psnr_db": -10 * math.log10(max(mse, 1e-12)),
            }
    record = {
        "phase": phase, "subject": subject, "seed": seed, "config": config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": train_seconds, "initial_loss": loss_trace[0],
        "final_loss": loss_trace[-1], "heldout": heldout,
        "mean_heldout_opacity_dice": float(np.mean([v["opacity_dice"] for v in heldout.values()])),
        "mean_heldout_psnr_db": float(np.mean([v["rgb_psnr_db"] for v in heldout.values()])),
        "mean_heldout_opacity_mse": float(np.mean([v["opacity_mse"] for v in heldout.values()])),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if phase == "confirmation":
        torch.save({"model": model.cpu().state_dict(), "config": config,
                    "subject": subject, "seed": seed}, run_dir / "model.pt")
    atomic_json(metrics_path, record)
    del model, optimizer
    torch.cuda.empty_cache()
    return record


def complete_records(root, phase, configs, subjects):
    records = []
    for config in configs:
        for subject in subjects:
            for seed in SEEDS:
                path = root / phase / config["name"] / subject / f"seed_{seed}" / "metrics.json"
                if not path.is_file():
                    return None
                records.append(json.loads(path.read_text()))
    return records


def screen_score(records):
    by_config = {}
    for record in records:
        by_config.setdefault(record["config"]["name"], []).append(record)
    scored = []
    for name, rows in by_config.items():
        by_subject = {}
        for row in rows:
            by_subject.setdefault(row["subject"], []).append(row)
        dice = [np.mean([r["mean_heldout_opacity_dice"] for r in group])
                for group in by_subject.values()]
        psnr = [np.mean([r["mean_heldout_psnr_db"] for r in group])
                for group in by_subject.values()]
        scored.append({"name": name, "mean_opacity_dice": float(np.mean(dice)),
                       "sd_opacity_dice": float(np.std(dice, ddof=1)),
                       "mean_psnr_db": float(np.mean(psnr)),
                       "parameter_count": rows[0]["parameter_count"]})
    return sorted(scored, key=lambda x: (-x["mean_opacity_dice"], -x["mean_psnr_db"], x["parameter_count"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/mitea")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/nerf_mitea_ablation_v1")
    parser.add_argument("--phase", choices=("all", "screening", "confirmation"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen ablation")
    device = torch.device("cuda")
    manifest = json.loads((ROOT / "benchmarks/splits/mitea_cohort_v1.json").read_text())
    fold0 = manifest["splits"]["folds"][0]
    if tuple(fold0[:8]) != SCREEN_SUBJECTS or tuple(fold0[8:18]) != CONFIRM_SUBJECTS:
        raise RuntimeError("Fold-0 subject order differs from the frozen protocol")
    configs = variants()
    study_config = {"protocol_sha256": sha256(PROTOCOL),
                    "runner_sha256": sha256(Path(__file__)),
                    "model_sha256": sha256(ROOT / "scripts/train_nerf_mitea.py"),
                    "manifest_sha256": sha256(ROOT / "benchmarks/splits/mitea_cohort_v1.json"),
                    "seed_list": SEEDS, "screen_subjects": SCREEN_SUBJECTS,
                    "confirm_subjects": CONFIRM_SUBJECTS, "variants": configs}
    config_path = args.output_dir / "study_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"{args.output_dir} exists; pass --resume to continue")
        if json.loads(config_path.read_text()) != study_config:
            raise RuntimeError("Study configuration/source hashes changed; refusing resume")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, study_config)

    if args.phase in ("all", "screening"):
        for subject in SCREEN_SUBJECTS:
            views = load_subject(args.data_dir, subject, device)
            for config in configs:
                for seed in SEEDS:
                    record_path = args.output_dir / "screening" / config["name"] / subject / f"seed_{seed}" / "metrics.json"
                    if args.resume and record_path.is_file():
                        continue
                    record = train_one(config, views, subject, seed, "screening", device, args.output_dir)
                    print(f"screen complete {config['name']} {subject} seed={seed}: "
                          f"Dice={record['mean_heldout_opacity_dice']:.4f} "
                          f"PSNR={record['mean_heldout_psnr_db']:.2f}dB", flush=True)
            del views
            torch.cuda.empty_cache()
        records = complete_records(args.output_dir, "screening", configs, SCREEN_SUBJECTS)
        if records is None:
            raise RuntimeError("Screening is incomplete; no configuration may be selected")
        ranking = screen_score(records)
        selected = [row["name"] for row in ranking[:2]]
        atomic_json(args.output_dir / "screening_ranking.json",
                    {"selection_rule": "top two by mean subject-level heldout opacity Dice; tie-break PSNR then parameters",
                     "ranking": ranking, "selected": selected})
        print(f"Screening selection frozen: {selected}", flush=True)

    if args.phase in ("all", "confirmation"):
        ranking_path = args.output_dir / "screening_ranking.json"
        if not ranking_path.is_file():
            raise FileNotFoundError("Complete screening first; missing screening_ranking.json")
        selected = json.loads(ranking_path.read_text())["selected"]
        confirm_configs = [c for c in configs if c["name"] == "baseline" or c["name"] in selected]
        for subject in CONFIRM_SUBJECTS:
            views = load_subject(args.data_dir, subject, device)
            for config in confirm_configs:
                for seed in SEEDS:
                    record_path = args.output_dir / "confirmation" / config["name"] / subject / f"seed_{seed}" / "metrics.json"
                    if args.resume and record_path.is_file():
                        continue
                    record = train_one(config, views, subject, seed, "confirmation", device, args.output_dir)
                    print(f"confirm complete {config['name']} {subject} seed={seed}: "
                          f"Dice={record['mean_heldout_opacity_dice']:.4f} "
                          f"PSNR={record['mean_heldout_psnr_db']:.2f}dB", flush=True)
            del views
            torch.cuda.empty_cache()
        confirmation = complete_records(args.output_dir, "confirmation", confirm_configs,
                                        CONFIRM_SUBJECTS)
        if confirmation is not None:
            atomic_json(args.output_dir / "confirmation_summary.json",
                        {"selected": selected, "scores": screen_score(confirmation),
                         "records": len(confirmation),
                         "warning": "Development-fold synthetic rendering only; no sparse-plane or clinical inference."})
            print("Confirmation phase complete.", flush=True)


if __name__ == "__main__":
    main()
