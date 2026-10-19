import argparse
import json
import os
import sys
import time
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
import tqdm

from code_gaussian.src.data_loaders.mitea_loader import MITEADataset
from code_gaussian.src.utils.coord_utils import sample_volume_at_points
from code_gaussian.src.occupancy import evaluate_gaussian_occupancy
from code_gaussian.src.models.subject_field import SubjectGaussianField
from code_gaussian.src.metrics import binary_metrics
from cardiac_common.sparse import evaluate_grid, reconstruction_metrics

def image_to_world(points_img, affine, device):
    points_h = torch.cat([points_img, torch.ones(points_img.shape[0], 1, device=device)], dim=-1)
    return (points_h @ affine.to(device).T)[:, :3]


def initialize_params(volume, affine, num_gaussians, device):
    occ_indices = torch.nonzero(volume)
    if len(occ_indices) == 0:
        raise ValueError("Selected sample has no occupied voxels.")

    selected = occ_indices[torch.randint(0, len(occ_indices), (num_gaussians,))]
    pts_img = selected[:, [2, 1, 0]].float().to(device)
    means = image_to_world(pts_img, affine, device)

    means = torch.nn.Parameter(means + torch.randn_like(means) * 0.4)
    log_scales = torch.nn.Parameter(torch.full((num_gaussians, 3), 0.9, device=device))
    raw_opacities = torch.nn.Parameter(torch.full((num_gaussians, 1), 1.4, device=device))
    return means, log_scales, raw_opacities


def gaussian_params(means, log_scales, raw_opacities):
    return {
        "means": means.unsqueeze(0),
        "scales": F.softplus(log_scales).unsqueeze(0) + 0.35,
        "opacities": torch.sigmoid(raw_opacities).unsqueeze(0),
    }


def sample_points(volume, affine, num_samples, device, recipe="stabilized"):
    """May uses half foreground/half uniform; stabilized adds centroid samples."""
    if recipe not in {"may2026", "stabilized"}:
        raise ValueError("recipe must be may2026 or stabilized")
    occ_indices = torch.nonzero(volume)
    num_pos = num_samples // (2 if recipe == "may2026" else 3)
    num_cavity = 0 if recipe == "may2026" else num_samples // 3
    num_uniform = num_samples - num_pos - num_cavity

    selected = occ_indices[torch.randint(0, len(occ_indices), (num_pos,))]
    pts_pos_img = selected[:, [2, 1, 0]].float().to(device)
    pts_pos_img = pts_pos_img + torch.randn_like(pts_pos_img) * 1.5

    pts_cavity_img = torch.empty(0, 3, device=device)
    if num_cavity:
        centroid = occ_indices.float().mean(dim=0)[[2, 1, 0]].to(device)
        spread = occ_indices.float().std(dim=0, unbiased=False)[[2, 1, 0]].to(device).clamp_min(4.0)
        pts_cavity_img = centroid + torch.randn(num_cavity, 3, device=device) * spread * 0.15

    D, H, W = volume.shape
    pts_uniform_img = torch.stack([
        torch.rand(num_uniform, device=device) * (W - 1),
        torch.rand(num_uniform, device=device) * (H - 1),
        torch.rand(num_uniform, device=device) * (D - 1),
    ], dim=-1)

    points_img = torch.cat([pts_pos_img, pts_cavity_img, pts_uniform_img], dim=0)
    points_world = image_to_world(points_img, affine, device)
    sample_weights = torch.cat([
        torch.full((num_pos,), 1.5, device=device),
        torch.full((num_cavity,), 8.0, device=device),
        torch.full((num_uniform,), 2.0, device=device),
    ]).unsqueeze(0)
    return points_world.unsqueeze(0), sample_weights


def fit(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    args.split_policy = args.split_policy or ("may2026" if args.recipe == "may2026" else "shuffled")
    if os.path.exists(os.path.join(args.run_dir, "subject_fit.pth")):
        raise FileExistsError("Choose a new run directory; a subject checkpoint already exists here")
    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = MITEADataset(args.data_dir, split=args.split, split_seed=args.split_seed,
                          split_policy=args.split_policy)
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError(f"sample index must be in [0, {len(dataset) - 1}]")
    sample = dataset[args.sample_index]
    volume = sample["occupancy_ed"].to(device)
    affine = sample["affine"].to(device)

    means, log_scales, raw_opacities = initialize_params(volume, affine, args.num_gaussians, device)
    optimizer = torch.optim.Adam([means, log_scales, raw_opacities], lr=args.lr)
    print(f"{sample['subject']} · {args.recipe} · full 3D label supervision · {args.num_gaussians} Gaussians", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    pbar = tqdm.trange(args.steps, desc="Fitting subject", disable=not sys.stdout.isatty())
    for step in pbar:
        points, sample_weights = sample_points(volume, affine, args.num_samples, device, args.recipe)
        params = gaussian_params(means, log_scales, raw_opacities)
        pred = evaluate_gaussian_occupancy(
            points, params["means"], params["scales"], params["opacities"]
        )
        target = sample_volume_at_points([volume], points, affine.unsqueeze(0))
        loss_occ = F.binary_cross_entropy(pred, target, weight=sample_weights)
        loss_scale = torch.mean(F.softplus(log_scales))
        loss_opacity = torch.mean(torch.sigmoid(raw_opacities))
        loss = loss_occ + args.scale_loss_weight * loss_scale + args.opacity_loss_weight * loss_opacity
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step + 1}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "occ": f"{loss_occ.item():.4f}"})
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"  step {step + 1}/{args.steps}: weighted BCE={loss_occ.item():.6f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - started
    params = gaussian_params(means, log_scales, raw_opacities)
    torch.save({
        "type": "subject_gaussian_fit",
        "subject": sample["subject"], "supervision": "full_3d_labels",
        "affine": sample["affine"], "volume_shape_zyx": list(volume.shape),
        "split": args.split,
        "sample_index": args.sample_index,
        "num_gaussians": args.num_gaussians,
        "means": params["means"].detach().cpu(),
        "scales": params["scales"].detach().cpu(),
        "opacities": params["opacities"].detach().cpu(),
        "config": vars(args),
        "seed": args.seed,
    }, os.path.join(args.run_dir, "subject_fit.pth"))
    if args.evaluate:
        evaluate_reference(args, sample, params, device, fit_seconds)


def evaluate_reference(args, sample, params, device, fit_seconds):
    """Native-grid scores plus the historical sampled-point diagnostic."""
    from code_gaussian.scripts.evaluate_3d import sample_metric_points

    target = sample["occupancy_ed"].cpu()
    field = SubjectGaussianField(params, target.shape, sample["affine"]).to(device)
    started = time.perf_counter()
    probability = evaluate_grid(field, target.shape, device, args.eval_chunk_size)
    metrics = reconstruction_metrics(probability, target, torch.ones_like(target, dtype=torch.bool),
                                     sample["affine"], args.threshold)
    evaluation_seconds = time.perf_counter() - started
    torch.manual_seed(args.seed + 1)
    points, _ = sample_metric_points(target.numpy(), sample["affine"], args.metric_points, device)
    with torch.no_grad():
        sampled_pred = evaluate_gaussian_occupancy(points.unsqueeze(0), **params).cpu().numpy()
        sampled_target = sample_volume_at_points([target], points.unsqueeze(0),
                                                sample["affine"].unsqueeze(0)).cpu().numpy()
    report = {"protocol": "full_label_subject_fit_" + args.recipe,
              "subject": sample["subject"], "scan": 1, "phase": "ED", "seed": args.seed,
              "supervision": "full_3d_labels", "split": args.split, "split_policy": args.split_policy,
              "sample_index": args.sample_index, "num_gaussians": args.num_gaussians,
              "steps": args.steps, "threshold": args.threshold, "eval_stride": 1,
              "volume_shape_zyx": list(target.shape), "fit_seconds": fit_seconds,
              "evaluation_seconds": evaluation_seconds, "metrics": metrics,
              "historical_sampled_diagnostic": {
                  "sampling": "one_third_foreground_one_third_centroid_one_third_uniform",
                  "points": args.metric_points,
                  "metrics": binary_metrics(sampled_pred, sampled_target, args.threshold)},
              "config": vars(args)}
    Path(args.run_dir, "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(probability, Path(args.run_dir, "predicted_occupancy.pt"))
    print(f"Native-grid Dice={metrics['whole_grid']['dice']:.4f}, IoU={metrics['whole_grid']['iou']:.4f}", flush=True)
    print(f"Historical sampled IoU={report['historical_sampled_diagnostic']['metrics']['iou']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--recipe", choices=("may2026", "stabilized"), default="may2026")
    parser.add_argument("--run-dir", type=str, default="runs/subject_fit_v01")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split-policy", choices=("may2026", "shuffled"), default=None)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--num-gaussians", type=int, default=1800)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--scale-loss-weight", type=float, default=0.002)
    parser.add_argument("--opacity-loss-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--evaluate", action="store_true", help="Save native-resolution occupancy and geometry metrics")
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument("--metric-points", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.num_gaussians, args.num_samples, args.eval_chunk_size,
           args.metric_points, args.cpu_threads, args.lr) <= 0 or not 0 < args.threshold < 1:
        parser.error("counts and learning rate must be positive; threshold must be between 0 and 1")
    fit(args)
