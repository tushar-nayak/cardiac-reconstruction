"""Exploratory NeRF pilot on synthetic views rendered from one MITEA volume.

This trains a density/radiance field from three synthetic orthographic views
of a development subject. It is a renderer smoke experiment, not the frozen
sparse-section benchmark: the view targets are generated from the full volume.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def encode(x: torch.Tensor, frequencies: int) -> torch.Tensor:
    parts = [x]
    for i in range(frequencies):
        scaled = x * (2**i) * math.pi
        parts.extend((torch.sin(scaled), torch.cos(scaled)))
    return torch.cat(parts, dim=-1)


class RadianceField(nn.Module):
    """Small coordinate MLP predicting volume density and grayscale radiance."""

    def __init__(self, position_frequencies=8, direction_frequencies=4, width=128,
                 depth=4, use_view_directions=True):
        super().__init__()
        self.position_frequencies = position_frequencies
        self.direction_frequencies = direction_frequencies
        self.use_view_directions = use_view_directions
        in_dim = 3 * (1 + 2 * position_frequencies)
        if use_view_directions:
            in_dim += 3 * (1 + 2 * direction_frequencies)
        layers = []
        for _ in range(depth):
            layers.extend((nn.Linear(in_dim, width), nn.ReLU(inplace=True)))
            in_dim = width
        self.trunk = nn.Sequential(*layers)
        self.density = nn.Linear(width, 1)
        color_in = width + (3 if use_view_directions else 0)
        self.color = nn.Sequential(nn.Linear(color_in, width // 2), nn.ReLU(inplace=True),
                                   nn.Linear(width // 2, 1), nn.Sigmoid())

    def forward(self, xyz, direction):
        features = [encode(xyz, self.position_frequencies)]
        if self.use_view_directions:
            features.append(encode(direction, self.direction_frequencies))
        h = self.trunk(torch.cat(features, dim=-1))
        sigma = F.softplus(self.density(h))
        color_input = torch.cat((h, direction), dim=-1) if self.use_view_directions else h
        color = self.color(color_input)
        return sigma.squeeze(-1), color.squeeze(-1)


def make_rays(shape_zyx, axis, sign, device):
    """Orthographic rays spanning the two axes perpendicular to ``axis``."""
    shape_xyz = tuple(reversed(shape_zyx))
    other = [i for i in range(3) if i != axis]
    axes = [torch.linspace(-1, 1, shape_xyz[i], device=device) for i in other]
    a, b = torch.meshgrid(*axes, indexing="ij")
    origins = torch.zeros((a.numel(), 3), device=device)
    origins[:, other[0]] = a.reshape(-1)
    origins[:, other[1]] = b.reshape(-1)
    origins[:, axis] = -1.0 if sign > 0 else 1.0
    directions = torch.zeros_like(origins)
    directions[:, axis] = float(sign)
    return origins, directions


@torch.no_grad()
def render_reference(image, mask, origins, directions, n_samples, chunk=256):
    device = image.device
    t = torch.linspace(0, 2, n_samples, device=device)
    delta = 2.0 / (n_samples - 1)
    rgb_out, opacity_out = [], []
    for start in range(0, origins.shape[0], chunk):
        o = origins[start:start + chunk]
        d = directions[start:start + chunk]
        points = o[:, None, :] + d[:, None, :] * t[None, :, None]
        grid = points.reshape(1, 1, -1, n_samples, 3)
        c = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True).reshape(-1, n_samples)
        m = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True).reshape(-1, n_samples)
        sigma = 18.0 * m.clamp(0, 1)
        alpha = 1.0 - torch.exp(-sigma * delta)
        trans = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]),
                                         (1.0 - alpha + 1e-8)[:, :-1]), dim=1), dim=1)
        weights = trans * alpha
        rgb_out.append((weights * c).sum(dim=1))
        opacity_out.append(weights.sum(dim=1))
    return torch.cat(rgb_out), torch.cat(opacity_out)


def volume_render(model, origins, directions, n_samples, chunk=128):
    device = origins.device
    t = torch.linspace(0, 2, n_samples, device=device)
    delta = 2.0 / (n_samples - 1)
    rgb_out, opacity_out = [], []
    for start in range(0, origins.shape[0], chunk):
        o, d = origins[start:start + chunk], directions[start:start + chunk]
        points = o[:, None, :] + d[:, None, :] * t[None, :, None]
        dirs = d[:, None, :].expand_as(points)
        sigma, color = model(points.reshape(-1, 3), dirs.reshape(-1, 3))
        sigma = sigma.reshape(-1, n_samples)
        color = color.reshape(-1, n_samples)
        alpha = 1.0 - torch.exp(-sigma * delta)
        trans = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]),
                                         (1.0 - alpha + 1e-8)[:, :-1]), dim=1), dim=1)
        weights = trans * alpha
        rgb_out.append((weights * color).sum(dim=1))
        opacity_out.append(weights.sum(dim=1))
    return torch.cat(rgb_out), torch.cat(opacity_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/mitea")
    parser.add_argument("--subject", default="MITEA_007", help="Use development subjects only")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--rays-per-step", type=int, default=256)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nerf_mitea_pilot")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot is configured for the available CUDA GPU")
    device = torch.device("cuda")

    manifest = json.loads((ROOT / "benchmarks/splits/mitea_cohort_v1.json").read_text())
    fold0 = set(manifest["splits"]["folds"][0])
    if args.subject not in fold0:
        raise ValueError(f"{args.subject} is not in development fold 0; refusing test-subject use")
    image_path = args.data_dir / "images" / f"{args.subject}_scan1_ED.nii.gz"
    label_path = args.data_dir / "labels" / f"{args.subject}_scan1_ED.nii.gz"
    image_nii, label_nii = nib.load(str(image_path)), nib.load(str(label_path))
    if image_nii.shape != label_nii.shape or not np.allclose(image_nii.affine, label_nii.affine):
        raise ValueError("Image/label geometry mismatch")
    image = image_nii.get_fdata(dtype=np.float32)
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)
    label = (label_nii.get_fdata(dtype=np.float32) > 0).astype(np.float32)
    # NIfTI xyz -> grid_sample's D,H,W = z,y,x; ray coordinates remain xyz.
    image = torch.from_numpy(np.transpose(image, (2, 1, 0)).copy())[None, None].to(device)
    label = torch.from_numpy(np.transpose(label, (2, 1, 0)).copy())[None, None].to(device)
    shape = tuple(image.shape[-3:])

    # Three positive-axis synthetic views are training views; reverse views are
    # held out for a simple novel-view check. Targets are volume-rendered from
    # the complete subject volume, so this is not sparse-plane reconstruction.
    train_views = []
    for axis in range(3):
        origins, directions = make_rays(shape, axis, +1, device)
        target_rgb, target_opacity = render_reference(
            image, label, origins, directions, args.samples)
        train_views.append((origins, directions, target_rgb, target_opacity))
    all_o = torch.cat([v[0] for v in train_views])
    all_d = torch.cat([v[1] for v in train_views])
    all_rgb = torch.cat([v[2] for v in train_views])
    all_opacity = torch.cat([v[3] for v in train_views])

    model = RadianceField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        ids = torch.randint(len(all_o), (args.rays_per_step,), device=device)
        pred_rgb, pred_opacity = volume_render(
            model, all_o[ids], all_d[ids], args.samples)
        loss_rgb = F.mse_loss(pred_rgb, all_rgb[ids])
        loss_opacity = F.mse_loss(pred_opacity, all_opacity[ids])
        loss = loss_rgb + 0.25 * loss_opacity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % 100 == 0:
            print(f"step {step + 1}/{args.steps}: loss={loss.item():.6f} "
                  f"rgb={loss_rgb.item():.6f} opacity={loss_opacity.item():.6f}", flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started

    model.eval()
    results = {"subject": args.subject, "split": "development_fold_0",
               "experiment": "synthetic_orthographic_volume_rendering",
               "note": "Not the sparse-plane reconstruction benchmark; target views are generated from the full 3D volume.",
               "train_directions": ["+x", "+y", "+z"],
               "heldout_directions": ["-x", "-y", "-z"], "steps": args.steps,
               "samples_per_ray": args.samples, "rays_per_step": args.rays_per_step,
               "seed": args.seed, "training_seconds": train_seconds,
               "final_train_loss": losses[-1], "initial_train_loss": losses[0],
               "views": {}}
    with torch.no_grad():
        for axis in range(3):
            for sign in (+1, -1):
                origins, directions = make_rays(shape, axis, sign, device)
                gt_rgb, gt_opacity = render_reference(
                    image, label, origins, directions, args.samples)
                pred_rgb, pred_opacity = volume_render(
                    model, origins, directions, args.samples)
                mse = float(F.mse_loss(pred_rgb, gt_rgb).cpu())
                op_mse = float(F.mse_loss(pred_opacity, gt_opacity).cpu())
                results["views"][f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"] = {
                    "rgb_mse": mse, "rgb_psnr_db": -10 * math.log10(max(mse, 1e-12)),
                    "opacity_mse": op_mse,
                }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "args": vars(args)}, args.output / "model.pt")
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    np.save(args.output / "loss.npy", np.asarray(losses, dtype=np.float32))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
