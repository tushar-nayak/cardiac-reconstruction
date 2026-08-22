import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
import tqdm

from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.utils.coord_utils import sample_volume_at_points
from src_code.src.occupancy import evaluate_gaussian_occupancy

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


def sample_points(volume, affine, num_samples, device):
    occ_indices = torch.nonzero(volume)
    num_pos = num_samples // 3
    num_cavity = num_samples // 3
    num_uniform = num_samples - num_pos - num_cavity

    selected = occ_indices[torch.randint(0, len(occ_indices), (num_pos,))]
    pts_pos_img = selected[:, [2, 1, 0]].float().to(device)
    pts_pos_img = pts_pos_img + torch.randn_like(pts_pos_img) * 1.5

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
    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MITEADataset(args.data_dir, split=args.split)
    sample = dataset[args.sample_index]
    volume = sample["occupancy_ed"].to(device)
    affine = sample["affine"].to(device)

    means, log_scales, raw_opacities = initialize_params(volume, affine, args.num_gaussians, device)
    optimizer = torch.optim.Adam([means, log_scales, raw_opacities], lr=args.lr)

    pbar = tqdm.trange(args.steps, desc="Fitting subject")
    for step in pbar:
        points, sample_weights = sample_points(volume, affine, args.num_samples, device)
        params = gaussian_params(means, log_scales, raw_opacities)
        pred = evaluate_gaussian_occupancy(
            points, params["means"], params["scales"], params["opacities"]
        )
        target = sample_volume_at_points([volume], points, affine.unsqueeze(0))
        loss_occ = F.binary_cross_entropy(pred, target, weight=sample_weights)
        loss_scale = torch.mean(F.softplus(log_scales))
        loss_opacity = torch.mean(torch.sigmoid(raw_opacities))
        loss = loss_occ + args.scale_loss_weight * loss_scale + args.opacity_loss_weight * loss_opacity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "occ": f"{loss_occ.item():.4f}"})

    params = gaussian_params(means, log_scales, raw_opacities)
    torch.save({
        "type": "subject_gaussian_fit",
        "split": args.split,
        "sample_index": args.sample_index,
        "num_gaussians": args.num_gaussians,
        "means": params["means"].detach().cpu(),
        "scales": params["scales"].detach().cpu(),
        "opacities": params["opacities"].detach().cpu(),
        "config": vars(args),
        "seed": args.seed,
    }, os.path.join(args.run_dir, "subject_fit.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default="runs/subject_fit_v01")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-gaussians", type=int, default=1800)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--scale-loss-weight", type=float, default=0.002)
    parser.add_argument("--opacity-loss-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    fit(parser.parse_args())
