"""Fit an implicit occupancy network from a small set of labeled 2D planes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code_gaussian.src.data_loaders.mitea_loader import MITEADataset
from code_gaussian.src.metrics import binary_metrics
from inr_model import ImplicitOccupancyNetwork


def normalized_coordinates(indices: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Map (z,y,x) voxel indices to the INR's normalized (x,y,z) space."""
    depth, height, width = shape
    xyz = indices[:, [2, 1, 0]].float()
    sizes = xyz.new_tensor([width, height, depth])
    return 2.0 * xyz / (sizes - 1.0).clamp_min(1.0) - 1.0


def choose_slices(volume: torch.Tensor, count: int, min_pixels: int) -> list[int]:
    counts = volume.sum(dim=(1, 2))
    valid = torch.where(counts >= min_pixels)[0]
    if valid.numel() < count:
        raise ValueError(
            f"Only {valid.numel()} axial slices have >= {min_pixels} foreground voxels; "
            f"cannot select {count}. Lower --min-slice-pixels or --num-slices."
        )
    positions = torch.linspace(0, valid.numel() - 1, count).round().long()
    return valid[positions].tolist()


def make_plane_samples(volume: torch.Tensor, slice_indices: list[int], device: torch.device):
    d, h, w = volume.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    point_sets, target_sets = [], []
    for z in slice_indices:
        indices = torch.stack((torch.full_like(xx, z), yy, xx), dim=-1).reshape(-1, 3)
        point_sets.append(normalized_coordinates(indices, (d, h, w)))
        target_sets.append(volume[z].reshape(-1, 1))
    return torch.cat(point_sets).to(device), torch.cat(target_sets).to(device)


def sample_full_volume(model, shape: tuple[int, int, int], device, chunk_size: int):
    d, h, w = shape
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, d * h * w, chunk_size):
            linear = torch.arange(start, min(start + chunk_size, d * h * w), device=device)
            z = torch.div(linear, h * w, rounding_mode="floor")
            y = torch.div(linear % (h * w), w, rounding_mode="floor")
            x = linear % w
            indices = torch.stack((z, y, x), dim=-1)
            coords = normalized_coordinates(indices, shape)
            predictions.append(model(coords).squeeze(-1).cpu())
    return torch.cat(predictions).reshape(shape)


def fit(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = MITEADataset(args.data_dir, split=args.split, split_seed=args.split_seed)
    sample = dataset[args.sample_index]
    target = sample["occupancy_ed"].float()
    selected = choose_slices(target, args.num_slices, args.min_slice_pixels)
    points, labels = make_plane_samples(target, selected, device)

    model = ImplicitOccupancyNetwork(
        hidden_dim=args.hidden_dim, layers=args.layers, frequencies=args.frequencies
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        batch = torch.randint(points.shape[0], (min(args.batch_size, points.shape[0]),), device=device)
        pred = model(points[batch]).squeeze(-1)
        loss = F.binary_cross_entropy(pred, labels[batch, 0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, args.steps // 10) == 0 or step == 0:
            print(f"step {step + 1}/{args.steps}: slice BCE={loss.item():.5f}")

    occupancy = sample_full_volume(model, tuple(target.shape), device, args.grid_chunk_size)
    predicted = occupancy >= args.threshold
    metrics = binary_metrics(predicted, target.bool())
    output_dir = Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "type": "subject_sparse_slice_inr",
        "model_config": {"hidden_dim": args.hidden_dim, "layers": args.layers, "frequencies": args.frequencies},
        "model_state_dict": model.cpu().state_dict(),
        "subject": sample["subject"],
        "split": args.split,
        "sample_index": args.sample_index,
        "slice_indices": selected,
        "volume_shape": list(target.shape),
        "config": vars(args),
    }
    torch.save(checkpoint, output_dir / "inr_checkpoint.pth")
    torch.save(occupancy, output_dir / "predicted_occupancy.pt")
    summary = {
        "method": "coordinate_mlp_sparse_slice_supervision",
        "subject": sample["subject"], "split": args.split,
        "slice_indices": selected, "threshold": args.threshold,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Extracted MITEA root containing images/ and labels/")
    parser.add_argument("--run-dir", default="runs/inr_subject_v01")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--min-slice-pixels", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--frequencies", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grid-chunk-size", type=int, default=65536)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
