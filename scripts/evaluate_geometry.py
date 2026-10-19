"""Evaluate a saved occupancy tensor against a NIfTI target without retraining."""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_common.geometry import geometry_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="Saved (z,y,x) probability tensor (.pt)")
    parser.add_argument("--target", required=True, help="Ground-truth NIfTI label volume")
    parser.add_argument("--output", required=True, help="New JSON file; existing files are rejected")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-stride", type=int, default=1, help="Must match the saved tensor's stride")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.eval_stride < 1:
        parser.error("threshold must be between 0 and 1 and eval stride must be positive")
    prediction = torch.load(args.prediction, map_location="cpu", weights_only=True)
    if not isinstance(prediction, torch.Tensor) or not torch.isfinite(prediction).all():
        raise ValueError("prediction must be a finite probability tensor")
    if (prediction < 0).any() or (prediction > 1).any():
        raise ValueError("prediction values must be probabilities in [0,1]")
    image = nib.load(args.target)
    stride = args.eval_stride
    target = np.transpose(image.get_fdata() > 0, (2, 1, 0))[::stride, ::stride, ::stride]
    affine = image.affine @ np.diag([stride, stride, stride, 1])
    metrics = geometry_comparison(prediction.detach().numpy() >= args.threshold, target, affine)
    report = {"prediction_path": args.prediction, "target_path": args.target,
              "threshold": args.threshold, "eval_stride": stride, "geometry": metrics}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Delta K: {metrics['delta_integrated_gaussian_curvature']} ({metrics['delta_k_status']})")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
