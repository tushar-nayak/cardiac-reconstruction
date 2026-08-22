"""Backward-compatible entry point for 3D reconstruction evaluation."""

import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src_code.scripts.evaluate_3d import evaluate_3d


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Gaussian cardiac reconstruction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=1000)
    parser.add_argument("--slice-resolution", type=int, default=128)
    parser.add_argument("--metric-points", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=0.5)
    evaluate_3d(parser.parse_args())


if __name__ == "__main__":
    main()
