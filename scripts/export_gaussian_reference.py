"""Export Previous Stable Recipe and refined full-label fits for the website."""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_gaussian_preview import export_mask, write_json
from scripts.benchmark_sparse import summarize


def load_runs(paths):
    rows = {}
    for path in paths:
        directory = Path(path)
        row = json.loads((directory / "metrics.json").read_text())
        if row["supervision"] != "full_3d_labels" or row["config"]["recipe"] != "may2026":
            raise ValueError("reference results require full-label May recipe runs")
        key = (row["subject"], row["seed"])
        if key in rows:
            raise ValueError(f"duplicate subject/seed: {key}")
        rows[key] = (directory, row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", default=["runs/gaussian_may2026_reference_v01",
                                                        "runs/gaussian_may2026_MITEA071_v01"])
    parser.add_argument("--candidate", nargs="+", default=["runs/gaussian_may2026_refined_v01",
                                                          "runs/gaussian_may2026_MITEA071_refined_v01"])
    parser.add_argument("--data-dir", default="dataset/mitea")
    parser.add_argument("--output", default="docs/assets/gaussian-reference")
    args = parser.parse_args()
    torch.set_num_threads(4)
    before, after = load_runs(args.control), load_runs(args.candidate)
    if not before or before.keys() != after.keys():
        raise ValueError("reference fits must have the same nonempty subject/seed pairs")
    for key in before:
        for field in ("protocol", "subject", "seed", "split", "split_policy", "sample_index",
                      "num_gaussians", "volume_shape_zyx", "threshold", "eval_stride"):
            if before[key][1][field] != after[key][1][field]:
                raise ValueError(f"unmatched {field}: {key}")
        if before[key][1]["steps"] != 600 or after[key][1]["steps"] != 1800:
            raise ValueError("website results compare the 600-step recipe with 1800-step refinement")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    subjects = {}
    for (subject, seed), (directory, result) in before.items():
        image = nib.load(Path(args.data_dir) / "labels" / f"{subject}_scan1_ED.nii.gz")
        target = np.transpose(image.get_fdata() > 0, (2, 1, 0))
        if subject not in subjects:
            subjects[subject] = {"shape": list(target.shape), "planes": [], "seeds": {},
                                 "target": export_mask(output, f"{subject}_target", target, image.affine)}
        variants = {}
        for name, (run_dir, row) in (("control", before[(subject, seed)]),
                                     ("candidate", after[(subject, seed)])):
            probability = torch.load(run_dir / "predicted_occupancy.pt", map_location="cpu", weights_only=True)
            if list(probability.shape) != list(target.shape) or not torch.isfinite(probability).all():
                raise ValueError(f"invalid prediction in {run_dir}")
            variants[name] = {
                **export_mask(output, f"{subject}_{seed}_{name}", probability.numpy() >= row["threshold"], image.affine),
                "metrics": row["metrics"], "fit_seconds": row["fit_seconds"],
                "sampled_iou": row["historical_sampled_diagnostic"]["metrics"]["iou"],
            }
        subjects[subject]["seeds"][str(seed)] = variants
        print(f"Exported {subject} seed {seed}", flush=True)
    summary = lambda rows: summarize([{**row, "method": "gaussian"} for _, row in rows.values()])["gaussian"]
    write_json(output / "index.json", {"subjects": subjects, "control_summary": summary(before),
               "candidate_summary": summary(after), "protocol": {"supervision": "full_3d_labels",
                "gaussians": 1800, "control_steps": 600, "candidate_steps": 1800, "threshold": 0.5}})
    print(f"Reference website assets saved to {output}")


if __name__ == "__main__":
    main()
