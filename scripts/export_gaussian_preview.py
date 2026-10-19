"""Export the experimental sparse-plane pilot outside the published website."""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_gaussian.src.metrics import volume_to_mesh


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))


def export_mask(output, name, mask, affine):
    mask = np.asarray(mask, dtype=bool)
    vertices, faces = volume_to_mesh(mask.astype(np.float32), affine)
    mesh_file, mask_file = f"{name}.json", f"{name}.bin"
    write_json(output / mesh_file, {
        **{key: np.round(vertices[:, axis], 3).tolist() for axis, key in enumerate("xyz")},
        **{key: faces[:, axis].tolist() for axis, key in enumerate("ijk")},
    })
    (output / mask_file).write_bytes(np.packbits(mask.reshape(-1), bitorder="little").tobytes())
    return {"mesh": mesh_file, "mask": mask_file}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", default="runs/gaussian_uniform_control_pilot_v01")
    parser.add_argument("--candidate", default="runs/gaussian_observed_lr3e3_pilot_v01")
    parser.add_argument("--data-dir", default="dataset/mitea")
    parser.add_argument("--output", default="runs/gaussian-pilot-web-export")
    args = parser.parse_args()
    torch.set_num_threads(4)
    control_dir, candidate_dir = Path(args.control), Path(args.candidate)
    control = json.loads((control_dir / "results.json").read_text())
    candidate = json.loads((candidate_dir / "results.json").read_text())
    key = lambda row: (row["subject"], row["seed"])
    before = {key(row): row for row in control["runs"]}
    after = {key(row): row for row in candidate["runs"]}
    if not before or before.keys() != after.keys():
        raise ValueError("pilot files must contain the same nonempty set of subject/seed pairs")
    for pair in before:
        for field in ("supervision_sha256", "planes_zyx_axis_index", "eval_stride", "threshold",
                      "volume_shape_zyx", "parameter_count", "optimizer_steps", "split", "scan", "phase"):
            if before[pair][field] != after[pair][field]:
                raise ValueError(f"unmatched {field} for {pair}")
        if before[pair]["eval_stride"] != 1:
            raise ValueError("preview currently requires native-resolution pilot predictions")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    subjects = {}
    for subject in dict.fromkeys(row["subject"] for row in control["runs"]):
        rows = [row for row in control["runs"] if row["subject"] == subject]
        first = rows[0]
        nii = nib.load(Path(args.data_dir) / "labels" / f"{subject}_scan1_ED.nii.gz")
        target = np.transpose(nii.get_fdata() > 0, (2, 1, 0))
        if list(target.shape) != first["volume_shape_zyx"]:
            raise ValueError(f"target shape differs for {subject}")
        record = {"shape": list(target.shape), "planes": first["planes_zyx_axis_index"],
                  "target": export_mask(output, f"{subject}_target", target, nii.affine), "seeds": {}}
        for row in rows:
            seed = row["seed"]
            pair = (subject, seed)
            variants = {}
            for name, directory, result in (("control", control_dir, before[pair]),
                                             ("candidate", candidate_dir, after[pair])):
                path = directory / "gaussian" / subject / f"seed_{seed}" / "predicted_occupancy.pt"
                probability = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(probability, torch.Tensor) or list(probability.shape) != list(target.shape):
                    raise ValueError(f"invalid occupancy tensor at {path}")
                if not torch.isfinite(probability).all():
                    raise ValueError(f"nonfinite occupancy at {path}")
                mask = probability.numpy() >= result["threshold"]
                variants[name] = {**export_mask(output, f"{subject}_{seed}_{name}", mask, nii.affine),
                                  "metrics": result["metrics"], "fit_seconds": result["fit_seconds"]}
            record["seeds"][str(seed)] = variants
            print(f"Exported {subject} seed {seed}", flush=True)
        subjects[subject] = record
    write_json(output / "index.json", {"subjects": subjects,
               "control_summary": control["summary"]["gaussian"],
               "candidate_summary": candidate["summary"]["gaussian"],
               "protocol": {"steps": 1000, "gaussians": 256, "threshold": 0.5,
                            "control": "Uniform centers · learning rate 0.001",
                            "candidate": "Observed foreground centers · learning rate 0.003"}})
    print(f"Preview assets saved to {output}")


if __name__ == "__main__":
    main()
