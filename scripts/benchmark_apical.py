"""Cavity volumes and EF from apical views: Simpson versus 3D reconstruction.

See ``benchmarks/mitea_apical_ef_v1.md``. For each subject, phase (ED, ES),
layout, and misplacement condition, every method reads the same misplaced views;
metrics compare predicted cavity volume and overlap with the full 3D cavity.
Held-out test subjects cannot be selected.
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_sparse import fit_field, write_json
from cardiac_common.apical import LEVELS, observe, view_constraints, view_planes
from cardiac_common.clinical import biplane_simpson_ml
from cardiac_common.cohort import LV_CAVITY, enumerate_volumes, load_label
from cardiac_common.population import check_not_trained_on
from cardiac_common.prior_inr import fit_prior_inr
from cardiac_common.rbf import choose_centres, evaluate_rbf, fit_rbf
from cardiac_common.sparse import coordinates, evaluate_grid
from cardiac_common.ssm import fit_ssm, mean_shape
from cardiac_common.unet import UNet3D, predict_unet
from cardiac_common.views import sample_plane
from code_inr.inr_model import ImplicitOccupancyNetwork

PHASES = ("ED", "ES")
LAYOUT_ORDER = ("biplane", "triplane")
CONDITIONS = {"biplane": [("none", "known"), ("mild", "known"), ("severe", "known"),
                          ("mild", "assumed"), ("severe", "assumed")],
              "triplane": [("none", "known"), ("severe", "known")]}
METHODS = ("simpson", "inr", "rbf", "ssm", "unet", "prior_inr", "mean_shape")
POPULATION = ("ssm", "unet", "prior_inr", "mean_shape")
INR = SimpleNamespace(steps=1000, batch_size=4096, lr=1e-3)  # v1 settings


def reconstruct(method, target, affine, intended, actual, pose, observed, values, models, seed, device):
    """Binary cavity prediction on the native grid."""
    shape = target.shape
    if method == "rbf":
        positions, constraint = view_constraints(target, affine, intended, actual, pose)
        chosen = choose_centres(constraint)
        fit = fit_rbf(positions[chosen], constraint[chosen], "r3", 0.01)
        return evaluate_rbf(fit, shape, device) >= 0
    if method == "mean_shape":
        return mean_shape(models["ssm"], shape)
    if method == "unet":
        return predict_unet(models["unet_net"], observed, values, shape, device) >= 0.5
    observed_zyx = np.argwhere(observed)
    labels = values[observed].astype(np.float32)
    if method == "ssm":
        return fit_ssm(models["ssm"], observed_zyx, labels, shape, 20, 0.001, device)[0]
    points = coordinates(torch.from_numpy(observed_zyx), shape)
    label_tensor = torch.from_numpy(labels)
    torch.manual_seed(seed)
    model = ImplicitOccupancyNetwork(64, 4, 4).to(device)
    if method == "inr":
        fit_field(model, points, label_tensor, INR, device, seed)
    else:
        prior, _ = fit_ssm(models["ssm"], observed_zyx, labels, shape, 20, 0.001, device)
        torch.manual_seed(seed)
        fit_prior_inr(model, points, label_tensor, torch.from_numpy(observed), prior, 1000, 1e-4, 4096, 0.0,
                      seed, device)
    return evaluate_grid(model, shape, device, 4096, 1).numpy() >= 0.5


def dice(prediction, truth):
    total = prediction.sum() + truth.sum()
    return float(2 * (prediction & truth).sum() / total) if total else 1.0


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(4)
    manifest = json.loads(Path(args.manifest).read_text())
    fold = int(args.subset.removeprefix("fold"))
    subjects = sorted(manifest["splits"]["folds"][fold])
    if set(subjects) & set(manifest["splits"]["test"]):
        raise RuntimeError("evaluation fold overlaps the held-out test split")
    if args.subjects:
        if set(args.subjects) - set(subjects):
            raise ValueError("subjects not in the evaluation fold")
        subjects = sorted(args.subjects)
    models = {}
    if set(args.methods) & {"ssm", "prior_inr", "mean_shape"}:
        models["ssm"] = torch.load(args.ssm_model, weights_only=False)
    if "unet" in args.methods:
        models["unet"] = torch.load(args.unet_model, weights_only=False)
        models["unet_net"] = UNet3D().to(device)
        models["unet_net"].load_state_dict(models["unet"]["state_dict"])
    for name in ("ssm", "unet"):
        if name in models and models[name].get("target") != "cavity":
            raise RuntimeError(f"{name} model was not trained on the cavity target")
    paths = {(v.subject, v.scan, v.phase): v.label_path for v in enumerate_volumes(args.data_dir)}
    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", vars(args))
    rows = []
    for subject in subjects:
        number = int(subject.rsplit("_", 1)[1])
        for method in set(args.methods) & set(POPULATION):
            check_not_trained_on(subject, models["unet" if method == "unet" else "ssm"])
        for phase_index, phase in enumerate(PHASES):
            label, affine = load_label(paths[(subject, 1, phase)])
            target = label == LV_CAVITY
            voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000
            for layout_index, layout in enumerate(LAYOUT_ORDER):
                for level, pose in CONDITIONS[layout]:
                    level_index = list(LEVELS).index(level)
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [2027, number, phase_index, layout_index, level_index]))
                    intended, actual = view_planes(label, affine, layout, *LEVELS[level], rng)
                    observed, values = observe(target, affine, intended, actual, pose)
                    digest = hashlib.sha256(observed.tobytes() + values.tobytes()).hexdigest()
                    for method in args.methods:
                        if method == "simpson" and layout != "biplane":
                            continue
                        row = {"subject": subject, "fold": fold, "category": manifest["strata"][subject],
                               "phase": phase, "layout": layout, "misplacement": level, "pose": pose,
                               "method": method, "seed": args.seed, "true_ml": float(target.sum() * voxel_ml),
                               "observed_voxels": int(observed.sum()), "observation_sha256": digest}
                        started = time.perf_counter()
                        try:
                            if method == "simpson":
                                views = [sample_plane(label, affine, plane, 0.5) for plane in actual]
                                row["predicted_ml"] = biplane_simpson_ml(*views, disks=20)[0]
                            else:
                                prediction = reconstruct(method, target, affine, intended, actual, pose, observed,
                                                         values, models, args.seed, device)
                                row.update(predicted_ml=float(prediction.sum() * voxel_ml),
                                           dice=dice(prediction, target),
                                           unobserved_dice=dice(prediction & ~observed, target & ~observed),
                                           empty_prediction=bool(not prediction.any()))
                            row["status"] = "ok"
                        except Exception as error:  # recorded, never silently dropped
                            row.update(status="failed", error=f"{type(error).__name__}: {error}",
                                       traceback=traceback.format_exc())
                            print(f"  FAILED {subject} {phase} {layout} {level}/{pose} {method}: {error}", flush=True)
                        row["seconds"] = time.perf_counter() - started
                        rows.append(row)
        write_json(output / "results.json", {"rows": rows})
        print(f"{subject}: {sum(r['subject'] == subject for r in rows)} rows", flush=True)
    write_json(output / "results.json", {"rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset", required=True, choices=("fold0", "fold1", "fold2", "fold3", "fold4"))
    parser.add_argument("--subjects", nargs="+", default=None, help="Restrict to these subjects (smoke checks)")
    parser.add_argument("--run-dir", required=True, help="New output directory; never overwrites a run")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--ssm-model", default=None)
    parser.add_argument("--unet-model", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if set(args.methods) & {"ssm", "prior_inr", "mean_shape"} and not args.ssm_model:
        parser.error("ssm/prior_inr/mean_shape require --ssm-model")
    if "unet" in args.methods and not args.unet_model:
        parser.error("unet requires --unet-model")
    run(args)


if __name__ == "__main__":
    main()
