"""Biplane Simpson volumes and ejection fraction from simulated apical views.

Each phase is measured independently: the long axis comes from that phase's
cavity label (an oracle view placement), and optional tilt/shift noise models
probe misplacement. This is the clinical reference that reconstruction methods
must beat using the same views.

Both scans of a subject share near-identical CMR-derived labels, so agreement
statistics average the two scans and count subjects, not scans.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_common.clinical import biplane_simpson_ml, ejection_fraction, label_volumes_ml
from cardiac_common.cohort import enumerate_volumes, fold_subjects, load_label
from cardiac_common.views import apical_planes, lv_long_axis, perturb_plane, sample_plane


def select_subjects(manifest, subset):
    splits = manifest["splits"]
    if subset == "test":
        return set(splits["test"])
    if subset == "development":
        return {s for fold in splits["folds"] for s in fold}
    return set(fold_subjects(splits, int(subset.removeprefix("fold")))[1])


def agreement(estimate, truth):
    difference = np.asarray(estimate) - np.asarray(truth)
    bias, spread = float(difference.mean()), float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    return {"n": len(difference), "bias": bias, "mae": float(np.abs(difference).mean()),
            "loa_low": bias - 1.96 * spread, "loa_high": bias + 1.96 * spread,
            "pearson_r": float(np.corrcoef(estimate, truth)[0, 1]) if len(difference) > 2 else None}


def measure(path, args, rng):
    label, affine = load_label(path)
    axis = lv_long_axis(label, affine)
    planes = apical_planes(axis, args.angles)
    if args.tilt_deg or args.shift_mm:
        planes = [perturb_plane(p, args.tilt_deg, args.shift_mm, rng) for p in planes]
    views = [sample_plane(label, affine, plane, args.spacing_mm) for plane in planes]
    estimate, info = biplane_simpson_ml(*views, disks=args.disks)
    return {"simpson_ml": estimate, "true_ml": label_volumes_ml(label, affine)["lv_cavity_ml"],
            "simpson_length_mm": info["length_mm"], "axis_length_mm": axis.length_mm,
            "apex_direction_world": axis.direction.round(4).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", default="benchmarks/splits/mitea_cohort_v1.json")
    parser.add_argument("--subset", default="development",
                        help="development, test, or foldK (validation subjects of fold K)")
    parser.add_argument("--output", required=True, help="New JSON file")
    parser.add_argument("--angles", nargs=2, type=float, default=[0.0, 90.0])
    parser.add_argument("--spacing-mm", type=float, default=0.5)
    parser.add_argument("--disks", type=int, default=20)
    parser.add_argument("--tilt-deg", type=float, default=0.0)
    parser.add_argument("--shift-mm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} exists")
    manifest = json.loads(Path(args.manifest).read_text())
    subjects = select_subjects(manifest, args.subset)
    rng = np.random.default_rng(args.seed)

    by_scan = {}
    for volume in enumerate_volumes(args.data_dir):
        if volume.subject in subjects:
            by_scan.setdefault((volume.subject, volume.scan), {})[volume.phase] = volume.label_path
    rows = []
    for (subject, scan), phases in sorted(by_scan.items()):
        if set(phases) != {"ED", "ES"}:
            continue
        ed, es = measure(phases["ED"], args, rng), measure(phases["ES"], args, rng)
        rows.append({
            "subject": subject, "scan": scan, "category": manifest["strata"][subject],
            "ED": ed, "ES": es,
            "true_ef": ejection_fraction(ed["true_ml"], es["true_ml"]),
            "simpson_ef": ejection_fraction(ed["simpson_ml"], es["simpson_ml"]),
        })
        print(f"{subject} scan{scan}: EF true={rows[-1]['true_ef']:.1f} "
              f"simpson={rows[-1]['simpson_ef']:.1f}", flush=True)
    def per_subject(value):
        groups = {}
        for row in rows:
            groups.setdefault(row["subject"], []).append(value(row))
        return [float(np.mean(groups[s])) for s in sorted(groups)]

    summary = {
        "edv_ml": agreement(per_subject(lambda r: r["ED"]["simpson_ml"]),
                            per_subject(lambda r: r["ED"]["true_ml"])),
        "esv_ml": agreement(per_subject(lambda r: r["ES"]["simpson_ml"]),
                            per_subject(lambda r: r["ES"]["true_ml"])),
        "ef_percent": agreement(per_subject(lambda r: r["simpson_ef"]),
                                per_subject(lambda r: r["true_ef"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": vars(args), "manifest_sha256": manifest["splits_sha256"],
                                  "summary": summary, "rows": rows}, indent=2) + "\n")
    for name, stats in summary.items():
        print(f"{name}: n={stats['n']} bias={stats['bias']:.2f} MAE={stats['mae']:.2f} "
              f"LoA=[{stats['loa_low']:.2f}, {stats['loa_high']:.2f}] r={stats['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
