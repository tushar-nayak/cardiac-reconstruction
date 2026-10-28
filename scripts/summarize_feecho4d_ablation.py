"""Audit and summarize all frozen FeEcho4D interpolation/count/coverage runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardiac_common.radial import COMMON_VIEWS, LAYOUTS, METHODS, SEEDS


def describe(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, .25)), "q3": float(np.quantile(values, .75))}


def bootstrap(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    rng = np.random.default_rng(2027)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def paired(first, second):
    subjects = sorted(set(first) & set(second))
    diff = np.array([first[s] - second[s] for s in subjects])
    if not len(diff):
        return {"n": 0}
    return {"n": len(diff), "mean_difference": float(diff.mean()),
            "median_difference": float(np.median(diff)), "bootstrap_95ci": bootstrap(diff),
            "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0,
            "first_better": int((diff > 0).sum()), "ties": int((diff == 0).sum())}


def audit_and_summarize(root):
    paths = sorted(root.glob("*/runs.jsonl"))
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    subjects = [f"Patient{i:03d}" for i in range(1, 53)]
    expected = {(subject, layout, method, seed)
                for subject in subjects for layout in LAYOUTS for method in METHODS
                for seed in (SEEDS if method in ("inr", "gaussian") else (None,))}
    keyed = {(r["subject"], r["layout"], r["method"], r["seed"]): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError("duplicate run keys")
    missing, extra = expected - set(keyed), set(keyed) - expected
    if missing or extra:
        raise ValueError(f"incomplete experiment: {len(missing)} missing / {len(extra)} unexpected runs")
    failed = [r for r in rows if r["status"] != "ok"]
    for subject in subjects:
        eval_hashes = set()
        for layout, obs_ids in LAYOUTS.items():
            cell = [r for r in rows if r["subject"] == subject and r["layout"] == layout]
            if len({r["observation_sha256"] for r in cell}) != 1:
                raise ValueError(f"observation mismatch: {subject} {layout}")
            for r in cell:
                if tuple(r["observed_views"]) != obs_ids:
                    raise ValueError("observation IDs do not match frozen layout")
                eval_hashes.add(r["evaluation_sha256"])
                if r["status"] == "ok":
                    query = {v["view_id"] for v in r["metrics"]["per_view"]}
                    if query != set(range(1, 38)) - set(obs_ids):
                        raise ValueError("query coverage mismatch")
                    common = query & set(COMMON_VIEWS)
                    if common != set(COMMON_VIEWS):
                        raise ValueError("common-view coverage mismatch")
        if len(eval_hashes) != 1:
            raise ValueError("evaluation-grid mismatch across methods/layouts")

    aggregated = []
    for subject in subjects:
        for layout in LAYOUTS:
            for method in METHODS:
                seeds = SEEDS if method in ("inr", "gaussian") else (None,)
                records = [keyed[(subject, layout, method, seed)] for seed in seeds]
                if any(r["status"] != "ok" for r in records):
                    continue  # report failed coverage below; never average partial seeds
                result = {"subject": subject, "diagnosis": records[0]["diagnosis"],
                          "layout": layout, "method": method, "seeds_averaged": len(records)}
                for subset in records[0]["metrics"]["subsets"]:
                    for metric in ("dice", "iou", "pixel_accuracy"):
                        values = [r["metrics"]["subsets"][subset] for r in records]
                        result[f"{subset}.{metric}"] = (float(np.mean([v[metric] for v in values]))
                                                        if all(v is not None for v in values) else None)
                for key in ("fit_seconds", "query_seconds"):
                    result[key] = float(np.mean([r[key] for r in records]))
                result["mean_view_dice"] = float(np.mean([r["metrics"]["common_mean_view_dice"] for r in records]))
                aggregated.append(result)

    cells = {}
    lookup = {}
    for layout in LAYOUTS:
        cells[layout] = {}
        for method in METHODS:
            data = [r for r in aggregated if r["layout"] == layout and r["method"] == method]
            lookup[(layout, method)] = {r["subject"]: r["common.dice"] for r in data}
            metric_names = [k for k in aggregated[0] if k not in
                            ("subject", "diagnosis", "layout", "method", "seeds_averaged")]
            cells[layout][method] = {key: describe([r[key] for r in data if r[key] is not None])
                                    for key in metric_names}
            cells[layout][method]["common_mean_ci"] = bootstrap([r["common.dice"] for r in data])

    primary = {layout: paired(lookup[(layout, "sdf")], lookup[(layout, "inr")]) for layout in LAYOUTS}
    sorted_keys = sorted(primary, key=lambda key: primary[key].get("wilcoxon_p", 1))
    bound = 0.0
    for i, key in enumerate(sorted_keys):
        bound = max(bound, min(1.0, (len(sorted_keys) - i) * primary[key].get("wilcoxon_p", 1)))
        primary[key]["holm_p"] = bound
    secondary = {layout: {f"{method}_minus_inr": paired(lookup[(layout, method)], lookup[(layout, "inr")])
                          for method in ("nearest", "gaussian")} for layout in LAYOUTS}
    count_effects = {method: {f"{hi}_minus_{lo}": paired(lookup[(hi, method)], lookup[(lo, method)])
                             for lo, hi in zip(("spread2", "spread3", "spread6"),
                                               ("spread3", "spread6", "spread12"))}
                     for method in METHODS}
    coverage = {method: paired(lookup[("spread3", method)], lookup[("clustered3", method)]) for method in METHODS}

    original_path = ROOT / "runs/feecho4d_external_20261023_final/runs.jsonl"
    original = {(r["subject"], r["method"], r["seed"]): r
                for r in map(json.loads, original_path.read_text().splitlines())}
    errors = []
    for row in rows:
        if row["layout"] == "spread3" and row["method"] in ("inr", "gaussian") and row["status"] == "ok":
            ref = original[(row["subject"], row["method"], row["seed"])]
            if row["observed_png_sha256"] != ref["observation_sha256"]:
                raise ValueError("original observation fingerprint mismatch")
            errors.append(abs(row["metrics"]["subsets"]["all_unobserved"]["dice"]
                              - ref["metrics"]["pooled"]["dice"]))
    reproduction = {"compared_runs": len(errors), "max_absolute_dice_difference": max(errors) if errors else None}
    if errors and max(errors) > 1e-6:
        raise ValueError(f"original recipe reproduction failed: {reproduction}")
    summary = {"protocol": "benchmarks/feecho4d_ablation_v1.md", "planned_subjects": 52,
               "planned_runs": len(expected), "successful_runs": len(rows) - len(failed),
               "failed_runs": len(failed), "failures": [{k: r.get(k) for k in
                                                        ("subject", "layout", "method", "seed", "error")} for r in failed],
               "empty_predictions": sum(r.get("empty_prediction", False) for r in rows),
               "complete_subject_method_cells": len(aggregated), "cells": cells,
               "primary_sdf_minus_inr": primary, "secondary": secondary,
               "annotation_count_effects": count_effects, "spread3_minus_clustered3": coverage,
               "original_recipe_reproduction": reproduction}
    return rows, aggregated, summary


def write_report(root, aggregated, summary):
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "subject_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregated[0]))
        writer.writeheader()
        writer.writerows(aggregated)
    lines = ["# FeEcho4D interpolation, annotation count, and coverage results", "",
             "Exploratory extension; primary endpoint pools the same 24 withheld angular views in every condition.", "",
             f"Completed {summary['successful_runs']}/{summary['planned_runs']} planned runs; "
             f"{summary['failed_runs']} failed, {summary['empty_predictions']} empty predictions.", "",
             "| Layout | Method | Mean ± SD Dice | Median [IQR] | Subjects |",
             "| --- | --- | --- | --- | ---: |"]
    for layout in LAYOUTS:
        for method in METHODS:
            c = summary["cells"][layout][method]["common.dice"]
            lines.append(f"| {layout} | {method} | {c['mean']:.4f} ± {c['sd']:.4f} | "
                         f"{c['median']:.4f} [{c['q1']:.4f}, {c['q3']:.4f}] | {c['n']} |")
    lines += ["", "Primary comparison: SDF minus INR (positive favors SDF). "
              "Bootstrap intervals resample subjects; Holm correction covers five layouts.", "",
              "| Layout | Paired mean difference | 95% bootstrap CI | Holm p | SDF higher / paired n |",
              "| --- | ---: | --- | ---: | --- |"]
    for layout, c in summary["primary_sdf_minus_inr"].items():
        low, high = c["bootstrap_95ci"]
        lines.append(f"| {layout} | {c['mean_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                     f"{c['holm_p']:.3g} | {c['first_better']}/{c['n']} |")
    lines += ["", "Coverage effect: spread3 minus clustered3 (positive favors spread views).", "",
              "| Method | Paired mean difference | 95% bootstrap CI |", "| --- | ---: | --- |"]
    for method, c in summary["spread3_minus_clustered3"].items():
        low, high = c["bootstrap_95ci"]
        lines.append(f"| {method} | {c['mean_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines += ["", f"Original 34-view recipe reproduction: {summary['original_recipe_reproduction']}.", "",
              "Source/run hashes are in worker config.json files. Counts 6 and 12 also expand the observed span. "
              "Scores use annotation x/y/angle coordinates and do not measure physical 3D reconstruction or clinical validity.", ""]
    (out / "results.md").write_text("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    _, aggregated, summary = audit_and_summarize(args.run_dir)
    out = write_report(args.run_dir, aggregated, summary)
    print(json.dumps({"successful_runs": summary["successful_runs"],
                      "failed_runs": summary["failed_runs"], "output": str(out),
                      "primary_sdf_minus_inr": summary["primary_sdf_minus_inr"]}, indent=2))


if __name__ == "__main__":
    main()
