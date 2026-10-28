"""Summarize a completed FeEcho4D run at the subject level."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from summarize_benchmark import bootstrap_ci


METRICS = (
    "pooled.dice", "pooled.iou", "pooled.pixel_accuracy",
    "mean_view.dice", "mean_view.iou", "mean_view.pixel_accuracy",
)


def metric_value(row, name):
    outer, inner = name.split(".")
    return row["metrics"][outer][inner]


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "median": float(np.median(values)), "q1": float(np.quantile(values, .25)),
            "q3": float(np.quantile(values, .75))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / "runs.jsonl").read_text().splitlines()]
    all_subjects = sorted({row["subject"] for row in rows})
    expected = {(subject, method, seed) for subject in all_subjects
                for method in ("inr", "gaussian") for seed in (17, 23, 42)}
    present = {(row["subject"], row["method"], row["seed"]) for row in rows}
    if len(present) != len(rows):
        raise ValueError("duplicate subject/method/seed run records")
    missing = sorted(expected - present)
    failures = [row for row in rows if row["status"] != "ok"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["method"])].append(row)
    if not failures and (missing or any(len(v) != 3 for v in grouped.values())):
        raise ValueError(f"incomplete seeds: missing={len(missing)}")

    summary = {"subjects": len(all_subjects), "planned_runs": len(expected),
               "completed_runs": len(rows) - len(failures), "failed_runs": len(failures),
               "failures": [{k: row.get(k) for k in ("subject", "method", "seed", "error")}
                            for row in failures],
               "methods": {}, "subgroups": {}}
    diagnosis = {row["subject"]: row.get("diagnosis", "unknown") for row in rows}
    subject_method = {}
    for method in ("inr", "gaussian"):
        records = {subject: grouped[(subject, method)] for subject in all_subjects
                   if (subject, method) in grouped}
        subject_metric = {}
        for name in METRICS:
            subject_metric[name] = {
                subject: float(np.mean([metric_value(r, name) for r in rs if r["status"] == "ok"]))
                for subject, rs in records.items() if any(r["status"] == "ok" for r in rs)
            }
        subject_metric["fit_seconds"] = {
            subject: float(np.mean([r["fit_seconds"] for r in rs if r["status"] == "ok"]))
            for subject, rs in records.items() if any(r["status"] == "ok" for r in rs)
        }
        subject_method[method] = subject_metric
        summary["methods"][method] = {
            metric: describe(list(values.values())) for metric, values in subject_metric.items()
        }

    for group, subjects in (
        ("healthy", [s for s in all_subjects if diagnosis.get(s) == "healthy"]),
        ("CHD", [s for s in all_subjects if diagnosis.get(s) != "healthy"]),
    ):
        summary["subgroups"][group] = {"n_subjects": len(subjects), "methods": {}}
        for method in ("inr", "gaussian"):
            values = [subject_method[method]["pooled.dice"][s]
                      for s in subjects if s in subject_method[method]["pooled.dice"]]
            summary["subgroups"][group]["methods"][method] = {
                "pooled_dice_mean": float(np.mean(values)) if values else None,
                "pooled_dice_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }

    common = set(subject_method["inr"]["pooled.dice"]) & set(subject_method["gaussian"]["pooled.dice"])
    summary["paired_complete_subjects"] = len(common)
    # Protocol: Gaussian minus INR, paired by subject, seeds already averaged.
    summary["paired_comparisons"] = {}
    for metric in ("pooled.dice", "mean_view.dice"):
        subjects = sorted(set(subject_method["inr"][metric]) & set(subject_method["gaussian"][metric]))
        differences = np.array([subject_method["gaussian"][metric][s] - subject_method["inr"][metric][s]
                                for s in subjects])
        low, high = bootstrap_ci(differences)
        summary["paired_comparisons"][metric] = {
            "comparison": "gaussian - inr", "n_subjects": len(subjects),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "bootstrap_95ci": [low, high], "bootstrap_replicates": 10000, "bootstrap_seed": 2027,
            "wilcoxon_p": float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0,
            "gaussian_better_subjects": int((differences > 0).sum()),
        }
    (args.run_dir / "summary_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"subjects": summary["subjects"], "planned_runs": summary["planned_runs"],
                      "completed_runs": summary["completed_runs"], "failed_runs": summary["failed_runs"],
                      "methods": {m: summary["methods"][m]["pooled.dice"]
                                  for m in ("inr", "gaussian")},
                      "paired_comparisons": summary["paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
