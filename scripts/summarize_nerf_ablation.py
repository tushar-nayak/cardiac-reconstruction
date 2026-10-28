"""Audit and summarize the frozen MITEA synthetic-rendering NeRF ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_nerf_ablation import (
    CONFIRM_SUBJECTS, SCREEN_SUBJECTS, SEEDS, complete_records, screen_score, variants,
)


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    n = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_ci(values, seed=2027, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def by_subject(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row["config"]["name"], {}).setdefault(row["subject"], []).append(row)
    output = {}
    for config, subjects in grouped.items():
        output[config] = {}
        for subject, rows in subjects.items():
            if len(rows) != len(SEEDS) or {row["seed"] for row in rows} != set(SEEDS):
                raise ValueError(f"Incomplete seed set for {config}/{subject}")
            output[config][subject] = {
                key: float(np.mean([r[field] for r in rows]))
                for key, field in (("opacity_dice", "mean_heldout_opacity_dice"),
                                   ("psnr_db", "mean_heldout_psnr_db"),
                                   ("opacity_mse", "mean_heldout_opacity_mse"),
                                   ("seconds", "training_seconds"))
            }
    return output


def descriptive(subject_scores):
    rows = []
    for config, subjects in subject_scores.items():
        values = [v["opacity_dice"] for v in subjects.values()]
        psnr = [v["psnr_db"] for v in subjects.values()]
        omse = [v["opacity_mse"] for v in subjects.values()]
        times = [v["seconds"] for v in subjects.values()]
        rows.append({"config": config, "n_subjects": len(values),
                     "opacity_dice_mean": float(np.mean(values)),
                     "opacity_dice_sd": float(np.std(values, ddof=1)),
                     "opacity_dice_median": float(np.median(values)),
                     "opacity_dice_q1": float(np.quantile(values, .25)),
                     "opacity_dice_q3": float(np.quantile(values, .75)),
                     "opacity_dice_bootstrap_ci": bootstrap_ci(values),
                     "psnr_mean_db": float(np.mean(psnr)),
                     "opacity_mse_mean": float(np.mean(omse)),
                     "training_seconds_median": float(np.median(times))})
    return rows


def paired(rows, baseline="baseline", family_correction=True):
    base = rows[baseline]
    names = [name for name in rows if name != baseline]
    raw = []
    for name in names:
        common = sorted(set(base) & set(rows[name]))
        diff = np.array([rows[name][s]["opacity_dice"] - base[s]["opacity_dice"] for s in common])
        raw.append({"config": name, "n_subjects": len(common),
                    "mean_difference": float(np.mean(diff)),
                    "median_difference": float(np.median(diff)),
                    "bootstrap_95ci": bootstrap_ci(diff),
                    "subjects_higher": int(np.sum(diff > 0)),
                    "subjects_tied": int(np.sum(diff == 0)),
                    "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0})
    if family_correction and raw:
        adjusted = holm([x["wilcoxon_p"] for x in raw])
        for row, p in zip(raw, adjusted):
            row["holm_p"] = float(p)
    return raw


def direction_summary(records):
    grouped = {}
    for row in records:
        for direction, metrics in row["heldout"].items():
            key = (row["phase"], row["config"]["name"], direction, row["subject"])
            grouped.setdefault(key, []).append(metrics)
    subject_level = {}
    for (phase, config, direction, subject), seed_rows in grouped.items():
        if len(seed_rows) != len(SEEDS):
            raise ValueError(f"Incomplete directional seed set: {phase}/{config}/{subject}/{direction}")
        subject_level.setdefault((phase, config, direction), []).append({
            metric: float(np.mean([r[metric] for r in seed_rows]))
            for metric in ("opacity_dice", "opacity_mse", "rgb_psnr_db")
        })
    result = []
    for (phase, config, direction), rows in sorted(subject_level.items()):
        result.append({"phase": phase, "config": config, "direction": direction,
                       "n_subjects": len(rows),
                       "opacity_dice_mean": float(np.mean([r["opacity_dice"] for r in rows])),
                       "opacity_dice_sd": float(np.std([r["opacity_dice"] for r in rows], ddof=1)),
                       "opacity_mse_mean": float(np.mean([r["opacity_mse"] for r in rows])),
                       "psnr_mean_db": float(np.mean([r["rgb_psnr_db"] for r in rows]))})
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    configs = variants()
    screening = complete_records(root, "screening", configs, SCREEN_SUBJECTS)
    if screening is None:
        raise RuntimeError("Screening run set is incomplete")
    ranking_path = root / "screening_ranking.json"
    if not ranking_path.is_file():
        raise RuntimeError("Missing frozen screening selection")
    selected = json.loads(ranking_path.read_text())["selected"]
    confirmation_configs = [c for c in configs if c["name"] == "baseline" or c["name"] in selected]
    confirmation = complete_records(root, "confirmation", confirmation_configs, CONFIRM_SUBJECTS)
    if confirmation is None:
        raise RuntimeError("Confirmation run set is incomplete")

    screen_scores = by_subject(screening)
    confirm_scores = by_subject(confirmation)
    screen_desc = descriptive(screen_scores)
    confirm_desc = descriptive(confirm_scores)
    screen_pairs = paired(screen_scores)
    confirm_pairs = paired(confirm_scores)
    output = {"scope": "synthetic MITEA volume rendering; not sparse-plane reconstruction",
              "screening_runs": len(screening), "confirmation_runs": len(confirmation),
              "screen_subjects": list(SCREEN_SUBJECTS),
              "confirmation_subjects": list(CONFIRM_SUBJECTS),
              "selected_configurations": selected,
              "screening_descriptive": screen_desc,
              "screening_paired_vs_baseline_exploratory": screen_pairs,
              "confirmation_descriptive": confirm_desc,
              "confirmation_paired_vs_baseline": confirm_pairs,
              "directional_metrics": direction_summary(screening + confirmation)}
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    write_csv(analysis / "screening_subject_scores.csv",
              [{"config": c, "subject": s, **v}
               for c, by_sub in screen_scores.items() for s, v in by_sub.items()])
    write_csv(analysis / "confirmation_subject_scores.csv",
              [{"config": c, "subject": s, **v}
               for c, by_sub in confirm_scores.items() for s, v in by_sub.items()])
    write_csv(analysis / "directional_metrics.csv", output["directional_metrics"])

    lines = ["# MITEA NeRF ablation results", "",
             "**Scope:** synthetic orthographic rendering from complete 3D scans; this is not sparse-plane reconstruction.", "",
             f"Screening: {len(screening)} successful fits across {len(SCREEN_SUBJECTS)} development subjects and {len(configs)} configurations.",
             f"Confirmation: {len(confirmation)} successful fits across {len(CONFIRM_SUBJECTS)} disjoint development subjects.",
             f"Selected after screening: {', '.join(selected)}.", "",
             "Primary outcome is opacity Dice on the three held-out reverse cardinal views. Values average seeds within subject before cohort summaries.", "",
             "## Screening", "",
             "| Configuration | Mean Dice | SD | Median [IQR] | 95% bootstrap CI | Mean PSNR (dB) | Median fit (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in screen_desc:
        ci = row["opacity_dice_bootstrap_ci"]
        lines.append(f"| {row['config']} | {row['opacity_dice_mean']:.4f} | {row['opacity_dice_sd']:.4f} | "
                     f"{row['opacity_dice_median']:.4f} [{row['opacity_dice_q1']:.4f}, {row['opacity_dice_q3']:.4f}] | "
                     f"[{ci[0]:.4f}, {ci[1]:.4f}] | {row['psnr_mean_db']:.2f} | {row['training_seconds_median']:.1f} |")
    lines += ["", "Screening paired differences versus baseline are exploratory because they are used for configuration selection.",
              "", "## Disjoint development confirmation", "",
              "| Configuration | Mean Dice | SD | Median [IQR] | 95% bootstrap CI | Mean PSNR (dB) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in confirm_desc:
        ci = row["opacity_dice_bootstrap_ci"]
        lines.append(f"| {row['config']} | {row['opacity_dice_mean']:.4f} | {row['opacity_dice_sd']:.4f} | "
                     f"{row['opacity_dice_median']:.4f} [{row['opacity_dice_q1']:.4f}, {row['opacity_dice_q3']:.4f}] | "
                     f"[{ci[0]:.4f}, {ci[1]:.4f}] | {row['psnr_mean_db']:.2f} |")
    lines += ["", "| Selected variant minus baseline | Mean paired Dice difference | 95% bootstrap CI | Holm p | Higher / n |",
              "|---|---:|---|---:|---:|"]
    for row in confirm_pairs:
        ci = row["bootstrap_95ci"]
        lines.append(f"| {row['config']} | {row['mean_difference']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{row['holm_p']:.3g} | {row['subjects_higher']}/{row['n_subjects']} |")
    lines += ["", "These results concern synthetic view rendering from full-volume-derived targets. They do not estimate sparse-view 3D reconstruction, generalization to new subjects, or clinical performance.", ""]
    (analysis / "results.md").write_text("\n".join(lines))
    print(json.dumps({"screening_runs": len(screening), "confirmation_runs": len(confirmation),
                      "selected": selected, "output": str(analysis)}, indent=2))


if __name__ == "__main__":
    main()
