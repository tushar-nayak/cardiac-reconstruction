"""Create review figures from the audited NeRF ablation summary."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root / "analysis"
    summary = json.loads((root / "summary.json").read_text())
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    rows = sorted(summary["screening_descriptive"], key=lambda x: x["opacity_dice_mean"])
    names = [r["config"] for r in rows]
    means = np.array([r["opacity_dice_mean"] for r in rows])
    lows = np.array([r["opacity_dice_bootstrap_ci"][0] for r in rows])
    highs = np.array([r["opacity_dice_bootstrap_ci"][1] for r in rows])
    colors = ["#D97706" if name == "baseline" else "#2563EB" for name in names]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(rows))
    ax.errorbar(means, y, xerr=np.vstack((means - lows, highs - means)), fmt="none",
                ecolor="#64748B", capsize=3, linewidth=1.4)
    ax.scatter(means, y, c=colors, s=42, zorder=3)
    base = summary["screening_descriptive"]
    baseline = next(r["opacity_dice_mean"] for r in base if r["config"] == "baseline")
    ax.axvline(baseline, color="#D97706", linestyle="--", linewidth=1.2, label="Baseline")
    ax.set_yticks(y, names)
    ax.set_xlabel("Held-out synthetic-view opacity Dice (subject bootstrap 95% CI)")
    ax.set_title("NeRF screening ablations (development subjects)")
    ax.grid(axis="x", alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "screening_forest.pdf", bbox_inches="tight")
    fig.savefig(out / "screening_forest.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in summary["screening_descriptive"]:
        color = "#D97706" if row["config"] == "baseline" else "#2563EB"
        ax.scatter(row["psnr_mean_db"], row["opacity_dice_mean"], color=color, s=42)
        ax.annotate(row["config"], (row["psnr_mean_db"], row["opacity_dice_mean"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean held-out grayscale PSNR (dB)")
    ax.set_ylabel("Mean held-out rendered-opacity Dice")
    ax.set_title("Rendering fidelity and silhouette agreement")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "screening_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(out / "screening_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    table = read_csv(root / "confirmation_subject_scores.csv")
    per_config = {}
    for row in table:
        per_config.setdefault(row["config"], {})[row["subject"]] = float(row["opacity_dice"])
    subjects = sorted(per_config["baseline"])
    choices = [name for name in summary["selected_configurations"]]
    fig, axes = plt.subplots(1, len(choices), figsize=(6 * len(choices), 5), squeeze=False)
    for ax, name in zip(axes[0], choices):
        x0, x1 = 0, 1
        for subject in subjects:
            ax.plot([x0, x1], [per_config["baseline"][subject], per_config[name][subject]],
                    color="#94A3B8", alpha=.55, linewidth=.8)
        ax.scatter(np.zeros(len(subjects)), [per_config["baseline"][s] for s in subjects],
                   color="#D97706", label="Baseline", zorder=3)
        ax.scatter(np.ones(len(subjects)), [per_config[name][s] for s in subjects],
                   color="#2563EB", label=name, zorder=3)
        ax.set_xticks([0, 1], ["Baseline", name], rotation=12)
        ax.set_ylabel("Held-out opacity Dice")
        ax.set_title(f"Confirmation: {name}")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("Paired confirmation on disjoint development subjects")
    fig.tight_layout()
    fig.savefig(out / "confirmation_paired.pdf", bbox_inches="tight")
    fig.savefig(out / "confirmation_paired.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Exported figures to {out}")


if __name__ == "__main__":
    main()
