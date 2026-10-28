"""Generate aggregate scientific plots and LaTeX tables from audited results."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac_common.radial import LAYOUTS, METHODS
from scripts.summarize_feecho4d_ablation import bootstrap

LABELS = {"nearest": "Nearest view", "sdf": "SDF interpolation", "inr": "INR", "gaussian": "Gaussian"}
COLORS = {"nearest": "#64748B", "sdf": "#16A34A", "inr": "#2563EB", "gaussian": "#D97706"}
LAYOUT_LABELS = {key: f"Spread, {len(views)} views" for key, views in LAYOUTS.items()}
LAYOUT_LABELS["clustered3"] = "Clustered, 3 views"


def p_text(value):
    if value >= .001:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    analysis = args.run_dir / "analysis"
    summary = json.loads((analysis / "summary.json").read_text())
    data = list(csv.DictReader((analysis / "subject_scores.csv").open()))
    figures, tables = analysis / "figures", analysis / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    rng = np.random.default_rng(2027)
    for i, method in enumerate(METHODS):
        vals = [float(r["common.dice"]) for r in data if r["layout"] == "spread3" and r["method"] == method]
        axes[0].scatter(i + rng.normal(0, .045, len(vals)), vals, color=COLORS[method],
                        s=13, alpha=.65, linewidths=0)
        axes[0].hlines(np.median(vals), i - .18, i + .18, color="black", linewidth=1.8)
    axes[0].set_xticks(range(4), [LABELS[m] for m in METHODS], rotation=18)
    axes[0].set_title("Three spread observations; n = 52 subjects")
    axes[0].set_ylabel("Dice on the common 24 withheld views")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=.2)
    x = [2, 3, 6, 12]
    for method in METHODS:
        cells = [summary["cells"][f"spread{k}"][method] for k in x]
        means = [c["common.dice"]["mean"] for c in cells]
        cis = np.asarray([c["common_mean_ci"] for c in cells])
        axes[1].plot(x, means, marker="o", markersize=4, color=COLORS[method], label=LABELS[method])
        axes[1].fill_between(x, cis[:, 0], cis[:, 1], color=COLORS[method], alpha=.12)
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Number of observed radial masks")
    axes[1].set_ylabel("Mean subject Dice; 95% bootstrap CI")
    axes[1].set_title("Nested observations; fixed evaluation views")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, loc="lower right")
    fig.savefig(figures / "feecho4d_baselines_budget.pdf")
    fig.savefig(figures / "feecho4d_baselines_budget.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for i, method in enumerate(METHODS):
        effect = summary["spread3_minus_clustered3"][method]
        value = effect["mean_difference"]
        low, high = effect["bootstrap_95ci"]
        axes[0].errorbar(value, i, xerr=[[value - low], [high - value]],
                         fmt="o", capsize=3, color=COLORS[method])
    axes[0].axvline(0, linestyle="--", color="#475569", linewidth=1)
    axes[0].set_yticks(range(4), [LABELS[m] for m in METHODS])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean paired Dice difference: spread − clustered")
    axes[0].set_title("Same annotation count: three views")
    axes[0].grid(axis="x", alpha=.2)
    bins = ["gap_0_15", "gap_15_30", "gap_30_60", "gap_gt60"]
    for method in METHODS:
        records = [r for r in data if r["layout"] == "clustered3" and r["method"] == method]
        values = [np.array([float(r[f"{b}.dice"]) for r in records if r[f"{b}.dice"]]) for b in bins]
        means = [v.mean() for v in values]
        cis = np.asarray([bootstrap(v) for v in values])
        axes[1].plot(range(4), means, marker="o", markersize=4, color=COLORS[method], label=LABELS[method])
        axes[1].fill_between(range(4), cis[:, 0], cis[:, 1], color=COLORS[method], alpha=.12)
    axes[1].set_xticks(range(4), ["≤15°", ">15–30°", ">30–60°", ">60°"])
    axes[1].set_xlabel("Distance to nearest observed angle")
    axes[1].set_ylabel("Mean subject pooled Dice")
    axes[1].set_title("Three clustered views: error away from inputs")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, loc="lower left")
    fig.savefig(figures / "feecho4d_coverage.pdf")
    fig.savefig(figures / "feecho4d_coverage.png", dpi=180)
    plt.close(fig)

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Layout & Nearest & SDF interpolation & INR & Gaussian \\", r"\midrule"]
    for layout in LAYOUTS:
        values = [summary["cells"][layout][method]["common.dice"] for method in METHODS]
        entries = [f"{v['mean']:.3f} $\\pm$ {v['sd']:.3f}" for v in values]
        lines.append(" & ".join([LAYOUT_LABELS[layout]] + entries) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (tables / "feecho4d_ablation.tex").write_text("\n".join(lines))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Layout & Mean SDF $-$ INR & 95\% bootstrap CI & Holm $p$ & SDF higher / $n$ \\", r"\midrule"]
    for layout, comparison in summary["primary_sdf_minus_inr"].items():
        low, high = comparison["bootstrap_95ci"]
        lines.append(f"{LAYOUT_LABELS[layout]} & {comparison['mean_difference']:+.3f} & [{low:+.3f}, {high:+.3f}] & "
                     f"{p_text(comparison['holm_p'])} & {comparison['first_better']}/{comparison['n']}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (tables / "feecho4d_comparisons.tex").write_text("\n".join(lines))
    print(f"Exported plots and LaTeX tables to {analysis}")


if __name__ == "__main__":
    main()
