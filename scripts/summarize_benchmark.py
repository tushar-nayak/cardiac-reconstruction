"""Subject-level statistics for the protocol-v1 sparse-plane benchmark.

``select-gaussian`` applies the fold-0 Gaussian recipe rule. ``report`` builds
the confirmatory summary: seeds are averaged within each subject first, then
paired Wilcoxon tests (Holm-corrected) and bootstrap CIs compare each method
against INR. See ``benchmarks/protocol_v1.md``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PRIMARY = "unobserved_dice"
METRICS = {
    "unobserved_dice": lambda m: (m["unobserved_voxels"] or {}).get("dice"),
    "whole_dice": lambda m: m["whole_grid"]["dice"],
    "whole_iou": lambda m: m["whole_grid"]["iou"],
    "hd95_mm": lambda m: m["surface"]["hd95_mm"],
    "assd_mm": lambda m: m["surface"]["assd_mm"],
    "absolute_volume_error_ml": lambda m: m["absolute_volume_error_ml"],
    "voxel_components_26": lambda m: m["voxel_components_26"],
    "absolute_delta_k": lambda m: m["geometry"]["absolute_delta_integrated_gaussian_curvature"],
}
LCC_METRICS = ("unobserved_dice", "whole_dice", "hd95_mm", "assd_mm", "absolute_volume_error_ml")


def load_runs(run_dirs):
    rows = []
    for run_dir in run_dirs:
        payload = json.loads((Path(run_dir) / "results.json").read_text())
        for row in payload["runs"]:
            rows.append(dict(row, run_dir=str(run_dir)))
    return rows


def check_pairing(rows):
    """Every method must see the same observations for a subject and layout."""
    fingerprints = defaultdict(set)
    for row in rows:
        fingerprints[(row["subject"], row["plane_layout"])].add(row["supervision_sha256"])
    mismatched = sorted(key for key, values in fingerprints.items() if len(values) > 1)
    if mismatched:
        raise ValueError(f"observation fingerprints differ across methods: {mismatched}")


def subject_means(rows, metrics=METRICS, lcc=False):
    """``{(method, layout): {subject: {metric: seed-mean or None}}}``."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("status") == "failed":
            continue
        source = row["metrics"]["largest_component_sensitivity"] if lcc else row["metrics"]
        for name, read in metrics.items():
            grouped[(row["method"], row["plane_layout"])][row["subject"]][name].append(read(source))
    result = {}
    for key, subjects in grouped.items():
        result[key] = {}
        for subject, values in subjects.items():
            result[key][subject] = {}
            for name, seeds in values.items():
                valid = [value for value in seeds if value is not None]
                # A subject's value is missing if any seed is unavailable, so missingness is visible.
                result[key][subject][name] = float(np.mean(valid)) if len(valid) == len(seeds) else None
    return result


def describe(values):
    valid = np.array([value for value in values if value is not None], dtype=float)
    if not len(valid):
        return {"n": 0, "missing": len(values)}
    q1, median, q3 = np.percentile(valid, [25, 50, 75])
    return {"n": int(len(valid)), "missing": len(values) - int(len(valid)),
            "median": float(median), "q1": float(q1), "q3": float(q3),
            "mean": float(valid.mean()), "sd": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0}


def bootstrap_ci(differences, replicates=10000, seed=2027):
    rng = np.random.default_rng(seed)
    differences = np.asarray(differences, dtype=float)
    samples = rng.integers(0, len(differences), size=(replicates, len(differences)))
    means = differences[samples].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_comparisons(means, reference="inr", metric=PRIMARY, methods=None):
    """Paired tests of ``methods`` (default: all) against ``reference``, Holm over the family."""
    comparisons = []
    for (method, layout) in sorted(means):
        if method == reference or (reference, layout) not in means:
            continue
        if methods is not None and method not in methods:
            continue
        ref, other = means[(reference, layout)], means[(method, layout)]
        subjects = sorted(s for s in set(ref) & set(other)
                          if ref[s][metric] is not None and other[s][metric] is not None)
        differences = np.array([other[s][metric] - ref[s][metric] for s in subjects])
        p_value = float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0
        low, high = bootstrap_ci(differences)
        comparisons.append({"method": method, "reference": reference, "layout": layout,
                            "metric": metric, "n_subjects": len(subjects),
                            "mean_difference": float(differences.mean()),
                            "median_difference": float(np.median(differences)),
                            "bootstrap_95ci": [low, high],
                            "subjects_improved": int((differences > 0).sum()),
                            "wilcoxon_p": p_value})
    for comparison, adjusted in zip(comparisons, holm([c["wilcoxon_p"] for c in comparisons])):
        comparison["holm_p"] = adjusted
    return comparisons


def report(args):
    rows = load_runs(args.runs)
    for row in rows:
        # v2 confirmatory runs are cross-fitted per fold, so their rows carry split fold1..fold4.
        if row.get("split") != args.expect_subset and not (
                args.family == "v2" and args.expect_subset == "confirmatory"
                and row.get("split") in ("fold1", "fold2", "fold3", "fold4")):
            raise ValueError(f"{row['run_dir']} contains split {row.get('split')!r}, "
                             f"expected {args.expect_subset!r}")
    check_pairing(rows)
    means = subject_means(rows)
    lcc_means = subject_means(rows, {name: METRICS[name] for name in LCC_METRICS}, lcc=True)
    categories = {row["subject"]: row.get("category") for row in rows}
    summary = {"descriptive": {}, "largest_component_sensitivity": {}, "subgroups": {}}
    for key, subjects in sorted(means.items()):
        label = f"{key[0]}/{key[1]}"
        summary["descriptive"][label] = {
            name: describe([values[name] for values in subjects.values()]) for name in METRICS}
        summary["largest_component_sensitivity"][label] = {
            name: describe([values[name] for values in lcc_means[key].values()]) for name in LCC_METRICS}
        by_category = defaultdict(list)
        for subject, values in subjects.items():
            by_category[categories[subject]].append(values[PRIMARY])
        summary["subgroups"][label] = {c: describe(v) for c, v in sorted(by_category.items())}
    failures = [{key: row.get(key) for key in ("method", "plane_layout", "subject", "seed", "error")}
                for row in rows if row.get("status") == "failed"]
    output = {"runs": args.runs, "subset": args.expect_subset, "primary_metric": PRIMARY,
              "subjects": len(categories), "failed_runs": failures, "family": args.family, **summary}
    if args.family == "v1":
        output["paired_vs_inr"] = paired_comparisons(means)
    else:
        # Protocol v2: Holm family is the five new methods vs INR on both layouts (10 tests);
        # the secondary pairs are descriptive (no Holm across pairs).
        output["paired_vs_inr"] = paired_comparisons(means, methods=V2_METHODS)
        if len(output["paired_vs_inr"]) != 2 * len(V2_METHODS):
            raise ValueError(f"v2 primary family needs {2 * len(V2_METHODS)} comparisons, "
                             f"found {len(output['paired_vs_inr'])}")
        output["secondary"] = {f"{method}_vs_{reference}": paired_comparisons(means, reference, methods=(method,))
                               for method, reference in V2_SECONDARY}
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    path.with_suffix(".tex").write_text(latex_table(summary["descriptive"]))
    for comparison in output["paired_vs_inr"]:
        print(f"{comparison['method']}/{comparison['layout']} - inr: "
              f"mean diff {comparison['mean_difference']:+.4f} "
              f"CI [{comparison['bootstrap_95ci'][0]:+.4f}, {comparison['bootstrap_95ci'][1]:+.4f}] "
              f"Holm p={comparison['holm_p']:.3g} (n={comparison['n_subjects']})")
    print(f"Wrote {path} and {path.with_suffix('.tex')}")


def latex_table(descriptive):
    columns = [("unobserved_dice", "Unobs. Dice", 3), ("whole_dice", "Dice", 3),
               ("hd95_mm", "HD95 (mm)", 1), ("assd_mm", "ASSD (mm)", 2),
               ("absolute_volume_error_ml", r"$|\Delta V|$ (mL)", 1)]
    lines = [r"\begin{tabular}{ll" + "r" * len(columns) + "}", r"\toprule",
             "Layout & Method & " + " & ".join(title for _, title, _ in columns) + r" \\", r"\midrule"]
    for label, stats in descriptive.items():
        method, layout = label.split("/")
        cells = []
        for name, _, digits in columns:
            item = stats[name]
            cells.append("--" if not item["n"] else
                         f"{item['median']:.{digits}f} [{item['q1']:.{digits}f}, {item['q3']:.{digits}f}]")
        lines.append(f"{layout} & {method} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


V2_METHODS = ("rbf", "gp", "ssm", "unet", "prior_inr")
V2_SECONDARY = (("rbf", "interpolation"), ("gp", "rbf"), ("unet", "ssm"), ("prior_inr", "ssm"))


def select_config(args):
    """Protocol v2 fold-0 rule: best layout-averaged unobserved Dice; ties (<= 0.002) go to
    the earlier candidate in grid order."""
    scores = []
    for name, orthogonal, axial in args.candidate:
        rows = load_runs([orthogonal, axial])
        if any(row.get("split") != "fold0" for row in rows):
            raise ValueError(f"{name}: selection must use fold-0 runs only")
        means = subject_means(rows)
        if sorted(layout for _, layout in means) != ["axial", "orthogonal"] or len(means) != 2:
            raise ValueError(f"{name}: expected one method on both layouts, found {sorted(means)}")
        (method, _), = {(m, None) for m, _ in means}
        layouts = [means[(method, layout)] for layout in ("orthogonal", "axial")]
        subjects = sorted(set(layouts[0]) & set(layouts[1]))
        per_layout = [float(np.mean([layout[s][PRIMARY] for s in subjects])) for layout in layouts]
        scores.append({"candidate": name, "method": method, "subjects": len(subjects),
                       "orthogonal": per_layout[0], "axial": per_layout[1],
                       "layout_averaged": float(np.mean([np.mean([l[s][PRIMARY] for l in layouts])
                                                          for s in subjects])),
                       "failed_runs": sum(row.get("status") == "failed" for row in rows)})
    best = max(score["layout_averaged"] for score in scores)
    selected = next(score for score in scores if score["layout_averaged"] >= best - 0.002)
    result = {"rule": "protocol_v2 fold-0 selection", "scores": scores, "selected": selected["candidate"]}
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for score in scores:
        print(f"{score['candidate']:>28}: orth {score['orthogonal']:.4f} axial {score['axial']:.4f} "
              f"avg {score['layout_averaged']:.4f} (n={score['subjects']}, failed={score['failed_runs']})")
    print("selected:", result["selected"])


def select_gaussian(args):
    """Fold-0 rule: mean over subjects of the layout-averaged, seed-averaged score."""
    scores = {}
    for recipe, run_dirs in (("uniform_lr0.001", args.uniform), ("observed_lr0.003", args.observed)):
        rows = [row for row in load_runs(run_dirs) if row["method"] == "gaussian"]
        if any(row.get("split") != "fold0" for row in rows):
            raise ValueError("Gaussian recipe selection must use fold-0 runs only")
        means = subject_means(rows)
        layouts = sorted(layout for _, layout in means)
        if layouts != ["axial", "orthogonal"]:
            raise ValueError(f"{recipe}: expected axial and orthogonal runs, found {layouts}")
        subjects = sorted(set.intersection(*(set(means[("gaussian", l)]) for l in layouts)))

        def score(metric, transform=lambda value: value):
            per_subject = [np.mean([transform(means[("gaussian", l)][s][metric]) for l in layouts])
                           for s in subjects]
            return float(np.mean(per_subject))
        scores[recipe] = {"subjects": len(subjects), "unobserved_dice": score(PRIMARY),
                          "whole_dice_error": score("whole_dice", lambda value: 1 - value),
                          "failed_runs": sum(row.get("status") == "failed" for row in rows)}
    uniform, observed = scores["uniform_lr0.001"], scores["observed_lr0.003"]
    difference = observed["unobserved_dice"] - uniform["unobserved_dice"]
    if abs(difference) > 0.002:
        selected, reason = ("observed_lr0.003" if difference > 0 else "uniform_lr0.001"), "primary"
    elif observed["whole_dice_error"] < uniform["whole_dice_error"]:
        selected, reason = "observed_lr0.003", "tie_break_whole_dice_error"
    else:
        selected, reason = "uniform_lr0.001", "tie_break_simpler_recipe"
    result = {"rule": "protocol_v1 fold-0 Gaussian recipe selection", "scores": scores,
              "unobserved_dice_difference_observed_minus_uniform": difference,
              "selected": selected, "decided_by": reason}
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select-gaussian", help="Apply the fold-0 Gaussian recipe rule")
    choose.add_argument("--uniform", nargs=2, required=True, metavar="RUN_DIR",
                        help="Orthogonal and axial run directories for the uniform recipe")
    choose.add_argument("--observed", nargs=2, required=True, metavar="RUN_DIR",
                        help="Orthogonal and axial run directories for the observed recipe")
    choose.add_argument("--output", required=True)
    summary = commands.add_parser("report", help="Confirmatory subject-level report")
    summary.add_argument("--runs", nargs="+", required=True)
    summary.add_argument("--expect-subset", default="confirmatory")
    summary.add_argument("--output", required=True)
    summary.add_argument("--family", choices=("v1", "v2"), default="v1",
                         help="Holm family: v1 (Gaussian, interpolation vs INR) or v2 (rbf, gp, ssm, unet, prior_inr vs INR)")
    config = commands.add_parser("select-config", help="Protocol v2 fold-0 configuration rule")
    config.add_argument("--candidate", nargs=3, action="append", required=True,
                        metavar=("NAME", "ORTH_RUN", "AXIAL_RUN"), help="Repeat in grid order")
    config.add_argument("--output", required=True)
    args = parser.parse_args()
    {"select-gaussian": select_gaussian, "report": report, "select-config": select_config}[args.command](args)


if __name__ == "__main__":
    main()
