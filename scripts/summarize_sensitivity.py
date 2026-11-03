"""Summaries for the sensitivity analyses (``benchmarks/mitea_sensitivity_v1.md``).

Descriptive only: subject-level values (seeds averaged within subject), 95%
percentile bootstrap CIs over subjects, and paired differences with the number
of subjects higher. A difference is flagged material when its CI excludes 0
and |mean| >= 0.01 Dice. No hypothesis tests. Works on partial output and
reports completeness, so it can also track a running study.

    .venv/bin/python scripts/summarize_sensitivity.py --root runs/mitea_sensitivity_v1
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from summarize_benchmark import METRICS, bootstrap_ci, describe, subject_means

PRIMARY = "unobserved_dice"
MATERIAL = 0.01
LAYOUTS = ("orthogonal", "axial")
FOLDS = (1, 2, 3, 4)
CONDITIONS = ("clean", "random2", "random4", "inside2", "outside2")
COUNTS = {"orthogonal": (1, 2, 3), "axial": (1, 2, 3, 4, 6)}
SIZES = (10, 25, 50)
PER_SUBJECT = ("inr", "rbf", "gp", "interpolation")
METRICS = dict(METRICS, signed_volume_error_ml=lambda m: m["predicted_volume_ml"] - m["target_volume_ml"])
METRICS.pop("absolute_delta_k")


def expected_units():
    units = [f"A_{c}_fold{k}_{layout}" for c in CONDITIONS for k in FOLDS for layout in LAYOUTS]
    units += [f"C_{layout}{n}_fold{k}" for k in FOLDS for layout, counts in COUNTS.items()
              for n in counts if n != 3]
    units += [f"B1_{prior}_fold{k}_{layout}" for k in FOLDS for prior in ("healthy66", "mixed66")
              for layout in LAYOUTS]
    units += [f"B2_size{n}_fold{k}_{layout}" for n in SIZES for k in FOLDS for layout in LAYOUTS]
    return units


def load(root):
    """``{(study, condition): rows}`` from completed units, plus integrity counts."""
    groups, integrity = defaultdict(list), {"failed_runs": [], "empty_predictions": 0, "fingerprint_conflicts": []}
    complete = [unit for unit in expected_units() if (root / unit / ".complete").exists()]
    for unit in complete:
        study, condition = unit.split("_")[:2]
        rows = json.loads((root / unit / "results.json").read_text())["runs"]
        fingerprints = defaultdict(set)
        for row in rows:
            fingerprints[row["subject"]].add(row["supervision_sha256"])
            if row.get("status") == "failed":
                integrity["failed_runs"].append({"unit": unit, "method": row["method"], "subject": row["subject"],
                                                 "error": row.get("error")})
            elif row.get("empty_prediction"):
                integrity["empty_predictions"] += 1
        integrity["fingerprint_conflicts"] += [f"{unit}/{s}" for s, v in fingerprints.items() if len(v) > 1]
        groups[(study, condition)].extend(rows)
    integrity.update(units_expected=len(expected_units()), units_complete=len(complete))
    return groups, integrity


def paired(a, b, metric=PRIMARY):
    """``a - b`` over subjects with both values; None when they share no subject."""
    subjects = sorted(s for s in set(a) & set(b) if a[s][metric] is not None and b[s][metric] is not None)
    if not subjects:
        return None
    differences = np.array([a[s][metric] - b[s][metric] for s in subjects])
    low, high = bootstrap_ci(differences)
    mean = float(differences.mean())
    return {"n_subjects": len(subjects), "mean_difference": mean, "bootstrap_95ci": [low, high],
            "subjects_higher": int((differences > 0).sum()),
            "material": bool((low > 0 or high < 0) and abs(mean) >= MATERIAL)}


def summary(values, metric=PRIMARY):
    """Describe one method's subject values, with a bootstrap CI of the mean."""
    valid = [v[metric] for v in values.values() if v[metric] is not None]
    result = describe(valid)
    if valid:
        result["bootstrap_95ci"] = list(bootstrap_ci(valid))
    return result


def restrict(values, subjects):
    return {s: v for s, v in values.items() if s in subjects}


def study_a(means):
    clean = means.get(("A", "clean"), {})
    result = {"descriptive": {}, "change_from_clean": {}, "signed_volume_error_ml": {}, "ordering": {}}
    for condition in CONDITIONS:
        current = means.get(("A", condition), {})
        for (method, layout), values in sorted(current.items()):
            result["descriptive"].setdefault(condition, {}).setdefault(layout, {})[method] = summary(values)
            result["signed_volume_error_ml"].setdefault(condition, {}).setdefault(layout, {})[method] = summary(
                values, "signed_volume_error_ml")
            if condition != "clean" and (method, layout) in clean:
                result["change_from_clean"].setdefault(condition, {}).setdefault(layout, {})[method] = paired(
                    values, clean[(method, layout)])
        for layout in LAYOUTS:
            ranked = sorted((s["mean"], m) for (m, lay), v in current.items()
                            if lay == layout and "mean" in (s := summary(v)))
            if ranked:
                result["ordering"].setdefault(condition, {})[layout] = [m for _, m in reversed(ranked)]
    result["primary"] = result["change_from_clean"].get("random2", {})
    return result


def emptied_sections(rows):
    """Sections with clean foreground that a perturbation emptied, once per subject and layout."""
    seen, counts = set(), Counter()
    for row in rows:
        key = (row["subject"], row["plane_layout"])
        perturbation = row.get("section_perturbation")
        if perturbation is None or key in seen:
            continue
        seen.add(key)
        counts[row["plane_layout"]] += sum(p["foreground_before"] > 0 and p["foreground_after"] == 0
                                           for p in perturbation["planes"])
    return dict(counts)


def study_b(means, groups):
    healthy, mixed = means.get(("B1", "healthy66"), {}), means.get(("B1", "mixed66"), {})
    reference = means.get(("A", "clean"), {})
    strata = {row["subject"]: row["category"] for rows in groups.values() for row in rows}
    b1 = {"healthy_minus_mixed": {}, "descriptive": {}, "by_category": {}, "healthy_minus_reference": {}}
    for (method, layout), values in sorted(healthy.items()):
        if (method, layout) not in mixed:
            continue
        b1["healthy_minus_mixed"].setdefault(layout, {})[method] = paired(values, mixed[(method, layout)])
        for name, source in (("healthy66", values), ("mixed66", mixed[(method, layout)])):
            b1["descriptive"].setdefault(name, {}).setdefault(layout, {})[method] = summary(source)
        if (method, layout) in reference:
            b1["healthy_minus_reference"].setdefault(layout, {})[method] = paired(
                values, restrict(reference[(method, layout)], set(values)))
        for category in sorted({strata[s] for s in values}):
            members = {s for s in values if strata[s] == category}
            b1["by_category"].setdefault(category, {}).setdefault(layout, {})[method] = paired(
                restrict(values, members), restrict(mixed[(method, layout)], members))
    b2 = {"descriptive": {}, "minus_full_pool": {}, "smallest_n_within_0.01": {}}
    for (method, layout), full in sorted(reference.items()):
        if method not in ("ssm", "prior_inr", "unet", "mean_shape"):
            continue
        full_mean = summary(full).get("mean")
        b2["descriptive"].setdefault(layout, {}).setdefault(method, {})["full"] = summary(full)
        within = None
        for n in SIZES:
            values = means.get(("B2", f"size{n}"), {}).get((method, layout))
            if values is None:
                continue
            mean = summary(values).get("mean")
            b2["descriptive"][layout][method][str(n)] = summary(values)
            b2["minus_full_pool"].setdefault(layout, {}).setdefault(method, {})[str(n)] = paired(values, full)
            if within is None and None not in (mean, full_mean) and abs(mean - full_mean) <= MATERIAL:
                within = n
        b2["smallest_n_within_0.01"].setdefault(layout, {})[method] = within
    return {"B1": b1, "B2": b2}


def study_c(means):
    """Plane-count curves; three planes are the Study A clean condition."""
    result = {"descriptive": {}, "minus_prior_inr": {}, "minus_mean_shape": {}, "smallest_count_within_0.01": {}}
    clean = means.get(("A", "clean"), {})
    zero = next((v for (m, lay), v in clean.items() if m == "mean_shape" and lay == "orthogonal"), None)
    if zero is not None:
        result["zero_planes_mean_shape_whole_dice"] = summary(zero, "whole_dice")
    for family, counts in COUNTS.items():
        for n in counts:
            current = ({k: v for k, v in clean.items() if k[1] == family} if n == 3
                       else means.get(("C", f"{family}{n}"), {}))
            for (method, _), values in sorted(current.items()):
                result["descriptive"].setdefault(family, {}).setdefault(method, {})[str(n)] = summary(values)
                for reference, key in (("prior_inr", "minus_prior_inr"), ("mean_shape", "minus_mean_shape")):
                    if method != reference and (reference, family) in current:
                        result[key].setdefault(family, {}).setdefault(method, {})[str(n)] = paired(
                            values, current[(reference, family)])
        for method in PER_SUBJECT:
            gaps = result["minus_prior_inr"].get(family, {}).get(method, {})
            result["smallest_count_within_0.01"].setdefault(family, {})[method] = next(
                (int(n) for n in sorted(gaps, key=int) if gaps[n] and gaps[n]["mean_difference"] >= -MATERIAL), None)
    return result


def fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def fmt_diff(diff):
    if diff is None:
        return "–"
    low, high = diff["bootstrap_95ci"]
    flag = " *" if diff["material"] else ""
    return f"{diff['mean_difference']:+.3f} [{low:+.3f}, {high:+.3f}] {diff['subjects_higher']}/{diff['n_subjects']}{flag}"


def markdown(result):
    lines = ["# MITEA sensitivity analyses v1: summary", "",
             f"Units complete: {result['integrity']['units_complete']}/{result['integrity']['units_expected']}. "
             f"Failed runs: {len(result['integrity']['failed_runs'])}. Empty predictions: "
             f"{result['integrity']['empty_predictions']}. Fingerprint conflicts: "
             f"{len(result['integrity']['fingerprint_conflicts'])}.", "",
             "Differences: mean [95% bootstrap CI] subjects higher/n; * = material (CI excludes 0 and |mean| ≥ 0.01).",
             "", "## Study A: change in unobserved Dice from clean sections", ""]
    a = result["A"]
    lines += ["| Layout | Method | Clean | " + " | ".join(CONDITIONS[1:]) + " |", "|---|---|---:|" + "---|" * 4]
    for layout in LAYOUTS:
        for method, clean in sorted(a["descriptive"].get("clean", {}).get(layout, {}).items()):
            cells = [fmt_diff(a["change_from_clean"].get(c, {}).get(layout, {}).get(method)) for c in CONDITIONS[1:]]
            lines.append(f"| {layout} | {method} | {fmt(clean.get('mean'))} | " + " | ".join(cells) + " |")
    lines += ["", "Signed volume error (mL, mean) by condition:", "",
              "| Layout | Method | " + " | ".join(CONDITIONS) + " |", "|---|---|" + "---:|" * len(CONDITIONS)]
    for layout in LAYOUTS:
        for method in sorted(a["signed_volume_error_ml"].get("clean", {}).get(layout, {})):
            cells = [fmt(a["signed_volume_error_ml"].get(c, {}).get(layout, {}).get(method, {}).get("mean"), 1)
                     for c in CONDITIONS]
            lines.append(f"| {layout} | {method} | " + " | ".join(cells) + " |")
    b1 = result["B"]["B1"]
    lines += ["", "## Study B1: healthy-only minus size-matched mixed prior (pathology subjects)", "",
              "| Layout | Method | Healthy66 | Mixed66 | Difference |", "|---|---|---:|---:|---|"]
    for layout, methods in b1["healthy_minus_mixed"].items():
        for method, diff in sorted(methods.items()):
            h = b1["descriptive"]["healthy66"][layout][method]["mean"]
            m = b1["descriptive"]["mixed66"][layout][method]["mean"]
            lines.append(f"| {layout} | {method} | {fmt(h)} | {fmt(m)} | {fmt_diff(diff)} |")
    b2 = result["B"]["B2"]
    lines += ["", "## Study B2: unobserved Dice by training-set size", "",
              "| Layout | Method | n=10 | n=25 | n=50 | Full pool | Smallest n within 0.01 |",
              "|---|---|---:|---:|---:|---:|---|"]
    for layout, methods in b2["descriptive"].items():
        for method, by_n in sorted(methods.items()):
            cells = [fmt(by_n.get(str(n), {}).get("mean")) for n in SIZES] + [fmt(by_n["full"]["mean"])]
            within = b2["smallest_n_within_0.01"][layout].get(method)
            lines.append(f"| {layout} | {method} | " + " | ".join(cells) + f" | {within or '>50 or pending'} |")
    c = result["C"]
    for family, counts in COUNTS.items():
        lines += ["", f"## Study C: {family} family, unobserved Dice by plane count", "",
                  "| Method | " + " | ".join(f"n={n}" for n in counts) + " |", "|---|" + "---:|" * len(counts)]
        for method, by_n in sorted(c["descriptive"].get(family, {}).items()):
            lines.append(f"| {method} | " + " | ".join(fmt(by_n.get(str(n), {}).get("mean")) for n in counts) + " |")
        lines += ["", f"Gap to `prior_inr` ({family}):", "",
                  "| Method | " + " | ".join(f"n={n}" for n in counts) + " |", "|---|" + "---|" * len(counts)]
        for method in PER_SUBJECT:
            gaps = c["minus_prior_inr"].get(family, {}).get(method, {})
            lines.append(f"| {method} | " + " | ".join(fmt_diff(gaps.get(str(n))) for n in counts) + " |")
    if "zero_planes_mean_shape_whole_dice" in c:
        lines += ["", f"Zero planes (`mean_shape`, whole-volume Dice): "
                      f"{fmt(c['zero_planes_mean_shape_whole_dice']['mean'])}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/mitea_sensitivity_v1"))
    args = parser.parse_args()
    groups, integrity = load(args.root)
    means = {}
    for key, rows in groups.items():
        for (method, layout), values in subject_means(rows, METRICS).items():
            means.setdefault(key, {})[(method, layout)] = values
    integrity["emptied_sections"] = {condition: emptied_sections(groups.get(("A", condition), []))
                                     for condition in CONDITIONS[1:]}
    result = {"protocol": "benchmarks/mitea_sensitivity_v1.md", "integrity": integrity,
              "A": study_a(means), "B": study_b(means, groups), "C": study_c(means)}
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "results.md").write_text(markdown(result))
    print(f"{integrity['units_complete']}/{integrity['units_expected']} units complete; wrote {output}")


if __name__ == "__main__":
    main()
