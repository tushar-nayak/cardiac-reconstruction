"""Summaries for the apical-view EF study (``benchmarks/mitea_apical_ef_v1.md``).

EF comes from each method's predicted ED and ES cavity volumes under the same
layout and condition. The primary comparison is the paired difference in
absolute EF error, method minus biplane Simpson, at each misplacement level
(biplane, known pose). Triplane reconstructions are compared with biplane
Simpson at the same level. Descriptive only: bootstrap CIs, no p-values.

    .venv/bin/python scripts/summarize_apical_ef.py --root runs/mitea_apical_ef_v1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from summarize_benchmark import bootstrap_ci

FOLDS = (1, 2, 3, 4)
MATERIAL = {"ef": 1.0, "edv": 5.0, "esv": 5.0}
RECONSTRUCTIONS = ("inr", "rbf", "ssm", "unet", "prior_inr")
CONDITIONS = [("biplane", "none", "known"), ("biplane", "mild", "known"), ("biplane", "severe", "known"),
              ("biplane", "mild", "assumed"), ("biplane", "severe", "assumed"),
              ("triplane", "none", "known"), ("triplane", "severe", "known")]


def load(root):
    rows, integrity = [], {"units_expected": len(FOLDS), "units_complete": 0, "failed": [],
                           "empty_predictions": 0, "fingerprint_conflicts": 0}
    for k in FOLDS:
        unit = root / f"E_fold{k}"
        if not (unit / ".complete").exists():
            continue
        integrity["units_complete"] += 1
        rows += json.loads((unit / "results.json").read_text())["rows"]
    digests = defaultdict(set)
    for row in rows:
        digests[(row["subject"], row["phase"], row["layout"], row["misplacement"], row["pose"])].add(
            row["observation_sha256"])
        if row["status"] != "ok":
            integrity["failed"].append({k: row.get(k) for k in ("subject", "phase", "layout", "misplacement",
                                                                  "pose", "method", "error")})
        elif row.get("empty_prediction"):
            integrity["empty_predictions"] += 1
    integrity["fingerprint_conflicts"] = sum(len(v) > 1 for v in digests.values())
    return rows, integrity


def subject_measures(rows):
    """``{(method, layout, level, pose): {subject: {edv, esv, ef, true_*, dice...}}}``."""
    volumes = defaultdict(dict)
    for row in rows:
        if row["status"] == "ok":
            volumes[(row["method"], row["layout"], row["misplacement"], row["pose"], row["subject"])][row["phase"]] = row
    result = defaultdict(dict)
    for (method, layout, level, pose, subject), phases in volumes.items():
        if set(phases) != {"ED", "ES"} or phases["ED"]["predicted_ml"] <= 0:
            continue
        ed, es = phases["ED"], phases["ES"]
        measures = {"edv": ed["predicted_ml"], "esv": es["predicted_ml"],
                    "ef": 100 * (ed["predicted_ml"] - es["predicted_ml"]) / ed["predicted_ml"],
                    "true_edv": ed["true_ml"], "true_esv": es["true_ml"],
                    "true_ef": 100 * (ed["true_ml"] - es["true_ml"]) / ed["true_ml"]}
        if "dice" in ed:
            measures.update(dice=(ed["dice"] + es["dice"]) / 2,
                            unobserved_dice=(ed["unobserved_dice"] + es["unobserved_dice"]) / 2)
        result[(method, layout, level, pose)][subject] = measures
    return result


def agreement(values, quantity):
    errors = np.array([v[quantity] - v[f"true_{quantity}"] for v in values.values()])
    bias, spread = float(errors.mean()), float(errors.std(ddof=1)) if len(errors) > 1 else 0.0
    return {"n": len(errors), "bias": bias, "mae": float(np.abs(errors).mean()),
            "mae_95ci": list(bootstrap_ci(np.abs(errors))), "loa": [bias - 1.96 * spread, bias + 1.96 * spread]}


def versus_simpson(values, simpson, quantity):
    """Paired difference in absolute error, method minus Simpson (negative favours the method)."""
    subjects = sorted(set(values) & set(simpson))
    if not subjects:
        return None
    error = lambda v: abs(v[quantity] - v[f"true_{quantity}"])
    differences = np.array([error(values[s]) - error(simpson[s]) for s in subjects])
    low, high = bootstrap_ci(differences)
    mean = float(differences.mean())
    return {"n_subjects": len(subjects), "mean_difference": mean, "bootstrap_95ci": [low, high],
            "subjects_better": int((differences < 0).sum()),
            "material": bool((low > 0 or high < 0) and abs(mean) >= MATERIAL[quantity])}


def summarize(rows):
    measures = subject_measures(rows)
    result = {"agreement": {}, "versus_simpson": {}, "dice": {}}
    for layout, level, pose in CONDITIONS:
        name = f"{layout}/{level}/{pose}"
        simpson = measures.get(("simpson", "biplane", level, "known"), {})
        for method in ("simpson",) + RECONSTRUCTIONS + ("mean_shape",):
            values = measures.get((method, layout, level, pose))
            if not values:
                continue
            if method != "mean_shape":
                result["agreement"].setdefault(name, {})[method] = {
                    q: agreement(values, q) for q in ("ef", "edv", "esv")}
            if method in RECONSTRUCTIONS:
                result["versus_simpson"].setdefault(name, {})[method] = {
                    q: versus_simpson(values, simpson, q) for q in ("ef", "edv", "esv")}
            if method != "simpson":
                dice = [v["dice"] for v in values.values()]
                unobserved = [v["unobserved_dice"] for v in values.values()]
                result["dice"].setdefault(name, {})[method] = {"dice": float(np.mean(dice)),
                                                               "unobserved_dice": float(np.mean(unobserved))}
    result["primary"] = {level: result["versus_simpson"].get(f"biplane/{level}/known", {})
                         for level in ("none", "mild", "severe")}
    return result


def markdown(result, integrity):
    lines = ["# MITEA apical-view EF study v1: summary", "",
             f"Units complete: {integrity['units_complete']}/{integrity['units_expected']}. Failed runs: "
             f"{len(integrity['failed'])}. Empty predictions: {integrity['empty_predictions']}. "
             f"Fingerprint conflicts: {integrity['fingerprint_conflicts']}.", "",
             "EF in percentage points. Versus Simpson: mean paired difference in absolute EF error "
             "[95% CI], subjects better/n; negative favours the method; * = material (≥ 1 point).", ""]
    for name, methods in result["agreement"].items():
        lines += [f"## {name}", "", "| Method | EF MAE [95% CI] | EF bias | EF LoA | EDV bias (mL) | ESV bias (mL) "
                  "| Δ|EF err| vs Simpson | Dice |", "|---|---|---:|---|---:|---:|---|---:|"]
        for method, stats in methods.items():
            ef = stats["ef"]
            vs = result["versus_simpson"].get(name, {}).get(method, {}).get("ef")
            vs_text = "–" if vs is None else (f"{vs['mean_difference']:+.2f} [{vs['bootstrap_95ci'][0]:+.2f}, "
                                               f"{vs['bootstrap_95ci'][1]:+.2f}] {vs['subjects_better']}/"
                                               f"{vs['n_subjects']}{' *' if vs['material'] else ''}")
            dice = result["dice"].get(name, {}).get(method, {}).get("dice")
            lines.append(f"| {method} | {ef['mae']:.2f} [{ef['mae_95ci'][0]:.2f}, {ef['mae_95ci'][1]:.2f}] | "
                         f"{ef['bias']:+.2f} | [{ef['loa'][0]:+.1f}, {ef['loa'][1]:+.1f}] | "
                         f"{stats['edv']['bias']:+.1f} | {stats['esv']['bias']:+.1f} | {vs_text} | "
                         f"{'–' if dice is None else f'{dice:.3f}'} |")
        lines.append("")
    lines += ["## Cavity Dice (mean of ED and ES)", "", "| Condition | Method | Dice | Unobserved Dice |",
              "|---|---|---:|---:|"]
    for name, methods in result["dice"].items():
        for method, stats in methods.items():
            lines.append(f"| {name} | {method} | {stats['dice']:.3f} | {stats['unobserved_dice']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/mitea_apical_ef_v1"))
    args = parser.parse_args()
    rows, integrity = load(args.root)
    result = summarize(rows)
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"protocol": "benchmarks/mitea_apical_ef_v1.md",
                                                     "integrity": integrity, **result}, indent=2) + "\n")
    (output / "results.md").write_text(markdown(result, integrity))
    print(f"{integrity['units_complete']}/{integrity['units_expected']} units complete; wrote {output}")


if __name__ == "__main__":
    main()
