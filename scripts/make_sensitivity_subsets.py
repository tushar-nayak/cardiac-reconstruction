"""Write the subject lists of the sensitivity study (benchmarks/mitea_sensitivity_v1.md).

Lists depend only on the frozen manifest, its pathology strata, and fixed RNG
seeds; they are written before any training. Re-running verifies existing
files and never overwrites them.

    .venv/bin/python scripts/make_sensitivity_subsets.py \
        --manifest benchmarks/splits/mitea_cohort_v1.json --output-dir runs/mitea_sensitivity_v1/subsets
"""

import argparse
import json
from pathlib import Path

import numpy as np

EVALUATION_FOLDS = (1, 2, 3, 4)
SIZES = (10, 25, 50)
MATCHED_SIZE = 66


def training_pool(manifest, fold):
    """The other four development folds of evaluation fold ``fold``."""
    return sorted(s for index, subjects in enumerate(manifest["splits"]["folds"]) if index != fold
                  for s in subjects)


def by_pathology(subjects, strata):
    return (sorted(s for s in subjects if strata[s] != "healthy"),
            sorted(s for s in subjects if strata[s] == "healthy"))


def subject_lists(manifest):
    """Name -> list record for Studies B1 and B2."""
    strata = manifest["strata"]
    development = sorted(s for fold in manifest["splits"]["folds"] for s in fold)
    lists = {"healthy66": {"rule": "all healthy development subjects", "evaluation_fold": None,
                           "subjects": by_pathology(development, strata)[1]}}
    for k in EVALUATION_FOLDS:
        pool = training_pool(manifest, k)
        pathology, healthy = by_pathology(pool, strata)
        fraction = len(pathology) / len(pool)
        rng = np.random.default_rng(2027 + k)
        count = round(MATCHED_SIZE * fraction)
        chosen = (list(rng.choice(pathology, count, replace=False))
                  + list(rng.choice(healthy, MATCHED_SIZE - count, replace=False)))
        lists[f"mixed66_fold{k}"] = {
            "rule": f"{MATCHED_SIZE} from the fold-{k} training pool at its pathology fraction, seed {2027 + k}",
            "evaluation_fold": k, "subjects": sorted(str(s) for s in chosen)}
        rng = np.random.default_rng(3027 + k)
        pathology_order, healthy_order = list(rng.permutation(pathology)), list(rng.permutation(healthy))
        for n in SIZES:
            count = round(n * fraction)
            lists[f"size{n}_fold{k}"] = {
                "rule": f"nested prefix of the fold-{k} pool, pathology and healthy shuffled separately, "
                        f"seed {3027 + k}",
                "evaluation_fold": k,
                "subjects": sorted(str(s) for s in pathology_order[:count] + healthy_order[:n - count])}
        lists[f"b1_eval_fold{k}"] = {"rule": f"pathology subjects of evaluation fold {k}", "evaluation_fold": k,
                                     "subjects": by_pathology(manifest["splits"]["folds"][k], strata)[0]}
    test = set(manifest["splits"]["test"])
    for name, record in lists.items():
        subjects = record["subjects"]
        record.update(name=name, pathology_count=sum(strata[s] != "healthy" for s in subjects),
                      manifest_sha256=manifest["splits_sha256"])
        if test & set(subjects) or len(set(subjects)) != len(subjects):
            raise RuntimeError(f"{name}: test subjects or duplicates")
        k = record["evaluation_fold"]
        if not name.startswith("b1_eval") and k is not None and set(subjects) & set(manifest["splits"]["folds"][k]):
            raise RuntimeError(f"{name}: training list contains evaluation-fold subjects")
    return lists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, record in subject_lists(manifest).items():
        path = output / f"{name}.json"
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise RuntimeError(f"{path} differs from the deterministic list; refusing to overwrite")
            continue
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{path}: {len(record['subjects'])} subjects, {record['pathology_count']} with pathology")


if __name__ == "__main__":
    main()
