"""Build the protocol-v2 PCA shape model from development folds.

Example (fold-0 development model):
    .venv/bin/python scripts/train_ssm.py --data-dir dataset/mitea \
        --manifest benchmarks/splits/mitea_cohort_v1.json --train-folds 1 2 3 4 \
        --output runs/protocol_v2_models/ssm_train1234.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_common.cohort import enumerate_volumes, load_label
from cardiac_common.population import listed_training_subjects, training_subjects
from cardiac_common.ssm import build_ssm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-folds", nargs="+", type=int)
    source.add_argument("--train-subjects-file", default=None,
                        help="JSON file with a \"subjects\" list of development subjects")
    parser.add_argument("--max-modes", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(args.manifest).read_text())
    if args.train_folds is not None:
        subjects = training_subjects(manifest, args.train_folds)
    else:
        listed = json.loads(Path(args.train_subjects_file).read_text())["subjects"]
        subjects = listed_training_subjects(manifest, listed)
    paths = {(v.subject, v.scan, v.phase): v.label_path for v in enumerate_volumes(args.data_dir)}
    started = time.perf_counter()
    masks = [load_label(paths[(subject, 1, "ED")])[0] > 0 for subject in subjects]
    artifact = build_ssm(masks, subjects, args.max_modes)
    artifact.update(method="ssm", train_folds=args.train_folds, subject_list=args.train_subjects_file,
                    manifest_sha256=manifest["splits_sha256"],
                    build_seconds=time.perf_counter() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    explained = (artifact["sd"] ** 2).cumsum(0) / (artifact["sd"] ** 2).sum()
    kept = len(artifact["sd"])  # at most len(subjects) - 1
    at = "/".join(f"{explained[min(k, kept) - 1]:.3f}" for k in (5, 10, 20))
    print(f"{len(subjects)} subjects, {kept} modes; cumulative variance of kept modes at "
          f"5/10/20 (capped at {kept}): {at}; "
          f"{artifact['build_seconds']:.0f}s -> {output}")


if __name__ == "__main__":
    main()
