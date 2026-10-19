"""MITEA cohort enumeration, multi-class labels, and subject-grouped splits.

Label values were checked against cardiac phase: label 2 shrinks from ED to
ES (ejection fractions of roughly 40-65%), while label 1 keeps a nearly
constant volume. Label 2 is therefore the LV cavity and label 1 the
myocardium. Every volume of a patient (both scans, both phases) stays in the
same split so that repeated acquisitions cannot leak across folds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

BACKGROUND, MYOCARDIUM, LV_CAVITY = 0, 1, 2
LABEL_NAMES = {BACKGROUND: "background", MYOCARDIUM: "myocardium", LV_CAVITY: "lv_cavity"}
_NAME = re.compile(r"^(MITEA_\d{3})_scan(\d)_(ED|ES)\.nii\.gz$")


@dataclass(frozen=True)
class Volume:
    subject: str
    scan: int
    phase: str
    image_path: Path
    label_path: Path


def enumerate_volumes(data_dir):
    """Return every labelled volume, sorted by subject, scan, and phase."""
    labels_dir = Path(data_dir) / "labels"
    images_dir = Path(data_dir) / "images"
    volumes = []
    for path in sorted(labels_dir.glob("MITEA_*.nii.gz")):
        match = _NAME.match(path.name)
        if not match:
            continue
        image = images_dir / path.name
        if not image.exists():
            raise FileNotFoundError(f"label without image: {path.name}")
        subject, scan, phase = match.groups()
        volumes.append(Volume(subject, int(scan), phase, image, path))
    if not volumes:
        raise FileNotFoundError(f"No MITEA labels found under {labels_dir}")
    return volumes


def load_demographics(data_dir):
    with open(Path(data_dir) / "demographics.csv", newline="") as handle:
        return {row["mitea_id"]: row for row in csv.DictReader(handle)}


def load_label(path):
    """Load a label as canonical ``(z, y, x)`` integers plus its NIfTI affine.

    The affine maps nibabel ``(x, y, z)`` voxel indices to world millimetres,
    matching the convention used throughout ``cardiac_common``.
    """
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    values = np.rint(data).astype(np.uint8)
    if not np.allclose(data, values) or not set(np.unique(values)) <= set(LABEL_NAMES):
        raise ValueError(f"unexpected label values in {path}")
    return np.ascontiguousarray(values.transpose(2, 1, 0)), image.affine.astype(np.float64)


def make_splits(subjects, strata, folds=5, test_fraction=0.2, seed=2027):
    """Hold out a stratified test set, then stratify the rest into CV folds.

    Within each stratum subjects are shuffled with ``seed`` and dealt
    round-robin, so small pathology groups are spread across folds.
    """
    if folds < 2 or not 0 < test_fraction < 1:
        raise ValueError("need at least two folds and 0 < test_fraction < 1")
    subjects = sorted(set(subjects))
    groups = defaultdict(list)
    for subject in subjects:
        groups[strata[subject]].append(subject)
    rng = random.Random(seed)
    test, development = [], []
    for stratum in sorted(groups):
        members = list(groups[stratum])
        rng.shuffle(members)
        held_out = int(round(len(members) * test_fraction))
        test.extend(members[:held_out])
        development.extend(members[held_out:])
    fold_members = [[] for _ in range(folds)]
    position = 0
    for stratum in sorted(groups):
        for subject in (s for s in development if strata[s] == stratum):
            fold_members[position % folds].append(subject)
            position += 1
    return {"test": sorted(test), "folds": [sorted(fold) for fold in fold_members]}


def fold_subjects(splits, fold):
    """Return ``(train, validation)`` subjects for one cross-validation fold."""
    folds = splits["folds"]
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold must be in [0, {len(folds) - 1}]")
    train = sorted(s for index, members in enumerate(folds) if index != fold for s in members)
    return train, list(folds[fold])


def build_manifest(data_dir, folds=5, test_fraction=0.2, seed=2027):
    volumes = enumerate_volumes(data_dir)
    demographics = load_demographics(data_dir)
    subjects = sorted({volume.subject for volume in volumes})
    missing = [subject for subject in subjects if subject not in demographics]
    if missing:
        raise ValueError(f"subjects missing demographics: {missing}")
    strata = {subject: demographics[subject]["category"] for subject in subjects}
    splits = make_splits(subjects, strata, folds, test_fraction, seed)
    fingerprint = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    return {
        "version": "mitea_cohort_v1",
        "seed": seed, "folds": folds, "test_fraction": test_fraction,
        "stratified_by": "demographics.csv category",
        "labels": {str(key): value for key, value in LABEL_NAMES.items()},
        "volumes_per_subject": {s: sum(v.subject == s for v in volumes) for s in subjects},
        "strata": strata,
        "splits": splits,
        "splits_sha256": fingerprint,
    }
