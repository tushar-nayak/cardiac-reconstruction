"""Observed-mask-only radial interpolation and fixed FeEcho4D ablation grids."""

from __future__ import annotations

import numpy as np

from cardiac_common.interpolation import signed_distance_2d


LAYOUT_ANGLES = {
    "spread2": (30, 150),
    "spread3": (30, 90, 150),
    "spread6": (0, 30, 60, 90, 120, 150),
    "spread12": tuple(range(0, 180, 15)),
    "clustered3": (75, 90, 105),
}
LAYOUTS = {key: tuple(angle // 5 + 1 for angle in angles)
           for key, angles in LAYOUT_ANGLES.items()}
COMMON_VIEWS = tuple(i for i in range(1, 38) if (i - 1) % 3 != 0)
SEEDS = (17, 23, 42)
METHODS = ("nearest", "sdf", "inr", "gaussian")


class RadialInterpolator:
    """Interpolate in index/angle space without accessing withheld labels.

    Each mask is in its original image coordinate frame. No rotation-center,
    image-center, or angular-periodicity assumption is introduced.
    """

    def __init__(self, view_ids, observed_masks, method="sdf"):
        ids = np.asarray(view_ids, dtype=float)
        masks = np.asarray(observed_masks, dtype=bool)
        if (ids.ndim != 1 or len(ids) == 0 or not np.isfinite(ids).all()
                or (ids < 1).any() or (ids > 37).any()
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("observed view IDs must be distinct finite values in [1,37]")
        if masks.ndim != 3 or masks.shape[0] != len(ids):
            raise ValueError("supply one 2D mask per observed view")
        if method not in ("nearest", "sdf"):
            raise ValueError("method must be nearest or sdf")
        order = np.argsort(ids)
        self.ids = ids[order]
        self.masks = masks[order].copy()
        self.method = method
        self.fields = (np.stack([signed_distance_2d(mask) for mask in self.masks])
                       if method == "sdf" else None)

    def predict(self, query_ids):
        query = np.asarray(query_ids, dtype=float)
        if (query.ndim != 1 or not np.isfinite(query).all()
                or (query < 1).any() or (query > 37).any()):
            raise ValueError("query IDs must be finite values in [1,37]")
        output = []
        for view_id in query:
            if self.method == "nearest":
                output.append(self.masks[np.abs(self.ids - view_id).argmin()])
                continue
            hi = int(np.searchsorted(self.ids, view_id))
            if hi == 0:
                field = self.fields[0]
            elif hi == len(self.ids):
                field = self.fields[-1]
            else:
                weight = (view_id - self.ids[hi - 1]) / (self.ids[hi] - self.ids[hi - 1])
                field = (1 - weight) * self.fields[hi - 1] + weight * self.fields[hi]
            output.append(field >= 0)
        return np.stack(output) if output else np.empty((0, *self.masks.shape[1:]), bool)


def pixel_counts(prediction, target):
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes differ")
    return {"tp": int(np.logical_and(pred, truth).sum()),
            "predicted": int(pred.sum()), "target": int(truth.sum()),
            "pixels": int(pred.size), "correct": int((pred == truth).sum())}


def pool_counts(records):
    if not records:
        return None
    total = {key: sum(row[key] for row in records)
             for key in ("tp", "predicted", "target", "pixels", "correct")}
    denominator = total["predicted"] + total["target"]
    union = denominator - total["tp"]
    return {**total, "dice": 2 * total["tp"] / denominator if denominator else 1.0,
            "iou": total["tp"] / union if union else 1.0,
            "pixel_accuracy": total["correct"] / total["pixels"]}


def radial_metrics(prediction, target, query_ids, observed_ids):
    """Score predictions only after fitting, preserving counts per query view."""
    if len(prediction) != len(query_ids) or len(target) != len(query_ids):
        raise ValueError("one prediction and target required per query ID")
    if set(query_ids) & set(observed_ids):
        raise ValueError("scored queries must be disjoint from observations")
    per_view = []
    for pred, truth, view_id in zip(prediction, target, query_ids):
        distance = min(abs(view_id - index) for index in observed_ids) * 5
        per_view.append({"view_id": int(view_id), "angle": int((view_id - 1) * 5),
                         "distance_degrees": int(distance),
                         "bracketed": bool(min(observed_ids) <= view_id <= max(observed_ids)),
                         **pixel_counts(pred, truth)})
    common = [row for row in per_view if row["view_id"] in COMMON_VIEWS]
    if {row["view_id"] for row in common} != set(COMMON_VIEWS):
        raise ValueError("all common evaluation views must be present")
    subsets = {
        "common": common,
        "common_interior": [row for row in common if 30 <= row["angle"] <= 150],
        "bracketed": [row for row in common if row["bracketed"]],
        "outside_range": [row for row in common if not row["bracketed"]],
        "gap_0_15": [row for row in common if row["distance_degrees"] <= 15],
        "gap_15_30": [row for row in common if 15 < row["distance_degrees"] <= 30],
        "gap_30_60": [row for row in common if 30 < row["distance_degrees"] <= 60],
        "gap_gt60": [row for row in common if row["distance_degrees"] > 60],
        "all_unobserved": per_view,
    }
    common_view_scores = [pool_counts([row]) for row in common]
    return {"subsets": {name: pool_counts(values) for name, values in subsets.items()},
            "common_mean_view_dice": float(np.mean([row["dice"] for row in common_view_scores])),
            "per_view": per_view}
