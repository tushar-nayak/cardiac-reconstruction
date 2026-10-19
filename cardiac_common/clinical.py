"""Surface distances, volumes, and the biplane Simpson ejection-fraction baseline."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from cardiac_common.cohort import LV_CAVITY, MYOCARDIUM
from cardiac_common.views import cavity_profile


def voxel_spacing_zyx(affine):
    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)[::-1]


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def surface_distances(prediction, target, affine):
    """Symmetric boundary distances in millimetres.

    HD95 is the larger of the two directed 95th percentiles; ASSD averages all
    boundary-to-boundary distances. Both are ``None`` if either mask is empty.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if not prediction.any() or not target.any():
        return {"hd95_mm": None, "assd_mm": None, "hausdorff_mm": None}
    spacing = voxel_spacing_zyx(affine)
    pred_surface, target_surface = _surface(prediction), _surface(target)
    to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[target_surface]
    return {
        "hd95_mm": float(max(np.percentile(to_target, 95), np.percentile(to_pred, 95))),
        "assd_mm": float(np.concatenate((to_target, to_pred)).mean()),
        "hausdorff_mm": float(max(to_target.max(), to_pred.max())),
    }


def label_volumes_ml(label, affine):
    voxel_ml = abs(np.linalg.det(np.asarray(affine)[:3, :3])) / 1000
    return {
        "lv_cavity_ml": float((label == LV_CAVITY).sum() * voxel_ml),
        "myocardium_ml": float((label == MYOCARDIUM).sum() * voxel_ml),
    }


def ejection_fraction(edv, esv):
    if edv <= 0:
        raise ValueError("end-diastolic volume must be positive")
    return 100.0 * (edv - esv) / edv


def biplane_simpson_ml(first, second, disks=20):
    """Modified Simpson method of disks from two apical ``PlaneSample`` views.

    Both planes must share the long axis as their ``u`` direction and origin,
    as produced by ``apical_planes``. Disk length spans the cavity along the
    axis in either view, matching the guideline choice of the longer length.
    Each disk is an ellipse with diameters taken from the two views.
    """
    if disks < 1:
        raise ValueError("disks must be positive")
    profiles = [cavity_profile(view) for view in (first, second)]
    present = np.concatenate([u[w > 0] for u, w in profiles])
    if not len(present):
        return 0.0, {"length_mm": 0.0, "disks": disks}
    spacing = max(float(np.diff(view.u_mm).mean()) if len(view.u_mm) > 1 else 1.0
                  for view in (first, second))
    start, stop = present.min() - spacing / 2, present.max() + spacing / 2
    height = (stop - start) / disks
    centres = start + height * (np.arange(disks) + 0.5)
    diameters = [np.interp(centres, u, w, left=0.0, right=0.0) for u, w in profiles]
    volume_mm3 = np.pi / 4 * np.sum(diameters[0] * diameters[1]) * height
    return float(volume_mm3 / 1000), {"length_mm": float(stop - start), "disks": disks}
