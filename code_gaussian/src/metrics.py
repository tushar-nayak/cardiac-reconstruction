"""Reconstruction metrics and mesh extraction utilities."""

from __future__ import annotations

import numpy as np


def binary_metrics(prediction, target, threshold=0.5):
    prediction = np.asarray(prediction) >= threshold
    target = np.asarray(target) >= 0.5
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    intersection = np.logical_and(prediction, target).sum()
    pred_sum = prediction.sum()
    target_sum = target.sum()
    union = np.logical_or(prediction, target).sum()
    return {
        "accuracy": float(np.equal(prediction, target).mean()),
        "iou": float(intersection / max(union, 1)),
        "dice": float(2 * intersection / max(pred_sum + target_sum, 1)),
        "pred_voxels": int(pred_sum),
        "target_voxels": int(target_sum),
    }


def volume_to_mesh(volume, affine, level=0.5):
    """Extract an affine-aware mesh from a canonical ``(z,y,x)`` volume."""
    from skimage.measure import marching_cubes

    volume = np.asarray(volume, dtype=np.float32)
    affine = np.asarray(affine, dtype=np.float32)
    if volume.ndim != 3 or affine.shape != (4, 4):
        raise ValueError("volume must be 3D and affine must be 4x4")
    if volume.min() > level or volume.max() < level:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    vertices_zyx, faces, _, _ = marching_cubes(volume, level=level)
    vertices_xyz = vertices_zyx[:, [2, 1, 0]]
    vertices_h = np.concatenate(
        [vertices_xyz, np.ones((len(vertices_xyz), 1), dtype=np.float32)], axis=1
    )
    vertices_world = (vertices_h @ affine.T)[:, :3]
    return vertices_world.astype(np.float32), faces.astype(np.int32)


def save_mesh_npz(path, vertices, faces):
    np.savez_compressed(path, vertices=np.asarray(vertices), faces=np.asarray(faces))
