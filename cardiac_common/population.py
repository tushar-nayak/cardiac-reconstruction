"""Shared pieces for population methods (protocol v2): crop frame and guards.

The crop is centred on the array centre and depends on array shape only, so
placing a subject in it never reads the label.
"""

import numpy as np
from scipy import ndimage

CROP_SHAPE = (128, 128, 160)
SDF_CLIP_MM = 30.0


def crop_origin(shape, crop_shape=CROP_SHAPE):
    """Array index of crop voxel 0 along each axis (may be negative)."""
    return tuple((size - 1) // 2 - (crop - 1) // 2 for size, crop in zip(shape, crop_shape))


def crop_slices(shape, crop_shape=CROP_SHAPE):
    """Paired (array, crop) slices for a crop centred at ``(shape - 1) / 2``.

    Where the array is smaller than the crop, the crop is padded; where it is
    larger, the array is cut. Uses array shape only.
    """
    source, target = [], []
    for size, crop, start in zip(shape, crop_shape, crop_origin(shape, crop_shape)):
        low, high = max(start, 0), min(start + crop, size)
        source.append(slice(low, high))
        target.append(slice(low - start, high - start))
    return tuple(source), tuple(target)


def to_crop(volume, crop_shape=CROP_SHAPE, fill=0):
    volume = np.asarray(volume)
    source, target = crop_slices(volume.shape, crop_shape)
    crop = np.full(crop_shape, fill, dtype=volume.dtype)
    crop[target] = volume[source]
    return crop


def from_crop(crop, shape, fill=0):
    """Paste a crop back onto a native grid; voxels outside the crop get ``fill``."""
    crop = np.asarray(crop)
    source, target = crop_slices(shape, crop.shape)
    volume = np.full(shape, fill, dtype=crop.dtype)
    volume[source] = crop[target]
    return volume


def signed_distance_3d(mask, clip=SDF_CLIP_MM):
    """Positive inside, negative outside, in voxels (1 mm), clipped to ±clip."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, -clip, dtype=np.float32)
    field = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    return np.clip(field, -clip, clip).astype(np.float32)


def observation_channels(observed, labels_volume, crop_shape=CROP_SHAPE):
    """Two crop-frame channels: observed label value and the observation mask."""
    observed = np.asarray(observed, dtype=bool)
    values = np.where(observed, np.asarray(labels_volume, dtype=np.float32), 0.0)
    return np.stack([to_crop(values.astype(np.float32), crop_shape),
                     to_crop(observed.astype(np.float32), crop_shape)])


def check_not_trained_on(subject, artifact):
    """Refuse to evaluate a subject the population model was trained on."""
    if subject in set(artifact["training_subjects"]):
        raise RuntimeError(f"{subject} is in the population model's training set")


def training_subjects(manifest, folds):
    """Subjects of the given development folds; the test split is never returned."""
    if len(set(folds)) != len(folds) or not set(folds) <= set(range(len(manifest["splits"]["folds"]))):
        raise ValueError(f"invalid training folds {folds}")
    subjects = sorted(s for fold in folds for s in manifest["splits"]["folds"][fold])
    if set(subjects) & set(manifest["splits"]["test"]):
        raise RuntimeError("training folds overlap the held-out test split")
    return subjects
