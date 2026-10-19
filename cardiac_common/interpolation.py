"""Nonlearned completion baselines that read only the observed sections.

Both functions take ``sections``: ``(axis, index, mask2d)`` tuples extracted at
the selected planes, never the full target volume.
"""

import numpy as np
from scipy import ndimage


def plane_sections(volume, planes):
    """Extract 2D binary sections at ``(axis, index)`` planes of a ``(z,y,x)`` volume."""
    volume = np.asarray(volume) >= 0.5
    return [(axis, index, np.take(volume, index, axis=axis).copy()) for axis, index in planes]


def section_extrusion_intersection(sections, shape):
    """Intersect each section extruded along its own normal axis (SEI).

    With one section per array axis this is the largest volume consistent with
    every section's foreground. It is not a silhouette visual hull.
    """
    axes = [axis for axis, _, _ in sections]
    if not sections or len(set(axes)) != len(axes):
        raise ValueError("SEI requires at most one section per array axis")
    result = np.ones(shape, dtype=bool)
    for axis, _, mask in sections:
        result &= np.expand_dims(np.asarray(mask, dtype=bool), axis)
    return result


def signed_distance_2d(mask):
    """Positive inside foreground, negative outside; constant for empty/full masks."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, -float(sum(mask.shape)))
    if mask.all():
        return np.full(mask.shape, float(sum(mask.shape)))
    return ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)


def axial_sdf_interpolation(sections, shape):
    """Linearly interpolate signed distances between parallel axial sections.

    Slices beyond the first/last section copy the nearest section's field.
    Values ``>= 0`` are foreground.
    """
    if not sections or any(axis != 0 for axis, _, _ in sections):
        raise ValueError("SDF interpolation requires axial (axis 0) sections")
    ordered = sorted(sections, key=lambda item: item[1])
    indices = np.array([index for _, index, _ in ordered], dtype=float)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("axial sections must have distinct indices")
    fields = np.stack([signed_distance_2d(mask) for _, _, mask in ordered])
    result = np.empty(shape, dtype=bool)
    for z in range(shape[0]):
        upper = int(np.searchsorted(indices, z))
        if upper == 0:
            field = fields[0]
        elif upper == len(indices):
            field = fields[-1]
        else:
            weight = (z - indices[upper - 1]) / (indices[upper] - indices[upper - 1])
            field = (1 - weight) * fields[upper - 1] + weight * fields[upper]
        result[z] = field >= 0
    return result


def interpolate_sections(sections, shape, layout):
    """Dispatch to the protocol baseline for ``layout``; returns (mask, variant)."""
    if layout == "orthogonal":
        return section_extrusion_intersection(sections, shape), "section_extrusion_intersection"
    if layout == "axial":
        return axial_sdf_interpolation(sections, shape), "axial_linear_signed_distance"
    raise ValueError(f"no interpolation baseline for layout {layout!r}")
