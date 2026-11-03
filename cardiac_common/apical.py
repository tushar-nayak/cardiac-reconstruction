"""Apical-view observations for the EF study (``benchmarks/mitea_apical_ef_v1.md``).

Views are planes through the phase's own cavity long axis, optionally misplaced.
Methods observe either the misplaced planes where they really are (known pose)
or the misplaced images placed on the intended planes (assumed pose). Simpson,
RBF, and the voxel methods all read the same misplaced views.
"""

import numpy as np

from cardiac_common.cohort import LV_CAVITY
from cardiac_common.interpolation import signed_distance_2d
from cardiac_common.rbf import CLIP_MM
from cardiac_common.views import (apical_planes, lv_long_axis, perturb_plane, sample_plane, voxel_to_world,
                                  world_to_voxel)

LAYOUTS = {"biplane": (0.0, 90.0), "triplane": (0.0, 60.0, 120.0)}
LEVELS = {"none": (0.0, 0.0), "mild": (5.0, 3.0), "severe": (10.0, 5.0)}  # tilt (deg), shift (mm)
HALF_THICKNESS_MM = 0.5


def view_planes(label, affine, layout, tilt_deg, shift_mm, rng):
    """(intended, actual) planes for ``layout``; actual planes are misplaced when requested."""
    intended = apical_planes(lv_long_axis(label, affine), LAYOUTS[layout])
    if not (tilt_deg or shift_mm):
        return intended, intended
    return intended, [perturb_plane(plane, tilt_deg, shift_mm, rng) for plane in intended]


def plane_distance(shape, affine, plane):
    """Signed distance (mm) of every voxel centre from ``plane``, built from separable terms."""
    affine = np.asarray(affine)
    coefficient = affine[:3, :3].T @ plane.normal  # per (x, y, z) voxel index
    z, y, x = (np.arange(n, dtype=np.float64) for n in shape)
    return ((coefficient[2] * z)[:, None, None] + (coefficient[1] * y)[None, :, None]
            + (coefficient[0] * x)[None, None, :] + float((affine[:3, 3] - plane.origin) @ plane.normal))


def observe(target, affine, intended, actual, pose):
    """(observed mask, labels on it) for a binary ``target`` under ``pose``.

    Known pose: voxels within half a voxel of an actual plane keep their own
    label. Assumed pose: voxels on the intended planes take the label found at
    the same in-plane position on the actual plane; the first-listed plane wins
    where intended planes overlap.
    """
    target = np.asarray(target, dtype=bool)
    shape = target.shape
    observed = np.zeros(shape, dtype=bool)
    values = np.zeros(shape, dtype=bool)
    for nominal, moved in reversed(list(zip(intended, actual))):
        near = np.abs(plane_distance(shape, affine, nominal if pose == "assumed" else moved)) <= HALF_THICKNESS_MM
        if pose == "known":
            observed |= near
            values[near] = target[near]
            continue
        if pose != "assumed":
            raise ValueError(f"unknown pose condition {pose!r}")
        indices = np.argwhere(near)
        offset = voxel_to_world(indices, affine) - nominal.origin
        coords = [offset @ axis for axis in (nominal.u, nominal.v, nominal.normal)]
        points = moved.origin + sum(c[:, None] * axis for c, axis in zip(coords, (moved.u, moved.v, moved.normal)))
        voxel = np.rint(world_to_voxel(points, affine)).astype(int)
        inside = np.all((voxel >= 0) & (voxel < np.array(shape)), axis=1)
        labels = np.zeros(len(indices), dtype=bool)
        labels[inside] = target[tuple(voxel[inside].T)]
        observed[tuple(indices.T)] = True
        values[tuple(indices.T)] = labels
    return observed, values


def view_constraints(target, affine, intended, actual, pose, spacing_mm=1.0):
    """RBF constraints: clipped in-plane signed distances of each actual view, at believed positions."""
    positions, values = [], []
    for nominal, moved in zip(intended, actual):
        sample = sample_plane(np.asarray(target, dtype=np.int16), affine, moved, spacing_mm)
        inside = sample.image >= 0
        field = np.clip(signed_distance_2d(sample.image == 1), -CLIP_MM, CLIP_MM)
        if pose == "known":
            believed = sample.voxel_zyx
        else:
            uu, vv = np.meshgrid(sample.u_mm, sample.v_mm, indexing="ij")
            believed = world_to_voxel(nominal.origin + uu[..., None] * nominal.u + vv[..., None] * nominal.v, affine)
        positions.append(believed[inside])
        values.append(field[inside])
    return np.concatenate(positions), np.concatenate(values)


def training_observer(labels, affines, rng_layouts=("biplane", "triplane"), max_tilt_deg=10.0, max_shift_mm=5.0):
    """``observe_fn(index, rng)`` for U-Net training: random layout and known-pose misplacement."""
    axes = [lv_long_axis(label, affine) for label, affine in zip(labels, affines)]
    cavities = [label == LV_CAVITY for label in labels]

    def observe_fn(index, rng):
        layout = rng_layouts[int(rng.integers(len(rng_layouts)))]
        planes = [perturb_plane(plane, rng.uniform(0, max_tilt_deg), rng.uniform(0, max_shift_mm), rng)
                  for plane in apical_planes(axes[index], LAYOUTS[layout])]
        return observe(cavities[index], affines[index], planes, planes, "known")

    return cavities, observe_fn
