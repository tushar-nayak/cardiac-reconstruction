"""Simulated apical echo views: oblique planes through the LV long axis.

The long axis is estimated from the ground-truth cavity, so plane placement is
an oracle assumption; ``perturb_plane`` models probe misplacement. The angular
reference about the long axis is the world y axis projected off the axis. The
labels contain no right ventricle or atria, so rotation angles approximate,
but do not identify, the anatomical four-, two-, and three-chamber views.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from cardiac_common.cohort import LV_CAVITY


@dataclass(frozen=True)
class Plane:
    """A plane through ``origin`` spanned by unit ``u`` (long axis) and ``v``."""

    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    name: str = ""

    @property
    def normal(self):
        return np.cross(self.u, self.v)


@dataclass(frozen=True)
class LongAxis:
    base: np.ndarray
    apex: np.ndarray

    @property
    def direction(self):
        """Unit vector from base toward apex."""
        delta = self.apex - self.base
        return delta / np.linalg.norm(delta)

    @property
    def length_mm(self):
        return float(np.linalg.norm(self.apex - self.base))


def voxel_to_world(indices_zyx, affine):
    xyz = np.asarray(indices_zyx, dtype=np.float64)[..., ::-1]
    return xyz @ affine[:3, :3].T + affine[:3, 3]


def world_to_voxel(points, affine):
    """World millimetres -> continuous canonical ``(z, y, x)`` indices."""
    inverse = np.linalg.inv(affine)
    xyz = np.asarray(points, dtype=np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    return xyz[..., ::-1]


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError("degenerate direction")
    return vector / norm


def lv_long_axis(label, affine, end_fraction=0.15):
    """Estimate the cavity long axis and orient it from base to apex.

    The axis is the principal direction of the cavity. The apex is the end
    whose outer ``end_fraction`` of the length holds fewer cavity voxels,
    since the cavity tapers toward the apex and is truncated at the valve plane.
    """
    points = voxel_to_world(np.argwhere(label == LV_CAVITY), affine)
    if len(points) < 10:
        raise ValueError("label has too few cavity voxels to define a long axis")
    centre = points.mean(axis=0)
    direction = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    projection = (points - centre) @ direction
    low, high = projection.min(), projection.max()
    margin = end_fraction * (high - low)
    if (projection > high - margin).sum() > (projection < low + margin).sum():
        direction, projection, (low, high) = -direction, -projection, (-high, -low)
    return LongAxis(base=centre + low * direction, apex=centre + high * direction)


def apical_planes(axis, angles_deg=(0.0, 60.0, 120.0), reference=(0.0, 1.0, 0.0)):
    """Planes containing the long axis, rotated about it by ``angles_deg``."""
    u = axis.direction
    reference = np.asarray(reference, dtype=np.float64)
    first = reference - (reference @ u) * u
    if np.linalg.norm(first) < 1e-6:
        first = np.eye(3)[np.argmin(np.abs(u))]
        first -= (first @ u) * u
    first = _unit(first)
    second = np.cross(u, first)
    planes = []
    for angle in angles_deg:
        theta = np.deg2rad(angle)
        v = np.cos(theta) * first + np.sin(theta) * second
        planes.append(Plane(axis.base.copy(), u.copy(), v, name=f"apical_{angle:g}"))
    return planes


def _rotation(axis, angle):
    axis = _unit(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def perturb_plane(plane, tilt_deg, shift_mm, rng):
    """Tilt by ``tilt_deg`` about a random in-plane axis and shift along the normal."""
    phi = rng.uniform(0, 2 * np.pi)
    hinge = np.cos(phi) * plane.u + np.sin(phi) * plane.v
    rotation = _rotation(hinge, np.deg2rad(tilt_deg))
    origin = plane.origin + shift_mm * plane.normal
    return Plane(origin, rotation @ plane.u, rotation @ plane.v, plane.name)


@dataclass(frozen=True)
class PlaneSample:
    """Nearest-neighbour labels on a regular in-plane grid.

    ``image`` is indexed ``[u, v]`` and holds -1 outside the volume.
    ``u_mm``/``v_mm`` give grid coordinates relative to the plane origin.
    """

    plane: Plane
    image: np.ndarray
    u_mm: np.ndarray
    v_mm: np.ndarray
    world: np.ndarray
    voxel_zyx: np.ndarray

    def valid_points(self):
        """Inside-volume ``(world_points, voxel_zyx, labels)`` as flat arrays."""
        mask = self.image >= 0
        return self.world[mask], self.voxel_zyx[mask], self.image[mask]


def sample_plane(label, affine, plane, spacing_mm=1.0, extent_mm=None):
    """Read a label along ``plane`` by nearest-neighbour lookup."""
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be positive")
    if extent_mm is None:
        extent_mm = float(np.linalg.norm(np.array(label.shape) * np.linalg.norm(affine[:3, :3], axis=0)[::-1]))
    steps = np.arange(-extent_mm, extent_mm + spacing_mm / 2, spacing_mm)
    uu, vv = np.meshgrid(steps, steps, indexing="ij")
    world = plane.origin + uu[..., None] * plane.u + vv[..., None] * plane.v
    voxel = world_to_voxel(world, affine)
    nearest = np.rint(voxel).astype(int)
    inside = np.all((nearest >= 0) & (nearest < np.array(label.shape)), axis=-1)
    image = np.full(uu.shape, -1, dtype=np.int16)
    z, y, x = nearest[inside].T
    image[inside] = label[z, y, x]
    rows, cols = np.nonzero(inside)
    if not len(rows):
        raise ValueError("plane does not intersect the volume")
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    return PlaneSample(plane, image[crop], steps[crop[0]], steps[crop[1]], world[crop], voxel[crop])


def observed_mask(shape, affine, planes, half_thickness_mm=0.5):
    """Voxels whose centres lie within ``half_thickness_mm`` of any plane."""
    indices = np.indices(shape).reshape(3, -1).T
    world = voxel_to_world(indices, affine)
    mask = np.zeros(len(world), dtype=bool)
    for plane in planes:
        mask |= np.abs((world - plane.origin) @ plane.normal) <= half_thickness_mm
    return mask.reshape(shape)


def cavity_profile(sample, label_value=LV_CAVITY):
    """Per-row cavity extent (mm) along ``v`` for each long-axis position ``u``."""
    cavity = sample.image == label_value
    spacing = float(sample.v_mm[1] - sample.v_mm[0]) if len(sample.v_mm) > 1 else 1.0
    widths = np.zeros(len(sample.u_mm))
    for row in np.nonzero(cavity.any(axis=1))[0]:
        # Keep the run containing the long axis so detached fragments do not widen a disk.
        runs, _ = ndimage.label(cavity[row])
        centre = np.argmin(np.abs(sample.v_mm))
        keep = runs[centre] if runs[centre] else np.bincount(runs.ravel())[1:].argmax() + 1
        cols = np.nonzero(runs == keep)[0]
        widths[row] = (cols.max() - cols.min() + 1) * spacing
    return sample.u_mm.copy(), widths
