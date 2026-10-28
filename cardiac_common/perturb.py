"""Simulated tracing error in observed sections (sensitivity study v1, Study A).

Only the observed sections are perturbed; the evaluation target never is. With
``d`` the signed distance to a section's boundary (midway between pixel
centres, positive inside, 1 mm pixels), the perturbed section is
``d + offset + sd * g >= 0``. ``g`` is Gaussian-filtered white noise scaled to
unit SD over the section, drawn once per subject, layout, and plane, so every
noise level reuses the same field.
"""

import numpy as np
from scipy import ndimage

SEED = 2027
LAYOUT_INDEX = {"orthogonal": 0, "axial": 1}


def boundary_distance(section):
    """Signed distance to the boundary midway between pixel centres; positive inside."""
    section = np.asarray(section, dtype=bool)
    return np.where(section, ndimage.distance_transform_edt(section) - 0.5,
                    -(ndimage.distance_transform_edt(~section) - 0.5))


def unit_field(shape, filter_mm, entropy):
    """White noise filtered with a Gaussian of SD ``filter_mm``, divided by its SD."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    field = ndimage.gaussian_filter(rng.standard_normal(shape), filter_mm)
    return field / field.std()


def perturb_section(section, offset_mm, sd_mm, field):
    """Perturbed copy of a 2D section; empty and full sections are left unchanged."""
    section = np.asarray(section, dtype=bool)
    if not section.any() or section.all():
        return section.copy()
    return boundary_distance(section) + offset_mm + sd_mm * field >= 0


def perturb_observations(volume, planes, subject, layout, offset_mm=0.0, sd_mm=0.0, filter_mm=5.0):
    """Copy of ``volume`` whose ``(axis, index)`` planes hold perturbed sections.

    Each section is perturbed from its clean version. Planes are written in
    reverse order, so where planes intersect the first-listed plane wins (for
    the orthogonal layout, the lowest array axis). Returns the volume and
    per-plane foreground counts in layout order.
    """
    volume = np.asarray(volume) >= 0.5
    perturbed = volume.copy()
    number = int(subject.rsplit("_", 1)[1])
    counts = [None] * len(planes)
    for position in reversed(range(len(planes))):
        axis, index = planes[position]
        section = np.take(volume, index, axis=axis)
        field = unit_field(section.shape, filter_mm, [SEED, number, LAYOUT_INDEX[layout], position])
        new = perturb_section(section, offset_mm, sd_mm, field)
        selection = [slice(None)] * 3
        selection[axis] = index
        perturbed[tuple(selection)] = new
        counts[position] = {"axis": int(axis), "index": int(index), "foreground_before": int(section.sum()),
                            "foreground_after": int(new.sum()), "changed_pixels": int((new != section).sum())}
    return perturbed, counts
