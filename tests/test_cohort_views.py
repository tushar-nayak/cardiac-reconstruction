import unittest

import numpy as np

from cardiac_common.clinical import (
    biplane_simpson_ml, ejection_fraction, label_volumes_ml, surface_distances,
)
from cardiac_common.cohort import LV_CAVITY, MYOCARDIUM, fold_subjects, make_splits
from cardiac_common.views import (
    apical_planes, lv_long_axis, observed_mask, perturb_plane, sample_plane, voxel_to_world,
)


def half_ellipsoid(long_mm=40, radius_mm=18, wall_mm=6, shape=(70, 64, 64)):
    """Truncated prolate cavity with base at z=10 and apex toward high z."""
    z, y, x = np.indices(shape).astype(float)
    z0, y0, x0 = 10, (shape[1] - 1) / 2, (shape[2] - 1) / 2
    def inside(a, b):
        return ((z - z0) / a) ** 2 + ((y - y0) / b) ** 2 + ((x - x0) / b) ** 2 <= 1
    label = np.zeros(shape, dtype=np.uint8)
    above_base = z >= z0
    label[inside(long_mm + wall_mm, radius_mm + wall_mm) & above_base] = MYOCARDIUM
    label[inside(long_mm, radius_mm) & above_base] = LV_CAVITY
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = (5.0, -3.0, 12.0)
    return label, affine


class SplitTests(unittest.TestCase):
    def test_splits_are_disjoint_complete_and_stratified(self):
        subjects = [f"S{i:03d}" for i in range(50)]
        strata = {s: ("rare" if i < 10 else "common") for i, s in enumerate(subjects)}
        splits = make_splits(subjects, strata, folds=5, test_fraction=0.2, seed=3)
        groups = [splits["test"], *splits["folds"]]
        flat = [s for group in groups for s in group]
        self.assertEqual(sorted(flat), subjects)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(sum(strata[s] == "rare" for s in splits["test"]), 2)
        for fold in splits["folds"]:
            self.assertGreaterEqual(sum(strata[s] == "rare" for s in fold), 1)
        train, val = fold_subjects(splits, 2)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train + val) & set(splits["test"]))
        self.assertEqual(splits, make_splits(subjects, strata, folds=5, test_fraction=0.2, seed=3))


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.label, self.affine = half_ellipsoid()

    def test_long_axis_points_from_base_to_apex(self):
        axis = lv_long_axis(self.label, self.affine)
        expected = voxel_to_world([[1, 0, 0]], self.affine)[0] - voxel_to_world([[0, 0, 0]], self.affine)[0]
        self.assertGreater(axis.direction @ expected, 0.99)
        self.assertAlmostEqual(axis.length_mm, 40, delta=2)

    def test_apical_planes_contain_axis_and_sample_cavity(self):
        axis = lv_long_axis(self.label, self.affine)
        planes = apical_planes(axis, (0, 60, 120))
        for plane in planes:
            self.assertAlmostEqual(abs(plane.normal @ axis.direction), 0, places=6)
            sample = sample_plane(self.label, self.affine, plane)
            _, _, labels = sample.valid_points()
            self.assertTrue((labels == LV_CAVITY).any())
        self.assertAlmostEqual(abs(planes[0].v @ planes[1].v), 0.5, places=6)

    def test_observed_mask_is_thin(self):
        planes = apical_planes(lv_long_axis(self.label, self.affine), (0,))
        mask = observed_mask(self.label.shape, self.affine, planes)
        self.assertTrue(0 < mask.mean() < 0.05)

    def test_perturbation_is_rigid_and_bounded(self):
        plane = apical_planes(lv_long_axis(self.label, self.affine), (0,))[0]
        moved = perturb_plane(plane, 10, 3, np.random.default_rng(0))
        self.assertAlmostEqual(moved.u @ moved.v, 0, places=9)
        self.assertAlmostEqual(np.degrees(np.arccos(abs(moved.normal @ plane.normal))), 10, places=6)
        self.assertAlmostEqual(np.linalg.norm(moved.origin - plane.origin), 3, places=9)


class ClinicalTests(unittest.TestCase):
    def test_biplane_simpson_matches_voxel_volume(self):
        label, affine = half_ellipsoid()
        planes = apical_planes(lv_long_axis(label, affine), (0, 90))
        views = [sample_plane(label, affine, plane, spacing_mm=0.5) for plane in planes]
        estimate, info = biplane_simpson_ml(*views)
        truth = label_volumes_ml(label, affine)["lv_cavity_ml"]
        self.assertAlmostEqual(estimate / truth, 1, delta=0.05)
        self.assertAlmostEqual(info["length_mm"], 41, delta=2)

    def test_surface_distances(self):
        target = np.zeros((20, 20, 20), dtype=bool)
        target[5:15, 5:15, 5:15] = True
        shifted = np.roll(target, 2, axis=2)
        affine = np.diag([2.0, 1.0, 1.0, 1.0])
        self.assertEqual(surface_distances(target, target, affine)["hd95_mm"], 0)
        result = surface_distances(shifted, target, affine)
        self.assertAlmostEqual(result["hausdorff_mm"], 4.0)
        self.assertIsNone(surface_distances(np.zeros_like(target), target, affine)["hd95_mm"])

    def test_ejection_fraction(self):
        self.assertAlmostEqual(ejection_fraction(120, 48), 60)


if __name__ == "__main__":
    unittest.main()
