import unittest

import numpy as np

from cardiac_common.apical import observe, plane_distance, view_constraints, view_planes
from cardiac_common.cohort import LV_CAVITY, MYOCARDIUM
from cardiac_common.views import Plane, observed_mask, perturb_plane


def ellipsoid_label(shape=(60, 64, 68)):
    """A long cavity along z inside a myocardial shell, as label values."""
    grid = np.indices(shape).transpose(1, 2, 3, 0).astype(float)
    centre = (np.array(shape) - 1) / 2
    scaled = (grid - centre) / np.array([24.0, 12.0, 12.0])
    radius = np.linalg.norm(scaled, axis=-1)
    label = np.zeros(shape, np.uint8)
    label[radius <= 1.25] = MYOCARDIUM
    label[radius <= 1.0] = LV_CAVITY
    return label


class ApicalObservationTests(unittest.TestCase):
    def setUp(self):
        self.label = ellipsoid_label()
        self.target = self.label == LV_CAVITY
        self.affine = np.eye(4)
        self.affine[:3, 3] = [3.137, -2.071, 1.293]  # off-grid, so no voxel sits exactly 0.5 mm from a plane

    def test_plane_distance_matches_observed_mask(self):
        intended, _ = view_planes(self.label, self.affine, "biplane", 0, 0, np.random.default_rng(0))
        for plane in intended:
            near = np.abs(plane_distance(self.target.shape, self.affine, plane)) <= 0.5
            np.testing.assert_array_equal(near, observed_mask(self.target.shape, self.affine, [plane]))

    def test_unperturbed_pose_conditions_agree(self):
        intended, actual = view_planes(self.label, self.affine, "triplane", 0, 0, np.random.default_rng(0))
        known = observe(self.target, self.affine, intended, actual, "known")
        assumed = observe(self.target, self.affine, intended, actual, "assumed")
        np.testing.assert_array_equal(known[0], assumed[0])
        np.testing.assert_array_equal(known[1], assumed[1])
        np.testing.assert_array_equal(known[1], self.target & known[0])
        self.assertTrue(known[1].any())

    def test_assumed_pose_reads_the_moved_plane(self):
        plane = Plane(np.array([33.0, 30.0, 30.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
        moved = Plane(plane.origin + 4.0 * plane.normal, plane.u, plane.v)
        observed, values = observe(self.target, self.affine, [plane], [moved], "assumed")
        known_observed, known_values = observe(self.target, self.affine, [moved], [moved], "known")
        # Same observed voxels as the intended plane, carrying the moved plane's labels.
        np.testing.assert_array_equal(observed, observe(self.target, self.affine, [plane], [plane], "known")[0])
        self.assertEqual(int(values.sum()), int(known_values.sum()))
        self.assertFalse(np.array_equal(values, known_values))

    def test_misplacement_is_seeded_and_changes_views(self):
        first = view_planes(self.label, self.affine, "biplane", 10, 5, np.random.default_rng(3))[1]
        again = view_planes(self.label, self.affine, "biplane", 10, 5, np.random.default_rng(3))[1]
        for a, b in zip(first, again):
            np.testing.assert_allclose(a.origin, b.origin)
            np.testing.assert_allclose(a.u, b.u)
        intended = view_planes(self.label, self.affine, "biplane", 0, 0, np.random.default_rng(3))[0]
        self.assertFalse(np.allclose(first[0].u, intended[0].u))

    def test_view_constraints_have_both_signs(self):
        intended, actual = view_planes(self.label, self.affine, "biplane", 5, 3, np.random.default_rng(1))
        for pose in ("known", "assumed"):
            positions, values = view_constraints(self.target, self.affine, intended, actual, pose)
            self.assertEqual(positions.shape[1], 3)
            self.assertEqual(len(positions), len(values))
            self.assertTrue((values > 0).any() and (values < 0).any())


if __name__ == "__main__":
    unittest.main()
