import unittest

import numpy as np

from cardiac_common.radial import (
    COMMON_VIEWS, LAYOUTS, RadialInterpolator, pixel_counts, pool_counts,
    radial_metrics,
)


def disk(radius, size=31):
    yy, xx = np.indices((size, size))
    return (xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2 <= radius ** 2


class RadialTests(unittest.TestCase):
    def test_common_test_grid_is_identical_and_never_observed(self):
        self.assertEqual(len(COMMON_VIEWS), 24)
        self.assertNotIn(1, COMMON_VIEWS)
        self.assertNotIn(37, COMMON_VIEWS)
        for ids in LAYOUTS.values():
            self.assertFalse(set(ids) & set(COMMON_VIEWS))
        for lower, upper in zip(("spread2", "spread3", "spread6"),
                                ("spread3", "spread6", "spread12")):
            self.assertTrue(set(LAYOUTS[lower]) <= set(LAYOUTS[upper]))

    def test_nearest_ties_use_lower_angle_even_with_unsorted_inputs(self):
        small, large = disk(3), disk(9)
        model = RadialInterpolator([31, 7], [large, small], "nearest")
        out = model.predict([1, 7, 19, 31, 37])
        np.testing.assert_array_equal(out[:3], np.stack([small] * 3))
        np.testing.assert_array_equal(out[3:], np.stack([large] * 2))

    def test_sdf_reproduces_sections_and_extends_endpoints(self):
        small, large = disk(3), disk(9)
        model = RadialInterpolator([7, 31], [small, large], "sdf")
        out = model.predict([1, 7, 13, 19, 25, 31, 37])
        np.testing.assert_array_equal(out[0], small)
        np.testing.assert_array_equal(out[1], small)
        np.testing.assert_array_equal(out[-2], large)
        np.testing.assert_array_equal(out[-1], large)
        self.assertTrue(np.all(np.diff(out.sum(axis=(1, 2))) >= 0))
        self.assertGreater(out[3].sum(), small.sum())
        self.assertLess(out[3].sum(), large.sum())

    def test_empty_and_full_observations_remain_well_defined(self):
        for value in (False, True):
            mask = np.full((5, 7), value)
            pred = RadialInterpolator([7, 31], [mask, mask]).predict([1, 19, 37])
            self.assertTrue((pred == value).all())

    def test_interpolator_owns_only_copies_of_observations(self):
        observed = np.stack([disk(3), disk(9)])
        model = RadialInterpolator([7, 31], observed)
        before = model.predict([19])
        observed[:] = ~observed
        np.testing.assert_array_equal(model.predict([19]), before)
        with self.assertRaises(ValueError):
            RadialInterpolator([7, 7], observed)

    def test_count_pooling_matches_direct_foreground_dice(self):
        truth = np.stack([disk(2), disk(9)])
        pred = np.stack([disk(4), disk(6)])
        rows = [pixel_counts(p, t) for p, t in zip(pred, truth)]
        score = pool_counts(rows)
        self.assertAlmostEqual(score["dice"],
                               2 * np.logical_and(pred, truth).sum() / (pred.sum() + truth.sum()))
        self.assertEqual(pool_counts([pixel_counts(np.zeros((2, 2)), np.zeros((2, 2)))])["dice"], 1)
        self.assertIsNone(pool_counts([]))

    def test_metric_subsets_respect_observation_support(self):
        ids = [i for i in range(1, 38) if i not in LAYOUTS["clustered3"]]
        target = np.stack([disk(4)] * len(ids))
        result = radial_metrics(target, target, ids, LAYOUTS["clustered3"])
        self.assertEqual(result["subsets"]["common"]["dice"], 1)
        self.assertEqual(result["subsets"]["common_interior"]["pixels"], 16 * 31 * 31)
        self.assertGreater(result["subsets"]["outside_range"]["pixels"], 0)
        with self.assertRaises(ValueError):
            radial_metrics(target[:1], target[:1], [19], LAYOUTS["clustered3"])


if __name__ == "__main__":
    unittest.main()
