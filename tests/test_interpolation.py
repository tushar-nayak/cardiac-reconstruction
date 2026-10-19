import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cardiac_common.interpolation import (
    axial_sdf_interpolation, interpolate_sections, plane_sections,
    section_extrusion_intersection, signed_distance_2d,
)
from cardiac_common.sparse import largest_component, reconstruction_metrics, select_planes
from scripts.benchmark_sparse import manifest_subjects
from scripts.summarize_benchmark import bootstrap_ci, holm, subject_means


def sphere(shape=(21, 23, 25), radius=7.5):
    centre = (np.array(shape) - 1) / 2
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    return np.linalg.norm(grid - centre, axis=-1) <= radius


class InterpolationTests(unittest.TestCase):
    def test_sei_of_central_sphere_planes_is_extrusion_intersection(self):
        volume = sphere()
        planes = select_planes(volume.shape, 3, "orthogonal")
        result = section_extrusion_intersection(plane_sections(volume, planes), volume.shape)
        expected = np.ones(volume.shape, bool)
        for axis, index in planes:
            expected &= np.expand_dims(np.take(volume, index, axis=axis), axis)
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result[volume].all())
        for axis, index in planes:
            np.testing.assert_array_equal(np.take(result, index, axis=axis),
                                          np.take(volume, index, axis=axis))

    def test_sdf_interpolation_between_identical_slices_is_extrusion(self):
        disk = sphere((1, 23, 25))[0]
        shape = (20, 23, 25)
        sections = [(0, 3, disk), (0, 15, disk)]
        result = axial_sdf_interpolation(sections, shape)
        for z in range(shape[0]):
            np.testing.assert_array_equal(result[z], disk)

    def test_sdf_interpolation_reproduces_observed_slices_and_blends_sizes(self):
        small, large = sphere((1, 31, 31), 4)[0], sphere((1, 31, 31), 12)[0]
        result = axial_sdf_interpolation([(0, 0, small), (0, 10, large)], (11, 31, 31))
        np.testing.assert_array_equal(result[0], small)
        np.testing.assert_array_equal(result[10], large)
        areas = result.sum(axis=(1, 2))
        self.assertTrue(np.all(np.diff(areas) >= 0))

    def test_signed_distance_sign_and_empty_sections(self):
        disk = sphere((1, 11, 11), 3)[0]
        field = signed_distance_2d(disk)
        self.assertTrue((field[disk] > 0).all() and (field[~disk] < 0).all())
        self.assertTrue((signed_distance_2d(np.zeros((4, 4))) < 0).all())

    def test_baselines_read_only_sections(self):
        volume = sphere()
        planes = select_planes(volume.shape, 3, "orthogonal")
        changed = volume.copy()
        observed = np.zeros(volume.shape, bool)
        for axis, index in planes:
            selection = [slice(None)] * 3
            selection[axis] = index
            observed[tuple(selection)] = True
        changed[~observed] = ~changed[~observed]
        first, _ = interpolate_sections(plane_sections(volume, planes), volume.shape, "orthogonal")
        second, _ = interpolate_sections(plane_sections(changed, planes), volume.shape, "orthogonal")
        np.testing.assert_array_equal(first, second)


class MetricExtensionTests(unittest.TestCase):
    def test_largest_component_sensitivity_and_surface_metrics(self):
        target = sphere()
        prediction = target.copy()
        prediction[0, 0, 0] = True
        metrics = reconstruction_metrics(prediction.astype(float), target,
                                         np.zeros(target.shape, bool), np.eye(4))
        self.assertEqual(metrics["voxel_components_26"], 2)
        self.assertEqual(metrics["largest_component_sensitivity"]["whole_grid"]["dice"], 1)
        self.assertEqual(metrics["largest_component_sensitivity"]["surface"]["hd95_mm"], 0)
        self.assertGreater(metrics["surface"]["hausdorff_mm"], 0)
        self.assertEqual(largest_component(np.zeros((3, 3, 3)))[1], 0)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"splits": {"test": ["T1", "T2"],
                                    "folds": [["A", "B"], ["C"], ["D"], ["E"], ["F"]]}}

    def test_confirmatory_excludes_tuning_fold_and_test(self):
        chosen = manifest_subjects(self.manifest, "confirmatory")
        self.assertEqual([s for s, _ in chosen], ["C", "D", "E", "F"])
        self.assertEqual(manifest_subjects(self.manifest, "fold0"), [("A", 0), ("B", 0)])

    def test_overlap_with_test_is_rejected(self):
        self.manifest["splits"]["folds"][2].append("T1")
        with self.assertRaises(RuntimeError):
            manifest_subjects(self.manifest, "fold2")

    def test_frozen_manifest_folds_never_contain_test_subjects(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks/splits/mitea_cohort_v1.json"
        manifest = json.loads(path.read_text())
        test = set(manifest["splits"]["test"])
        for subset in ("fold0", "fold1", "fold2", "fold3", "fold4", "confirmatory"):
            self.assertFalse(test & {s for s, _ in manifest_subjects(manifest, subset)})


class SummaryTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_capped(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])

    def test_bootstrap_is_reproducible(self):
        values = np.linspace(-1, 2, 30)
        self.assertEqual(bootstrap_ci(values), bootstrap_ci(values))
        low, high = bootstrap_ci(values)
        self.assertLess(low, values.mean())
        self.assertGreater(high, values.mean())

    def test_seeds_are_averaged_and_missing_values_stay_missing(self):
        def row(seed, dice, delta_k):
            return {"method": "inr", "plane_layout": "orthogonal", "subject": "S", "seed": seed,
                    "status": "ok", "metrics": {"whole_grid": {"dice": dice},
                                                "geometry": {"absolute_delta_integrated_gaussian_curvature": delta_k}}}
        metrics = {"whole_dice": lambda m: m["whole_grid"]["dice"],
                   "absolute_delta_k": lambda m: m["geometry"]["absolute_delta_integrated_gaussian_curvature"]}
        means = subject_means([row(1, 0.8, 1.0), row(2, 0.9, None)], metrics)
        self.assertAlmostEqual(means[("inr", "orthogonal")]["S"]["whole_dice"], 0.85)
        self.assertIsNone(means[("inr", "orthogonal")]["S"]["absolute_delta_k"])


if __name__ == "__main__":
    unittest.main()
