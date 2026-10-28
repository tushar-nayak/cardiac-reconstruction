import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from cardiac_common.perturb import boundary_distance, perturb_observations, perturb_section, unit_field
from cardiac_common.population import listed_training_subjects
from cardiac_common.sparse import select_planes
from cardiac_common.ssm import build_ssm, mean_shape

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "benchmarks/splits/mitea_cohort_v1.json").read_text())


def disk(shape, radius):
    centre = (np.array(shape) - 1) / 2
    grid = np.indices(shape).transpose(1, 2, 0)
    return np.linalg.norm(grid - centre, axis=-1) <= radius


def sphere(shape, radius):
    centre = (np.array(shape) - 1) / 2
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    return np.linalg.norm(grid - centre, axis=-1) <= radius


def load_subsets_module():
    spec = importlib.util.spec_from_file_location("make_sensitivity_subsets",
                                                  ROOT / "scripts/make_sensitivity_subsets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SectionPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.section = disk((101, 101), 20)
        self.field = unit_field(self.section.shape, 5.0, [2027, 7, 0, 0])

    def test_boundary_sits_midway_between_pixel_centres(self):
        distance = boundary_distance(self.section)
        self.assertGreaterEqual(distance[self.section].min(), 0.5)
        self.assertLessEqual(distance[~self.section].max(), -0.5)

    def test_zero_offset_and_noise_reproduce_the_section(self):
        np.testing.assert_array_equal(perturb_section(self.section, 0.0, 0.0, self.field), self.section)

    def test_offsets_dilate_and_erode_symmetrically(self):
        outside = perturb_section(self.section, 2.0, 0.0, self.field)
        inside = perturb_section(self.section, -2.0, 0.0, self.field)
        self.assertTrue(outside[self.section].all())
        self.assertFalse(inside[~self.section].any())
        radius = np.sqrt(self.section.sum() / np.pi)
        self.assertAlmostEqual(np.sqrt(outside.sum() / np.pi) - radius, 2.0, delta=0.3)
        self.assertAlmostEqual(radius - np.sqrt(inside.sum() / np.pi), 2.0, delta=0.3)

    def test_empty_and_full_sections_are_unchanged(self):
        for section in (np.zeros((40, 40), bool), np.ones((40, 40), bool)):
            np.testing.assert_array_equal(perturb_section(section, 2.0, 4.0, unit_field((40, 40), 5.0, [1])),
                                          section)

    def test_field_is_reproducible_with_unit_sd(self):
        again = unit_field(self.section.shape, 5.0, [2027, 7, 0, 0])
        np.testing.assert_array_equal(self.field, again)
        self.assertAlmostEqual(float(self.field.std()), 1.0, places=6)
        self.assertFalse(np.array_equal(self.field, unit_field(self.section.shape, 5.0, [2027, 7, 0, 1])))

    def test_noise_moves_the_boundary_by_about_its_sd(self):
        noisy = perturb_section(self.section, 0.0, 2.0, self.field)
        changed = noisy != self.section
        self.assertTrue(changed.any())
        self.assertLess(np.abs(boundary_distance(self.section))[changed].max(), 4 * 2.0 + 0.5)


class ObservationPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.volume = sphere((40, 48, 56), 15)
        self.planes = select_planes(self.volume.shape, 3, "orthogonal")

    def test_only_observed_planes_change(self):
        perturbed, counts = perturb_observations(self.volume, self.planes, "MITEA_007", "orthogonal", sd_mm=2.0)
        observed = np.zeros(self.volume.shape, bool)
        for axis, index in self.planes:
            selection = [slice(None)] * 3
            selection[axis] = index
            observed[tuple(selection)] = True
        np.testing.assert_array_equal(perturbed[~observed], self.volume[~observed])
        self.assertTrue((perturbed[observed] != self.volume[observed]).any())
        self.assertEqual([c["axis"] for c in counts], [0, 1, 2])
        self.assertTrue(all(c["foreground_before"] > 0 and c["changed_pixels"] > 0 for c in counts))

    def test_lowest_axis_plane_wins_at_intersections(self):
        perturbed, _ = perturb_observations(self.volume, self.planes, "MITEA_007", "orthogonal", sd_mm=3.0)
        (_, z), (_, y), (_, x) = self.planes
        field = unit_field(self.volume[z].shape, 5.0, [2027, 7, 0, 0])
        expected = perturb_section(self.volume[z], 0.0, 3.0, field)
        np.testing.assert_array_equal(perturbed[z, y, :], expected[y, :])
        np.testing.assert_array_equal(perturbed[z, :, x], expected[:, x])

    def test_clean_parameters_return_the_volume(self):
        perturbed, _ = perturb_observations(self.volume, self.planes, "MITEA_007", "orthogonal")
        np.testing.assert_array_equal(perturbed, self.volume)


class SubjectListTests(unittest.TestCase):
    def test_listed_subjects_are_validated_and_sorted(self):
        folds = MANIFEST["splits"]["folds"]
        chosen = [folds[2][0], folds[1][0]]
        self.assertEqual(listed_training_subjects(MANIFEST, chosen), sorted(chosen))
        with self.assertRaises(ValueError):
            listed_training_subjects(MANIFEST, [folds[1][0], folds[1][0]])
        with self.assertRaises(RuntimeError):
            listed_training_subjects(MANIFEST, [MANIFEST["splits"]["test"][0]])
        with self.assertRaises(ValueError):
            listed_training_subjects(MANIFEST, ["MITEA_999"])

    def test_sensitivity_lists_follow_the_protocol(self):
        lists = load_subsets_module().subject_lists(MANIFEST)
        self.assertEqual(lists, load_subsets_module().subject_lists(MANIFEST))
        strata, folds = MANIFEST["strata"], MANIFEST["splits"]["folds"]
        self.assertEqual(len(lists["healthy66"]["subjects"]), 66)
        self.assertEqual(lists["healthy66"]["pathology_count"], 0)
        self.assertEqual([lists[f"mixed66_fold{k}"]["pathology_count"] for k in range(1, 5)], [25, 25, 26, 27])
        self.assertEqual(sum(len(lists[f"b1_eval_fold{k}"]["subjects"]) for k in range(1, 5)), 33)
        for k in range(1, 5):
            self.assertEqual(len(lists[f"mixed66_fold{k}"]["subjects"]), 66)
            sizes = [set(lists[f"size{n}_fold{k}"]["subjects"]) for n in (10, 25, 50)]
            self.assertEqual([len(s) for s in sizes], [10, 25, 50])
            self.assertTrue(sizes[0] < sizes[1] < sizes[2])
            for name in (f"mixed66_fold{k}", f"size50_fold{k}"):
                self.assertFalse(set(lists[name]["subjects"]) & set(folds[k]))
            self.assertTrue(all(strata[s] != "healthy" for s in lists[f"b1_eval_fold{k}"]["subjects"]))


class MeanShapeTests(unittest.TestCase):
    def test_mean_shape_is_the_thresholded_mean_map(self):
        shape = (110, 120, 130)
        artifact = build_ssm([sphere(shape, r) for r in (18, 20, 22)], ["A", "B", "C"])
        prediction = mean_shape(artifact, shape)
        target = sphere(shape, 20)
        dice = 2 * (prediction & target).sum() / (prediction.sum() + target.sum())
        self.assertEqual(prediction.shape, shape)
        self.assertGreater(dice, 0.98)


if __name__ == "__main__":
    unittest.main()
