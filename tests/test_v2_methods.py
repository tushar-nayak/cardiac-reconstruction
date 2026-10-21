import unittest

import numpy as np
import torch

from cardiac_common.gp import gp_complete
from cardiac_common.interpolation import plane_sections
from cardiac_common.population import (CROP_SHAPE, check_not_trained_on, crop_slices, from_crop, to_crop,
                                       training_subjects)
from cardiac_common.rbf import rbf_complete
from cardiac_common.prior_inr import fit_prior_inr
from cardiac_common.sparse import plane_samples, select_planes
from cardiac_common.ssm import build_ssm, fit_ssm
from cardiac_common.unet import UNet3D, jittered_planes, observe
from code_inr.inr_model import ImplicitOccupancyNetwork


def sphere(shape, radius, centre=None):
    centre = (np.array(shape) - 1) / 2 if centre is None else np.asarray(centre)
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    return np.linalg.norm(grid - centre, axis=-1) <= radius


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


class CropTests(unittest.TestCase):
    def test_round_trip_keeps_array_centre_region(self):
        volume = np.random.default_rng(0).random((150, 120, 170)) > 0.5
        back = from_crop(to_crop(volume), volume.shape)
        source, _ = crop_slices(volume.shape)
        np.testing.assert_array_equal(back[source], volume[source])
        outside = np.ones(volume.shape, bool)
        outside[source] = False
        self.assertFalse(back[outside].any())

    def test_small_arrays_are_padded_and_centred(self):
        volume = np.ones((100, 101, 102), bool)
        crop = to_crop(volume)
        self.assertEqual(crop.shape, CROP_SHAPE)
        self.assertEqual(crop.sum(), volume.sum())
        centre = (np.array(CROP_SHAPE) - 1) // 2
        self.assertTrue(crop[tuple(centre)])


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"splits": {"test": ["T1"], "folds": [["A"], ["B"], ["C"], ["D"], ["E"]]}}

    def test_training_subjects_exclude_unlisted_folds_and_test(self):
        self.assertEqual(training_subjects(self.manifest, [1, 3]), ["B", "D"])
        with self.assertRaises(ValueError):
            training_subjects(self.manifest, [1, 1])
        self.manifest["splits"]["folds"][2].append("T1")
        with self.assertRaises(RuntimeError):
            training_subjects(self.manifest, [2])

    def test_runner_guard_refuses_training_subjects(self):
        with self.assertRaises(RuntimeError):
            check_not_trained_on("B", {"training_subjects": ["A", "B"]})
        check_not_trained_on("C", {"training_subjects": ["A", "B"]})


class RbfTests(unittest.TestCase):
    def test_sphere_from_orthogonal_sections(self):
        volume = sphere((41, 43, 45), 12)
        planes = select_planes(volume.shape, 3, "orthogonal")
        mask, centres = rbf_complete(plane_sections(volume, planes), volume.shape)
        self.assertGreater(dice(mask, volume), 0.9)
        self.assertLessEqual(centres, 4000)

    def test_reads_only_sections(self):
        volume = sphere((31, 33, 35), 9)
        planes = select_planes(volume.shape, 3, "orthogonal")
        observed = np.zeros(volume.shape, bool)
        for axis, index in planes:
            selection = [slice(None)] * 3
            selection[axis] = index
            observed[tuple(selection)] = True
        changed = volume.copy()
        changed[~observed] = ~changed[~observed]
        first, _ = rbf_complete(plane_sections(volume, planes), volume.shape)
        second, _ = rbf_complete(plane_sections(changed, planes), volume.shape)
        np.testing.assert_array_equal(first, second)


class GpTests(unittest.TestCase):
    def test_sphere_and_uncertainty_grows_away_from_sections(self):
        volume = sphere((41, 43, 45), 12)
        planes = select_planes(volume.shape, 3, "orthogonal")
        mask, mean, sd, centres = gp_complete(plane_sections(volume, planes), volume.shape, 10.0, -10.0)
        self.assertGreater(dice(mask, volume), 0.9)
        self.assertEqual(mean.shape, sd.shape)
        self.assertLess(sd[10, 10, 11], sd[2, 2, 2])  # (20, 20, 22) lies on a section; a corner does not
        self.assertLessEqual(centres, 4000)

    def test_reads_only_sections_and_reverts_to_prior_mean(self):
        volume = sphere((31, 33, 35), 6)
        planes = select_planes(volume.shape, 3, "orthogonal")
        observed, _ = observe(volume, planes)
        changed = volume.copy()
        changed[~observed] = ~changed[~observed]
        first = gp_complete(plane_sections(volume, planes), volume.shape, 5.0, -10.0, uncertainty=False)
        second = gp_complete(plane_sections(changed, planes), volume.shape, 5.0, -10.0, uncertainty=False)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertLess(first[1][0, 0, 0], -5)  # far corner: posterior mean near the -10 mm prior


class PriorInrTests(unittest.TestCase):
    def test_distillation_reproduces_prior_without_section_updates_leaking(self):
        shape = (24, 24, 24)
        target = sphere(shape, 7)
        prior = sphere(shape, 8)
        points, labels, observed = plane_samples(torch.from_numpy(target).float(),
                                                 select_planes(shape, 3, "orthogonal"))
        model = ImplicitOccupancyNetwork(32, 3, 3)
        seconds, section_bce, distill_bce = fit_prior_inr(
            model, points, labels, observed, prior, steps=50, lr=1e-3, batch_size=1024, prior_weight=0.5,
            seed=17, device=torch.device("cpu"))
        self.assertLess(distill_bce, 0.3)
        self.assertTrue(np.isfinite(section_bce) and seconds > 0)


class SsmTests(unittest.TestCase):
    def test_fits_intermediate_sphere_from_sections(self):
        shape = (110, 110, 140)
        training = [sphere(shape, radius) for radius in (14, 18, 22, 26, 30)]
        artifact = build_ssm(training, ["A", "B", "C", "D", "E"], max_modes=4)
        target = sphere(shape, 20)
        observed = np.zeros(shape, bool)
        for axis, index in select_planes(shape, 3, "orthogonal"):
            selection = [slice(None)] * 3
            selection[axis] = index
            observed[tuple(selection)] = True
        mask, info = fit_ssm(artifact, np.argwhere(observed), target[observed].astype(np.float32),
                             shape, 4, 0.001)
        self.assertGreater(dice(mask, target), 0.9)
        self.assertEqual(len(info["coefficients"]), 4)


class UnetTests(unittest.TestCase):
    def test_output_matches_input_size(self):
        output = UNet3D(base=4, levels=3)(torch.zeros(1, 2, 16, 24, 32))
        self.assertEqual(tuple(output.shape), (1, 1, 16, 24, 32))

    def test_jittered_planes_stay_valid_and_distinct(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            planes = jittered_planes((120, 120, 150), "axial", (0.375, 0.5, 0.625), rng)
            self.assertEqual(len(set(planes)), 3)
            self.assertTrue(all(0 <= index < 120 for _, index in planes))
        observed, values = observe(np.ones((8, 8, 8), bool), [(0, 2)])
        self.assertEqual(observed.sum(), 64)
        self.assertEqual(values.sum(), 64)


if __name__ == "__main__":
    unittest.main()
