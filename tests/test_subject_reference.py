import unittest
from unittest.mock import patch

import torch

from code_gaussian.scripts.fit_subject_reconstruction import initialize_params, gaussian_params, sample_points
from code_gaussian.src.data_loaders.mitea_loader import MITEADataset
from code_gaussian.src.models.subject_field import SubjectGaussianField


class SubjectReferenceTests(unittest.TestCase):
    def test_historical_split_reproduces_may_subject_boundaries(self):
        files = [f"/data/images/MITEA_{index:03d}_scan1_ED.nii.gz" for index in range(1, 135)]
        with patch("code_gaussian.src.data_loaders.mitea_loader.glob.glob", return_value=files):
            splits = {name: MITEADataset("/data", split=name, split_policy="may2026").subject_ids
                      for name in ("train", "val", "test")}
            current = MITEADataset("/data", split="val").subject_ids
        self.assertEqual([len(splits[name]) for name in ("train", "val", "test")], [110, 15, 9])
        self.assertEqual(splits["val"][0], "MITEA_111")
        self.assertNotEqual(current, splits["val"])
        self.assertEqual(len(set(sum(splits.values(), []))), 134)

    def test_world_query_uses_affine_and_xyz_coordinate_order(self):
        affine = torch.diag(torch.tensor([2., 3., 4., 1.]))
        affine[:3, 3] = torch.tensor([5., 6., 7.])
        params = {"means": torch.tensor([[[11., 12., 11.]]]),
                  "scales": torch.ones(1, 1, 3), "opacities": torch.ones(1, 1, 1)}
        field = SubjectGaussianField(params, (3, 5, 7), affine)
        points = torch.tensor([[0., 0., 0.], [-1., -1., -1.], [1., 1., 1.]])
        prediction = field(points).flatten()
        expected = -torch.expm1(-torch.exp(-0.5 * torch.tensor([0., 88., 88.])))
        torch.testing.assert_close(prediction, expected)

    def test_may_sampling_retains_positive_uniform_weights_without_centroid_term(self):
        volume = torch.zeros(12, 13, 14)
        volume[3:9, 3:10, 3:11] = 1
        points, weights = sample_points(volume, torch.eye(4), 101, torch.device("cpu"), "may2026")
        self.assertEqual(tuple(points.shape), (1, 101, 3))
        torch.testing.assert_close(weights[0, :50], torch.full((50,), 1.5))
        torch.testing.assert_close(weights[0, 50:], torch.full((51,), 2.0))
        _, stable_weights = sample_points(volume, torch.eye(4), 99, torch.device("cpu"), "stabilized")
        self.assertEqual(int((stable_weights == 8).sum()), 33)

    def test_may_initialization_keeps_physical_scales_and_high_opacity(self):
        volume = torch.ones(4, 4, 4)
        means, log_scales, raw_opacities = initialize_params(volume, torch.eye(4), 16, torch.device("cpu"))
        params = gaussian_params(means, log_scales, raw_opacities)
        self.assertAlmostEqual(params["scales"].mean().item(), 1.5911538747, places=5)
        self.assertAlmostEqual(params["opacities"].mean().item(), 0.8021838886, places=5)
        self.assertTrue(means.requires_grad)


if __name__ == "__main__":
    unittest.main()
