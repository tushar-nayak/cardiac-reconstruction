import unittest

import numpy as np
import torch

from cardiac_common.sparse import (
    coordinates, evaluate_grid, plane_samples, reconstruction_metrics, select_planes,
)
from code_gaussian.sparse_model import SparseGaussianField


class SparseBenchmarkTests(unittest.TestCase):
    def test_unobserved_labels_cannot_change_training_samples(self):
        target = torch.zeros(9, 8, 7)
        planes = select_planes(target.shape, 3, "orthogonal")
        points, labels, observed = plane_samples(target, planes)
        changed = target.clone()
        changed[~observed] = 1
        changed_points, changed_labels, changed_observed = plane_samples(changed, planes)
        self.assertTrue(torch.equal(points, changed_points))
        self.assertTrue(torch.equal(labels, changed_labels))
        self.assertTrue(torch.equal(observed, changed_observed))

    def test_plane_intersections_are_not_oversampled(self):
        points, labels, observed = plane_samples(torch.ones(3, 3, 3), [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(len(points), 19)
        self.assertEqual(len(points.unique(dim=0)), len(points))
        self.assertEqual(int(observed.sum()), len(labels))

    def test_axial_planes_are_distinct_even_for_small_volumes(self):
        self.assertEqual(select_planes((3, 5, 7), 3), [(0, 0), (0, 1), (0, 2)])
        with self.assertRaises(ValueError):
            select_planes((3, 5, 7), 4)

    def test_grid_sampling_preserves_xyz_order_and_stride(self):
        class CoordinateField(torch.nn.Module):
            def forward(self, points):
                return (points[:, 0] + points[:, 1] * 2 + points[:, 2] * 3).unsqueeze(-1)
        shape = (4, 5, 6)
        axes = torch.meshgrid(*(torch.arange(size) for size in shape), indexing="ij")
        xyz = coordinates(torch.stack(axes, dim=-1), shape)
        expected = xyz[..., 0] + 2 * xyz[..., 1] + 3 * xyz[..., 2]
        prediction = evaluate_grid(CoordinateField(), shape, torch.device("cpu"), 7, 2)
        torch.testing.assert_close(prediction, expected[::2, ::2, ::2])

    def test_unseen_overlap_and_physical_volume_error(self):
        target = np.ones((2, 2, 2))
        observed = np.zeros_like(target, dtype=bool)
        observed[0] = True
        prediction = observed.astype(float)
        metrics = reconstruction_metrics(prediction, target, observed, np.diag([2, 3, 4, 1]))
        self.assertAlmostEqual(metrics["whole_grid"]["dice"], 2 / 3)
        self.assertEqual(metrics["unobserved_voxels"]["dice"], 0)
        self.assertAlmostEqual(metrics["absolute_volume_error_ml"], 4 * 24 / 1000)

    def test_empty_masks_agree_and_fully_observed_has_no_unseen_score(self):
        empty = np.zeros((2, 2, 2))
        metrics = reconstruction_metrics(empty, empty, np.ones_like(empty), np.eye(4))
        self.assertEqual(metrics["whole_grid"]["dice"], 1)
        self.assertIsNone(metrics["unobserved_voxels"])

    def test_gaussian_has_finite_gradients_from_plane_samples(self):
        model = SparseGaussianField(8)
        points = torch.rand(17, 3) * 2 - 1
        prediction = model(points)
        self.assertEqual(tuple(prediction.shape), (17, 1))
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())
        prediction.sum().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
