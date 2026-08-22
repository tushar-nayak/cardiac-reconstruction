import unittest

import numpy as np

from src_code.src.metrics import binary_metrics, volume_to_mesh


class MetricsTests(unittest.TestCase):
    def test_binary_metrics(self):
        result = binary_metrics([[1, 0], [1, 0]], [[1, 1], [0, 0]])
        self.assertAlmostEqual(result["dice"], 0.5)
        self.assertAlmostEqual(result["iou"], 1 / 3)

    def test_mesh_vertices_are_transformed_to_world(self):
        volume = np.zeros((5, 5, 5), dtype=np.float32)
        volume[1:4, 1:4, 1:4] = 1
        affine = np.eye(4, dtype=np.float32)
        affine[:3, 3] = [10, 20, 30]
        vertices, faces = volume_to_mesh(volume, affine)
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
        self.assertTrue(np.all(vertices[:, 0] >= 10))
