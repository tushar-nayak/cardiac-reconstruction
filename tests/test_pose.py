import torch
import unittest

from code_gaussian.src.models.pose_model import axis_angle_to_matrix


class PoseTests(unittest.TestCase):
    def test_zero_axis_angle_is_identity(self):
        rotation = axis_angle_to_matrix(torch.zeros(4, 3))
        self.assertTrue(torch.allclose(rotation, torch.eye(3).expand(4, -1, -1), atol=1e-6))
