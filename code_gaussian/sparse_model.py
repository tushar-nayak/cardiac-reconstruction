"""Gaussian field for the shared sparse-plane benchmark."""

import torch
from torch import nn
from torch.nn import functional as F

from code_gaussian.src.occupancy import evaluate_gaussian_occupancy


class SparseGaussianField(nn.Module):
    """Axis-aligned Gaussian field initialized without access to target labels."""

    def __init__(self, num_gaussians=256):
        super().__init__()
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be positive")
        self.means = nn.Parameter(torch.rand(1, num_gaussians, 3) * 2 - 1)
        self.raw_scales = nn.Parameter(torch.full((1, num_gaussians, 3), -1.8))
        self.raw_opacities = nn.Parameter(torch.full((1, num_gaussians, 1), -2.0))

    def forward(self, points):
        return evaluate_gaussian_occupancy(
            points.reshape(1, -1, 3), self.means,
            F.softplus(self.raw_scales) + 1e-4,
            torch.sigmoid(self.raw_opacities),
        ).reshape(*points.shape[:-1], 1)
