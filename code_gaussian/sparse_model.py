"""Gaussian field for the shared sparse-plane benchmark."""

import torch
from torch import nn
from torch.nn import functional as F

from code_gaussian.src.occupancy import evaluate_gaussian_occupancy


class SparseGaussianField(nn.Module):
    """Axis-aligned Gaussian field with optional observed-plane initialization."""

    def __init__(self, num_gaussians=256):
        super().__init__()
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be positive")
        self.means = nn.Parameter(torch.rand(1, num_gaussians, 3) * 2 - 1)
        self.raw_scales = nn.Parameter(torch.full((1, num_gaussians, 3), -1.8))
        self.raw_opacities = nn.Parameter(torch.full((1, num_gaussians, 1), -2.0))

    @torch.no_grad()
    def initialize_from_observations(self, points, labels):
        """Seed centers from positive observed points; never request a 3D mask.

        Small spatial jitter spreads centers around the observed planes. With
        no observed foreground, retain the constructor's uniform initialization.
        Returns whether positive observations were available.
        """
        labels = labels.reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(labels):
            raise ValueError("expected (N,3) observed coordinates and N labels")
        if not torch.isfinite(points).all() or not torch.isfinite(labels).all():
            raise ValueError("observations must be finite")
        positive = points[labels > 0.5].to(self.means)
        if not len(positive):
            return False
        chosen = torch.randint(len(positive), (self.means.shape[1],), device=positive.device)
        centers = positive[chosen].unsqueeze(0) + torch.randn_like(self.means) * 0.05
        self.means.copy_(centers.clamp(-1, 1))
        return True

    def forward(self, points):
        return evaluate_gaussian_occupancy(
            points.reshape(1, -1, 3), self.means,
            F.softplus(self.raw_scales) + 1e-4,
            torch.sigmoid(self.raw_opacities),
        ).reshape(*points.shape[:-1], 1)
