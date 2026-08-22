"""Shared Gaussian occupancy operations.

The subject-fit and encoder paths must use the same field definition.  Keeping
this implementation in one place prevents small differences in scale or
chunking from changing reported metrics.
"""

from __future__ import annotations

import torch


def evaluate_gaussian_occupancy(
    points: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Evaluate ``1 - exp(-sum(alpha * gaussian))`` at world-space points.

    Args:
        points: ``(B, N, 3)`` query points.
        means/scales/opacities: batched Gaussian parameters. Opacities may be
            ``(B, G)`` or ``(B, G, 1)``.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape (B, G, 3), got {tuple(means.shape)}")
    if scales.shape != means.shape:
        raise ValueError("scales must have the same shape as means")
    if opacities.ndim == 2:
        opacities = opacities.unsqueeze(-1)
    if opacities.shape[:2] != means.shape[:2]:
        raise ValueError("opacities must have one value per Gaussian")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    values = []
    safe_scales = scales.clamp_min(torch.finfo(scales.dtype).eps)
    for start in range(0, points.shape[1], chunk_size):
        chunk = points[:, start : start + chunk_size]
        diff = (chunk.unsqueeze(2) - means.unsqueeze(1)) / safe_scales.unsqueeze(1)
        weights = torch.exp(-0.5 * diff.square().sum(dim=-1))
        density = (weights * opacities.squeeze(-1).unsqueeze(1)).sum(dim=-1)
        values.append(-torch.expm1(-density))
    return torch.cat(values, dim=1)
