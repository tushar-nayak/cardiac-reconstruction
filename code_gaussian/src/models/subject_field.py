"""Query a fitted physical-space Gaussian field on normalized voxel coordinates."""

import torch
from torch import nn

from code_gaussian.src.occupancy import evaluate_gaussian_occupancy


class SubjectGaussianField(nn.Module):
    def __init__(self, params, shape_zyx, affine):
        super().__init__()
        for key in ("means", "scales", "opacities"):
            self.register_buffer(key, params[key].detach())
        self.register_buffer("affine", torch.as_tensor(affine).float())
        self.register_buffer("voxel_extent_xyz", torch.tensor(list(reversed(shape_zyx))).float() - 1)

    def forward(self, normalized_xyz):
        shape = normalized_xyz.shape[:-1]
        voxel = (normalized_xyz.reshape(-1, 3) + 1) * self.voxel_extent_xyz / 2
        homogeneous = torch.cat((voxel, torch.ones_like(voxel[:, :1])), dim=-1)
        world = (homogeneous @ self.affine.T)[:, :3]
        return evaluate_gaussian_occupancy(
            world.unsqueeze(0), self.means, self.scales, self.opacities,
        ).reshape(*shape, 1)
