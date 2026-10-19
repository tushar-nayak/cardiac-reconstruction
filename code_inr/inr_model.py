"""Coordinate based implicit occupancy field used by the INR baseline."""

from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, frequencies: int = 4):
        super().__init__()
        if frequencies < 1:
            raise ValueError("frequencies must be positive")
        self.frequencies = frequencies

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        encoded = []
        for index in range(self.frequencies):
            scaled = coordinates * (2.0**index) * torch.pi
            encoded.extend((torch.sin(scaled), torch.cos(scaled)))
        return torch.cat(encoded, dim=-1)


class ImplicitOccupancyNetwork(nn.Module):
    """Predict occupancy in [0, 1] from normalized xyz coordinates."""

    def __init__(self, hidden_dim: int = 64, layers: int = 4, frequencies: int = 4):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        self.encoding = PositionalEncoding(frequencies)
        width = 3 * 2 * frequencies
        blocks: list[nn.Module] = []
        for index in range(layers):
            output_width = 1 if index == layers - 1 else hidden_dim
            blocks.append(nn.Linear(width, output_width))
            if index < layers - 1:
                blocks.append(nn.ReLU(inplace=True))
            width = output_width
        self.network = nn.Sequential(*blocks)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        shape = coordinates.shape[:-1]
        encoded = self.encoding(coordinates.reshape(-1, 3))
        return torch.sigmoid(self.network(encoded)).reshape(*shape, 1)
