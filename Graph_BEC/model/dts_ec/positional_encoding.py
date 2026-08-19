"""Sinusoidal temporal position encoding for DTS-EC."""

from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(hidden_dim, dtype=torch.float32)
        angles = positions / torch.pow(10000.0, 2 * torch.floor(dimensions / 2) / hidden_dim)
        table = torch.empty(max_length, hidden_dim)
        table[:, 0::2] = torch.sin(angles[:, 0::2])
        table[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.table[:, : inputs.shape[1]]


__all__ = ["PositionalEncoding"]
