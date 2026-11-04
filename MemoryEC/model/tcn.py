"""Shared lightweight temporal convolutional encoder."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(self.activation(self.norm(self.conv(x))))
        return self.pool(x + residual)


class SharedTCN(nn.Module):
    """Apply one temporal encoder to every ROI independently."""

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(
            TemporalBlock(hidden_dim, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, dilation=2, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,T,N], got {tuple(x.shape)}")
        batch, _, nodes = x.shape
        x = x.permute(0, 2, 1).reshape(batch * nodes, 1, x.shape[1])
        x = self.blocks(self.input_projection(x))
        time = x.shape[-1]
        return x.reshape(batch, nodes, -1, time).permute(0, 3, 1, 2)
