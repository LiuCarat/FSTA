"""Shared causal temporal convolution for each ROI."""

from __future__ import annotations

import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.convolution(inputs)
        if self.padding:
            outputs = outputs[..., :-self.padding]
        outputs = self.dropout(self.activation(outputs))
        return self.norm(outputs + residual)


class CausalTCN(nn.Module):
    """One shared TCN applied independently to all ROI sequences."""

    def __init__(
        self,
        hidden_dim: int = 32,
        layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CausalBlock(hidden_dim, kernel_size, 2**index, dropout)
                for index in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [batch, time, roi, hidden]")
        batch, time, roi_count, hidden_dim = inputs.shape
        outputs = inputs.permute(0, 2, 3, 1).reshape(
            batch * roi_count, hidden_dim, time
        )
        for block in self.blocks:
            outputs = block(outputs)
        outputs = outputs.reshape(batch, roi_count, hidden_dim, time)
        outputs = outputs.permute(0, 3, 1, 2)
        return self.output_norm(outputs)
