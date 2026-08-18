"""Learnable Fourier-temporal encoder for fixed-length ROI windows."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, window_length: int, hidden_dim: int):
        super().__init__()
        positions = torch.arange(window_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(window_length, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0).unsqueeze(2), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.shape[1]]


class FourierEncoder(nn.Module):
    """Apply a trainable complex frequency filter along the time axis."""

    def __init__(
        self,
        window_length: int = 78,
        roi_count: int = 90,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_length = window_length
        self.roi_count = roi_count
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(1, hidden_dim)
        frequency_count = window_length // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(1, frequency_count, roi_count, hidden_dim, 2) * 0.02
        )
        self.position = SinusoidalPositionEncoding(window_length, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, roi]")
        batch, time, roi_count = inputs.shape
        if time != self.window_length:
            raise ValueError(
                f"Expected time length {self.window_length}, got {time}"
            )
        if roi_count != self.roi_count:
            raise ValueError(
                f"Expected {self.roi_count} ROIs, got {roi_count}"
            )

        embedded = self.embedding(inputs.unsqueeze(-1))
        spectrum = torch.fft.rfft(embedded, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight.contiguous())
        filtered = torch.fft.irfft(spectrum * weight, n=time, dim=1, norm="ortho")
        return self.norm(self.position(embedded + self.dropout(filtered)))
