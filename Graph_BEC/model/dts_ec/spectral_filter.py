"""Learnable Fourier-domain spectral feature extractor for DTS-EC."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralFilter(nn.Module):
    """Apply learnable temporal Fourier filtering followed by channel refinement."""

    def __init__(
        self,
        window_length: int,
        roi_count: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, window_length // 2 + 1, roi_count, hidden_dim, 2) * 0.02
        )
        self.filter_dropout = nn.Dropout(dropout)
        self.filter_norm = nn.LayerNorm(hidden_dim)
        self.expand = nn.Linear(hidden_dim, hidden_dim * 4)
        self.project = nn.Linear(hidden_dim * 4, hidden_dim)
        self.feature_dropout = nn.Dropout(dropout)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spectral = torch.fft.rfft(inputs, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        filtered = torch.fft.irfft(spectral * weight, n=inputs.shape[1], dim=1, norm="ortho")
        filtered = self.filter_norm(inputs + self.filter_dropout(filtered))
        refined = self.project(F.gelu(self.expand(filtered)))
        return self.feature_norm(filtered + self.feature_dropout(refined))


__all__ = ["SpectralFilter"]
