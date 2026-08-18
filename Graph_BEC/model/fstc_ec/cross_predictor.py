"""Shared cross-ROI residual prediction head."""

from __future__ import annotations

from torch import nn


class CrossPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, messages):
        return self.network(messages).squeeze(-1)
