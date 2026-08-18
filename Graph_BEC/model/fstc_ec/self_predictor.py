"""Shared ROI self-prediction head."""

from __future__ import annotations

from torch import nn


class SelfPredictor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        return self.projection(features).squeeze(-1)
