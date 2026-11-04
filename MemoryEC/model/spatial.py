"""Permutation-equivariant attention across ROIs."""

from __future__ import annotations

import torch
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,N,H], got {tuple(x.shape)}")
        batch, time, nodes, hidden = x.shape
        flat = x.reshape(batch * time, nodes, hidden)
        attended, weights = self.attention(
            flat,
            flat,
            flat,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        output = self.norm(flat + self.dropout(attended)).reshape(
            batch, time, nodes, hidden
        )
        if return_attention:
            return output, weights.reshape(batch, time, -1, nodes, nodes)
        return output
