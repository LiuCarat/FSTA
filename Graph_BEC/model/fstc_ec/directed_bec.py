"""Lagged directed effective-connectivity generation and propagation."""

from __future__ import annotations

import math

import torch
from torch import nn


class DirectedBECGenerator(nn.Module):
    """Estimate lagged directed influence and propagate source states."""

    def __init__(self, hidden_dim: int = 32, edge_dim: int = 16):
        super().__init__()
        self.source_projection = nn.Linear(hidden_dim, edge_dim, bias=False)
        self.target_projection = nn.Linear(hidden_dim, edge_dim, bias=False)
        self.causal_projection = nn.Linear(hidden_dim, edge_dim, bias=False)

    def forward(self, features: torch.Tensor):
        if features.ndim != 4:
            raise ValueError("features must have shape [batch, time, roi, hidden]")
        if features.shape[1] < 2:
            raise ValueError("features must contain at least two time points")

        source_features = features[:, :-1]
        target_features = features[:, 1:]
        source = self.source_projection(source_features)
        target = self.target_projection(target_features)
        window_logits = torch.einsum("btid,btjd->btij", source, target)
        window_logits = window_logits / math.sqrt(source.shape[-1])

        time_specific_bec = torch.tanh(window_logits)
        time_specific_bec = time_specific_bec.masked_fill(
            torch.eye(
                time_specific_bec.shape[-1],
                dtype=torch.bool,
                device=time_specific_bec.device,
            ).view(1, 1, time_specific_bec.shape[-1], time_specific_bec.shape[-1]),
            0.0,
        )
        bec_logits = window_logits.mean(dim=1)
        bec = torch.tanh(bec_logits)
        diagonal = torch.eye(
            bec.shape[-1], dtype=torch.bool, device=bec.device
        ).unsqueeze(0)
        bec = bec.masked_fill(diagonal, 0.0)

        causal_features = self.causal_projection(source_features)
        messages = torch.einsum(
            "btij,btid->btjd", time_specific_bec, causal_features
        )
        return {
            "bec": bec,
            "bec_logits": bec_logits,
            "window_logits": window_logits,
            "time_specific_bec": time_specific_bec,
            "messages": messages,
        }
