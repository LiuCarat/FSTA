"""Key-value memories for global and directed-pair retrieval."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class KeyValueMemory(nn.Module):
    def __init__(self, hidden_dim: int = 32, slots: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slots = slots
        self.keys = nn.Parameter(torch.empty(slots, hidden_dim))
        self.values = nn.Parameter(torch.empty(slots, hidden_dim))
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_query = F.normalize(query, dim=-1)
        normalized_keys = F.normalize(self.keys, dim=-1)
        scores = torch.matmul(normalized_query, normalized_keys.transpose(0, 1))
        scores = scores * math.sqrt(self.hidden_dim)
        weights = scores.softmax(dim=-1)
        return torch.matmul(weights, self.values), weights

    def forward(
        self,
        pair_query: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_query.ndim != 4:
            raise ValueError(
                f"Expected pair_query [B,N,N,H], got {tuple(pair_query.shape)}"
            )
        if mode == "pair":
            return self.retrieve(pair_query)
        if mode != "global":
            raise ValueError(f"Unknown memory mode: {mode}")

        nodes = pair_query.shape[1]
        mask = ~torch.eye(nodes, dtype=torch.bool, device=pair_query.device)
        global_query = pair_query[:, mask].mean(dim=1)
        retrieved, weights = self.retrieve(global_query)
        retrieved = retrieved[:, None, None, :].expand_as(pair_query)
        return retrieved, weights
