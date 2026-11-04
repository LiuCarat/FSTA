"""Lightweight spatiotemporal MemoryEC model."""

from __future__ import annotations

import torch
from torch import nn

from model.memory import KeyValueMemory
from model.spatial import SpatialAttention
from model.tcn import SharedTCN
from utils import normalize_per_roi


MEMORY_MODES = {
    "st_no_memory": "none",
    "st_global_memory": "global",
    "st_pair_memory": "pair",
}


class DirectedPairQuery(nn.Module):
    def __init__(self, hidden_dim: int = 32, max_lag: int = 3) -> None:
        super().__init__()
        self.max_lag = max_lag
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 * max_lag, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def lag_features(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for lag in range(1, self.max_lag + 1):
            earlier = x[:, :-lag]
            later = x[:, lag:]
            source_to_target = torch.einsum("bti,btj->bij", earlier, later)
            source_to_target = source_to_target / earlier.shape[1]
            target_to_source = source_to_target.transpose(1, 2)
            features.extend((source_to_target, target_to_source))
        return torch.stack(features, dim=-1)

    def forward(self, node_features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        nodes = node_features.shape[1]
        source = node_features[:, :, None, :].expand(-1, -1, nodes, -1)
        target = node_features[:, None, :, :].expand(-1, nodes, -1, -1)
        lag = self.lag_features(x)
        return self.projection(torch.cat((source, target, lag), dim=-1))


class MemoryEC(nn.Module):
    def __init__(
        self,
        model_type: str = "st_pair_memory",
        hidden_dim: int = 32,
        memory_slots: int = 8,
        attention_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if model_type not in MEMORY_MODES:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.model_type = model_type
        self.memory_mode = MEMORY_MODES[model_type]
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.attention_heads = attention_heads
        self.dropout_rate = dropout

        self.tcn = SharedTCN(hidden_dim, dropout)
        self.spatial = SpatialAttention(hidden_dim, attention_heads, dropout)
        self.pair_query = DirectedPairQuery(hidden_dim, max_lag=3)
        if self.memory_mode != "none":
            self.memory = KeyValueMemory(hidden_dim, memory_slots)
            self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.memory = None
            self.gate = None
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def config(self) -> dict:
        return {
            "model_type": self.model_type,
            "hidden_dim": self.hidden_dim,
            "memory_slots": self.memory_slots,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout_rate,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        normalized = normalize_per_roi(x)
        temporal = self.tcn(normalized)
        spatial = self.spatial(temporal)
        pooled = spatial.mean(dim=1)
        query = self.pair_query(pooled, normalized)

        memory_weights = None
        gate = None
        features = query
        if self.memory is not None and self.gate is not None:
            retrieved, memory_weights = self.memory(query, self.memory_mode)
            gate = torch.sigmoid(self.gate(torch.cat((query, retrieved), dim=-1)))
            features = query + gate * retrieved

        logits = self.decoder(features).squeeze(-1)
        if not return_aux:
            return logits
        return logits, {
            "normalized": normalized,
            "temporal": temporal,
            "spatial": spatial,
            "pooled": pooled,
            "pair_query": query,
            "memory_weights": memory_weights,
            "gate": gate,
        }
