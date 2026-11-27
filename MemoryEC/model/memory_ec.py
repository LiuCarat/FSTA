"""Lightweight spatiotemporal MemoryEC model."""

from __future__ import annotations

import torch
from torch import nn

from MemoryEC.model.memory import KeyValueMemory
from MemoryEC.model.spatial import SpatialAttention
from MemoryEC.model.tcn import SharedTCN
from MemoryEC.utils import normalize_per_roi


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
            nn.Linear(4 * hidden_dim + 3 * max_lag, hidden_dim),
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
            features.append(source_to_target)
        return torch.stack(features, dim=-1)

    def forward(self, node_features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        nodes = node_features.shape[1]
        source = node_features[:, :, None, :].expand(-1, -1, nodes, -1)
        target = node_features[:, None, :, :].expand(-1, nodes, -1, -1)
        forward_lag = self.lag_features(x)
        reverse_lag = forward_lag.transpose(1, 2)
        global_context = node_features.mean(dim=1)[:, None, None, :].expand_as(source)
        return self.projection(torch.cat(
            (
                source,
                target,
                source - target,
                forward_lag,
                reverse_lag,
                forward_lag - reverse_lag,
                global_context,
            ),
            dim=-1,
        ))


class MemoryEC(nn.Module):
    def __init__(
        self,
        model_type: str = "st_pair_memory",
        hidden_dim: int = 32,
        memory_slots: int = 8,
        memory_temperature: float = 1.0,
        attention_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.memory_mode = MEMORY_MODES[model_type]
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.memory_temperature = memory_temperature
        self.attention_heads = attention_heads
        self.dropout_rate = dropout

        self.tcn = SharedTCN(hidden_dim, dropout)
        self.spatial = SpatialAttention(hidden_dim, attention_heads, dropout)
        self.pair_query = DirectedPairQuery(hidden_dim, max_lag=3)
        if self.memory_mode != "none":
            self.memory = KeyValueMemory(
                hidden_dim,
                memory_slots,
                memory_temperature,
                adaptive_relations=self.memory_mode == "pair",
            )
            self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.memory = None
            self.gate = None
        decoder_input = hidden_dim + 1 if self.memory_mode == "pair" else hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def enforce_relation_exchange(logits: torch.Tensor) -> torch.Tensor:
        reverse = logits.transpose(1, 2)
        return torch.stack(
            (
                (logits[..., 0] + reverse[..., 0]) / 2,
                (logits[..., 1] + reverse[..., 2]) / 2,
                (logits[..., 2] + reverse[..., 1]) / 2,
                (logits[..., 3] + reverse[..., 3]) / 2,
            ),
            dim=-1,
        )

    def pair_memory_forward(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reverse_query = query.transpose(1, 2)
        symmetric_query = (query + reverse_query) / 2
        candidates = torch.stack(
            (symmetric_query, query, reverse_query, symmetric_query),
            dim=-2,
        )
        relation_types = torch.tensor((0, 1, 1, 2), device=query.device)
        retrieved, memory_weights, memory_evidence = self.memory.retrieve_relations(
            candidates,
            relation_types,
        )
        gate = torch.sigmoid(self.gate(torch.cat((candidates, retrieved), dim=-1)))
        fused = candidates + gate * retrieved
        raw_relation_logits = self.decoder(
            torch.cat((fused, memory_evidence.unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        relation_logits = self.enforce_relation_exchange(raw_relation_logits)
        memory_evidence = self.enforce_relation_exchange(memory_evidence)
        edge_logits = torch.logsumexp(relation_logits[..., (1, 3)], dim=-1)
        edge_logits = edge_logits - torch.logsumexp(
            relation_logits[..., (0, 2)],
            dim=-1,
        )
        return edge_logits, {
            "relation_logits": relation_logits,
            "relation_probs": relation_logits.softmax(dim=-1),
            "memory_logits": memory_evidence,
            "memory_weights": memory_weights,
            "gate": gate,
        }

    def config(self) -> dict:
        return {
            "model_type": self.model_type,
            "hidden_dim": self.hidden_dim,
            "memory_slots": self.memory_slots,
            "memory_temperature": self.memory_temperature,
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

        if self.memory_mode == "pair":
            logits, pair_aux = self.pair_memory_forward(query)
            if not return_aux:
                return logits
            return logits, {
                "normalized": normalized,
                "temporal": temporal,
                "spatial": spatial,
                "pooled": pooled,
                "pair_query": query,
                **pair_aux,
            }

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
