"""Key-value memories for global and directed-pair retrieval."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class KeyValueMemory(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        slots: int = 8,
        temperature: float = 1.0,
        adaptive_relations: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slots = slots
        self.temperature = temperature
        self.keys = nn.Parameter(torch.empty(slots, hidden_dim))
        self.values = nn.Parameter(torch.empty(slots, hidden_dim))
        self.relation_assignment = (
            nn.Parameter(torch.randn(slots, 3) * 0.01)
            if adaptive_relations else None
        )
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_query = F.normalize(query, dim=-1)
        normalized_keys = F.normalize(self.keys, dim=-1)
        scores = torch.matmul(normalized_query, normalized_keys.transpose(0, 1))
        scores = scores * (math.sqrt(self.hidden_dim) / self.temperature)
        weights = scores.softmax(dim=-1)
        return torch.matmul(weights, self.values), weights

    def retrieve_relations(
        self,
        queries: torch.Tensor,
        relation_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_queries = F.normalize(queries, dim=-1)
        normalized_keys = F.normalize(self.keys, dim=-1)
        cosine_scores = torch.matmul(
            normalized_queries,
            normalized_keys.transpose(0, 1),
        )
        cosine_scores = cosine_scores * (
            math.sqrt(self.hidden_dim) / self.temperature
        )
        log_pi = F.log_softmax(self.relation_assignment, dim=-1)
        relation_log_pi = log_pi.transpose(0, 1)[relation_types]
        scores = cosine_scores + relation_log_pi
        weights = scores.softmax(dim=-1)
        retrieved = torch.matmul(weights, self.values)
        evidence = torch.logsumexp(scores, dim=-1) - torch.logsumexp(
            relation_log_pi,
            dim=-1,
        )
        return retrieved, weights, evidence

    def assignment_statistics(self) -> dict[str, list]:
        if self.relation_assignment is None:
            return {}
        assignment = self.relation_assignment.softmax(dim=-1)
        return {
            "relation_assignment_logits": self.relation_assignment.detach().cpu().tolist(),
            "relation_assignment": assignment.detach().cpu().tolist(),
            "effective_memory_capacity": assignment.sum(dim=0).detach().cpu().tolist(),
        }

    def forward(
        self,
        pair_query: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "pair":
            return self.retrieve(pair_query)
        if mode == "global":
            nodes = pair_query.shape[1]
            mask = ~torch.eye(nodes, dtype=torch.bool, device=pair_query.device)
            global_query = pair_query[:, mask].mean(dim=1)
            retrieved, weights = self.retrieve(global_query)
            retrieved = retrieved[:, None, None, :].expand_as(pair_query)
            return retrieved, weights
