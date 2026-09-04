"""QC-guided self-supervised weak refinement for directed BEC matrices."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Graph_BEC.downstream.brainnetcnn import DirectedE2E


class QSRBECRefiner(nn.Module):
    """Refine BEC with fold-level QC sensitivity and neighbor context."""

    def __init__(self, nodes_num=90, hidden_channels=8, gate_max=0.5):
        super().__init__()
        self.nodes_num = int(nodes_num)
        self.gate_max = float(gate_max)
        self.input_projection = nn.Conv2d(4, hidden_channels, kernel_size=1)
        self.context_first = DirectedE2E(hidden_channels, hidden_channels, self.nodes_num)
        self.context_second = DirectedE2E(hidden_channels, hidden_channels, self.nodes_num)
        self.gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.direction_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, current_bec, neighbor_bec, qc_sensitive_map, return_parts=False):
        if current_bec.shape != neighbor_bec.shape:
            raise ValueError("current_bec and neighbor_bec must have matching shapes")
        if current_bec.ndim != 3 or current_bec.shape[-1] != self.nodes_num:
            raise ValueError("BEC inputs must have shape [subjects, nodes, nodes]")
        qc_map = torch.as_tensor(
            qc_sensitive_map, device=current_bec.device, dtype=current_bec.dtype
        )
        if qc_map.ndim == 2:
            qc_map = qc_map.unsqueeze(0).expand(current_bec.shape[0], -1, -1)
        if qc_map.shape != current_bec.shape:
            raise ValueError("qc_sensitive_map must be [nodes, nodes] or match BEC inputs")
        difference = neighbor_bec - current_bec
        inputs = torch.stack((current_bec, neighbor_bec, difference.abs(), qc_map), dim=1)
        features = F.leaky_relu(self.input_projection(inputs), negative_slope=0.1)
        features = F.leaky_relu(self.context_first(features), negative_slope=0.1)
        features = F.leaky_relu(self.context_second(features), negative_slope=0.1)
        gate = self.gate_max * torch.sigmoid(self.gate_head(features)).squeeze(1)
        direction = torch.tanh(self.direction_head(features)).squeeze(1)
        diagonal = torch.eye(self.nodes_num, device=current_bec.device, dtype=torch.bool)[None]
        gate = gate.masked_fill(diagonal, 0.0)
        direction = direction.masked_fill(diagonal, 0.0)
        refined = (current_bec + gate * direction * difference).masked_fill(diagonal, 0.0)
        if return_parts:
            return refined, gate, direction, difference
        return refined


def qsr_refinement_loss(
    original_refined,
    corrupted_refined,
    pseudo_target,
    original_bec,
    original_gate,
    corrupted_gate,
    variance_retention=0.85,
    gate_weight=0.1,
    variance_weight=0.1,
):
    """Combine pseudo-target restoration, sparse gating, and variance retention."""
    pseudo = F.smooth_l1_loss(original_refined, pseudo_target)
    restore = F.smooth_l1_loss(corrupted_refined, pseudo_target)
    gate = 0.5 * (original_gate.abs().mean() + corrupted_gate.abs().mean())
    original_variance = original_bec.flatten(1).var(dim=0, unbiased=False).mean()
    refined_variance = original_refined.flatten(1).var(dim=0, unbiased=False).mean()
    variance = F.relu(variance_retention * original_variance - refined_variance)
    total = pseudo + restore + gate_weight * gate + variance_weight * variance
    return total, {
        "pseudo_loss": pseudo,
        "restore_loss": restore,
        "gate_loss": gate,
        "variance_loss": variance,
    }
