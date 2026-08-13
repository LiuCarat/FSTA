"""方案一：PGR-BEC-Static，表型引导的静态边级门控修正。

该模块只使用初始 BEC 和表型邻域参考 BEC，不接收诊断标签。
它适合作为已有 subject_bec.npz 的稳定基线。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGRBECStatic(nn.Module):
    """Predict a subject-specific edge gate from [A, N, |N-A|]."""
    def __init__(self, nodes_num=90, hidden_channels=16, gate_max=0.2):
        super().__init__()
        self.nodes_num = nodes_num
        self.gate_max = float(gate_max)
        self.gate_network = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, initial_bec, neighbor_bec, gate_scale=None, return_parts=False):
        difference = neighbor_bec - initial_bec
        gate_input = torch.stack(
            (initial_bec, neighbor_bec, difference.abs()), dim=1
        )
        gate = self.gate_max * torch.sigmoid(self.gate_network(gate_input)).squeeze(1)
        if gate_scale is not None:
            scale = torch.as_tensor(gate_scale, device=gate.device, dtype=gate.dtype)
            if scale.numel() != initial_bec.shape[0]:
                raise ValueError("gate_scale must contain one value per subject")
            gate = gate * scale.reshape(-1, 1, 1)
        diagonal = torch.eye(self.nodes_num, device=initial_bec.device, dtype=torch.bool)[None]
        gate = gate.masked_fill(diagonal, 0.0)
        refined = (initial_bec + gate * difference).masked_fill(diagonal, 0.0)
        if return_parts:
            return refined, gate, difference
        return refined


def static_refinement_loss(refined, initial, gate, variance_retention=0.85,
                           anchor_weight=1.0, gate_weight=1e-3,
                           variance_weight=1.0):
    """Unsupervised loss; no ASD/TC term is present."""
    anchor = F.smooth_l1_loss(refined, initial.detach())
    sparse = gate.abs().mean()
    original_variance = initial.flatten(1).var(dim=0, unbiased=False).mean()
    refined_variance = refined.flatten(1).var(dim=0, unbiased=False).mean()
    variance = F.relu(variance_retention * original_variance - refined_variance)
    total = anchor_weight * anchor + gate_weight * sparse + variance_weight * variance
    return total, {"anchor_loss": anchor, "gate_loss": sparse, "variance_loss": variance}
