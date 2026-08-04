"""方案二：PGR-BEC-Dynamic，加入 FSTA 动态一致性约束。

与 Static 方案相比，这里要求 refined BEC 回到冻结的 FSTA 重建路径，
让修正后的 BEC 仍然能够解释受试者自己的 ROI 时间序列。
输入 BEC 必须是 FSTA 原始 attention 空间，不能是分类阶段标准化后的 BEC。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .pgr_bec_static import PGRBECStatic, static_refinement_loss


class PGRBECDynamic(PGRBECStatic):
    """Static edge gate plus a frozen FSTA reconstruction path."""
    def __init__(self, fsta_model, nodes_num=90, hidden_channels=16, gate_max=0.2):
        super().__init__(nodes_num, hidden_channels, gate_max)
        self.fsta_model = fsta_model
        for parameter in self.fsta_model.parameters():
            parameter.requires_grad_(False)
        self.fsta_model.eval()

    def train(self, mode=True):
        """Train only the gate network; keep the pretrained FSTA frozen/eval."""
        super().train(mode)
        self.fsta_model.eval()
        return self

    def forward(self, initial_bec, neighbor_bec, windows, return_parts=False):
        refined, gate, difference = super().forward(
            initial_bec, neighbor_bec, return_parts=True
        )
        reconstruction = self.fsta_model.forward_with_bec(windows, refined)
        if return_parts:
            return refined, reconstruction, gate, difference
        return refined, reconstruction


def dynamic_refinement_loss(refined, reconstruction, target, initial, gate,
                            anchor_weight=1.0, gate_weight=1e-3,
                            variance_weight=1.0, variance_retention=0.85,
                            dynamic_weight=1.0):
    """Combine subject-level reconstruction with conservative BEC constraints."""
    dynamic = F.mse_loss(reconstruction, target)
    static_total, static_parts = static_refinement_loss(
        refined, initial, gate,
        variance_retention=variance_retention,
        anchor_weight=anchor_weight,
        gate_weight=gate_weight,
        variance_weight=variance_weight,
    )
    total = dynamic_weight * dynamic + static_total
    return total, {"dynamic_loss": dynamic, **static_parts}
