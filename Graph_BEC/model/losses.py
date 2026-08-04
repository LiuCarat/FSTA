"""Unsupervised FSTA and BEC refinement losses."""
from Graph_BEC.normative_bec import (
    anchor_loss, gate_sparsity_loss, variance_retention_loss,
)
from torch import nn
import torch


class FSTAWindowLoss(nn.Module):
    MODES = ("original", "entropy")

    def __init__(self, mode="entropy", alpha=0.01, node_count=90, eps=1e-8):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown FSTA loss mode: {mode}")
        self.mode, self.alpha, self.node_count, self.eps = mode, alpha, node_count, eps
        self.reconstruction = nn.MSELoss()

    def forward(self, reconstruction, attention, target):
        prediction = self.reconstruction(reconstruction, target)
        if self.mode == "original":
            regularizer = attention.sum()
        else:
            probability = attention.clamp_min(self.eps)
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            regularizer = (-(probability * probability.log()).sum(dim=-1)).mean() / torch.log(torch.tensor(float(self.node_count), device=attention.device))
        return prediction + self.alpha * regularizer, prediction, regularizer
