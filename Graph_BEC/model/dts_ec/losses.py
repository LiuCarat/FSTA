"""Reconstruction and directed-EC regularization losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def reconstruction_stage_loss(output, target, entropy_weight=0.005):
    reconstruction = F.mse_loss(output["reconstruction"], target)
    bec = output["bec"].clamp_min(1e-8)
    entropy = -(bec * bec.log()).sum(dim=-1).mean()
    entropy = entropy / torch.log(
        torch.tensor(float(bec.shape[-1] - 1), device=bec.device)
    ).clamp_min(1e-8)
    total = reconstruction + entropy_weight * entropy
    return total, {"reconstruction": reconstruction, "entropy": entropy}


__all__ = ["reconstruction_stage_loss"]
