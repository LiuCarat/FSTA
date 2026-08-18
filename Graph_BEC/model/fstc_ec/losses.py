"""Training losses for the lagged FSTC-EC model."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sparse_bec_loss(bec):
    """Mean off-diagonal L1 penalty for directed BEC."""
    if bec.ndim != 3 or bec.shape[-1] != bec.shape[-2]:
        raise ValueError("bec must have shape [batch, roi, roi]")
    roi_count = bec.shape[-1]
    mask = ~torch.eye(roi_count, dtype=torch.bool, device=bec.device)
    return bec[:, mask].abs().mean()


def fstc_ec_loss(output, delta_target, lambda_sparse: float = 1e-6):
    """Predict one-step BOLD changes from lagged directed EC."""
    delta_loss = F.mse_loss(output["delta_prediction"], delta_target)
    sparse_loss = sparse_bec_loss(output["bec"])
    total = delta_loss + lambda_sparse * sparse_loss
    return total, {"delta": delta_loss, "sparse": sparse_loss}


__all__ = ["fstc_ec_loss", "sparse_bec_loss"]
