"""Parameterized BEC estimation: FSTA composition, loss, and extraction.

Merged from: fsta_graph_bec, bec_extractor, losses, bec_refiner.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from Graph_BEC.data import fixed_window_starts
from Graph_BEC.normative_bec import (
    MatrixGateRefiner,
    anchor_loss,
    gate_sparsity_loss,
    variance_retention_loss,
)

from .fsta_components import FSTA


# ---------------------------------------------------------------------------
# 1.  FSTA + phenotype-guided BEC refinement  (was fsta_graph_bec)
# ---------------------------------------------------------------------------

class FSTAGraphBEC(nn.Module):
    """The proposed model; classification is intentionally outside this class."""

    def __init__(
        self,
        fsta_options,
        window_length=78,
        nodes_num=90,
        d_model=16,
        d_inner=64,
        n_head=2,
        d_k=8,
        d_v=8,
        dropout=0.2,
        gate_init=-3.0,
        gate_max=0.2,
        share_gate=False,
    ):
        super().__init__()
        self.fsta = FSTA(
            fsta_options, window_length, d_model, d_inner, n_head, d_k, d_v, dropout
        )
        self.refiner = MatrixGateRefiner(nodes_num, gate_init, gate_max, share_gate)

    def forward_fsta(self, windows):
        return self.fsta(windows)

    def refine_bec(self, bec, phenotype_deviation, return_parts=False):
        return self.refiner(bec, phenotype_deviation, return_parts=return_parts)


# ---------------------------------------------------------------------------
# 2.  Unsupervised FSTA window loss  (was losses)
# ---------------------------------------------------------------------------

class FSTAWindowLoss(nn.Module):
    MODES = ("original", "entropy")

    def __init__(self, mode="entropy", alpha=0.01, node_count=90, eps=1e-8):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown FSTA loss mode: {mode}")
        self.mode = mode
        self.alpha = alpha
        self.node_count = node_count
        self.eps = eps
        self.reconstruction = nn.MSELoss()

    def forward(self, reconstruction, attention, target):
        prediction = self.reconstruction(reconstruction, target)
        if self.mode == "original":
            regularizer = attention.sum()
        else:
            probability = attention.clamp_min(self.eps)
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            regularizer = (
                (-(probability * probability.log()).sum(dim=-1)).mean()
                / torch.log(
                    torch.tensor(float(self.node_count), device=attention.device)
                )
            )
        return prediction + self.alpha * regularizer, prediction, regularizer


# ---------------------------------------------------------------------------
# 3.  Subject-level BEC extraction  (was bec_extractor)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_subject_bec(model, records, time_series, window_length, stride, device):
    model.eval()
    all_bec, all_mse = [], []
    for index, (record, series) in enumerate(zip(records, time_series), 1):
        attentions, errors = [], []
        for start in fixed_window_starts(series.shape[0], window_length, stride):
            window = (
                torch.from_numpy(series[start : start + window_length])
                .float()
                .unsqueeze(0)
                .to(device)
            )
            reconstruction, attention = model(window)
            attentions.append(attention.squeeze(0).cpu().numpy())
            errors.append(
                float((reconstruction - window).pow(2).mean().item())
            )
        bec = np.mean(np.stack(attentions), axis=0).T.astype(np.float32)
        np.fill_diagonal(bec, 0.0)
        all_bec.append(bec)
        all_mse.append(np.mean(errors))
        if index == 1 or index == len(records) or index % 100 == 0:
            print(
                f"BEC [{index}/{len(records)}] subject={record.subject_id} "
                f"mse={np.mean(errors):.6f}"
            )
    return {
        "bec": np.stack(all_bec),
        "reconstruction_mse": np.asarray(all_mse, dtype=np.float32),
    }
