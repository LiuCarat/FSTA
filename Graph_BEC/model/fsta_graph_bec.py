"""Composition of FSTA BEC estimation and label-free phenotype refinement."""
from __future__ import annotations
import torch.nn as nn
from .fsta_components import FSTA
from .bec_refiner import MatrixGateRefiner


class FSTAGraphBEC(nn.Module):
    """The proposed model; classification is intentionally outside this class."""
    def __init__(self, fsta_options, window_length=78, nodes_num=90, d_model=16,
                 d_inner=64, n_head=2, d_k=8, d_v=8, dropout=0.2,
                 gate_init=-3.0, gate_max=0.2, share_gate=False):
        super().__init__()
        self.fsta = FSTA(fsta_options, window_length, d_model, d_inner, n_head, d_k, d_v, dropout)
        self.refiner = MatrixGateRefiner(nodes_num, gate_init, gate_max, share_gate)

    def forward_fsta(self, windows):
        return self.fsta(windows)

    def refine_bec(self, bec, phenotype_deviation, return_parts=False):
        return self.refiner(bec, phenotype_deviation, return_parts=return_parts)
