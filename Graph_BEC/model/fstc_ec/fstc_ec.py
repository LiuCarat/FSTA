"""Fourier-STC lagged effective-connectivity model."""

from __future__ import annotations

import torch
from torch import nn

from .causal_tcn import CausalTCN
from .cross_predictor import CrossPredictor
from .directed_bec import DirectedBECGenerator
from .fourier_encoder import FourierEncoder


class FSTCEC(nn.Module):
    """Encode ROI histories, estimate lagged EC, and predict BOLD changes."""

    def __init__(
        self,
        window_length: int = 78,
        roi_count: int = 90,
        hidden_dim: int = 32,
        edge_dim: int = 16,
        tcn_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_length = window_length
        self.roi_count = roi_count
        self.encoder = FourierEncoder(
            window_length=window_length,
            roi_count=roi_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal_encoder = CausalTCN(
            hidden_dim=hidden_dim,
            layers=tcn_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.bec_generator = DirectedBECGenerator(hidden_dim, edge_dim)
        self.delta_predictor = CrossPredictor(
            edge_dim, hidden_dim=max(1, hidden_dim // 2)
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.temporal_encoder(self.encoder(inputs))

    def forward(self, inputs: torch.Tensor):
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, roi]")
        features = self.encode(inputs)
        ec_output = self.bec_generator(features)
        delta_prediction = self.delta_predictor(ec_output["messages"])
        future_prediction = inputs[:, :-1] + delta_prediction
        return {
            "features": features,
            **ec_output,
            "delta_prediction": delta_prediction,
            "future_prediction": future_prediction,
        }
