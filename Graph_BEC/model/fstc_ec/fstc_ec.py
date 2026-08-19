"""Factorized FSTA temporal encoder with a directed spatial EC adapter."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .fourier_att import FourierAtt
from .st_multi_head_att import (
    PositionwiseFeedForward,
    PositionalEncoding,
    STMultiHeadAtt,
)


class FSTCECReconstruction(nn.Module):
    """Estimate directed window BEC and use it for signal reconstruction.

    ``bec`` uses source-by-target orientation: ``bec[:, source, target]``.
    """

    def __init__(
        self,
        window_length: int = 78,
        roi_count: int = 90,
        hidden_dim: int = 32,
        n_heads: int = 4,
        ec_dim: int = 16,
        ec_temperature: float = 0.25,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if ec_dim <= 0:
            raise ValueError("ec_dim must be positive")
        if ec_temperature <= 0:
            raise ValueError("ec_temperature must be positive")

        self.window_length = window_length
        self.roi_count = roi_count
        self.hidden_dim = hidden_dim
        self.ec_dim = ec_dim
        self.ec_temperature = ec_temperature
        head_dim = hidden_dim // n_heads
        fourier_options = SimpleNamespace(
            nodes_num=roi_count,
            time_num=window_length,
            d_model=hidden_dim,
            num_hidden_layers=1,
            no_filters=False,
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            initializer_range=0.02,
        )

        self.input_projection = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.position_encoding = PositionalEncoding(
            d_hid=hidden_dim, n_position=window_length
        )
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.fourier_attention = FourierAtt(fourier_options)
        self.temporal_attention = STMultiHeadAtt(
            hidden_dim, n_heads, head_dim, head_dim, dropout=dropout
        )
        self.temporal_feed_forward = PositionwiseFeedForward(
            hidden_dim, hidden_dim * 4, dropout=dropout
        )

        self.source_projection = nn.Linear(hidden_dim, ec_dim, bias=False)
        self.target_projection = nn.Linear(hidden_dim, ec_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.signal_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.decoder_feed_forward = PositionwiseFeedForward(
            hidden_dim, hidden_dim * 4, dropout=dropout
        )
        self.output_projection = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def _temporal_features(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(inputs.unsqueeze(1)).permute(0, 2, 3, 1)
        position = self.position_encoding(embedded).unsqueeze(2).expand_as(embedded)
        encoded = self.input_norm(self.input_dropout(embedded + position))
        fourier_features = self.fourier_attention(encoded)
        temporal_input = fourier_features.transpose(1, 2)
        temporal_features, _ = self.temporal_attention(temporal_input)
        temporal_features = self.temporal_feed_forward(temporal_features)
        return temporal_features.transpose(1, 2)

    def _directed_ec(self, temporal_features: torch.Tensor) -> torch.Tensor:
        source = self.source_projection(temporal_features)
        target = self.target_projection(temporal_features)
        logits = torch.einsum("btid,btjd->btij", source, target)
        logits = logits / (self.ec_dim**0.5)
        diagonal = torch.eye(self.roi_count, device=logits.device, dtype=torch.bool)
        aggregated_logits = logits.mean(dim=1)
        aggregated_logits = aggregated_logits.masked_fill(
            diagonal, torch.finfo(aggregated_logits.dtype).min
        )
        bec = torch.softmax(aggregated_logits / self.ec_temperature, dim=-1)
        return bec.masked_fill(diagonal, 0.0)

    def _signal_flow(
        self, temporal_features: torch.Tensor, bec: torch.Tensor
    ) -> torch.Tensor:
        values = self.value_projection(temporal_features)
        propagated = torch.einsum("bij,btid->btjd", bec, values)
        return self.signal_norm(propagated)

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_feed_forward(features)
        return self.output_projection(decoded.permute(0, 3, 1, 2)).squeeze(1)

    def forward(self, inputs: torch.Tensor):
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, roi]")
        if inputs.shape[1] != self.window_length:
            raise ValueError(
                f"Expected time length {self.window_length}, got {inputs.shape[1]}"
            )
        if inputs.shape[2] != self.roi_count:
            raise ValueError(f"Expected {self.roi_count} ROIs, got {inputs.shape[2]}")

        temporal_features = self._temporal_features(inputs)
        bec = self._directed_ec(temporal_features)
        flowed_features = self._signal_flow(temporal_features, bec)
        reconstruction = self._decode(flowed_features)
        return {"bec": bec, "reconstruction": reconstruction}
