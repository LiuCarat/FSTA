"""DTS-EC: temporal dynamics and directed effective connectivity."""

from __future__ import annotations

import torch
from torch import nn

from .positional_encoding import PositionalEncoding
from .spectral_filter import SpectralFilter
from .temporal_mixer import TemporalDynamicsMixer


class DTSEC(nn.Module):
    """Estimate directed window BEC and use it for signal reconstruction.

    ``bec`` uses source-by-target orientation: ``bec[:, source, target]``.
    """

    def __init__(
        self,
        window_length: int = 78,
        roi_count: int = 90,
        hidden_dim: int = 32,
        temporal_dim: int | None = None,
        ec_dim: int = 16,
        decoder_hidden_dim: int = 0,
        ec_temperature: float = 0.25,
        dropout: float = 0.2,
    ):
        super().__init__()
        if ec_dim <= 0:
            raise ValueError("ec_dim must be positive")
        if temporal_dim is not None and temporal_dim <= 0:
            raise ValueError("temporal_dim must be positive when provided")
        if decoder_hidden_dim < 0:
            raise ValueError("decoder_hidden_dim must be non-negative")
        if ec_temperature <= 0:
            raise ValueError("ec_temperature must be positive")

        self.window_length = window_length
        self.roi_count = roi_count
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim or hidden_dim
        self.ec_dim = ec_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.ec_temperature = ec_temperature
        self.input_projection = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.position_encoding = PositionalEncoding(
            hidden_dim=hidden_dim, max_length=window_length
        )
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.spectral_filter = SpectralFilter(
            window_length=window_length,
            roi_count=roi_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal_encoder = TemporalDynamicsMixer(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal_projection = (
            nn.Identity()
            if self.temporal_dim == hidden_dim
            else nn.Linear(hidden_dim, self.temporal_dim, bias=False)
        )

        self.source_projection = nn.Linear(self.temporal_dim, ec_dim, bias=False)
        self.target_projection = nn.Linear(self.temporal_dim, ec_dim, bias=False)
        self.value_projection = nn.Linear(self.temporal_dim, ec_dim, bias=False)
        self.output_projection = self._make_decoder(decoder_hidden_dim, dropout)

    def _make_decoder(self, decoder_hidden_dim: int, dropout: float) -> nn.Module:
        if decoder_hidden_dim == 0:
            return nn.Linear(self.ec_dim, 1, bias=False)
        return nn.Sequential(
            nn.Linear(self.ec_dim, decoder_hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, 1, bias=False),
        )

    def _temporal_features(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(inputs.unsqueeze(1)).permute(0, 2, 3, 1)
        position = self.position_encoding(embedded).unsqueeze(2).expand_as(embedded)
        encoded = self.input_norm(self.input_dropout(embedded + position))
        spectral_features = self.spectral_filter(encoded)
        return self.temporal_projection(self.temporal_encoder(spectral_features))

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
        return torch.einsum("bij,btid->btjd", bec, values)

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_projection(features).squeeze(-1)

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
