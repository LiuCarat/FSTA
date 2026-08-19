"""ROI-wise temporal dynamics mixer for DTS-EC."""

from __future__ import annotations

import torch
from torch import nn


class TemporalDynamicsMixer(nn.Module):
    """Shared temporal convolutions applied independently to each ROI."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.local_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=False,
        )
        self.context_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=hidden_dim,
            bias=False,
        )
        self.channel_mixer = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
            bias=False,
        )
        self.gate = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
            groups=hidden_dim,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [batch, time, roi, hidden]")
        batch_size, time_points, roi_count, hidden_dim = inputs.shape
        roi_sequences = inputs.permute(0, 2, 3, 1).reshape(
            batch_size * roi_count, hidden_dim, time_points
        )
        local = self.local_conv(roi_sequences)
        context = self.context_conv(roi_sequences)
        update = self.activation(local + context)
        update = self.channel_mixer(update)
        gate = torch.sigmoid(self.gate(roi_sequences))
        mixed = roi_sequences + self.dropout(gate * update)
        outputs = mixed.reshape(batch_size, roi_count, hidden_dim, time_points)
        outputs = outputs.permute(0, 3, 1, 2)
        return self.norm(outputs)


__all__ = ["TemporalDynamicsMixer"]
