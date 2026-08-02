import torch
import torch.nn as nn

from model.FourierAtt import FourierAtt
from downstream_bec.models.subject_attention import (
    PositionalEncoding,
    PositionwiseFeedForward,
    SubjectSTMultiHeadAtt,
)


class SubjectFSTA(nn.Module):
    def __init__(
        self,
        opt,
        time_num,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
    ):
        super().__init__()
        self.fourier_attention = FourierAtt(opt)
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=1)
        self.position_enc = PositionalEncoding(d_model, time_num)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.slf_attn = SubjectSTMultiHeadAtt(
            d_model,
            n_head,
            d_k,
            d_v,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model,
            d_inner,
            dropout=dropout,
        )
        self.conv2 = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, inputs, slf_attn_mask=None):
        embedded = self.conv1(inputs.unsqueeze(1))
        embedded = embedded.permute(0, 2, 3, 1)

        position = self.position_enc(embedded)
        position = position.unsqueeze(2).expand_as(embedded)
        encoded = self.layer_norm(self.dropout(embedded + position))

        spatial_features = self.fourier_attention(encoded)

        temporal_input = spatial_features.transpose(1, 2)
        temporal_output, _ = self.slf_attn(temporal_input, slf_attn_mask)
        temporal_output = self.pos_ffn(temporal_output).transpose(1, 2)

        _, spatial_attention = self.slf_attn(
            spatial_features,
            slf_attn_mask,
        )

        temporal_output = temporal_output.transpose(2, 3)
        fused = torch.einsum(
            "btdn,bnm->btdm",
            temporal_output,
            spatial_attention,
        )

        fused = fused.transpose(2, 3)
        fused = self.pos_ffn(fused + spatial_features)

        reconstruction = fused.permute(0, 3, 1, 2)
        reconstruction = self.conv2(reconstruction).squeeze(1)
        return reconstruction, spatial_attention


class SubjectFSTALoss(nn.Module):
    MODES = ("original_sum", "entropy")

    def __init__(self, mode="original_sum", alpha=0.8, eps=1e-8):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown loss mode {mode!r}; choose from {self.MODES}")
        self.mode = mode
        self.alpha = alpha
        self.eps = eps
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, reconstruction, target, subject_attention):
        prediction_loss = self.reconstruction_loss(reconstruction, target)

        if self.mode == "original_sum":
            regularizer = subject_attention.mean(dim=0).sum()
        else:
            probabilities = subject_attention.clamp_min(self.eps)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            normalizer = torch.log(
                torch.tensor(
                    probabilities.size(-1),
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
            )
            regularizer = (entropy / normalizer).mean()

        total_loss = prediction_loss + self.alpha * regularizer
        return total_loss, prediction_loss, regularizer
