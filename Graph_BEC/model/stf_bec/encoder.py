"""STF-BEC encoder for subject-specific directed BEC estimation."""
import torch
import torch.nn as nn
from .temporal_attention import PositionalEncoding, STMultiHeadAtt
from .feed_forward import PositionwiseFeedForward
from .spectral_block import FourierAtt


class STFEncoder(nn.Module):
    def __init__(self, opt, time_num, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.FA = FourierAtt(opt)
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=1)
        self.position_enc = PositionalEncoding(d_hid=d_model, n_position=time_num)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.slf_attn = STMultiHeadAtt(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.conv2 = nn.Conv2d(d_model, 1, 1)
        self.norm_adj = nn.InstanceNorm2d(1)
        self.complex_weight = nn.Parameter(
            torch.randn(1, time_num // 2 + 1, opt.nodes_num, d_model, 2) * 0.02
        )

    def _encode(self, inputs, slf_attn_mask=None):
        """Compute the shared STF-BEC representation before spatial fusion."""
        embedded = inputs.unsqueeze(0).permute(1, 0, 2, 3)
        embedded = self.conv1(embedded).permute(0, 2, 3, 1)
        position = self.position_enc(embedded).unsqueeze(2).expand(embedded.shape)
        encoded = self.layer_norm(self.dropout(embedded + position))
        fourier_output = self.FA(encoded)
        temporal_input = fourier_output.transpose(1, 2)
        temporal_output, _ = self.slf_attn(temporal_input, slf_attn_mask)
        temporal_output = self.pos_ffn(temporal_output).transpose(1, 2)
        return fourier_output, temporal_output

    def _decode_with_spatial_attention(self, spatial_features, temporal_features, spatial_attention):
        temporal_features = temporal_features.transpose(2, 3)
        fused = torch.matmul(temporal_features, spatial_attention)
        fused = fused.transpose(2, 3)
        fused = self.pos_ffn(fused + spatial_features)
        output = self.conv2(fused.permute(0, 3, 1, 2)).squeeze(1)
        return output

    def forward(self, inputs, slf_attn_mask=None):
        spatial_features, temporal_features = self._encode(inputs, slf_attn_mask)
        _, spatial_attention = self.slf_attn(spatial_features, slf_attn_mask)
        reconstruction = self._decode_with_spatial_attention(
            spatial_features, temporal_features, spatial_attention
        )
        return reconstruction, spatial_attention

