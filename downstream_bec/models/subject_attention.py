import torch.nn as nn
import torch.nn.functional as F

from model.MultiHeadAtt import MultiHeadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position):
        super().__init__()
        import numpy as np
        import torch

        positions = np.arange(n_position)[:, None]
        dimensions = np.arange(d_hid)[None, :]
        angles = positions / np.power(10000, 2 * (dimensions // 2) / d_hid)
        table = np.zeros((n_position, d_hid), dtype=np.float32)
        table[:, 0::2] = np.sin(angles[:, 0::2])
        table[:, 1::2] = np.cos(angles[:, 1::2])
        self.register_buffer("pos_table", torch.from_numpy(table).unsqueeze(0))

    def forward(self, inputs):
        return self.pos_table[:, : inputs.size(1)].clone().detach()


class SubjectSTMultiHeadAtt(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
        )

    def forward(self, enc_input, slf_attn_mask=None):
        output, attention = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            mask=slf_attn_mask,
        )
        subject_attention = attention.mean(dim=(1, 2))
        return output, subject_attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        output = self.w_2(F.relu(self.w_1(inputs)))
        output = self.dropout(output)
        return self.layer_norm(output + residual)
