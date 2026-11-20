import torch.nn as nn
from .attention_modules import ScaledDotProductAttention


class SpatialMultiHeadAttention(nn.Module):
    

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):

        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len1_q, len1_k, len1_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len2_q, len2_v, len2_k = q.size(2), k.size(2), v.size(2)
        residual = q


        q = self.w_qs(q).view(sz_b, len1_q, len2_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len1_k, len2_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len1_v, len2_v, n_head, d_v)

        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)



        q = q.transpose(2, 3).contiguous().view(sz_b, len1_q, len2_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
