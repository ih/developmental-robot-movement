import torch
import torch.nn as nn


class EncoderLayerWithAttn(nn.Module):
    """Transformer encoder layer that optionally returns attention maps."""

    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward * d_model),
            nn.GELU(),
            nn.Linear(dim_feedforward * d_model, d_model),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None, record_attn=False):
        attn_out, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=record_attn,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        y = self.ffn(x)
        x = self.norm2(x + self.dropout(y))
        return (x, attn) if record_attn else (x, None)
