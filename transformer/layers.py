"""Encoder / decoder stacks and position-wise FFN (Sections 3.1, 3.3).

Post-LN residual (Section 3.1):

    LayerNorm(x + Dropout(Sublayer(x)))

Dropout sits on the sub-layer output, before the residual add (Section 5.4).

FFN (Eq. 2): max(0, x W_1 + b_1) W_2 + b_2  with inner width d_ff=2048.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from transformer.attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))


class ResidualLayerNorm(nn.Module):
    """LayerNorm(x + Dropout(sublayer(x)))."""

    def __init__(self, d_model: int, dropout: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer_out: Tensor) -> Tensor:
        return self.norm(x + self.dropout(sublayer_out))


class EncoderLayer(nn.Module):
    """Self-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attention_dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, attention_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.residual_attn = ResidualLayerNorm(d_model, dropout, layer_norm_eps)
        self.residual_ffn = ResidualLayerNorm(d_model, dropout, layer_norm_eps)

    def forward(self, x: Tensor, src_mask: Tensor | None) -> Tensor:
        x = self.residual_attn(x, self.self_attn(x, x, x, src_mask))
        return self.residual_ffn(x, self.ffn(x))


class DecoderLayer(nn.Module):
    """Masked self-attention + encoder-decoder attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attention_dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, attention_dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, attention_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.residual_self = ResidualLayerNorm(d_model, dropout, layer_norm_eps)
        self.residual_cross = ResidualLayerNorm(d_model, dropout, layer_norm_eps)
        self.residual_ffn = ResidualLayerNorm(d_model, dropout, layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        x = self.residual_self(x, self.self_attn(x, x, x, tgt_mask))
        x = self.residual_cross(x, self.cross_attn(x, memory, memory, src_mask))
        return self.residual_ffn(x, self.ffn(x))


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_clone(layer) for _ in range(n_layers)])

    def forward(self, x: Tensor, src_mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_clone(layer) for _ in range(n_layers)])

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


def _clone(module: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(module)
