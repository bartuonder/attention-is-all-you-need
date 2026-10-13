"""Scaled Dot-Product Attention and Multi-Head Attention (Section 3.2).

    Attention(Q, K, V) = softmax(Q K^T / √d_k) V                         (Eq. 1)

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Projections are bias-free: the paper parameterizes them as matrices W, not
affine maps. Attention dropout after softmax is the "attention dropout"
mentioned in Section 6.3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer.masking import MASK_FILL


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """Eq. 1. query/key/value: (batch, heads, seq, dim). mask broadcasts to scores."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, MASK_FILL)
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    """Section 3.2.2. Default h=8, d_k=d_v=d_model/h=64."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.w_o = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Tensor | None = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch = query.size(0)
        q = self._split_heads(self.w_q(query), self.d_k, batch)
        k = self._split_heads(self.w_k(key), self.d_k, batch)
        v = self._split_heads(self.w_v(value), self.d_v, batch)
        context, self.attn_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        concat = context.transpose(1, 2).contiguous().view(batch, -1, self.n_heads * self.d_v)
        return self.w_o(concat)

    def _split_heads(self, x: Tensor, head_dim: int, batch: int) -> Tensor:
        return x.view(batch, -1, self.n_heads, head_dim).transpose(1, 2)
