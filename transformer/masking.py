"""Padding and causal masks.

Decoder self-attention (Section 3.1, 3.2.3): illegal future connections are
set to -∞ before softmax so position i can depend only on positions < i.
"""

from __future__ import annotations

import torch
from torch import Tensor

MASK_FILL = -1e9


def padding_mask(tokens: Tensor, pad_id: int = 0) -> Tensor:
    """Boolean pad mask. True = PAD key that must not be attended to.

    Args:
        tokens: (batch, seq_len) token ids.
    Returns:
        (batch, seq_len) bool.
    """
    return tokens == pad_id


def subsequent_mask(size: int, device: torch.device | None = None) -> Tensor:
    """Causal mask of shape (1, 1, size, size). True = blocked (j > i)."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1).view(
        1, 1, size, size
    )


def make_src_mask(src: Tensor, pad_id: int = 0) -> Tensor:
    """Encoder / cross-attention key mask: (batch, 1, 1, src_len)."""
    return padding_mask(src, pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: Tensor, pad_id: int = 0) -> Tensor:
    """Decoder self-attention mask: padding ∪ subsequent, (batch, 1, tgt_len, tgt_len)."""
    tgt_len = tgt.size(1)
    pad = padding_mask(tgt, pad_id).unsqueeze(1).unsqueeze(2)
    causal = subsequent_mask(tgt_len, device=tgt.device)
    return pad | causal
