"""Token embeddings and sinusoidal positional encoding (Sections 3.4–3.5).

Embeddings are multiplied by √d_model (Section 3.4). Positional encodings:

    PE(pos, 2i)     = sin(pos / 10000^{2i / d_model})
    PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model})

Dropout is applied to embedding + PE (Section 5.4 Residual Dropout).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embed(tokens) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoids; registered as a buffer so they move with the module device."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
