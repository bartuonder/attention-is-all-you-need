"""Synthetic reverse-sequence transduction.

The architecture is sequence-to-sequence; reversing a token stream forces
encoder-decoder attention to learn a non-trivial alignment without WMT/BPE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from transformer.config import TransformerConfig


@dataclass
class Batch:
    src: Tensor
    tgt_in: Tensor
    tgt_out: Tensor


def make_reverse_batch(
    batch_size: int,
    vocab_size: int,
    min_len: int,
    max_len: int,
    config: TransformerConfig,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> Batch:
    first_token = max(config.eos_id, config.bos_id, config.pad_id) + 1
    if vocab_size <= first_token:
        raise ValueError("vocab_size is too small for special tokens")
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), generator=generator)
    src_len = int(lengths.max())
    src = torch.full((batch_size, src_len), config.pad_id, dtype=torch.long)
    tgt = torch.full((batch_size, src_len + 2), config.pad_id, dtype=torch.long)
    for i, length in enumerate(lengths.tolist()):
        tokens = torch.randint(first_token, vocab_size, (length,), generator=generator)
        src[i, :length] = tokens
        tgt[i, 0] = config.bos_id
        tgt[i, 1 : length + 1] = tokens.flip(0)
        tgt[i, length + 1] = config.eos_id
    if device is not None:
        src = src.to(device)
        tgt = tgt.to(device)
    return Batch(src=src, tgt_in=tgt[:, :-1], tgt_out=tgt[:, 1:])


def token_accuracy(logits: Tensor, target: Tensor, pad_id: int) -> float:
    pred = logits.argmax(dim=-1)
    mask = target != pad_id
    if int(mask.sum()) == 0:
        return 0.0
    return float((pred[mask] == target[mask]).float().mean())
