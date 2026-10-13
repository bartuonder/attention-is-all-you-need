"""Label-smoothed NLL (Section 5.4, Szegedy et al. 2016).

True class mass is 1 - ε_ls; the remaining ε_ls is spread over the other
non-pad tokens. Pad positions contribute nothing. ε_ls = 0.1 in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            logits: (batch, seq, vocab) or (N, vocab)
            target: (batch, seq) or (N,)
        """
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if self.smoothing == 0.0:
            return F.nll_loss(log_probs, target, ignore_index=self.padding_idx, reduction="mean")

        true_dist = torch.full_like(log_probs, self.smoothing / max(self.vocab_size - 2, 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0.0
        pad = target == self.padding_idx
        true_dist[pad] = 0.0
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        n_tokens = (~pad).sum().clamp(min=1)
        return loss.sum() / n_tokens
