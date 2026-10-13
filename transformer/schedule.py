"""Adam learning-rate schedule (Section 5.3, Eq. 3).

    lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

Linear warmup for `warmup_steps` (4000 in the paper), then inverse-sqrt decay.
Adam: β1=0.9, β2=0.98, ε=10^{-9}.
"""

from __future__ import annotations

import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformer.config import TransformerConfig


def transformer_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def build_optimizer(model: torch.nn.Module, config: TransformerConfig) -> Optimizer:
    return Adam(
        model.parameters(),
        lr=1.0,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )


def build_scheduler(optimizer: Optimizer, config: TransformerConfig) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: transformer_lr(step, config.d_model, config.warmup_steps),
    )
