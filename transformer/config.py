"""Hyperparameters from Vaswani et al. 2017, Tables 2–3 and Section 5."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Paper-faithful Transformer configuration.

    Base model (Table 3, first row):
        N=6, d_model=512, d_ff=2048, h=8, d_k=d_v=64, P_drop=0.1, ε_ls=0.1
    """

    src_vocab_size: int = 37000
    tgt_vocab_size: int = 37000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    d_k: int | None = None
    d_v: int | None = None
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_len: int = 5000
    label_smoothing: float = 0.1
    warmup_steps: int = 4000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    share_embeddings: bool = True
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_k is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads when d_k is omitted")
            self.d_k = self.d_model // self.n_heads
        if self.d_v is None:
            self.d_v = self.d_model // self.n_heads
        if self.share_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError("shared embeddings require src_vocab_size == tgt_vocab_size")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding")

    @classmethod
    def base(cls, vocab_size: int = 37000) -> TransformerConfig:
        """Section 3 / Table 3 base model (~65M params on WMT EN-DE vocab)."""
        return cls(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size)

    @classmethod
    def big(cls, vocab_size: int = 37000) -> TransformerConfig:
        """Table 3 bottom row: d_model=1024, d_ff=4096, h=16, P_drop=0.3."""
        return cls(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=1024,
            d_ff=4096,
            n_heads=16,
            dropout=0.3,
            attention_dropout=0.3,
        )

    @classmethod
    def tiny(cls, vocab_size: int = 50) -> TransformerConfig:
        """Small model for unit tests and the reverse-sequence overfit check."""
        return cls(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.0,
            attention_dropout=0.0,
            max_len=256,
            warmup_steps=50,
            label_smoothing=0.0,
        )
