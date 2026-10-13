"""Full encoder-decoder Transformer (Figure 1, Sections 3.1–3.5).

Source embedding, target embedding, and the pre-softmax projection share one
weight matrix when `share_embeddings=True` (Section 3.4, citing Press & Wolf).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer.config import TransformerConfig
from transformer.embeddings import SinusoidalPositionalEncoding, TokenEmbedding
from transformer.layers import Decoder, DecoderLayer, Encoder, EncoderLayer
from transformer.masking import make_src_mask, make_tgt_mask


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.src_embed = TokenEmbedding(config.src_vocab_size, config.d_model, config.pad_id)
        if config.share_embeddings:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = TokenEmbedding(config.tgt_vocab_size, config.d_model, config.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.encoder = Encoder(
            EncoderLayer(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.d_k,  # type: ignore[arg-type]
                config.d_v,  # type: ignore[arg-type]
                config.dropout,
                config.attention_dropout,
                config.layer_norm_eps,
            ),
            config.n_layers,
        )
        self.decoder = Decoder(
            DecoderLayer(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.d_k,  # type: ignore[arg-type]
                config.d_v,  # type: ignore[arg-type]
                config.dropout,
                config.attention_dropout,
                config.layer_norm_eps,
            ),
            config.n_layers,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.src_embed.embed.weight[self.config.pad_id].zero_()
            if self.tgt_embed is not self.src_embed:
                self.tgt_embed.embed.weight[self.config.pad_id].zero_()

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        if src_mask is None:
            src_mask = make_src_mask(src, self.config.pad_id)
        return self.encoder(self.pos_enc(self.src_embed(src)), src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        if tgt_mask is None:
            tgt_mask = make_tgt_mask(tgt, self.config.pad_id)
        return self.decoder(self.pos_enc(self.tgt_embed(tgt)), memory, src_mask, tgt_mask)

    def generator(self, decoder_out: Tensor) -> Tensor:
        """Pre-softmax linear map. Tied to the target embedding matrix (Section 3.4)."""
        return F.linear(decoder_out, self.tgt_embed.embed.weight)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """Teacher-forced logits of shape (batch, tgt_len, vocab)."""
        src_mask = make_src_mask(src, self.config.pad_id)
        tgt_mask = make_tgt_mask(tgt, self.config.pad_id)
        memory = self.encode(src, src_mask)
        dec = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(dec)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
