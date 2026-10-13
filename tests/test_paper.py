"""Checks that pin the implementation to Vaswani et al. 2017, not just shapes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from transformer.attention import scaled_dot_product_attention
from transformer.config import TransformerConfig
from transformer.data import make_reverse_batch, token_accuracy
from transformer.decode import greedy_decode, length_penalty
from transformer.embeddings import SinusoidalPositionalEncoding
from transformer.loss import LabelSmoothingLoss
from transformer.masking import make_src_mask, make_tgt_mask, subsequent_mask
from transformer.model import Transformer
from transformer.schedule import transformer_lr


def test_positional_encoding_matches_paper_formula() -> None:
    d_model, pos, i = 64, 13, 5
    pe = SinusoidalPositionalEncoding(d_model, max_len=32, dropout=0.0)
    angle = pos / (10000 ** (2 * i / d_model))
    row = pe.pe[0, pos]
    assert torch.allclose(row[2 * i], torch.tensor(math.sin(angle)), atol=1e-5)
    assert torch.allclose(row[2 * i + 1], torch.tensor(math.cos(angle)), atol=1e-5)


def test_lr_warmup_and_decay_meet_at_warmup_steps() -> None:
    d_model, warmup = 512, 4000
    peak = transformer_lr(warmup, d_model, warmup)
    assert transformer_lr(1, d_model, warmup) < peak
    assert transformer_lr(warmup * 2, d_model, warmup) < peak
    linear = d_model**-0.5 * warmup * warmup**-1.5
    decay = d_model**-0.5 * warmup**-0.5
    assert abs(linear - decay) < 1e-12
    assert abs(peak - linear) < 1e-12


def test_causal_mask_blocks_future() -> None:
    mask = subsequent_mask(4)
    allowed = ~mask[0, 0]
    assert torch.equal(allowed, torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    ))


def test_scaled_dot_product_is_softmax_of_scaled_scores() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 8)
    out, weights = scaled_dot_product_attention(q, k, v)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
    expected = F.softmax(scores, dim=-1)
    assert torch.allclose(weights, expected, atol=1e-6)
    assert torch.allclose(out, expected @ v, atol=1e-6)


def test_padding_keys_receive_near_zero_attention() -> None:
    q = torch.randn(1, 1, 2, 8)
    k = torch.randn(1, 1, 3, 8)
    v = torch.randn(1, 1, 3, 8)
    mask = torch.tensor([[[[False, False, True]]]])
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    assert weights[0, 0, :, 2].abs().max() < 1e-6


def test_embeddings_scaled_and_tied_to_generator() -> None:
    cfg = TransformerConfig.tiny(32)
    model = Transformer(cfg)
    tokens = torch.tensor([[1, 4, 5]])
    raw = model.src_embed.embed(tokens)
    scaled = model.src_embed(tokens)
    assert torch.allclose(scaled, raw * math.sqrt(cfg.d_model))
    assert model.tgt_embed is model.src_embed
    assert model.generator(torch.randn(1, 3, cfg.d_model)).shape == (1, 3, cfg.tgt_vocab_size)


def test_decoder_is_causal_no_future_leak() -> None:
    torch.manual_seed(1)
    cfg = TransformerConfig.tiny(40)
    model = Transformer(cfg)
    model.eval()
    src = torch.tensor([[4, 5, 6, 7, 0]])
    tgt_a = torch.tensor([[1, 8, 9, 10]])
    tgt_b = tgt_a.clone()
    tgt_b[0, -1] = 11
    with torch.no_grad():
        logits_a = model(src, tgt_a)
        logits_b = model(src, tgt_b)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)


def test_encoder_is_bidirectional() -> None:
    torch.manual_seed(2)
    cfg = TransformerConfig.tiny(40)
    model = Transformer(cfg)
    model.eval()
    src_a = torch.tensor([[4, 5, 6, 7]])
    src_b = src_a.clone()
    src_b[0, -1] = 8
    with torch.no_grad():
        mem_a = model.encode(src_a)
        mem_b = model.encode(src_b)
    assert not torch.allclose(mem_a[:, 0], mem_b[:, 0], atol=1e-5)


def test_label_smoothing_mass() -> None:
    vocab, pad, eps = 10, 0, 0.1
    loss_fn = LabelSmoothingLoss(vocab, pad, eps)
    logits = torch.zeros(1, vocab)
    logits[0, 3] = 10.0
    target = torch.tensor([3])
    loss = loss_fn(logits, target)
    assert loss.item() > 0.0
    pad_loss = loss_fn(logits, torch.tensor([pad]))
    assert pad_loss.item() == 0.0


def test_base_param_count_near_paper() -> None:
    """Table 3 lists 65M. Shared 37k embeddings + 6+6 layers land ~63M."""
    model = Transformer(TransformerConfig.base(37000))
    n = model.param_count()
    assert 60_000_000 < n < 70_000_000


def test_length_penalty_gnmt() -> None:
    assert abs(length_penalty(1, 0.6) - 1.0) < 1e-6
    assert length_penalty(10, 0.6) > 1.0


def test_tiny_overfit_reverse_task() -> None:
    torch.manual_seed(0)
    cfg = TransformerConfig.tiny(32)
    model = Transformer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = LabelSmoothingLoss(cfg.tgt_vocab_size, cfg.pad_id, smoothing=0.0)
    batch = make_reverse_batch(16, cfg.src_vocab_size, 4, 8, cfg, generator=torch.Generator().manual_seed(0))
    model.train()
    acc = 0.0
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        logits = model(batch.src, batch.tgt_in)
        loss = criterion(logits, batch.tgt_out)
        loss.backward()
        opt.step()
        acc = token_accuracy(logits, batch.tgt_out, cfg.pad_id)
    assert acc > 0.95

    model.eval()
    pred = greedy_decode(model, batch.src[:4], max_len=12)
    gold = torch.cat([torch.full((4, 1), cfg.bos_id), batch.tgt_out[:4]], dim=1)
    matches = 0
    for i in range(4):
        p = [t for t in pred[i].tolist() if t != cfg.pad_id]
        g = [t for t in gold[i].tolist() if t != cfg.pad_id]
        if p[: len(g)] == g or (p and p[-1] == cfg.eos_id and p[:-1] == g[: len(p) - 1]):
            matches += 1
        else:
            eos_p = p.index(cfg.eos_id) + 1 if cfg.eos_id in p else len(p)
            eos_g = g.index(cfg.eos_id) + 1 if cfg.eos_id in g else len(g)
            if p[:eos_p] == g[:eos_g]:
                matches += 1
    assert matches >= 3
