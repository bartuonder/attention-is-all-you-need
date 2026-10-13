"""Inference: greedy decoding and beam search (Section 6.1).

Paper settings: beam size 4, length penalty α=0.6 (Wu et al. 2016 / GNMT),
max output length = input length + 50.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from transformer.masking import make_src_mask, make_tgt_mask
from transformer.model import Transformer


def length_penalty(length: int, alpha: float) -> float:
    """GNMT length penalty: ((5 + |Y|) / 6)^α."""
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def greedy_decode(model: Transformer, src: Tensor, max_len: int | None = None) -> Tensor:
    model.eval()
    cfg = model.config
    if max_len is None:
        max_len = src.size(1) + 50
    src_mask = make_src_mask(src, cfg.pad_id)
    memory = model.encode(src, src_mask)
    batch = src.size(0)
    ys = torch.full((batch, 1), cfg.bos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        out = model.decode(ys, memory, src_mask, make_tgt_mask(ys, cfg.pad_id))
        next_tok = model.generator(out[:, -1]).argmax(dim=-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, cfg.pad_id), next_tok)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == cfg.eos_id)
        if bool(finished.all()):
            break
    return ys


@torch.no_grad()
def beam_search(
    model: Transformer,
    src: Tensor,
    beam_size: int = 4,
    alpha: float = 0.6,
    max_len: int | None = None,
) -> Tensor:
    """Batched by looping examples. Returns (batch, hyp_len) token ids including BOS."""
    model.eval()
    cfg = model.config
    if max_len is None:
        max_len = src.size(1) + 50
    src_mask = make_src_mask(src, cfg.pad_id)
    memory = model.encode(src, src_mask)
    hyps = [
        _beam_one(
            model,
            memory[i : i + 1],
            src_mask[i : i + 1],
            beam_size,
            alpha,
            max_len,
        )
        for i in range(src.size(0))
    ]
    max_hyp = max(h.size(0) for h in hyps)
    out = src.new_full((src.size(0), max_hyp), cfg.pad_id)
    for i, hyp in enumerate(hyps):
        out[i, : hyp.size(0)] = hyp
    return out


def _beam_one(
    model: Transformer,
    memory: Tensor,
    src_mask: Tensor,
    beam_size: int,
    alpha: float,
    max_len: int,
) -> Tensor:
    cfg = model.config
    device = memory.device
    live = torch.full((1, 1), cfg.bos_id, dtype=torch.long, device=device)
    live_scores = torch.zeros(1, device=device)
    finished: list[tuple[float, Tensor]] = []

    for step in range(max_len - 1):
        n = live.size(0)
        mem = memory.expand(n, -1, -1)
        sm = src_mask.expand(n, -1, -1, -1)
        logits = model.generator(model.decode(live, mem, sm, make_tgt_mask(live, cfg.pad_id))[:, -1])
        log_probs = F.log_softmax(logits, dim=-1)
        cand = live_scores.unsqueeze(1) + log_probs
        vocab = cand.size(-1)

        if step == 0:
            k = min(beam_size, vocab)
            top_scores, top_idx = torch.topk(cand[0], k)
            prev = torch.zeros(k, dtype=torch.long, device=device)
            tokens = top_idx
        else:
            k = min(beam_size * 2, cand.numel())
            top_scores, top_flat = torch.topk(cand.reshape(-1), k)
            prev = top_flat // vocab
            tokens = top_flat % vocab

        new_live_seq: list[Tensor] = []
        new_live_scores: list[Tensor] = []
        for score, p, tok in zip(top_scores, prev, tokens):
            seq = torch.cat([live[p], tok.view(1)])
            if int(tok) == cfg.eos_id:
                lp = length_penalty(seq.size(0), alpha)
                finished.append((float(score) / lp, seq))
            elif len(new_live_seq) < beam_size:
                new_live_seq.append(seq)
                new_live_scores.append(score)
            if len(new_live_seq) >= beam_size and len(finished) >= beam_size:
                break

        if not new_live_seq:
            break
        live = torch.stack(new_live_seq, dim=0)
        live_scores = torch.stack(new_live_scores, dim=0)
        if len(finished) >= beam_size:
            best_fin = max(s for s, _ in finished)
            best_live = max(
                float(s) / length_penalty(live.size(1) + (max_len - 1 - step), alpha)
                for s in live_scores
            )
            if best_fin >= best_live:
                break

    if not finished:
        best = live[int(live_scores.argmax())]
        return best
    finished.sort(key=lambda x: x[0], reverse=True)
    return finished[0][1]
