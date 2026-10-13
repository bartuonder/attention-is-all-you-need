"""Train the paper Transformer on a reverse-sequence toy task.

WMT 2014 + BPE is the original experimental setup (Section 5.1). This script
verifies that the architecture actually learns an alignment, which is the
prerequisite before scaling to machine translation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam

from transformer.config import TransformerConfig
from transformer.data import Batch, make_reverse_batch, token_accuracy
from transformer.decode import greedy_decode
from transformer.loss import LabelSmoothingLoss
from transformer.model import Transformer
from transformer.schedule import build_optimizer, build_scheduler


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(name: str, vocab_size: int) -> TransformerConfig:
    factories = {
        "tiny": TransformerConfig.tiny,
        "base": TransformerConfig.base,
        "big": TransformerConfig.big,
    }
    if name not in factories:
        raise ValueError(f"unknown config {name!r}")
    return factories[name](vocab_size)


def strip_special(tokens: list[int], cfg: TransformerConfig) -> list[int]:
    out: list[int] = []
    for tok in tokens:
        if tok == cfg.eos_id:
            break
        if tok not in (cfg.pad_id, cfg.bos_id):
            out.append(tok)
    return out


def subsample(data: Batch, idx: torch.Tensor) -> Batch:
    return Batch(src=data.src[idx], tgt_in=data.tgt_in[idx], tgt_out=data.tgt_out[idx])


def decode_report(model: Transformer, data: Batch, cfg: TransformerConfig, n: int, title: str) -> None:
    pred = greedy_decode(model, data.src[:n], max_len=data.tgt_out.size(1) + 2)
    print(f"\n{title} (src → gold reverse / pred):")
    hits = 0
    for i in range(n):
        src = strip_special(data.src[i].tolist(), cfg)
        gold = list(reversed(src))
        hyp = strip_special(pred[i].tolist(), cfg)
        ok = hyp == gold
        hits += int(ok)
        print(f"  {src}  →  {gold}  /  {hyp}  {'ok' if ok else 'miss'}")
    print(f"  exact match {hits}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("tiny", "base", "big"), default="tiny")
    parser.add_argument("--vocab-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3, help="used unless --paper-schedule")
    parser.add_argument(
        "--paper-schedule",
        action="store_true",
        help="use Eq. 3 warmup/decay instead of a constant Adam lr",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/reverse.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    cfg = build_config(args.config, args.vocab_size)
    model = Transformer(cfg).to(device)
    criterion = LabelSmoothingLoss(cfg.tgt_vocab_size, cfg.pad_id, cfg.label_smoothing)

    if args.paper_schedule:
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
    else:
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
        scheduler = None

    train = make_reverse_batch(
        args.n_train, cfg.src_vocab_size, args.min_len, args.max_len, cfg, device,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    heldout = make_reverse_batch(
        32, cfg.src_vocab_size, args.min_len, args.max_len, cfg, device,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 1),
    )

    print(
        f"device={device}  config={args.config}  params={model.param_count():,}  "
        f"n_train={args.n_train}  schedule={'paper Eq.3' if scheduler else f'adam {args.lr}'}"
    )
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    log_n = 0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, args.n_train, (args.batch_size,), device=device)
        batch = subsample(train, idx)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.src, batch.tgt_in)
        loss = criterion(logits, batch.tgt_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        running_acc += token_accuracy(logits, batch.tgt_out, cfg.pad_id)
        log_n += 1
        if step % args.log_every == 0 or step == args.steps:
            lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
            print(
                f"step {step:5d}/{args.steps}  "
                f"loss={running_loss / log_n:.4f}  "
                f"acc={running_acc / log_n:.3f}  "
                f"lr={lr:.6e}"
            )
            running_loss = 0.0
            running_acc = 0.0
            log_n = 0

    model.eval()
    decode_report(model, train, cfg, n=4, title="train samples")
    decode_report(model, heldout, cfg, n=4, title="held-out samples")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, args.checkpoint)
    print(f"\nsaved {args.checkpoint}")


if __name__ == "__main__":
    main()
