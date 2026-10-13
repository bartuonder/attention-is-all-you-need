# Attention Is All You Need (2017) — paper to code

Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin.
**Attention Is All You Need**, NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

This repo is a from-scratch PyTorch implementation of the original Transformer,
written to match the paper rather than `nn.Transformer` or later variants
(no Pre-LN, no GELU, no RoPE, no extra bias on attention projections).

## Architecture map

| Paper | Code |
| --- | --- |
| Eq. 1 scaled dot-product attention | `transformer/attention.py` |
| §3.2.2 multi-head attention | `MultiHeadAttention` |
| Eq. 2 position-wise FFN (ReLU) | `transformer/layers.py` |
| §3.1 Post-LN residual `LayerNorm(x + Sublayer(x))` | `ResidualLayerNorm` |
| §3.4 embeddings × √d_model, tied softmax | `TokenEmbedding`, `Transformer.generator` |
| §3.5 sinusoidal PE | `SinusoidalPositionalEncoding` |
| §3.1 / §3.2.3 causal + padding masks | `transformer/masking.py` |
| Eq. 3 Adam warmup schedule | `transformer/schedule.py` |
| §5.4 residual dropout, ε_ls = 0.1 | `config.dropout`, `LabelSmoothingLoss` |
| §6.1 beam 4, length penalty α = 0.6 | `transformer/decode.py` |

Base hyperparameters (Table 3):

```
N=6  d_model=512  d_ff=2048  h=8  d_k=d_v=64  P_drop=0.1  ε_ls=0.1
Adam β1=0.9  β2=0.98  ε=1e-9  warmup=4000
```

## What is faithful, what is not

Implemented exactly as specified: Post-LN, sinusoidal PE, shared embedding /
pre-softmax weights, scaled dot-product + multi-head, ReLU FFN, residual
dropout, label smoothing, the inverse-sqrt LR schedule, greedy and GNMT beam
search.

Not reproduced here (data / systems, not the model): WMT 2014, BPE vocab,
8×P100 training, checkpoint averaging. Those are experimental protocol, not
architecture. The training entry point uses a reverse-sequence task so you can
confirm the model learns an encoder–decoder alignment on a laptop.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# architecture tests, including a tiny reverse-task overfit
python -m pytest tests/ -q

# train tiny Transformer on reverse sequences (CPU/MPS/CUDA)
python train.py --config tiny

# paper Eq. 3 warmup schedule (designed for base/big, not the toy)
python train.py --config tiny --paper-schedule --steps 4000
```

Parameter counts with a 37k shared vocabulary (Table 3: base 65M, big 213M):

```
tiny   169,088
base    63,045,632
big    214,171,648
```

## Shapes (batch B, source S, target T)

```
src ids          (B, S)
tgt ids          (B, T)
embeddings       (B, ·, d_model)
Q,K,V per head   (B, h, ·, d_k or d_v)
encoder memory   (B, S, d_model)
logits           (B, T, vocab)
```
