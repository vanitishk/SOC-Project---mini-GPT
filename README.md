# MiniGPT — A GPT-Style Transformer Built From Scratch

An educational, from-scratch implementation of a **decoder-only Transformer**
(the architecture behind GPT-2/3/4, LLaMA, and similar models) in pure
PyTorch. No `transformers` library, no `nn.Transformer`, no pretrained
weights — every component (tokenizer, embeddings, masked self-attention,
feed-forward network, Transformer blocks) is implemented and documented
by hand so the internals are fully visible and inspectable.

This project follows the standard "GPT from scratch" approach popularized
by Andrej Karpathy's tutorial and explained visually in
[Jay Alammar's *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).
 With a small dataset
and a small architecture, expect it to learn the *style* and *vocabulary*
of its training text — character names, short exchanges, sentence
rhythm — not long-range coherent reasoning.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Dataset](#dataset)
3. [Architecture Overview](#architecture-overview)
4. [How It Works, Step by Step](#how-it-works-step-by-step)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Configuration](#configuration)
9. [Sample Output](#sample-output)
10. [Further Reading](#further-reading)

---

## What This Project Does

MiniGPT is trained to do one thing: **predict the next character**, given
all the characters that came before it. That's it. Do this well enough,
over and over, and you can generate entire passages of text one character
at a time — this is exactly how GPT-style models work, just at a much
larger scale and using subword tokens instead of characters.

The pipeline:

```
raw text  →  tokenizer  →  training (next-token prediction)  →  trained model  →  text generation
```

---

## Dataset

The training corpus is built from the **Cornell Movie-Dialogs Corpus**
(Danescu-Niculescu-Mizil & Lee, 2011), a public research dataset of
movie dialogue released by Cornell University. Rather than scraping full
copyrighted screenplays, `data/prepare_dataset.py` downloads this
existing corpus and filters it down to a curated set of **16 sci-fi /
space-adventure films**, centered on the **Star Wars original trilogy**:

- Star Wars: A New Hope, The Empire Strikes Back, Return of the Jedi
- Star Trek V, Galaxy Quest, Tron, Superman
- The Terminator, Terminator 2, X-Men
- Blade Runner, Back to the Future, The Matrix
- Jurassic Park, Aliens, Alien

Conversations are reconstructed **in their original turn-by-turn order**
(not just concatenated random lines) and formatted screenplay-style:

```
HUDSON: This floor's freezing.
APONE: Christ.  I never saw such a buncha old women.  You want me to fetch your slippers, Hudson?
HUDSON: Would you, Sir?
```

This yields roughly **400,000 characters** (~85 unique characters in the
vocabulary) — large enough for a small Transformer to learn real
character-level patterns, small enough to train in well under half an
hour on a laptop CPU.

To rebuild the dataset from scratch:
```bash
python data/prepare_dataset.py
```

---

## Architecture Overview

```
Input text: "LUKE: Help me"
     │
     ▼
┌─────────────────┐
│  Tokenizer       │  each character → integer id
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Token Embedding │  id → learned vector
│  + Positional    │  + learned "position" vector
│    Embedding     │
└─────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Transformer Block  × N              │
│  ┌─────────────────────────────┐    │
│  │ LayerNorm                    │    │
│  │ Causal Multi-Head Attention  │    │
│  │ + residual connection        │    │
│  ├─────────────────────────────┤    │
│  │ LayerNorm                    │    │
│  │ Feed-Forward Network (GELU)  │    │
│  │ + residual connection        │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│  Final LayerNorm │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Linear Head     │  vector → probability over vocabulary
└─────────────────┘
     │
     ▼
Next-character prediction
```

This is called **decoder-only** because it consists purely of masked
self-attention blocks operating on one sequence — there's no separate
encoder and no cross-attention to a source sequence, unlike the original
2017 "Attention Is All You Need" translation architecture. GPT models
dropped the encoder entirely and just stacked decoder blocks, which turns
out to be sufficient (and simpler) for general-purpose language modeling.

---

## How It Works, Step by Step

### 1. Tokenization (`minigpt/tokenizer.py`)

We use **character-level** tokenization: every unique character in the
training corpus (letters, punctuation, spaces, newlines) gets its own
integer id. This is simpler than the subword tokenizers (like
Byte-Pair Encoding) that real GPT models use, at the cost of longer
sequences and a harder learning problem (the model has to learn spelling
before it can learn words). It's the right tradeoff for a from-scratch
educational project — no separate tokenizer training step needed.

### 2. Embeddings (`minigpt/embeddings.py`)

Each token id is mapped to a learned vector (the **token embedding**).
Since self-attention has no inherent sense of sequence order, we also add
a **positional embedding** — a separate learned vector for each position
0, 1, 2, ... — so the model can tell "the first character" from "the
tenth character":

```
embedding[t] = token_embedding[token_id[t]] + position_embedding[t]
```

### 3. Causal Multi-Head Self-Attention (`minigpt/attention.py`)

This is the core mechanism. For every position in the sequence, attention
lets the model look back at *all previous positions* and decide, via
learned weights, how much to "pay attention to" each one.

Each token produces three vectors via learned linear projections:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain?"
- **Value (V)** — "what do I actually offer if attended to?"

The attention formula:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Step by step:
1. `Q @ Kᵀ` — dot product between every query and every key, giving a
   similarity score for every (query position, key position) pair.
2. Divide by `√d_k` — scales the scores down so the softmax doesn't
   saturate and produce vanishing gradients.
3. **Causal masking** — since this is a *decoder* generating text
   left-to-right, position *t* must never see positions after *t*
   (that would be cheating — looking at the answer). We set all
   "future" scores to `-∞` before the softmax, so they get 0 probability.
   This is implemented with a lower-triangular mask (`torch.tril`).
4. `softmax(...)` — converts scores into a probability distribution
   over which positions to attend to.
5. `@ V` — take a weighted sum of value vectors according to those
   probabilities. This is the attention output.

**Multi-head**: instead of doing this once, we split the embedding into
`n_head` smaller chunks and run the whole process independently on each,
then concatenate the results. This lets different heads specialize —
e.g. one head might learn to track "who is currently speaking", another
might track punctuation or nearby characters.

### 4. Feed-Forward Network (`minigpt/feedforward.py`)

After attention lets tokens exchange information, a simple two-layer MLP
(`Linear → GELU → Linear`) processes each position independently,
expanding to 4× the embedding dimension internally before projecting
back down. If attention is where tokens "communicate", the FFN is where
each token "thinks" about what it just gathered.

### 5. Residual Connections & LayerNorm (`minigpt/block.py`)

Each sub-layer (attention, FFN) is wrapped as:

```
x = x + SubLayer(LayerNorm(x))
```

This is the **Pre-LayerNorm** design used by GPT-2 and virtually all
modern Transformers. The `x + ...` residual connection means each layer
only has to learn what to *add* or *change*, not reconstruct its input
from scratch — and critically, it gives gradients a direct, unobstructed
path back through arbitrarily many layers during backpropagation, which
is what makes deep Transformers trainable at all.

### 6. Training: Next-Token Prediction (`minigpt/train.py`)

Given a chunk of `block_size` tokens, the model predicts, *at every
position simultaneously*, what the next token should be:

```
input:   [L, U, K, E, :]
target:  [U, K, E, :, ' ']   (input shifted one position to the right)
```

Loss is **cross-entropy** between predicted and actual next-token
distributions, averaged over every position in every sequence in the
batch. Training uses AdamW with a linear warmup + cosine decay learning
rate schedule (standard GPT-2/3 practice) and gradient clipping for
stability.

### 7. Autoregressive Generation (`minigpt/generate.py`)

To generate text, we run the trained model in a loop:

1. Feed in the current sequence.
2. Look at the logits (predictions) for the *last* position only.
3. Convert to a sampling distribution using **temperature** and
   **top-k** filtering (below).
4. Sample the next token.
5. Append it and repeat.

**Temperature** scales the logits before sampling
(`logits / temperature`): low temperature (e.g. 0.5) makes the model
more confident and repetitive; high temperature (e.g. 1.3) makes it more
random and diverse (sometimes incoherent).

**Top-k sampling** restricts sampling to only the *k* most likely tokens
at each step, masking out the long tail of implausible characters — this
avoids occasional very-low-probability tokens derailing generation while
still allowing controlled randomness.

---

## Project Structure

```
minigpt/
├── README.md                  ← you are here
├── requirements.txt
├── config.py                  # ModelConfig + TrainConfig dataclasses
├── data/
│   ├── prepare_dataset.py     # builds input.txt from the Cornell corpus
│   └── input.txt              # the training corpus (~400KB)
├── minigpt/
│   ├── tokenizer.py           # character-level tokenizer
│   ├── dataset.py             # (context, target) sequence chunking
│   ├── embeddings.py          # token + positional embeddings
│   ├── attention.py           # causal multi-head self-attention (from scratch)
│   ├── feedforward.py         # position-wise MLP
│   ├── block.py               # Pre-LN Transformer block
│   ├── model.py                # full GPT model assembly
│   ├── generate.py            # autoregressive sampling (temperature + top-k)
│   ├── train.py                # training/validation loop, LR schedule
│   ├── checkpoint.py           # save/load model + optimizer + config
│   └── utils.py                 # seeding, device detection, loss plotting
├── scripts/
│   ├── train_model.py          # CLI: train a model
│   └── generate_text.py        # CLI: generate text from a checkpoint
└── outputs/
    ├── checkpoints/            # saved model weights + tokenizer vocab
    └── plots/                  # loss curve PNGs
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. Runs on CPU, CUDA, or Apple
Silicon (MPS) — device is auto-detected.

---

## Usage

### 1. Build the dataset (only needed once)

```bash
python data/prepare_dataset.py
```

### 2. Train

```bash
python scripts/train_model.py
```

With default settings (a ~4.83M parameter model: 6 layers, 4 heads,
256-dim embeddings, 128-token context window), training for 8000
iterations takes roughly **30-40 minutes on a laptop CPU**. Checkpoints
are saved to `outputs/checkpoints/` every `eval_interval` iterations
(both the best-so-far model and the latest), and a loss curve is saved
to `outputs/plots/` at the end.

Useful flags:
```bash
python scripts/train_model.py --max_iters 1000          # quicker run
python scripts/train_model.py --n_layer 6 --n_embd 256  # bigger model (slower)
python scripts/train_model.py --device cuda              # force GPU
```

Run `python scripts/train_model.py --help` for the full list of options.

### 3. Generate text

```bash
python scripts/generate_text.py --prompt "LUKE:" --max_new_tokens 300
```

```bash
python scripts/generate_text.py \
  --prompt "HAN:" \
  --temperature 0.7 \
  --top_k 30 \
  --max_new_tokens 200
```

---

## Configuration

All hyperparameters live in `config.py` as two dataclasses:

**`ModelConfig`** (architecture — changes model shape, affects checkpoint compatibility):

| Field | Default | Meaning |
|---|---|---|
| `block_size` | 256 | Max context length (tokens) |
| `n_embd` | 256 | Embedding / residual stream dimensionality |
| `n_head` | 8 | Number of attention heads |
| `n_layer` | 6 | Number of stacked Transformer blocks |
| `dropout` | 0.15 | Dropout probability |

**`TrainConfig`** (training process):

| Field | Default | Meaning |
|---|---|---|
| `batch_size` | 32 | Sequences per training step |
| `max_iters` | 8000 | Total training iterations |
| `learning_rate` | 3e-4 | Peak learning rate |
| `warmup_iters` | 400 | Linear warmup steps |
| `eval_interval` | 200 | Iterations between validation checks |

These defaults were empirically tuned to train comfortably on a
CPU-only machine in well under half an hour. Increase model size
(`n_embd`, `n_layer`) and `max_iters` for better quality if you have a
GPU available.

---

## Sample Output

*(Generated with the trained checkpoint included in this repo — see
`outputs/checkpoints/`. Output quality depends heavily on model size and
training time; this is a small model trained briefly, so expect
dialogue-flavored, sometimes-nonsensical text rather than coherent
scenes.)*

```
$ python scripts/generate_text.py --prompt "LUKE:" --max_new_tokens 300 --temperature 0.8 --top_k 40

LUKE: A can to to get to ake...  think come ant him and to bet now going
you like where though read on me sting everat the mester not cark it's
a do want you wan.  We for as and in of the this you want for me are is
this is you compen mand you hour you thong forlby here here salter,
here good you real be

$ python scripts/generate_text.py --prompt "VADER:" --max_new_tokens 300 --temperature 0.7 --top_k 30

VADER: I my ou're go leng to the forl, the the the farr light wher as
the are, the all way're shey this have oner to what the ging the und
it sif the fist.

LEIA: I do his shat a the posty on of me ut on what we thos lack?
Have est that's that with te the good looke now ware und the out
finere.
STANDARD:
```

This is a small model (~0.8M parameters, 4 layers) trained for only
2,500 iterations, achieving a validation loss of **2.02** (down from
4.40 at initialization — roughly the loss of guessing uniformly at
random over 85 characters). The results show exactly what's expected at
this scale: it has correctly learned character names and the
`NAME: dialogue` screenplay format, produces mostly real short English
words and plausible letter sequences, and picks up on the dialogue-like
rhythm and punctuation of the training data — without full sentence-level
coherence, which would require a much larger model and far more training.



---

## Further Reading

- Vaswani et al., 2017 — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (the original Transformer paper)
- Jay Alammar — [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) (visual, intuitive explanation)
- Andrej Karpathy — ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY) (the video this project's approach is modeled after)
- 3Blue1Brown — [Neural Networks / Attention series](https://www.3blue1brown.com/topics/neural-networks) (visual/mathematical intuition)
- Danescu-Niculescu-Mizil & Lee, 2011 — [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) (the dataset used here)
