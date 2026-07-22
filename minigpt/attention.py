"""
attention.py
============
Causal (masked) multi-head self-attention, implemented from scratch.

This is the heart of the Transformer. Self-attention lets every position
in a sequence look at (attend to) every other position and decide, via
learned weights, how much information to pull from each of them.

The core operation, "Scaled Dot-Product Attention" (Vaswani et al., 2017):

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Where:
    Q (queries): "what am I looking for?"      shape (..., seq_len, d_k)
    K (keys):    "what do I contain?"          shape (..., seq_len, d_k)
    V (values):  "what do I actually offer?"   shape (..., seq_len, d_k)

Q, K, and V are all linear projections of the same input for
self-attention (as opposed to cross-attention, where Q comes from one
sequence and K/V from another).

Step by step:
    1. scores = Q @ K^T                  -- how well does each query
                                             match each key? shape
                                             (seq_len, seq_len)
    2. scores /= sqrt(d_k)               -- scale down so softmax doesn't
                                             saturate for large d_k
                                             (gradients would otherwise
                                             vanish)
    3. apply causal mask                 -- for GPT-style autoregressive
                                             generation, position t must
                                             not see positions > t. We set
                                             those scores to -inf before
                                             softmax so they become 0
                                             probability.
    4. weights = softmax(scores, dim=-1) -- convert scores to a probability
                                             distribution over positions to
                                             attend to
    5. output = weights @ V              -- weighted sum of value vectors

Multi-Head Attention runs this whole process `n_head` times in parallel,
each with its own learned Q/K/V projections into a smaller subspace
(d_k = n_embd / n_head), then concatenates the results. This lets
different heads specialize in different kinds of relationships (e.g. one
head might learn "attend to the previous noun", another "attend to the
matching quotation mark").

See: https://jalammar.github.io/illustrated-transformer/ for the visual
explanation this implementation follows.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, built from individual matmuls
    so every step of scaled dot-product attention is explicit and
    inspectable (no black-box nn.MultiheadAttention).

    Attributes:
        n_head: Number of attention heads.
        n_embd: Total embedding dimensionality (split across heads).
        head_dim: Dimensionality of each individual head (n_embd / n_head).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 dropout: float = 0.1, bias: bool = True) -> None:
        """
        Args:
            n_embd: Embedding dimensionality of the residual stream.
            n_head: Number of attention heads. Must divide n_embd evenly.
            block_size: Maximum sequence length, used to size the causal
                mask buffer.
            dropout: Dropout probability applied to attention weights and
                to the output projection.
            bias: Whether the Q/K/V and output Linear layers use a bias.
        """
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # A single Linear layer that projects the input into Q, K, and V
        # simultaneously (3x n_embd output features), which is more
        # efficient than three separate Linear layers.
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output projection applied after concatenating all heads back
        # together, mixing information across heads.
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: a lower-triangular matrix of 1s. Position i is
        # allowed to attend to position j only if j <= i. Registered as a
        # non-trainable buffer so it moves with the model to GPU/CPU but
        # is not updated during backprop.
        causal_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, n_embd).

        Returns:
            Output tensor, shape (batch, seq_len, n_embd), the same shape
            as the input (attention preserves the residual stream shape).
        """
        batch, seq_len, n_embd = x.shape

        # Project to combined Q, K, V then split.
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (batch, seq_len, n_embd)

        # Reshape into (batch, n_head, seq_len, head_dim) so each head
        # attends independently. transpose(1, 2) swaps seq_len and n_head.
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # --- Scaled dot-product attention (explicit, step by step) ---
        # (batch, n_head, seq_len, head_dim) @ (batch, n_head, head_dim, seq_len)
        # -> (batch, n_head, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply the causal mask: forbidden positions get -inf so that after
        # softmax they receive ~0 probability mass.
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (batch, n_head, seq_len, seq_len) @
        # (batch, n_head, seq_len, head_dim) -> (batch, n_head, seq_len, head_dim)
        out = attn_weights @ v

        # Recombine all heads: (batch, n_head, seq_len, head_dim) ->
        # (batch, seq_len, n_head, head_dim) -> (batch, seq_len, n_embd)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)

        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
