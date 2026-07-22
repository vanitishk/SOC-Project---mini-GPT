"""
block.py
========
A single Transformer decoder block: the repeating unit that gets stacked
`n_layer` times to build the full GPT model.

This implementation uses the **Pre-LayerNorm** architecture (as in GPT-2
and nearly all modern Transformers), rather than the original
"Attention Is All You Need" Post-LayerNorm design:

    Pre-LN (used here):
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Post-LN (original Transformer paper):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FeedForward(x))

Why Pre-LN? In Post-LN, gradients have to flow back through the LayerNorm
at every block, which becomes numerically unstable as networks get deeper
and typically requires a careful learning-rate warmup to train at all.
Pre-LN keeps an unobstructed "residual highway": the raw input x is
always added back after each sub-layer, so gradients can flow directly
from the output all the way back to the input, layer after layer. This
makes deep Transformers noticeably easier and more stable to train,
which is why virtually every modern LLM (GPT-2/3, LLaMA, etc.) uses it.

Residual connections themselves (the `x + ...` additions) exist because a
network sub-layer only needs to learn a *change* to make to its input,
not the whole transformation from scratch -- and, as above, they give
gradients a direct path back through arbitrarily many blocks.
"""

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """One Pre-LN Transformer decoder block: LN -> Attention -> residual,
    then LN -> FeedForward -> residual.

    Attributes:
        ln1: LayerNorm applied before self-attention.
        attn: The causal multi-head self-attention sub-layer.
        ln2: LayerNorm applied before the feed-forward network.
        ffn: The position-wise feed-forward sub-layer.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 dropout: float = 0.1, bias: bool = True) -> None:
        """
        Args:
            n_embd: Embedding dimensionality of the residual stream.
            n_head: Number of attention heads.
            block_size: Maximum sequence length (for the causal mask).
            dropout: Dropout probability used throughout the block.
            bias: Whether Linear/LayerNorm layers use a bias term.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)
        self.ffn = FeedForward(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, n_embd).

        Returns:
            Output tensor, same shape as input.
        """
        # Residual connection around attention: the block only has to
        # learn what to *add* to x, not replace it.
        x = x + self.attn(self.ln1(x))
        # Residual connection around the feed-forward network.
        x = x + self.ffn(self.ln2(x))
        return x
