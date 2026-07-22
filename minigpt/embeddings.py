"""
embeddings.py
=============
Token and positional embeddings.

A Transformer has no inherent notion of sequence order -- self-attention
treats its input as an unordered set of vectors unless we inject position
information explicitly. GPT-style models do this with a *learned*
positional embedding table (as opposed to the fixed sinusoidal encoding
from the original "Attention Is All You Need" paper): a separate
embedding vector for each position 0..block_size-1, added elementwise to
the token embedding.

    final_embedding[t] = token_embedding[token_id[t]] + position_embedding[t]

This sum becomes the initial "residual stream" that flows through every
Transformer block in the network.
"""

import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    """Combines token embeddings with learned positional embeddings.

    Attributes:
        token_embedding: Lookup table mapping token id -> embedding vector.
        position_embedding: Lookup table mapping position index -> embedding
            vector, learned jointly with the rest of the model.
        dropout: Dropout applied to the summed embeddings (as in GPT-2).
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int,
                 dropout: float = 0.1) -> None:
        """
        Args:
            vocab_size: Number of unique tokens in the vocabulary.
            block_size: Maximum sequence length the model supports.
            n_embd: Embedding dimensionality.
            dropout: Dropout probability applied after summing embeddings.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: LongTensor of token ids, shape (batch, seq_len).

        Returns:
            Embedded input, shape (batch, seq_len, n_embd).
        """
        batch, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.block_size}"
            )

        tok_emb = self.token_embedding(idx)  # (batch, seq_len, n_embd)

        positions = torch.arange(seq_len, device=idx.device)  # (seq_len,)
        pos_emb = self.position_embedding(positions)  # (seq_len, n_embd)

        # pos_emb broadcasts across the batch dimension.
        x = tok_emb + pos_emb
        return self.dropout(x)
