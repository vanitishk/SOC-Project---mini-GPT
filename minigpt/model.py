"""
model.py
========
The full MiniGPT model: a decoder-only Transformer for autoregressive
next-token prediction, assembled from the components in this package.

Architecture overview:

    input token ids (batch, seq_len)
        |
        v
    GPTEmbedding            (token + positional embeddings)
        |
        v
    TransformerBlock x N    (causal self-attention + FFN, x n_layer)
        |
        v
    Final LayerNorm
        |
        v
    Linear head -> logits   (batch, seq_len, vocab_size)

At training time, logits at every position are compared against the
"next token" target via cross-entropy loss (next-token prediction). At
inference time, only the logits at the *last* position are used to sample
the next token, which is then fed back in for the next step
(autoregressive generation) -- see generate.py.

This is "decoder-only" because it consists solely of masked
self-attention blocks (no encoder, no cross-attention to a separate
source sequence) -- the same family as GPT-2/3/4, LLaMA, etc.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .embeddings import GPTEmbedding
from config import ModelConfig


class MiniGPT(nn.Module):
    """Decoder-only Transformer language model.

    Attributes:
        config: The ModelConfig this model was built from.
        embedding: Token + positional embedding layer.
        blocks: Stack of TransformerBlock modules.
        ln_f: Final LayerNorm applied before the output head.
        head: Linear layer projecting to vocabulary-sized logits.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config: A ModelConfig describing the desired architecture.
                config.vocab_size must be set (from the tokenizer) before
                constructing the model.
        """
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("config.vocab_size must be set before building the model")

        self.config = config

        self.embedding = GPTEmbedding(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_embd=config.n_embd,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                block_size=config.block_size,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the token embedding matrix with the output
        # projection. This is standard practice in GPT-2 and reduces
        # parameter count while often improving quality, since it forces
        # the input and output token representations into a shared space.
        self.embedding.token_embedding.weight = self.head.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MiniGPT] initialized with {n_params / 1e6:.2f}M parameters")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 style weight initialization: small normal init for
        Linear/Embedding layers, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: LongTensor of input token ids, shape (batch, seq_len).
            targets: Optional LongTensor of target token ids (next-token
                labels), shape (batch, seq_len). If provided, the
                cross-entropy loss is computed and returned.

        Returns:
            logits: FloatTensor, shape (batch, seq_len, vocab_size).
            loss: Scalar cross-entropy loss if `targets` was given,
                else None.
        """
        x = self.embedding(idx)  # (batch, seq_len, n_embd)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross_entropy,
            # which expects (N, C) logits and (N,) targets.
            batch, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch * seq_len, vocab_size),
                targets.view(batch * seq_len),
            )

        return logits, loss

    def num_parameters(self) -> int:
        """Total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
