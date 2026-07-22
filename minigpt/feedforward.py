"""
feedforward.py
==============
The position-wise Feed-Forward Network (FFN) used inside each Transformer
block.

While attention lets tokens exchange information with each other, the FFN
processes each position *independently* (hence "position-wise"): the same
two-layer MLP is applied to every token's vector separately. Conceptually,
attention is where tokens "communicate", and the FFN is where each token
"thinks" about what it gathered.

Architecture (following GPT-2):
    Linear(n_embd -> 4 * n_embd) -> GELU -> Linear(4 * n_embd -> n_embd) -> Dropout

The 4x expansion in the hidden layer is standard practice: it gives the
network more capacity to compute a richer intermediate representation
before projecting back down to the residual stream's dimensionality.
GELU (Gaussian Error Linear Unit) is used instead of ReLU because it is
smoother and empirically works better for Transformers.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation, applied independently to each
    position in the sequence.

    Attributes:
        net: The sequential MLP (expand -> activate -> project -> dropout).
    """

    def __init__(self, n_embd: int, dropout: float = 0.1, bias: bool = True) -> None:
        """
        Args:
            n_embd: Input/output embedding dimensionality.
            dropout: Dropout probability applied after the final projection.
            bias: Whether the Linear layers use a bias term.
        """
        super().__init__()
        hidden_dim = 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, n_embd).

        Returns:
            Output tensor, same shape as input.
        """
        return self.net(x)
