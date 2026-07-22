"""
config.py
=========
Centralized configuration for MiniGPT.

Keeping all hyperparameters in one place (rather than scattered magic
numbers through the codebase) makes the project easy to read, tune, and
reproduce. Both dataclasses are saved alongside model checkpoints so a
generation run always knows exactly what architecture produced the
weights it is loading.

Two configs:
    - ModelConfig: everything that defines the network's architecture.
      Changing any of these values changes the *shape* of the model,
      so a checkpoint trained with one ModelConfig cannot be loaded
      into a model built with a different one.
    - TrainConfig: everything about *how* we train (batch size, learning
      rate, schedule, etc). These can differ between runs without
      affecting checkpoint compatibility.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the decoder-only Transformer.

    Attributes:
        vocab_size: Number of unique tokens (characters) in the vocabulary.
            Set automatically from the tokenizer after it is built.
        block_size: Maximum context length (in tokens) the model can
            attend over. Also called the "context window".
        n_embd: Dimensionality of token/positional embeddings and the
            residual stream running through the whole network.
        n_head: Number of attention heads. Must evenly divide n_embd.
        n_layer: Number of stacked Transformer blocks.
        dropout: Dropout probability applied in attention, FFN, and
            embedding layers, for regularization.
        bias: Whether Linear/LayerNorm layers use a bias term. GPT-2 style
            models use bias=True; some modern variants disable it.
    """

    vocab_size: int = 0  # filled in after tokenizer is built
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.15
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Hyperparameters controlling the training/optimization process.

    Attributes:
        batch_size: Number of sequences per optimization step.
        max_iters: Total number of training iterations (optimizer steps).
        eval_interval: How often (in iterations) to run validation and log.
        eval_iters: Number of batches to average over when estimating loss.
        learning_rate: Peak learning rate for AdamW.
        min_lr: Final learning rate at the end of cosine decay.
        warmup_iters: Number of iterations for linear LR warmup.
        weight_decay: AdamW weight decay coefficient.
        grad_clip: Max gradient norm for gradient clipping (0 disables it).
        train_val_split: Fraction of data used for training (rest is val).
        seed: Random seed for reproducibility.
        device: 'cuda', 'mps', or 'cpu'. Auto-detected if left as 'auto'.
        checkpoint_dir: Directory to save model checkpoints.
        plot_dir: Directory to save loss curve plots.
    """

    batch_size: int = 32
    max_iters: int = 8000
    eval_interval: int = 250
    eval_iters: int = 200
    learning_rate: float = 3e-4
    min_lr: float  = 3e-5
    warmup_iters: int = 400
    weight_decay: float = 0.15
    grad_clip: float = 1.0
    train_val_split: float = 0.9
    seed: int = 1337
    device: str = "auto"
    checkpoint_dir: str = "outputs/checkpoints"
    plot_dir: str = "outputs/plots"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
