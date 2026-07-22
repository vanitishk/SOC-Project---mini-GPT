"""
dataset.py
==========
Turns a long stream of tokenized text into (context, target) training
examples for next-token prediction.

The core idea behind training a GPT-style model is simple: given a chunk
of `block_size` consecutive tokens, predict the token that comes next at
*every position* in the chunk simultaneously. For a chunk
    [x0, x1, x2, x3]
the model is trained to predict:
    x1 given [x0]
    x2 given [x0, x1]
    x3 given [x0, x1, x2]
    x4 given [x0, x1, x2, x3]
all in a single forward pass, thanks to causal (masked) self-attention.
This is why the "target" sequence is simply the "context" sequence shifted
one position to the right.
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """A Dataset of fixed-length (context, target) token id sequences.

    Given a 1D tensor of token ids of length N and a `block_size`, this
    produces N - block_size overlapping training examples via a sliding
    window. Each example is a pair of tensors of shape (block_size,):
    the input sequence and the target sequence (input shifted by one).
    """

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        """
        Args:
            data: 1D LongTensor of token ids for this split (train or val).
            block_size: Length of each training sequence (context window).
        """
        if len(data) <= block_size:
            raise ValueError(
                f"Dataset split has only {len(data)} tokens, which is not "
                f"more than block_size ({block_size}). Use a larger dataset "
                f"or a smaller block_size."
            )
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of distinct starting positions for a full block_size window.
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


def load_and_split(text: str, tokenizer, train_val_split: float
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode raw text and split it into train/validation token tensors.

    Args:
        text: Full raw training corpus.
        tokenizer: A fitted CharTokenizer (or compatible) with .encode().
        train_val_split: Fraction of tokens to use for training (the
            remainder is used for validation). E.g. 0.9 -> 90% train.

    Returns:
        (train_data, val_data): two 1D LongTensors of token ids.
    """
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(train_val_split * len(ids))
    train_data = ids[:split_idx]
    val_data = ids[split_idx:]
    return train_data, val_data
