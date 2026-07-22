"""
checkpoint.py
=============
Save and load training checkpoints.

A checkpoint bundles everything needed to resume training or run
inference later, without needing to remember which hyperparameters were
used:
    - model weights (state_dict)
    - optimizer state (so training can resume with correct momentum, etc.)
    - the ModelConfig and TrainConfig used for this run
    - the current training iteration and best validation loss so far

The tokenizer's vocabulary is saved separately as tokenizer.json (see
tokenizer.py's save/load) since it's needed for both training and
generation and is more natural to store as plain JSON.
"""

import os
from typing import Any, Dict, Optional

import torch

from config import ModelConfig, TrainConfig


def save_checkpoint(model: torch.nn.Module,
                     optimizer: Optional[torch.optim.Optimizer],
                     model_config: ModelConfig,
                     train_config: TrainConfig,
                     iteration: int,
                     best_val_loss: float,
                     path: str) -> None:
    """Save a full training checkpoint to disk.

    Args:
        model: The MiniGPT model whose weights should be saved.
        optimizer: The optimizer whose state should be saved (pass None
            to skip, e.g. for a "final model only" export).
        model_config: The ModelConfig used to build `model`.
        train_config: The TrainConfig used for this training run.
        iteration: Current training iteration number.
        best_val_loss: Best validation loss observed so far.
        path: File path to save the checkpoint to (e.g. 'model.pt').
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "iteration": iteration,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    print(f"[checkpoint] saved to {path} (iter {iteration}, val_loss {best_val_loss:.4f})")


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load a checkpoint dict from disk.

    Args:
        path: File path to the saved checkpoint.
        device: Device to map tensors to when loading.

    Returns:
        The raw checkpoint dictionary (weights, configs, metadata). Use
        model_config = ModelConfig(**checkpoint["model_config"]) to
        reconstruct the config, then build a MiniGPT and call
        model.load_state_dict(checkpoint["model_state_dict"]).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    print(f"[checkpoint] loaded from {path} (iter {checkpoint.get('iteration', '?')})")
    return checkpoint
