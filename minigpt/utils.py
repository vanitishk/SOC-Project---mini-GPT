"""
utils.py
========
Small shared utilities: reproducibility (seeding), device auto-detection,
and loss curve plotting. Kept separate from the modeling code so those
files stay focused purely on architecture.
"""

import os
import random
from typing import List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility.

    Note that full bitwise reproducibility across different hardware or
    cuDNN versions is not guaranteed, but this eliminates run-to-run
    variance on the same machine.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Pick the best available compute device.

    Args:
        preference: 'auto' to auto-detect the best device, or an explicit
            device string ('cuda', 'mps', 'cpu') to force one.

    Returns:
        A torch.device.
    """
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPUs
        return torch.device("mps")
    return torch.device("cpu")


def plot_losses(train_losses: List[float], val_losses: List[float],
                 eval_interval: int, save_path: str) -> None:
    """Plot training and validation loss curves and save to disk.

    Args:
        train_losses: List of training loss values, one per evaluation.
        val_losses: List of validation loss values, one per evaluation
            (same length and cadence as train_losses).
        eval_interval: Number of training iterations between each
            recorded loss value (used to build the x-axis).
        save_path: File path (e.g. 'outputs/plots/loss.png') to save the
            resulting figure to.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
    import matplotlib.pyplot as plt

    iterations = [i * eval_interval for i in range(len(train_losses))]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, train_losses, label="train loss")
    plt.plot(iterations, val_losses, label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.title("MiniGPT training progress")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved loss curve to {save_path}")
