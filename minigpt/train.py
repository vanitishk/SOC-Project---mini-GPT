"""
train.py
========
The training loop: optimization, learning rate scheduling, periodic
validation, checkpointing, and loss tracking for MiniGPT.

Training procedure at a glance:
    1. Sample a random batch of (context, target) sequences.
    2. Forward pass: compute logits and cross-entropy loss (next-token
       prediction loss, averaged over every position in every sequence
       in the batch).
    3. Backward pass: compute gradients via backpropagation.
    4. Clip gradients to a max norm (prevents occasional large gradients
       from destabilizing training).
    5. Optimizer step (AdamW) updates model weights.
    6. Repeat for `max_iters` iterations, periodically evaluating on a
       held-out validation set and saving checkpoints.

--- Learning rate schedule: warmup + cosine decay ---
Training starts with a linear warmup from ~0 to the peak learning rate
over `warmup_iters` steps. Starting at full learning rate immediately can
cause instability early in training when the model's weights are still
close to their random initialization. After warmup, the learning rate
follows a cosine decay curve down to `min_lr`, which tends to produce
better final loss than a constant or step-decayed schedule and is the
standard choice used in GPT-2/3 style training.
"""

import math
import time
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from .checkpoint import save_checkpoint
from .dataset import CharDataset
from .model import MiniGPT
from .utils import get_device, plot_losses, set_seed


def get_lr(iteration: int, train_config: TrainConfig) -> float:
    """Compute the learning rate for the current iteration under a
    linear-warmup + cosine-decay schedule.

    Args:
        iteration: Current training iteration (0-indexed).
        train_config: TrainConfig holding learning_rate, min_lr,
            warmup_iters, and max_iters.

    Returns:
        The learning rate to use for this iteration.
    """
    # 1) Linear warmup for the first `warmup_iters` steps.
    if iteration < train_config.warmup_iters:
        return train_config.learning_rate * (iteration + 1) / train_config.warmup_iters

    # 2) After max_iters, hold at the minimum learning rate.
    if iteration > train_config.max_iters:
        return train_config.min_lr

    # 3) Cosine decay in between, from learning_rate down to min_lr.
    decay_ratio = (iteration - train_config.warmup_iters) / (
        train_config.max_iters - train_config.warmup_iters
    )
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges 1 -> 0
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)


@torch.no_grad()
def estimate_loss(model: MiniGPT,
                   loaders: dict,
                   eval_iters: int,
                   device: torch.device) -> dict:
    """Estimate average loss on train and validation splits.

    Averaging over multiple batches gives a much less noisy loss estimate
    than a single batch, since individual batches can vary a lot.

    Args:
        model: The model to evaluate.
        loaders: Dict with 'train' and 'val' DataLoaders (already
            configured to yield random batches, e.g. via shuffle=True).
        eval_iters: Number of batches to average over for each split.
        device: Device to run evaluation on.

    Returns:
        Dict mapping split name ('train'/'val') to average loss (float).
    """
    model.eval()
    results = {}
    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        data_iter = iter(loader)
        for i in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results


def train(model: MiniGPT,
          train_data: torch.Tensor,
          val_data: torch.Tensor,
          model_config: ModelConfig,
          train_config: TrainConfig,
          run_name: str = "minigpt") -> Tuple[List[float], List[float]]:
    """Run the full training loop.

    Args:
        model: A freshly constructed (or resumed) MiniGPT model.
        train_data: 1D LongTensor of training token ids.
        val_data: 1D LongTensor of validation token ids.
        model_config: The ModelConfig used to build `model`.
        train_config: Training hyperparameters.
        run_name: Base filename (without extension) used for saved
            checkpoints and plots.

    Returns:
        (train_losses, val_losses): lists of loss values recorded every
        `eval_interval` iterations, useful for plotting.
    """
    set_seed(train_config.seed)
    device = get_device(train_config.device)
    print(f"[train] using device: {device}")

    model = model.to(device)

    train_dataset = CharDataset(train_data, model_config.block_size)
    val_dataset = CharDataset(val_data, model_config.block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size, shuffle=True,
        drop_last=True,
    )
    loaders = {"train": train_loader, "val": val_loader}

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")

    train_iter = iter(train_loader)
    start_time = time.time()

    for iteration in range(train_config.max_iters + 1):
        # Update learning rate for this step.
        lr = get_lr(iteration, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation + checkpointing.
        if iteration % train_config.eval_interval == 0:
            losses = estimate_loss(model, loaders, train_config.eval_iters, device)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            elapsed = time.time() - start_time
            print(
                f"[iter {iteration:5d}] train_loss {losses['train']:.4f} | "
                f"val_loss {losses['val']:.4f} | lr {lr:.2e} | "
                f"elapsed {elapsed:.1f}s"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model, optimizer, model_config, train_config,
                    iteration, best_val_loss,
                    path=f"{train_config.checkpoint_dir}/{run_name}_best.pt",
                )

            save_checkpoint(
                model, optimizer, model_config, train_config,
                iteration, best_val_loss,
                path=f"{train_config.checkpoint_dir}/{run_name}_latest.pt",
            )

        if iteration == train_config.max_iters:
            break

        # --- One training step ---
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        optimizer.step()

    plot_losses(
        train_losses, val_losses, train_config.eval_interval,
        save_path=f"{train_config.plot_dir}/{run_name}_loss.png",
    )

    print(f"[train] finished. best val_loss: {best_val_loss:.4f}")
    return train_losses, val_losses
