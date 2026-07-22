"""
generate.py
===========
Autoregressive text generation: temperature scaling and top-k sampling.

Once trained, MiniGPT generates text one token at a time:
    1. Feed the current sequence into the model.
    2. Take the logits at the *last* position -- this is the model's
       prediction for what comes next.
    3. Convert logits into a sampling distribution (with temperature and
       top-k adjustments, see below).
    4. Sample the next token from that distribution.
    5. Append it to the sequence and repeat.

This is "autoregressive" because each new token is generated conditioned
on all previously generated tokens (auto = self, regressive = feeding
back on itself).

--- Temperature ---
Before sampling, logits are divided by a temperature T:
    scaled_logits = logits / T
- T < 1.0 sharpens the distribution (more confident, more repetitive,
  closer to always picking the single most likely token as T -> 0).
- T = 1.0 leaves the distribution as the model learned it.
- T > 1.0 flattens the distribution (more random, more diverse, but can
  produce less coherent text as T grows).

--- Top-k sampling ---
Rather than sampling from the full vocabulary (which includes many very
low-probability, often nonsensical tokens), top-k sampling restricts
sampling to only the k highest-probability tokens at each step, masking
out everything else. This avoids the "long tail" of unlikely tokens
occasionally derailing generation, while still allowing controlled
randomness among plausible continuations.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .model import MiniGPT
from .tokenizer import CharTokenizer


@torch.no_grad()
def generate(model: MiniGPT,
             tokenizer: CharTokenizer,
             prompt: str,
             max_new_tokens: int = 300,
             temperature: float = 0.8,
             top_k: Optional[int] = 40,
             device: Optional[torch.device] = None) -> str:
    """Generate text autoregressively from a trained MiniGPT model.

    Args:
        model: A trained MiniGPT model (should be in eval() mode).
        tokenizer: The CharTokenizer used to encode/decode text.
        prompt: The seed string to condition generation on. Can be empty.
        max_new_tokens: Number of new tokens (characters) to generate.
        temperature: Sampling temperature. Lower = more deterministic,
            higher = more random. Must be > 0.
        top_k: If set, restrict sampling to the top_k most likely tokens
            at each step. Set to None to disable (sample from the full
            distribution).
        device: Device to run generation on. Defaults to the model's
            current device.

    Returns:
        The generated text, including the original prompt.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    model.eval()
    device = device or next(model.parameters()).device
    block_size = model.config.block_size

    # Encode the prompt into a (1, seq_len) tensor. An empty prompt starts
    # from a single "unconditional" seed token so the model has something
    # to attend to; here we just require a non-empty prompt for simplicity
    # and clarity in this educational implementation.
    if len(prompt) == 0:
        raise ValueError("prompt must be non-empty for this implementation")

    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # The model only supports sequences up to block_size tokens, so
        # crop the context to the last block_size tokens if it's grown
        # longer than that.
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        logits, _ = model(idx_cond)  # (1, seq_len, vocab_size)

        # Only the logits for the last position matter -- that's the
        # model's prediction for the next token.
        last_logits = logits[:, -1, :]  # (1, vocab_size)

        # --- Temperature scaling ---
        last_logits = last_logits / temperature

        # --- Top-k filtering ---
        if top_k is not None:
            top_k = min(top_k, last_logits.size(-1))
            # Get the value of the k-th largest logit; anything smaller
            # gets masked out to -inf so it receives ~0 probability.
            kth_value = torch.topk(last_logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
            last_logits = last_logits.masked_fill(last_logits < kth_value, float("-inf"))

        probs = F.softmax(last_logits, dim=-1)  # (1, vocab_size)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        idx = torch.cat([idx, next_id], dim=1)

    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)
