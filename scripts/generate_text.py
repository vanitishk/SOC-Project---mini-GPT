"""
scripts/generate_text.py
=========================
Command-line entry point for generating text from a trained MiniGPT
checkpoint.

Usage:
    python scripts/generate_text.py --prompt "LUKE:" --max_new_tokens 300
    python scripts/generate_text.py --prompt "HAN:" --temperature 1.0 --top_k 20
    python scripts/generate_text.py --checkpoint outputs/checkpoints/minigpt_best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from config import ModelConfig
from minigpt.checkpoint import load_checkpoint
from minigpt.generate import generate
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer
from minigpt.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained MiniGPT model.")
    parser.add_argument("--checkpoint", type=str,
                         default="outputs/checkpoints/minigpt_best.pt",
                         help="Path to the model checkpoint.")
    parser.add_argument("--tokenizer", type=str,
                         default="outputs/checkpoints/tokenizer.json",
                         help="Path to the saved tokenizer vocabulary.")
    parser.add_argument("--prompt", type=str, default="LUKE:",
                         help="Seed text to condition generation on.")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                         help="Number of new characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="Sampling temperature (higher = more random).")
    parser.add_argument("--top_k", type=int, default=40,
                         help="Restrict sampling to the top_k most likely tokens.")
    parser.add_argument("--device", type=str, default="auto",
                         help="'auto', 'cuda', 'mps', or 'cpu'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    tokenizer = CharTokenizer.load(args.tokenizer)
    print(f"[tokenizer] loaded vocab size: {tokenizer.vocab_size}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = ModelConfig(**checkpoint["model_config"])

    model = MiniGPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"\n--- Generating (temperature={args.temperature}, top_k={args.top_k}) ---\n")
    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(output)


if __name__ == "__main__":
    main()
