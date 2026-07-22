"""
scripts/train_model.py
=======================
Command-line entry point for training MiniGPT.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --max_iters 3000 --batch_size 32
    python scripts/train_model.py --n_layer 4 --n_embd 128 --n_head 4

Run `python scripts/train_model.py --help` for all options.
"""

import argparse
import os
import sys

# Allow running this script directly (python scripts/train_model.py)
# by adding the project root to the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig, TrainConfig
from minigpt.dataset import load_and_split
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer
from minigpt.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniGPT on a text corpus.")

    # Data
    parser.add_argument("--data_path", type=str, default="data/input.txt",
                         help="Path to the training text file.")

    # Model architecture
    parser.add_argument("--block_size", type=int, default=256,
                         help="Context window size (tokens).")
    parser.add_argument("--n_embd", type=int, default=256,
                         help="Embedding dimensionality.")
    parser.add_argument("--n_head", type=int, default=8,
                         help="Number of attention heads.")
    parser.add_argument("--n_layer", type=int, default=6,
                         help="Number of Transformer blocks.")
    parser.add_argument("--dropout", type=float, default=0.25,
                         help="Dropout probability.")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=8000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=400)
    parser.add_argument("--weight_decay", type=float, default=0.2)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_val_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto",
                         help="'auto', 'cuda', 'mps', or 'cpu'.")
    parser.add_argument("--run_name", type=str, default="minigpt",
                         help="Base name for saved checkpoints/plots.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.data_path, encoding="utf-8") as f:
        text = f.read()
    print(f"[data] loaded {len(text):,} characters from {args.data_path}")

    tokenizer = CharTokenizer(text)
    print(f"[tokenizer] vocab size: {tokenizer.vocab_size}")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    tokenizer.save("outputs/checkpoints/tokenizer.json")

    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        train_val_split=args.train_val_split,
        seed=args.seed,
        device=args.device,
    )

    train_data, val_data = load_and_split(text, tokenizer, train_config.train_val_split)
    print(f"[data] train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    model = MiniGPT(model_config)

    train(model, train_data, val_data, model_config, train_config, run_name=args.run_name)


if __name__ == "__main__":
    main()
