"""
tokenizer.py
============
A character-level tokenizer.

Modern LLMs (GPT-2/3/4, LLaMA, etc.) use subword tokenizers like Byte-Pair
Encoding (BPE), which split text into frequent chunks ("tok", "eniz", "ation")
to keep vocabularies compact while covering arbitrary text efficiently.

Each unique character in the training corpus becomes one entry in the
vocabulary. Encoding maps a string to a list of integers; decoding maps
integers back to a string.
"""

import json
from typing import Dict, List


class CharTokenizer:
    """A simple, invertible character <-> integer tokenizer.

    Attributes:
        stoi: Mapping from character to integer id.
        itos: Mapping from integer id back to character.
        vocab_size: Number of unique characters known to this tokenizer.
    """

    def __init__(self, text: str) -> None:
        """Build the vocabulary from all unique characters in `text`.

        Args:
            text: The full training corpus. Every character that appears
                anywhere in this string becomes a vocabulary entry.
        """
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size: int = len(chars)

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token ids.

        Args:
            text: Input string. Characters not seen during vocabulary
                construction will raise a KeyError -- this is intentional,
                so unknown-character bugs surface loudly during an
                educational project rather than silently corrupting data.

        Returns:
            List of integer token ids, one per character.
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token ids back into a string.

        Args:
            ids: List (or 1D tensor-like sequence) of integer token ids.

        Returns:
            The decoded string.
        """
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        """Persist the vocabulary to a JSON file so it can be reloaded
        exactly (character order matters for id consistency)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Reconstruct a tokenizer from a saved vocabulary JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.stoi = {ch: int(i) for ch, i in data["stoi"].items()}
        obj.itos = {int(i): ch for ch, i in obj.stoi.items()}
        obj.vocab_size = len(obj.stoi)
        return obj

    def __len__(self) -> int:
        return self.vocab_size
