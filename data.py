"""
TinyShakespeare character-level dataset.
Downloads ~1 MB on first run, then caches locally.
"""

import os
import urllib.request

import numpy as np


_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
_CACHE = "data/shakespeare.txt"


def load_tinyshakespeare(cache_path: str = _CACHE):
    """
    Returns:
        train_data: np.ndarray[int32]
        val_data:   np.ndarray[int32]
        vocab_size: int
        itos:       dict[int, str]
        stoi:       dict[str, int]
    """
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Downloading TinyShakespeare → {cache_path} ...")
        urllib.request.urlretrieve(_URL, cache_path)

    with open(cache_path) as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    data = np.array([stoi[c] for c in text], dtype=np.int32)
    split = int(0.9 * len(data))
    return data[:split], data[split:], vocab_size, itos, stoi


def get_batch(
    data: np.ndarray,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
):
    """Sample a random batch. Returns numpy arrays (x, y)."""
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s : s + seq_len]     for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y
