"""HuggingFace dataset streaming with prechunking for language modeling.

Downloads TinyStories in streaming mode, tokenizes with GPT-2 BPE,
and yields fixed-length chunks suitable for batched training.
"""

import os
import numpy as np
from datasets import load_dataset
from tokenizer import BPETokenizer


def stream_and_chunk(
    dataset_name: str = "roneneldan/TinyStories",
    split: str = "train",
    seq_len: int = 128,
    tokenizer: BPETokenizer | None = None,
    cache_dir: str = "data/tokenized",
    max_stories: int = 50000,
    chunk_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream dataset, tokenize, chunk into fixed-length sequences.

    Returns (train_tokens, val_tokens, vocab_size) as numpy arrays.
    Caches tokenized data to avoid re-downloading.
    """
    tok = tokenizer or BPETokenizer()

    suffix = f"_max{max_stories}" if max_stories else ""
    train_path = os.path.join(cache_dir, f"train_{seq_len}{suffix}.npy")
    val_path = os.path.join(cache_dir, f"val_{seq_len}{suffix}.npy")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"  Loading cached tokenized data from {cache_dir}")
        train_data = np.load(train_path)
        val_data = np.load(val_path)
        print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
        return train_data, val_data, tok.vocab_size

    os.makedirs(cache_dir, exist_ok=True)
    print(f"  Streaming {dataset_name} ({split}), max_stories={max_stories:,}...")

    ds = load_dataset(dataset_name, split=split, streaming=True)

    all_tokens: list[int] = []
    count = 0
    for example in ds:
        text = example["text"]
        tokens = tok.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(tok.eos_token_id)
        count += 1
        if count % chunk_size == 0:
            print(f"    {count:,} stories → {len(all_tokens):,} tokens")
        if max_stories and count >= max_stories:
            break

    tokens_array = np.array(all_tokens, dtype=np.int32)
    del all_tokens

    split_idx = int(0.95 * len(tokens_array))
    train_data = tokens_array[:split_idx]
    val_data = tokens_array[split_idx:]

    np.save(train_path, train_data)
    np.save(val_path, val_data)
    print(f"  Cached: Train {len(train_data):,} | Val {len(val_data):,} tokens")
    print(f"  Vocab size: {tok.vocab_size}")

    return train_data, val_data, tok.vocab_size


def get_batch(
    data: np.ndarray,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random batch of (x, y) pairs from tokenized data using vectorized slicing."""
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    # Vectorized indexing: [batch, seq_len]
    indices = starts[:, None] + np.arange(seq_len)
    x = data[indices]
    y = data[indices + 1]
    return x, y
