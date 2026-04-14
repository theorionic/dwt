"""
DWA experiment entry point.

Runs a full DWA vs Dense LM comparison on TinyShakespeare.
Edit LMConfig fields below to adjust scale / speed.
"""

import jax

from config import LMConfig
from data import load_tinyshakespeare
from experiment import run_experiment


def main() -> None:
    print(f"JAX devices: {jax.devices()}")

    train_data, val_data, vocab_size, itos, stoi = load_tinyshakespeare()
    print(f"Dataset: {len(train_data):,} train  /  {len(val_data):,} val tokens  |  vocab={vocab_size}")

    cfg = LMConfig(vocab_size=vocab_size)
    run_experiment(cfg, train_data, val_data, itos, stoi)


if __name__ == "__main__":
    main()
