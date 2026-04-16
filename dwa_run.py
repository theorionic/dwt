"""
Entry point — train DWA + dense baseline and report results.

Usage:
    python dwa_run.py                          # fresh run
    python dwa_run.py --resume                 # resume latest checkpoint
    python dwa_run.py --skip-dense             # DWA only
    python dwa_run.py --dataset HuggingFaceFW/fineweb-edu --max-docs 100000
"""

import argparse
import json

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from dwa_model import LMConfig, DWALanguageModel, DenseLanguageModel, count_params, generate, small_config
from dwa_train import load_and_chunk, get_tokenizer, train_dwa, train_dense, N_DEVICES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="roneneldan/TinyStories")
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-docs",   type=int, default=None)
    p.add_argument("--seq-len",    type=int, default=512)
    p.add_argument("--steps",      type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--ckpt-dir",   default="ckpts")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--skip-dwa",   action="store_true")
    p.add_argument("--skip-dense", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}  ({N_DEVICES} devices)")

    # --- Data ---
    train_data, val_data, vocab_size = load_and_chunk(
        dataset_name=args.dataset,
        text_field=args.text_field,
        seq_len=args.seq_len,
        max_docs=args.max_docs,
    )

    # --- Config ---
    cfg = small_config(vocab_size=vocab_size)
    cfg.seq_len    = args.seq_len
    cfg.batch_size = (args.batch_size or 64) * N_DEVICES
    if args.steps:
        cfg.max_steps = args.steps

    print(f"Config: d={cfg.d_model} N={cfg.N} D={cfg.D} r={cfg.r} "
          f"k_max={cfg.k_max} steps={cfg.max_steps} batch={cfg.batch_size}")

    # --- Train ---
    dwa_ppl, dense_ppl = None, None

    if not args.skip_dwa:
        _, _, alpha_ema, dwa_ppl, _ = train_dwa(
            cfg, train_data, val_data,
            ckpt_dir=f"{args.ckpt_dir}/dwa",
            resume=args.resume,
        )

    if not args.skip_dense:
        _, _, dense_ppl, _ = train_dense(
            cfg, train_data, val_data,
            ckpt_dir=f"{args.ckpt_dir}/dense",
            resume=args.resume,
        )

    # --- Summary ---
    print(f"\n{'=' * 64}")
    print("  RESULTS")
    print(f"{'=' * 64}")
    if dwa_ppl is not None:
        n = count_params(DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(0))))
        print(f"  DWA   ppl={dwa_ppl:.2f}  ({n:,} params)")
    if dense_ppl is not None:
        n = count_params(DenseLanguageModel(cfg, nnx.Rngs(params=jax.random.key(0))))
        print(f"  Dense ppl={dense_ppl:.2f}  ({n:,} params)")
    if dwa_ppl and dense_ppl:
        delta = dense_ppl - dwa_ppl
        winner = "DWA" if delta > 0 else "Dense"
        print(f"  {winner} wins by {abs(delta):.2f} ppl points")

    results = {"dwa_ppl": dwa_ppl, "dense_ppl": dense_ppl,
               "config": {"d_model": cfg.d_model, "N": cfg.N, "steps": cfg.max_steps}}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
