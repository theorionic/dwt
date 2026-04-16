"""DWA training entry point — TinyStories + GPT-2 BPE tokenizer.

Usage:
    python run_train.py                          # fresh start
    python run_train.py --resume                  # resume from latest checkpoint
    python run_train.py --resume ckpts/dwa/step_000500

Multi-device: automatically uses all visible devices (TPU cores/GPUs) with data-parallel sharding.
"""

import argparse
import json
import os
import time

import jax
import numpy as np

from config import LMConfig
from tokenizer import get_tokenizer
from dataset import stream_and_chunk, get_batch
from trainer import (
    train_dwa_model,
    train_dense_model,
    N_DEVICES,
    evaluate_ppl,
    shard_batch,
)
from lm_model import DWALanguageModel, DenseLanguageModel
from experiment import (
    count_params,
    _eval_dwa_batch,
    _eval_dense_batch,
    generate,
    make_optimizer,
)
from pool_store import VectorPoolStore, DWAInferenceModel
from train import get_lambda_sharp, get_aux_scale, update_ema

import jax.numpy as jnp
import flax.nnx as nnx
import optax


def parse_args():
    p = argparse.ArgumentParser(description="DWA training on TinyStories")
    p.add_argument(
        "--resume",
        nargs="?",
        const="__latest__",
        default=None,
        help="Resume from checkpoint dir (or 'latest' if flag given)",
    )
    p.add_argument("--ckpt-dir", default="ckpts", help="Base checkpoint directory")
    p.add_argument("--skip-dwa", action="store_true", help="Skip DWA training")
    p.add_argument("--skip-dense", action="store_true", help="Skip dense training")
    p.add_argument("--skip-eval", action="store_true", help="Skip dynamic fetch eval")
    p.add_argument(
        "--max-steps", type=int, default=None, help="Override max_steps in config"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size (auto-scales per GPU)",
    )
    return p.parse_args()


def dynamic_fetch_eval(dwa, cfg, val_data, itos, tok, np_rng):
    print(f"\n{'=' * 64}")
    print("  Dynamic Fetch Benchmark")
    print(f"{'=' * 64}")

    pool_dir = "data/dwa_pool"
    store = VectorPoolStore(pool_dir, N=cfg.N, D=cfg.D, dtype=np.float16)
    store.save(dwa)

    print("\n  Disk footprint:")
    for name in ["pool.npy", "keys.npy", "meta.npz"]:
        fpath = os.path.join(pool_dir, name)
        if os.path.exists(fpath):
            print(f"    {name}: {os.path.getsize(fpath) / 1e6:.1f} MB")

    inf_model = DWAInferenceModel(dwa, store, cfg)
    prompt = tok.encode("Once upon a time")

    print("\n  Generation benchmarks:")
    for k in [4, 8, 16, 32, 64]:
        t0 = time.perf_counter()
        text, fts = inf_model.generate(
            prompt, max_new=50, tok=tok, temperature=0.8, k=k
        )
        elapsed = time.perf_counter() - t0
        ms_per_tok = elapsed / 50 * 1000
        avg_fetch = np.mean(fts) * 1000
        gen_text = text[len("Once upon a time") : 40]
        print(f"    k={k:3d}: {ms_per_tok:.0f} ms/tok, fetch={avg_fetch:.1f} ms/tok")

    t0 = time.perf_counter()
    full_ids = generate(
        dwa,
        prompt,
        max_new=50,
        vocab_size=cfg.vocab_size,
        is_dwa=True,
        seq_len=cfg.seq_len,
        temperature=0.8,
    )
    t_full = time.perf_counter() - t0
    print(f"    Full VRAM: {t_full / 50 * 1000:.0f} ms/tok")

    print("\n  Top-k quality (simulates dynamic fetch):")
    vocab_size = cfg.vocab_size

    def make_eval_topk(k_val):
        @nnx.jit
        def _eval_topk(model, x, y):
            cfg_m = model.cfg
            B, T = x.shape
            d, r = cfg_m.d_model, cfg_m.r
            h = model.tok_emb(x) + model.pos_emb.value[:T]
            for block in model.blocks_A:
                h = block(h)
            h = model.ln_mid(h)
            h_flat = h.reshape(B * T, d)
            pool_vecs = model.pool.value
            alpha, _, _ = model.retrieval(h_flat, pool_vecs, 0.0, 1.0)
            top_idx = jnp.argsort(alpha, axis=-1)[:, -k_val:]
            top_alpha = jnp.take_along_axis(alpha, top_idx, axis=-1)
            alpha_k = jax.nn.softmax(top_alpha, axis=-1)
            top_vecs = pool_vecs[top_idx]
            u_end, v_end, b_end = d * r, d * r + r * d, d * r + r * d + d
            U_k = top_vecs[:, :, :u_end].reshape(B * T, k_val, d, r)
            V_k = top_vecs[:, :, u_end:v_end].reshape(B * T, k_val, r, d)
            bias_k = top_vecs[:, :, v_end:b_end]
            UV_k = jnp.einsum("bkdr,bkre->bkde", U_k, V_k)
            W_delta = jnp.einsum("bk,bkde->bde", alpha_k, UV_k)
            b_delta = jnp.einsum("bk,bkd->bd", alpha_k, bias_k)
            W_assembled = model.middle.W_base.value[None] + W_delta
            b_assembled = model.middle.b_base.value[None] + b_delta
            h_mid = model.middle(h_flat, W_assembled, b_assembled)
            h = h_mid.reshape(B, T, d)
            for block in model.blocks_B:
                h = block(h)
            logits = model.head(model.ln_f(h))
            return optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(B * T, cfg_m.vocab_size), y.reshape(B * T)
            ).mean()

        return _eval_topk

    results = {}
    eval_batch_size = max(8, cfg.batch_size // N_DEVICES) # much smaller for eval to save memory
    for k_val in [4, 8, 16, 32, 64, cfg.N]:
        eval_fn = make_eval_topk(k_val)
        losses = []
        rng_eval = np.random.default_rng(99)
        for _ in range(20):
            x, y = get_batch(val_data, cfg.seq_len, eval_batch_size, rng_eval)
            x_sharded, y_sharded = shard_batch(x, y)
            loss = eval_fn(dwa, x_sharded, y_sharded)
            losses.append(float(loss))
        ppl_k = float(np.exp(np.mean(losses)))
        label = f"k={k_val:3d}" if k_val < cfg.N else f"k={k_val:3d} (all)"
        print(f"    {label} → ppl={ppl_k:.2f}")
        results[k_val] = ppl_k

    return results


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}  ({N_DEVICES} devices, data-parallel)")

    tok = get_tokenizer()
    print(f"Tokenizer: GPT-2 BPE, vocab_size={tok.vocab_size}")

    train_data, val_data, vocab_size = stream_and_chunk(
        seq_len=128,
        tokenizer=tok,
        max_stories=50000,
    )
    print(f"Tokens: {len(train_data):,} train | {len(val_data):,} val")

    base_batch = 128
    batch_size = (args.batch_size or base_batch) * N_DEVICES

    cfg = LMConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers_A=2,
        n_layers_B=2,
        seq_len=128,
        N=256,
        D=2048,
        r=4,
        S=2,
        d_k=32,
        k_max=16,
        batch_size=batch_size,
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=300,
        max_steps=3000,
        eval_every=500,
        eval_steps=20,
        phase1_end=500,
        phase2_end=2000,
        grad_clip=1.0,
    )
    if args.max_steps:
        cfg = LMConfig(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers_A=cfg.n_layers_A,
            n_layers_B=cfg.n_layers_B,
            seq_len=cfg.seq_len,
            N=cfg.N,
            D=cfg.D,
            r=cfg.r,
            S=cfg.S,
            d_k=cfg.d_k,
            k_max=cfg.k_max,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
            max_steps=args.max_steps,
            eval_every=cfg.eval_every,
            eval_steps=cfg.eval_steps,
            phase1_end=cfg.phase1_end,
            phase2_end=cfg.phase2_end,
            grad_clip=cfg.grad_clip,
        )

    print(
        f"\nConfig: d_model={cfg.d_model} N={cfg.N} D={cfg.D} r={cfg.r} "
        f"k_max={cfg.k_max} steps={cfg.max_steps} batch={cfg.batch_size}"
    )

    itos = {i: tok.decode([i]) for i in range(min(100, vocab_size))}
    stoi = {c: i for i, c in enumerate(range(vocab_size))}

    dwa_ppl = None
    dwa = None
    if not args.skip_dwa:
        dwa_ckpt = os.path.join(args.ckpt_dir, "dwa")
        dwa, opt_dwa, alpha_ema, dwa_ppl, dwa_log = train_dwa_model(
            cfg,
            train_data,
            val_data,
            ckpt_dir=dwa_ckpt,
            resume=bool(args.resume),
        )

    dense_ppl = None
    dense = None
    if not args.skip_dense:
        dense_ckpt = os.path.join(args.ckpt_dir, "dense")
        dense, opt_dense, dense_ppl, dense_log = train_dense_model(
            cfg,
            train_data,
            val_data,
            ckpt_dir=dense_ckpt,
            resume=bool(args.resume),
        )

    topk_results = None
    if dwa is not None and not args.skip_eval:
        topk_results = dynamic_fetch_eval(
            dwa, cfg, val_data, itos, tok, np.random.default_rng(99)
        )

    print(f"\n{'=' * 64}")
    print("  SUMMARY")
    print(f"{'=' * 64}")
    if dwa_ppl is not None:
        print(
            f"  DWA ppl:   {dwa_ppl:.2f}  ({count_params(DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(0)))):,} params)"
        )
    if dense_ppl is not None:
        print(
            f"  Dense ppl: {dense_ppl:.2f}  ({count_params(DenseLanguageModel(cfg, nnx.Rngs(params=jax.random.key(0)))):,} params)"
        )
    if dwa_ppl is not None and dense_ppl is not None:
        delta = dense_ppl - dwa_ppl
        print(
            f"  {'DWA beats dense' if delta > 0 else 'Dense beats DWA'} by {abs(delta):.2f} ppl points"
        )
    if topk_results:
        for k, ppl in topk_results.items():
            print(f"  k={k:3d} → ppl={ppl:.2f}")

    results = {
        "config": {
            "d_model": cfg.d_model,
            "N": cfg.N,
            "D": cfg.D,
            "steps": cfg.max_steps,
            "batch_size": cfg.batch_size,
            "n_devices": N_DEVICES,
        },
        "dwa_ppl": dwa_ppl,
        "dense_ppl": dense_ppl,
        "topk_ppl": {str(k): v for k, v in (topk_results or {}).items()},
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results.json")


if __name__ == "__main__":
    main()
