"""
Scaled-up DWA experiment: d_model=256, N=1024, r=8, S=4, 20K steps.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from config import LMConfig
from data import load_tinyshakespeare, get_batch
from lm_model import DWALanguageModel, DenseLanguageModel
from experiment import (
    cross_entropy, make_dwa_step, make_dense_step,
    make_optimizer, generate, count_params,
)
from train import get_lambda_sharp, get_aux_scale, update_ema


def evaluate(model, val_data, cfg, np_rng, is_dwa):
    """Quick perplexity estimate on 30 val batches."""
    from experiment import _eval_dwa_batch, _eval_dense_batch
    losses = []
    for _ in range(30):
        x, y = get_batch(val_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)
        loss = _eval_dwa_batch(model, x, y) if is_dwa else _eval_dense_batch(model, x, y)
        losses.append(float(loss))
    return float(np.exp(np.mean(losses)))


def main():
    print(f"JAX devices: {jax.devices()}")

    train_data, val_data, vocab_size, itos, stoi = load_tinyshakespeare()
    print(f"Dataset: {len(train_data):,} train / {len(val_data):,} val | vocab={vocab_size}")

    # Scaled-up config
    # r=8, D=8192: U=256*8=2048, V=8*256=2048, b=256 → 4352 <= 8192 ✓
    # Full rank: k_max=32, 32*8=256=d_model ✓
    cfg = LMConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=8,
        n_layers_A=2,
        n_layers_B=2,
        seq_len=128,
        N=1024,
        D=8192,
        r=8,
        S=4,
        d_k=32,
        k_max=32,
        gamma_init=0.01,
        tau_init=0.0,
        T_temperature=1.0,
        lambda_sharp_phase2_end=5.0,
        lambda_sharp_final=10.0,
        lambda_util=0.01,
        lambda_div=0.01,
        lambda_norm=0.001,
        lambda_sparse=0.01,
        beta_util=0.1,
        ema_decay=0.99,
        batch_size=8,
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=1000,
        max_steps=20000,
        eval_every=500,
        eval_steps=30,
        phase1_end=2000,
        phase2_end=12000,
        grad_clip=1.0,
    )
    print(f"\nConfig: d_model={cfg.d_model} N={cfg.N} D={cfg.D} r={cfg.r} "
          f"S={cfg.S} k_max={cfg.k_max} batch={cfg.batch_size} "
          f"steps={cfg.max_steps}")
    print(f"Effective assembly rank: k_max*r = {cfg.k_max}*{cfg.r} = {cfg.k_max * cfg.r} "
          f"(d_model={cfg.d_model}, {'full rank ✓' if cfg.k_max * cfg.r >= cfg.d_model else 'low rank'})")

    np_rng = np.random.default_rng(42)

    # ---- DWA ----
    print("\n" + "=" * 64)
    print("  [1/2] Training DWA model (d_model=256, N=1024, r=8)")
    print("=" * 64)
    dwa = DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(0)))
    opt = make_optimizer(dwa, cfg)
    dwa_step = make_dwa_step(cfg)
    alpha_ema = jnp.ones(cfg.N) / cfg.N
    dwa_params = count_params(dwa)
    print(f"  DWA: {dwa_params:,} parameters")

    dwa_log = []
    t0 = time.perf_counter()
    for s in range(cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)
        ls = jnp.array(get_lambda_sharp(s, cfg))
        aux = jnp.array(get_aux_scale(s, cfg))
        total, bd, alpha = dwa_step(
            dwa, opt, x, y, alpha_ema, ls, jnp.array(cfg.T_temperature), aux,
        )
        alpha_ema = update_ema(alpha_ema, alpha, cfg.ema_decay)

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl = evaluate(dwa, val_data, cfg, np_rng, is_dwa=True)
            phase = ("warmup" if s < cfg.phase1_end else
                     "gate_on" if s < cfg.phase2_end else "sharpen")
            elapsed = time.perf_counter() - t0
            entry = {"step": s, "phase": phase, "ce": float(bd["ce"]),
                     "ppl": ppl, "lambda": float(ls), "elapsed": elapsed}
            dwa_log.append(entry)
            print(f"  step {s:5d} [{phase:8s}]  ce={entry['ce']:.3f}  "
                  f"ppl={ppl:7.2f}  λ={entry['lambda']:.2f}  ({elapsed:.0f}s)")

    # ---- Dense ----
    print("\n" + "=" * 64)
    print("  [2/2] Training Dense baseline")
    print("=" * 64)
    dense = DenseLanguageModel(cfg, nnx.Rngs(params=jax.random.key(1)))
    opt_d = make_optimizer(dense, cfg)
    dense_step = make_dense_step()
    dense_params = count_params(dense)
    print(f"  Dense: {dense_params:,} parameters")

    dense_log = []
    t0 = time.perf_counter()
    for s in range(cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)
        loss = dense_step(dense, opt_d, x, y)

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl = evaluate(dense, val_data, cfg, np_rng, is_dwa=False)
            elapsed = time.perf_counter() - t0
            entry = {"step": s, "ce": float(loss), "ppl": ppl, "elapsed": elapsed}
            dense_log.append(entry)
            print(f"  step {s:5d}              ce={entry['ce']:.3f}  "
                  f"ppl={ppl:7.2f}  ({elapsed:.0f}s)")

    # ---- Results ----
    dwa_ppl = dwa_log[-1]["ppl"]
    dense_ppl = dense_log[-1]["ppl"]
    delta = dense_ppl - dwa_ppl

    print("\n" + "=" * 64)
    print("  RESULTS  (d_model=256, N=1024, r=8, 20K steps)")
    print("=" * 64)
    print(f"  {'':20s}  {'DWA':>10s}  {'Dense':>10s}")
    print(f"  {'Val perplexity':20s}  {dwa_ppl:>10.2f}  {dense_ppl:>10.2f}")
    print(f"  {'Parameters':20s}  {dwa_params:>10,}  {dense_params:>10,}")
    if delta > 0:
        print(f"\n  DWA beats dense by {delta:.2f} ppl points.")
    elif delta < 0:
        print(f"\n  Dense beats DWA by {-delta:.2f} ppl points.")
    else:
        print("\n  Models are equivalent.")

    # ---- Generation ----
    prompt = [stoi[c] for c in "ROMEO:"]
    print("\n" + "=" * 64)
    print("  GENERATED TEXT (prompt: \"ROMEO:\")")
    print("=" * 64)
    for label, model, is_dwa in [("DWA", dwa, True), ("Dense", dense, False)]:
        ids = generate(model, prompt, max_new=300, vocab_size=vocab_size,
                       is_dwa=is_dwa, seq_len=cfg.seq_len, temperature=0.8)
        text = "".join(itos[i] for i in ids)
        print(f"\n  [{label}]\n  {text}\n")


if __name__ == "__main__":
    main()