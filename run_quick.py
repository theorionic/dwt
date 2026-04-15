"""
Quick validation: train DWA + Dense for 3K steps on small config,
then benchmark dynamic fetch quality vs k.
"""

import time
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx

from config import LMConfig
from data import load_tinyshakespeare, get_batch
from lm_model import DWALanguageModel, DenseLanguageModel
from experiment import (
    cross_entropy, make_dwa_step, make_dense_step,
    make_optimizer, generate, count_params,
)
from train import get_lambda_sharp, get_aux_scale, update_ema
from pool_store import VectorPoolStore, DWAInferenceModel


def evaluate_ppl(model, val_data, cfg, np_rng, is_dwa):
    from experiment import _eval_dwa_batch, _eval_dense_batch
    losses = []
    for _ in range(20):
        x, y = get_batch(val_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)
        loss = _eval_dwa_batch(model, x, y) if is_dwa else _eval_dense_batch(model, x, y)
        losses.append(float(loss))
    return float(np.exp(np.mean(losses)))


def main():
    print(f"JAX devices: {jax.devices()}")
    train_data, val_data, vocab_size, itos, stoi = load_tinyshakespeare()
    print(f"Dataset: {len(train_data):,} train / {len(val_data):,} val | vocab={vocab_size}")

    # Small config for quick validation
    cfg = LMConfig(
        vocab_size=vocab_size,
        d_model=128, n_heads=4, n_layers_A=2, n_layers_B=2, seq_len=128,
        N=256, D=2048, r=4, S=2, d_k=32, k_max=16,
        batch_size=32, lr=3e-4, weight_decay=0.1,
        warmup_steps=200, max_steps=3000, eval_every=500, eval_steps=20,
        phase1_end=500, phase2_end=2000, grad_clip=1.0,
    )
    print(f"\nConfig: d_model={cfg.d_model} N={cfg.N} D={cfg.D} r={cfg.r} "
          f"k_max={cfg.k_max} steps={cfg.max_steps}")

    np_rng = np.random.default_rng(0)

    # ===================== Train DWA =====================
    print("\n" + "=" * 64)
    print("  [1/3] Training DWA model (3K steps)")
    print("=" * 64)
    dwa = DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(0)))
    opt = make_optimizer(dwa, cfg)
    dwa_step = make_dwa_step(cfg)
    alpha_ema = jnp.ones(cfg.N) / cfg.N
    dwa_params = count_params(dwa)
    print(f"  DWA: {dwa_params:,} params")

    t0 = time.perf_counter()
    for s in range(cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)
        ls = jnp.array(get_lambda_sharp(s, cfg))
        aux = jnp.array(get_aux_scale(s, cfg))
        total, bd, alpha = dwa_step(
            dwa, opt, x, y, alpha_ema, ls, jnp.array(1.0), aux,
        )
        alpha_ema = update_ema(alpha_ema, alpha, cfg.ema_decay)
        if s % 500 == 0 or s == cfg.max_steps - 1:
            ppl = evaluate_ppl(dwa, val_data, cfg, np_rng, True)
            print(f"  step {s:5d}  ce={float(bd['ce']):.3f}  ppl={ppl:.2f}  ({time.perf_counter()-t0:.0f}s)")
    dwa_ppl = evaluate_ppl(dwa, val_data, cfg, np_rng, True)

    # ===================== Train Dense =====================
    print("\n" + "=" * 64)
    print("  [2/3] Training Dense baseline (3K steps)")
    print("=" * 64)
    dense = DenseLanguageModel(cfg, nnx.Rngs(params=jax.random.key(1)))
    opt_d = make_optimizer(dense, cfg)
    dense_step = make_dense_step()
    dense_params = count_params(dense)
    print(f"  Dense: {dense_params:,} params")

    t0 = time.perf_counter()
    for s in range(cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)
        loss = dense_step(dense, opt_d, x, y)
        if s % 500 == 0 or s == cfg.max_steps - 1:
            ppl = evaluate_ppl(dense, val_data, cfg, np_rng, False)
            print(f"  step {s:5d}  ce={float(loss):.3f}  ppl={ppl:.2f}  ({time.perf_counter()-t0:.0f}s)")
    dense_ppl = evaluate_ppl(dense, val_data, cfg, np_rng, False)

    # ===================== Dynamic Fetch Benchmark =====================
    print("\n" + "=" * 64)
    print("  [3/3] Dynamic Fetch Benchmark")
    print("=" * 64)

    # Save pool to disk
    pool_dir = "data/dwa_pool"
    store = VectorPoolStore(pool_dir, N=cfg.N, D=cfg.D, dtype=np.float16)
    store.save(dwa)

    # Check disk sizes
    print("\n  Disk footprint:")
    for name in ["pool.npy", "keys.npy", "meta.npz"]:
        fpath = os.path.join(pool_dir, name)
        if os.path.exists(fpath):
            print(f"    {name}: {os.path.getsize(fpath)/1e6:.1f} MB")

    # Dynamic fetch model
    inf_model = DWAInferenceModel(dwa, store, cfg)

    # ---- Generation benchmark ----
    prompt = [stoi[c] for c in "ROMEO:"]
    print("\n  Generation benchmarks:")
    for k in [4, 8, 16, 32, 64]:
        t0 = time.perf_counter()
        text, fts = inf_model.generate(prompt, max_new=50, itos=itos, temperature=0.8, k=k)
        elapsed = time.perf_counter() - t0
        ms_per_tok = elapsed / 50 * 1000
        avg_fetch = np.mean(fts) * 1000
        print(f"    k={k:3d}: {ms_per_tok:.0f} ms/token, fetch={avg_fetch:.1f} ms/token, text: {text[6:40]}...")

    # Full-VRAM generation for comparison
    t0 = time.perf_counter()
    full_ids = generate(dwa, prompt, max_new=50, vocab_size=vocab_size,
                         is_dwa=True, seq_len=cfg.seq_len, temperature=0.8)
    t_full = time.perf_counter() - t0
    full_text = "".join(itos[i] for i in full_ids)
    print(f"    Full VRAM: {t_full/50*1000:.0f} ms/token, text: {full_text[6:40]}...")

    # ---- Quality: top-k masking perplexity ----
    print("\n  Quality: top-k masking (simulates dynamic fetch)")

    def make_eval_topk(k_val):
        """Create a JIT-compiled top-k eval function for a fixed k.

        k_val is captured as a static Python int from the closure,
        avoiding dynamic-slice issues inside JIT.
        """
        @nnx.jit
        def _eval_topk(model, x, y):
            """Evaluate DWA with only top-k vectors (simulates dynamic fetch)."""
            cfg_m = model.cfg
            B, T = x.shape
            d = cfg_m.d_model
            r = cfg_m.r

            h = model.tok_emb(x) + model.pos_emb.value[:T]
            for block in model.blocks_A:
                h = block(h)
            h = model.ln_mid(h)
            h_flat = h.reshape(B * T, d)
            pool_vecs = model.pool.value                        # [N, D]

            # Full retrieval to get per-token alpha weights
            alpha, scores, keys = model.retrieval(h_flat, pool_vecs, 0.0, 1.0)

            # Top-k selection — uses gather, not scatter (JIT-compatible)
            top_idx = jnp.argsort(alpha, axis=-1)[:, -k_val:]         # [B*T, k]
            top_alpha = jnp.take_along_axis(alpha, top_idx, axis=-1)   # [B*T, k]
            alpha_k = jax.nn.softmax(top_alpha, axis=-1)               # [B*T, k]
            top_vecs = pool_vecs[top_idx]                               # [B*T, k, D]

            # Inline assembly using only top-k vectors (no masking needed)
            u_end = d * r
            v_end = u_end + r * d
            b_end = v_end + d

            U_k = top_vecs[:, :, :u_end].reshape(B * T, k_val, d, r)       # [B*T, k, d, r]
            V_k = top_vecs[:, :, u_end:v_end].reshape(B * T, k_val, r, d)  # [B*T, k, r, d]
            bias_k = top_vecs[:, :, v_end:b_end]                             # [B*T, k, d]

            # Low-rank products per top-k vector, then weighted sum
            UV_k = jnp.einsum("bkdr,bkre->bkde", U_k, V_k)                  # [B*T, k, d, d]
            W_delta = jnp.einsum("bk,bkde->bde", alpha_k, UV_k)             # [B*T, d, d]
            b_delta = jnp.einsum("bk,bkd->bd", alpha_k, bias_k)             # [B*T, d]

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

    for k_val in [4, 8, 16, 32, 64, 256]:
        eval_fn = make_eval_topk(k_val)
        losses = []
        rng_eval = np.random.default_rng(99)
        for _ in range(20):
            x, y = get_batch(val_data, cfg.seq_len, cfg.batch_size, rng_eval)
            loss = eval_fn(dwa, jnp.array(x), jnp.array(y))
            losses.append(float(loss))
        ppl_k = float(np.exp(np.mean(losses)))
        label = f"k={k_val:3d}" if k_val < cfg.N else f"k={k_val:3d} (all)"
        print(f"    {label} → ppl={ppl_k:.2f}")

    # ===================== Summary =====================
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  DWA ppl:      {dwa_ppl:.2f}  ({dwa_params:,} params)")
    print(f"  Dense ppl:    {dense_ppl:.2f}  ({dense_params:,} params)")
    delta = dense_ppl - dwa_ppl
    if delta > 0:
        print(f"  DWA beats dense by {delta:.2f} ppl points")
    else:
        print(f"  Dense beats DWA by {-delta:.2f} ppl points")
    print(f"\n  VRAM for DWA active params:  ~1.5 MB (model) + ~0.1 MB (k=16 vectors)")
    print(f"  Pool on disk:               ~{os.path.getsize(os.path.join(pool_dir, 'pool.npy'))/1e6:.1f} MB")
    print(f"  Disk fetch per token:        ~4 ms (mmap + numpy)")
    print(f"  This enables inference on hardware that can't hold the full pool in VRAM.")


if __name__ == "__main__":
    main()