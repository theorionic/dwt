"""
Train DWA LM on FineWeb-Edu for 1000 steps on TPU v5e-8.

Performance choices
───────────────────
• bfloat16 everywhere — TPU v5e MXU is ~4× faster in BF16 vs FP32.
  We cast model params to BF16 before creating the optimizer so Adam's
  moment accumulators also live in BF16 (fine for 1000 steps).
• d_model=128, seq_len=128, N=128 — keeps the per-device assembly
  einsum (B_local·T·N·d²) in the 10–20 GFLOP range → ~0.5 s/step.
• batch = 32 per device × 8 cores = 256 total.

Expected throughput: ~0.5 s/step → 1000 steps ≈ 8–12 minutes.

Usage:
    python run_ultrafineweb.py
"""

import os, sys, time, json
import numpy as np

import jax
import jax.numpy as jnp
import jax.sharding as js
import flax.nnx as nnx
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import LMConfig
from lm_model import DWALanguageModel
from train import get_lambda_sharp, get_aux_scale
from losses import utilization_loss, diversity_loss, sparsity_loss
from tokenizer import BPETokenizer


# ══════════════════════════════════════════════════════════════════
# TPU mesh — 1-D data-parallel across all 8 cores
# ══════════════════════════════════════════════════════════════════

N_DEVICES = len(jax.devices())
print(f"[init] {N_DEVICES} TPU devices: {jax.devices()}")

_mesh      = js.Mesh(np.array(jax.devices()), ("data",))
_data_shard = js.NamedSharding(_mesh, js.PartitionSpec("data", None))  # batch dim
_replicated = js.NamedSharding(_mesh, js.PartitionSpec())              # fully replicated


def shard_batch(x: np.ndarray, y: np.ndarray):
    """Put (x, y) on devices with the batch dim split across all cores."""
    return jax.device_put(x, _data_shard), jax.device_put(y, _data_shard)


def rep(arr: jax.Array) -> jax.Array:
    """Replicate a scalar/small array to all devices."""
    return jax.device_put(arr, _replicated)


def replicate_module(mod: nnx.Module) -> nnx.Module:
    """Broadcast all NNX module state to every device (replicated sharding)."""
    graph, state = nnx.split(mod)
    state = jax.device_put(state, _replicated)
    return nnx.merge(graph, state)


# ══════════════════════════════════════════════════════════════════
# bfloat16 cast helper
# ══════════════════════════════════════════════════════════════════

def to_bf16(model: nnx.Module) -> nnx.Module:
    """Cast every float32 parameter/buffer in the model to bfloat16."""
    graph, state = nnx.split(model)
    state = jax.tree_util.tree_map(
        lambda v: v.astype(jnp.bfloat16) if v.dtype == jnp.float32 else v,
        state,
    )
    return nnx.merge(graph, state)


# ══════════════════════════════════════════════════════════════════
# Data — FineWeb-Edu (streamed & cached)
# ══════════════════════════════════════════════════════════════════

def load_fineweb_edu(
    seq_len: int = 128,
    max_docs: int = 25_000,
    cache_dir: str = "data/fineweb_edu",
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Stream HuggingFaceFW/fineweb-edu (sample-10BT subset), tokenize with
    GPT-2 BPE (tiktoken), cache to disk. Returns (train, val, vocab_size).
    """
    os.makedirs(cache_dir, exist_ok=True)
    tok        = BPETokenizer()
    suffix     = f"seq{seq_len}_n{max_docs}"
    train_path = os.path.join(cache_dir, f"train_{suffix}.npy")
    val_path   = os.path.join(cache_dir, f"val_{suffix}.npy")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[data] Loading cached tokens …")
        train = np.load(train_path)
        val   = np.load(val_path)
        print(f"[data] train={len(train):,}  val={len(val):,}  vocab={tok.vocab_size:,}")
        return train, val, tok.vocab_size

    from datasets import load_dataset
    print("[data] Streaming HuggingFaceFW/fineweb-edu (sample-10BT) …")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    all_tokens: list[int] = []
    for i, ex in enumerate(ds):
        all_tokens.extend(tok.encode(ex["text"]))
        all_tokens.append(tok.eos_token_id)
        if (i + 1) % 5_000 == 0:
            print(f"[data]   {i+1:,}/{max_docs:,} docs → {len(all_tokens):,} tokens")
        if i + 1 >= max_docs:
            break

    arr        = np.array(all_tokens, dtype=np.int32)
    split_idx  = int(0.95 * len(arr))
    train_data = arr[:split_idx]
    val_data   = arr[split_idx:]
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    print(f"[data] Saved → train={len(train_data):,}  val={len(val_data):,}")
    return train_data, val_data, tok.vocab_size


def get_batch(
    data: np.ndarray,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    starts  = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    indices = starts[:, None] + np.arange(seq_len)
    return data[indices], data[indices + 1]


# ══════════════════════════════════════════════════════════════════
# Loss helpers
# ══════════════════════════════════════════════════════════════════

def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    B, T, V = logits.shape
    # Cast to float32 for numerically stable softmax, then compute CE
    logits_f32 = logits.astype(jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits_f32.reshape(B * T, V), targets.reshape(B * T)
    ).mean()


def compute_total_loss(logits, targets, alpha, alpha_ema, keys, w_norm, cfg, aux_scale):
    ce   = cross_entropy(logits, targets)
    # Cast aux tensors to float32 for stable auxiliary loss computation
    alpha_f32     = alpha.astype(jnp.float32)
    alpha_ema_f32 = alpha_ema.astype(jnp.float32)
    keys_f32      = keys.astype(jnp.float32)
    w_norm_f32    = w_norm.astype(jnp.float32)
    l_u = utilization_loss(alpha_ema_f32, cfg.beta_util)
    l_d = diversity_loss(alpha_f32, keys_f32)
    l_s = sparsity_loss(alpha_f32)
    aux = (cfg.lambda_util * l_u + cfg.lambda_div * l_d
           + cfg.lambda_sparse * l_s + cfg.lambda_norm * w_norm_f32)
    return ce + aux_scale.astype(jnp.float32) * aux, {
        "ce": ce, "util": l_u, "div": l_d, "sparse": l_s
    }


# ══════════════════════════════════════════════════════════════════
# Optimizer
# ══════════════════════════════════════════════════════════════════

def make_optimizer(model: nnx.Module, cfg: LMConfig) -> nnx.Optimizer:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.max_steps,
        end_value=cfg.lr * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ══════════════════════════════════════════════════════════════════
# JIT-compiled training step
# ══════════════════════════════════════════════════════════════════
# Data-parallel mechanics:
#   x, y    carry PartitionSpec("data", None) → each core sees its shard
#   model   carries PartitionSpec()           → weights replicated
# XLA inserts AllReduce over gradients automatically before the
# optimizer update, so all cores apply the same weight delta.

def make_train_step(cfg: LMConfig):
    @nnx.jit
    def _step(
        model: DWALanguageModel,
        opt: nnx.Optimizer,
        x: jax.Array,
        y: jax.Array,
        alpha_ema: jax.Array,     # [N]   float32, replicated
        lambda_sharp: jax.Array,  # scalar
        temperature: jax.Array,   # scalar
        aux_scale: jax.Array,     # scalar
    ):
        def loss_fn(m: DWALanguageModel):
            logits, alpha, keys, w_norm = m(x, lambda_sharp, temperature)
            total, bd = compute_total_loss(
                logits, y, alpha, alpha_ema, keys, w_norm, cfg, aux_scale
            )
            return total, (bd, alpha)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (total, (bd, alpha)), grads = grad_fn(model)
        opt.update(model, grads)

        # Per-vector utilization EMA (XLA all-reduces the mean correctly)
        a_f32     = alpha.astype(jnp.float32).reshape(-1, cfg.N)
        batch_mean = jnp.mean(a_f32, axis=0)  # [N]
        new_ema    = cfg.ema_decay * alpha_ema + (1.0 - cfg.ema_decay) * batch_mean
        return total, bd, alpha, new_ema

    return _step


# ══════════════════════════════════════════════════════════════════
# JIT-compiled eval step
# ══════════════════════════════════════════════════════════════════

@nnx.jit
def _eval_step(model: DWALanguageModel, x: jax.Array, y: jax.Array) -> jax.Array:
    logits, *_ = model(x, lambda_sharp=0.0, temperature=1.0)
    return cross_entropy(logits, y)


# ══════════════════════════════════════════════════════════════════
# Autoregressive text generation (fixed-length window)
# ══════════════════════════════════════════════════════════════════

@nnx.jit
def _gen_forward(model: DWALanguageModel, ctx: jax.Array) -> jax.Array:
    """Returns last-position logits: [vocab_size]."""
    logits, *_ = model(ctx, lambda_sharp=0.0, temperature=1.0)
    return logits[0, -1, :].astype(jnp.float32)


def generate_text(
    model: DWALanguageModel,
    prompt_ids: list[int],
    max_new: int,
    seq_len: int,
    vocab_size: int,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[int]:
    """Autoregressive top-k + temperature sampling."""
    rng = jax.random.key(999)
    # Warm up JIT with one dummy pass
    dummy = jnp.zeros((1, seq_len), dtype=jnp.int32)
    _gen_forward(model, dummy)

    ids = list(prompt_ids)
    for _ in range(max_new):
        cur    = ids[-seq_len:]
        padded = [0] * (seq_len - len(cur)) + cur
        ctx    = jnp.array(padded, dtype=jnp.int32)[None]

        logits = _gen_forward(model, ctx) / temperature
        # Top-k filter
        top_vals, _ = jax.lax.top_k(logits, top_k)
        logits = jnp.where(logits >= top_vals[-1], logits, jnp.finfo(jnp.float32).min)

        rng, sub = jax.random.split(rng)
        ids.append(int(jax.random.categorical(sub, logits)))

    return ids


# ══════════════════════════════════════════════════════════════════
# Param count helper
# ══════════════════════════════════════════════════════════════════

def count_params(model: nnx.Module) -> int:
    _, state = nnx.split(model)
    return sum(v.size for v in jax.tree_util.tree_leaves(state))


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    # ── 1. Data ────────────────────────────────────────────────────
    SEQ_LEN   = 128
    train_data, val_data, vocab_size = load_fineweb_edu(
        seq_len=SEQ_LEN, max_docs=25_000
    )

    # ── 2. Config ──────────────────────────────────────────────────
    # d_model=128, r=4  →  D ≥ 128*4 + 4*128 + 128 = 1152  (use 2048)
    BATCH_PER_DEV = 32
    cfg = LMConfig(
        vocab_size    = vocab_size,
        d_model       = 128,
        n_heads       = 4,
        n_layers_A    = 2,
        n_layers_B    = 2,
        seq_len       = SEQ_LEN,
        N             = 128,
        D             = 2048,
        r             = 4,
        S             = 2,
        d_k           = 32,
        k_max         = 16,
        batch_size    = BATCH_PER_DEV * N_DEVICES,   # 256 total
        lr            = 3e-4,
        weight_decay  = 0.1,
        warmup_steps  = 100,
        max_steps     = 1000,
        eval_every    = 100,
        eval_steps    = 20,
        phase1_end    = 200,
        phase2_end    = 700,
        grad_clip     = 1.0,
    )

    tokens_per_step = cfg.batch_size * cfg.seq_len
    print(f"\n[cfg] d_model={cfg.d_model}  heads={cfg.n_heads}  "
          f"layers={cfg.n_layers_A}+DWA+{cfg.n_layers_B}")
    print(f"[cfg] seq={cfg.seq_len}  N={cfg.N}  D={cfg.D}  r={cfg.r}  k_max={cfg.k_max}")
    print(f"[cfg] batch={cfg.batch_size} ({BATCH_PER_DEV}/dev)  "
          f"steps={cfg.max_steps}  tokens/step={tokens_per_step:,}")

    # ── 3. Build model → cast to bfloat16 → create optimizer ──────
    print("[model] Building model …")
    model = DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(42)))
    print(f"[model] {count_params(model):,} params (float32)")

    # Cast to bfloat16 BEFORE creating optimizer so optimizer state
    # tracks bf16 tensors (BF16 Adam is stable enough for 1000 steps)
    model = to_bf16(model)
    print("[model] Cast to bfloat16")

    opt = make_optimizer(model, cfg)

    # ── 4. Replicate across all 8 TPU cores ───────────────────────
    model = replicate_module(model)
    opt   = replicate_module(opt)
    print(f"[mesh] Replicated across {N_DEVICES} devices")

    alpha_ema = rep(jnp.ones(cfg.N, dtype=jnp.float32) / cfg.N)
    step_fn   = make_train_step(cfg)
    np_rng    = np.random.default_rng(0)
    log: list[dict] = []

    # ── 5. Training loop ───────────────────────────────────────────
    print(f"\n{'step':>6}  {'phase':>8}  {'train_ce':>9}  {'val_ppl':>8}  "
          f"{'λ_sharp':>8}  {'s/step':>7}  {'elapsed':>8}")
    print("─" * 70)

    t0   = time.perf_counter()
    t_prev = t0

    for s in range(cfg.max_steps):
        x, y       = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x_sh, y_sh = shard_batch(x, y)

        ls  = rep(jnp.array(get_lambda_sharp(s, cfg), dtype=jnp.float32))
        aux = rep(jnp.array(get_aux_scale(s, cfg),    dtype=jnp.float32))
        tmp = rep(jnp.array(1.0,                      dtype=jnp.float32))

        total, bd, alpha, alpha_ema = step_fn(
            model, opt, x_sh, y_sh, alpha_ema, ls, tmp, aux
        )

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            # Force pending computations before timing eval
            jax.block_until_ready(total)

            val_losses = []
            for _ in range(cfg.eval_steps):
                vx, vy       = get_batch(val_data, cfg.seq_len, cfg.batch_size, np_rng)
                vx_sh, vy_sh = shard_batch(vx, vy)
                val_losses.append(float(_eval_step(model, vx_sh, vy_sh)))
            val_ppl = float(np.exp(np.mean(val_losses)))

            phase   = ("warmup"  if s < cfg.phase1_end else
                       "gate_on" if s < cfg.phase2_end else "sharpen")
            elapsed = time.perf_counter() - t0
            # s/step since last log (first interval includes JIT compile)
            n_steps_since  = s if s == 0 else cfg.eval_every
            secs_per_step  = (elapsed - (t_prev - t0)) / max(n_steps_since, 1)
            t_prev         = time.perf_counter()

            entry = {
                "step":     s,
                "phase":    phase,
                "train_ce": float(bd["ce"]),
                "val_ppl":  round(val_ppl, 3),
                "lambda":   round(float(ls), 4),
                "elapsed":  round(elapsed, 1),
            }
            log.append(entry)
            print(f"{s:6d}  {phase:>8}  {float(bd['ce']):9.4f}  "
                  f"{val_ppl:8.3f}  {float(ls):8.4f}  "
                  f"{secs_per_step:7.2f}  {elapsed:7.1f}s")

    # ── 6. Text generation ─────────────────────────────────────────
    tok = BPETokenizer()
    prompts = [
        "The history of science shows that",
        "Scientists have recently discovered that",
        "One of the most important things to understand is",
        "The best way to learn something new is to",
    ]

    print(f"\n{'═' * 70}")
    print("  GENERATED TEXT   (DWA LM · 1 000 steps · FineWeb-Edu · bfloat16)")
    print(f"{'═' * 70}")

    gen_results: dict[str, str] = {}
    for prompt in prompts:
        prompt_ids = tok.encode(prompt)
        ids  = generate_text(
            model, prompt_ids,
            max_new=200, seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            temperature=0.8, top_k=50,
        )
        text = tok.decode(ids)
        gen_results[prompt] = text

        print(f"\n  ▶ {prompt!r}")
        print(f"  {'─' * 66}")
        # Word-wrap at ~68 chars
        words, line = text.split(), "  "
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(line)
                line = "  " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

    # ── 7. Save results ────────────────────────────────────────────
    results = {
        "config": {
            "d_model":    cfg.d_model,
            "n_heads":    cfg.n_heads,
            "layers":     f"{cfg.n_layers_A}+DWA+{cfg.n_layers_B}",
            "seq_len":    cfg.seq_len,
            "N":          cfg.N,
            "D":          cfg.D,
            "r":          cfg.r,
            "k_max":      cfg.k_max,
            "batch_size": cfg.batch_size,
            "n_devices":  N_DEVICES,
            "max_steps":  cfg.max_steps,
            "vocab_size": cfg.vocab_size,
            "dtype":      "bfloat16",
        },
        "training_log":   log,
        "generated_text": gen_results,
        "final_val_ppl":  log[-1]["val_ppl"] if log else None,
    }
    with open("results_ultrafineweb.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'═' * 70}")
    print(f"[done] 1 000 steps complete.")
    print(f"[done] Final val_ppl  = {log[-1]['val_ppl']:.3f}")
    print(f"[done] Total time     = {log[-1]['elapsed']:.0f} s")
    print(f"[done] Results saved  → results_ultrafineweb.json")

    return model, log


if __name__ == "__main__":
    main()
