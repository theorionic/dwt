"""
DWA vs Dense transformer LM experiment on TinyShakespeare.

Trains both models for the same number of steps and reports:
  - validation perplexity curve
  - parameter count comparison
  - sample generated text
"""

import time
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import optax

from config import LMConfig
from data import get_batch
from lm_model import DWALanguageModel, DenseLanguageModel
from losses import utilization_loss, diversity_loss, sparsity_loss
from train import get_lambda_sharp, get_aux_scale, update_ema


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """
    logits:  [B, T, vocab_size]
    targets: [B, T]  integer token ids
    """
    B, T, V = logits.shape
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
    )
    return loss.mean()


def dwa_total_loss(
    logits: jax.Array,
    targets: jax.Array,
    alpha: jax.Array,       # [B*T, N]
    alpha_ema: jax.Array,   # [N]
    keys: jax.Array,        # [N, S, d_k]
    w_norm: jax.Array,      # scalar
    cfg: LMConfig,
    aux_scale: jax.Array,   # 0 in phase 1, 1 thereafter
) -> Tuple[jax.Array, dict]:
    ce   = cross_entropy(logits, targets)
    l_u  = utilization_loss(alpha_ema, cfg.beta_util)
    l_d  = diversity_loss(alpha, keys)
    l_s  = sparsity_loss(alpha)
    l_n  = w_norm  # already the mean Frobenius norm scalar

    aux  = (
        cfg.lambda_util   * l_u
        + cfg.lambda_div  * l_d
        + cfg.lambda_sparse * l_s
        + cfg.lambda_norm * l_n
    )
    total = ce + aux_scale * aux
    return total, {"ce": ce, "util": l_u, "div": l_d, "sparse": l_s, "norm": l_n}


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(model: nnx.Module, cfg: LMConfig) -> nnx.Optimizer:
    warmup = min(cfg.warmup_steps, max(1, cfg.max_steps - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=warmup,
        decay_steps=cfg.max_steps,
        end_value=cfg.lr * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ---------------------------------------------------------------------------
# JIT-compiled training steps
# ---------------------------------------------------------------------------

def make_dwa_step(cfg: LMConfig):
    @nnx.jit
    def step(
        model: DWALanguageModel,
        opt: nnx.Optimizer,
        x: jax.Array,
        y: jax.Array,
        alpha_ema: jax.Array,
        lambda_sharp: jax.Array,
        temperature: jax.Array,
        aux_scale: jax.Array,
    ) -> Tuple[jax.Array, dict, jax.Array, jax.Array]:
        def loss_fn(m: DWALanguageModel):
            logits, alpha, keys, w_norm = m(x, lambda_sharp, temperature)
            total, breakdown = dwa_total_loss(
                logits, y, alpha, alpha_ema, keys, w_norm, cfg, aux_scale
            )
            return total, (breakdown, alpha)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (total, (breakdown, alpha)), grads = grad_fn(model)
        opt.update(model, grads)

        # Fuse EMA update into the same JIT call
        batch_mean = jnp.mean(alpha.reshape(-1, cfg.N), axis=0)
        new_alpha_ema = cfg.ema_decay * alpha_ema + (1.0 - cfg.ema_decay) * batch_mean

        return total, breakdown, alpha, new_alpha_ema

    return step


def make_dense_step():
    @nnx.jit
    def step(
        model: DenseLanguageModel,
        opt: nnx.Optimizer,
        x: jax.Array,
        y: jax.Array,
    ) -> jax.Array:
        def loss_fn(m: DenseLanguageModel):
            logits = m(x)
            return cross_entropy(logits, y)

        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model)
        opt.update(model, grads)
        return loss

    return step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@nnx.jit
def _eval_dwa_batch(model: DWALanguageModel, x: jax.Array, y: jax.Array) -> jax.Array:
    logits, *_ = model(x, lambda_sharp=0.0)
    return cross_entropy(logits, y)


@nnx.jit
def _eval_dense_batch(model: DenseLanguageModel, x: jax.Array, y: jax.Array) -> jax.Array:
    return cross_entropy(model(x), y)


def evaluate(model, val_data: np.ndarray, cfg: LMConfig, rng: np.random.Generator, is_dwa: bool) -> float:
    """Returns perplexity on `eval_steps` random val batches."""
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = get_batch(val_data, cfg.seq_len, cfg.batch_size, rng)
        x, y = jnp.array(x), jnp.array(y)
        loss = _eval_dwa_batch(model, x, y) if is_dwa else _eval_dense_batch(model, x, y)
        losses.append(float(loss))
    return float(np.exp(np.mean(losses)))


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate(
    model,
    prompt_ids: list[int],
    max_new: int,
    vocab_size: int,
    is_dwa: bool,
    seq_len: int,
    temperature: float = 0.8,
) -> list[int]:
    """Autoregressive generation with temperature sampling.

    Always feeds the model a fixed-size [1, seq_len] context so JIT
    compiles once (no recompilation per token).  Positions before the
    prompt are zero-padded; the last valid position indexes the output.
    """
    rng_key = jax.random.key(42)
    pad_id = 0

    # Pre-compile: one call with full seq_len so JIT caches the graph
    dummy = jnp.zeros((1, seq_len), dtype=jnp.int32)
    if is_dwa:
        model(dummy, lambda_sharp=0.0)
    else:
        model(dummy)

    ids = list(prompt_ids)
    for _ in range(max_new):
        cur = ids[-seq_len:]
        padded = [pad_id] * (seq_len - len(cur)) + cur
        context = jnp.array(padded, dtype=jnp.int32)[None]
        pos = min(len(cur) - 1, seq_len - 1)

        if is_dwa:
            logits, *_ = model(context, lambda_sharp=0.0)
        else:
            logits = model(context)

        last_logits = logits[0, pos, :] / temperature
        rng_key, sub = jax.random.split(rng_key)
        next_id = int(jax.random.categorical(sub, last_logits))
        ids.append(next_id)

    return ids


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def count_params(model: nnx.Module) -> int:
    _, state = nnx.split(model)
    return sum(x.size for x in jax.tree_util.tree_leaves(state))


def train_dwa(
    cfg: LMConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    seed: int = 0,
) -> Tuple[DWALanguageModel, list[dict]]:
    rng_model = jax.random.key(seed)
    model = DWALanguageModel(cfg, nnx.Rngs(params=rng_model))
    opt   = make_optimizer(model, cfg)
    step  = make_dwa_step(cfg)

    alpha_ema = jnp.ones(cfg.N) / cfg.N
    np_rng    = np.random.default_rng(seed)
    log: list[dict] = []

    print(f"  DWA model — {count_params(model):,} parameters")
    t0 = time.perf_counter()

    for s in range(cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)

        ls  = jnp.array(get_lambda_sharp(s, cfg))       # phase schedule
        aux = jnp.array(get_aux_scale(s, cfg))

        total, breakdown, alpha = step(
            model, opt, x, y,
            alpha_ema, ls, jnp.array(cfg.T_temperature), aux,
        )
        alpha_ema = update_ema(alpha_ema, alpha, cfg.ema_decay)

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl = evaluate(model, val_data, cfg, np_rng, is_dwa=True)
            phase = "warmup" if s < cfg.phase1_end else ("gate_on" if s < cfg.phase2_end else "sharpen")
            elapsed = time.perf_counter() - t0
            entry = {
                "step": s, "phase": phase,
                "train_ce": float(breakdown["ce"]),
                "val_ppl": ppl,
                "lambda": float(ls),
                "elapsed": elapsed,
            }
            log.append(entry)
            print(
                f"  step {s:5d} [{phase:8s}]  "
                f"train_ce={entry['train_ce']:.3f}  "
                f"val_ppl={ppl:7.2f}  "
                f"lambda={entry['lambda']:.2f}  "
                f"({elapsed:.0f}s)"
            )

    return model, log


def train_dense(
    cfg: LMConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    seed: int = 1,
) -> Tuple[DenseLanguageModel, list[dict]]:
    rng_model = jax.random.key(seed)
    model = DenseLanguageModel(cfg, nnx.Rngs(params=rng_model))
    opt   = make_optimizer(model, cfg)
    step  = make_dense_step()

    np_rng = np.random.default_rng(seed)
    log: list[dict] = []

    print(f"  Dense model — {count_params(model):,} parameters")
    t0 = time.perf_counter()

    for s in range(cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x, y = jnp.array(x), jnp.array(y)

        loss = step(model, opt, x, y)

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl     = evaluate(model, val_data, cfg, np_rng, is_dwa=False)
            elapsed = time.perf_counter() - t0
            entry   = {"step": s, "train_ce": float(loss), "val_ppl": ppl, "elapsed": elapsed}
            log.append(entry)
            print(
                f"  step {s:5d}              "
                f"train_ce={entry['train_ce']:.3f}  "
                f"val_ppl={ppl:7.2f}  "
                f"({elapsed:.0f}s)"
            )

    return model, log


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: LMConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    itos: dict,
    stoi: dict,
) -> None:
    prompt_text = "ROMEO:"
    prompt_ids  = [stoi[c] for c in prompt_text]

    print("\n" + "=" * 64)
    print("  DWA vs Dense LM — TinyShakespeare character-level")
    print(f"  d_model={cfg.d_model}  n_layers={cfg.n_layers_A}+DWA+{cfg.n_layers_B}"
          f"  seq={cfg.seq_len}  steps={cfg.max_steps}")
    print(f"  DWA: N={cfg.N}  D={cfg.D}  r={cfg.r}  S={cfg.S}")
    print("=" * 64)

    print("\n[1/2] Training DWA model ...")
    dwa_model, dwa_log = train_dwa(cfg, train_data, val_data)

    print("\n[2/2] Training Dense baseline ...")
    dense_model, dense_log = train_dense(cfg, train_data, val_data)

    # ---- Summary ----
    dwa_final_ppl   = dwa_log[-1]["val_ppl"]
    dense_final_ppl = dense_log[-1]["val_ppl"]
    dwa_params      = count_params(dwa_model)
    dense_params    = count_params(dense_model)

    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  {'':20s}  {'DWA':>10s}  {'Dense':>10s}")
    print(f"  {'Val perplexity':20s}  {dwa_final_ppl:>10.2f}  {dense_final_ppl:>10.2f}")
    print(f"  {'Parameters':20s}  {dwa_params:>10,}  {dense_params:>10,}")
    print(f"  {'Train steps':20s}  {cfg.max_steps:>10,}  {cfg.max_steps:>10,}")

    delta = dense_final_ppl - dwa_final_ppl
    if delta > 0:
        print(f"\n  DWA beats dense by {delta:.2f} ppl points.")
    elif delta < 0:
        print(f"\n  Dense beats DWA by {-delta:.2f} ppl points.")
    else:
        print("\n  Models are equivalent.")

    # ---- Sample text ----
    print("\n" + "=" * 64)
    print("  GENERATED TEXT  (prompt: \"ROMEO:\")")
    print("=" * 64)

    for label, model, is_dwa in [("DWA", dwa_model, True), ("Dense", dense_model, False)]:
        ids   = generate(model, prompt_ids, max_new=100, vocab_size=cfg.vocab_size,
                         is_dwa=is_dwa, seq_len=cfg.seq_len)
        text  = "".join(itos[i] for i in ids)
        print(f"\n  [{label}]\n  {text}\n")
