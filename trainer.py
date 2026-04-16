"""Production training loop for DWA language model.

Features:
  - Multi-device data-parallel via JAX mesh sharding
  - Three-phase DWA training schedule (warmup → gate → sharpen)
  - Periodic validation perplexity
  - Checkpoint save/resume
  - Structured logging
"""

import os
import time

import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
import flax.nnx as nnx

from config import LMConfig
from lm_model import DWALanguageModel, DenseLanguageModel
from experiment import make_dwa_step, make_dense_step, make_optimizer, count_params
from experiment import _eval_dwa_batch, _eval_dense_batch
from train import get_lambda_sharp, get_aux_scale, update_ema
from checkpoint import save_checkpoint, load_checkpoint
from dataset import get_batch


_mesh = js.Mesh(jax.devices(), ("data",))
_data_sharding = js.NamedSharding(_mesh, js.PartitionSpec("data", None))
_replicate_sharding = js.NamedSharding(_mesh, js.PartitionSpec())
N_DEVICES = len(jax.devices())


def shard_batch(x: np.ndarray, y: np.ndarray):
    """Move numpy batch to devices with data-parallel sharding."""
    return jax.device_put(x, _data_sharding), jax.device_put(y, _data_sharding)


def replicate_model_state(model: nnx.Module):
    """Replicate model parameters across all devices."""
    graph, state = nnx.split(model)
    state = jax.device_put(state, _replicate_sharding)
    return nnx.merge(graph, state)


def replicate_optimizer_state(opt: nnx.Optimizer):
    """Replicate optimizer state across all devices."""
    graph, state = nnx.split(opt)
    state = jax.device_put(state, _replicate_sharding)
    return nnx.merge(graph, state)


def evaluate_ppl(model, val_data, cfg, np_rng, is_dwa):
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = get_batch(val_data, cfg.seq_len, cfg.batch_size, np_rng)
        x_sharded, y_sharded = shard_batch(x, y)
        loss = (
            _eval_dwa_batch(model, x_sharded, y_sharded) if is_dwa else _eval_dense_batch(model, x_sharded, y_sharded)
        )
        losses.append(float(loss))
    return float(np.exp(np.mean(losses)))


def train_dwa_model(
    cfg: LMConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    ckpt_dir: str | None = None,
    resume: bool = False,
):
    print(f"\n{'=' * 64}")
    print("  Training DWA model")
    print(f"{'=' * 64}")

    model = DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(0)))
    optimizer = make_optimizer(model, cfg)
    
    # Replicate model and optimizer state across TPU cores
    model = replicate_model_state(model)
    optimizer = replicate_optimizer_state(optimizer)
    
    step_fn = make_dwa_step(cfg)
    alpha_ema = jnp.ones(cfg.N) / cfg.N
    alpha_ema = jax.device_put(alpha_ema, _replicate_sharding)
    start_step = 0

    print(
        f"  {count_params(model):,} params | {N_DEVICES} devices | batch={cfg.batch_size}"
    )

    if resume and ckpt_dir:
        latest = _latest_checkpoint(ckpt_dir)
        if latest:
            start_step, alpha_ema = load_checkpoint(
                latest, model, optimizer, alpha_ema_shape=(cfg.N,)
            )
            # Ensure loaded state is sharded
            model = replicate_model_state(model)
            optimizer = replicate_optimizer_state(optimizer)
            alpha_ema = jax.device_put(alpha_ema, _replicate_sharding)
            start_step += 1
            print(f"  Resumed from step {start_step}")

    np_rng = np.random.default_rng(start_step)
    log: list[dict] = []
    t0 = time.perf_counter()

    for s in range(start_step, cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x_sharded, y_sharded = shard_batch(x, y)

        ls = jnp.array(get_lambda_sharp(s, cfg))
        aux = jnp.array(get_aux_scale(s, cfg))
        total, bd, alpha, alpha_ema = step_fn(
            model, optimizer, x_sharded, y_sharded, alpha_ema, ls, jnp.array(1.0), aux
        )

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl = evaluate_ppl(model, val_data, cfg, np_rng, True)
            phase = (
                "warmup"
                if s < cfg.phase1_end
                else ("gate_on" if s < cfg.phase2_end else "sharpen")
            )
            elapsed = time.perf_counter() - t0
            entry = {
                "step": s,
                "phase": phase,
                "train_ce": float(bd["ce"]),
                "val_ppl": ppl,
                "lambda": float(ls),
                "elapsed": elapsed,
            }
            log.append(entry)
            print(
                f"  step {s:5d} [{phase:8s}] ce={entry['train_ce']:.3f} ppl={ppl:7.2f} lambda={float(ls):.2f} ({elapsed:.0f}s)"
            )

            if ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, f"step_{s:06d}")
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    s,
                    alpha_ema,
                    {"phase": phase, "val_ppl": ppl},
                )

    final_ppl = evaluate_ppl(model, val_data, cfg, np_rng, True)
    return model, optimizer, alpha_ema, final_ppl, log


def train_dense_model(
    cfg: LMConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    ckpt_dir: str | None = None,
    resume: bool = False,
):
    print(f"\n{'=' * 64}")
    print("  Training Dense baseline")
    print(f"{'=' * 64}")

    model = DenseLanguageModel(cfg, nnx.Rngs(params=jax.random.key(1)))
    optimizer = make_optimizer(model, cfg)
    
    # Replicate model and optimizer state across TPU cores
    model = replicate_model_state(model)
    optimizer = replicate_optimizer_state(optimizer)
    
    step_fn = make_dense_step()
    start_step = 0

    print(
        f"  {count_params(model):,} params | {N_DEVICES} devices | batch={cfg.batch_size}"
    )

    if resume and ckpt_dir:
        latest = _latest_checkpoint(ckpt_dir)
        if latest:
            start_step, _ = load_checkpoint(latest, model, optimizer)
            # Ensure loaded state is sharded
            model = replicate_model_state(model)
            optimizer = replicate_optimizer_state(optimizer)
            start_step += 1
            print(f"  Resumed from step {start_step}")

    np_rng = np.random.default_rng(start_step + 10000)
    log: list[dict] = []
    t0 = time.perf_counter()

    for s in range(start_step, cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x_sharded, y_sharded = shard_batch(x, y)

        loss = step_fn(model, optimizer, x_sharded, y_sharded)

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl = evaluate_ppl(model, val_data, cfg, np_rng, False)
            elapsed = time.perf_counter() - t0
            entry = {
                "step": s,
                "train_ce": float(loss),
                "val_ppl": ppl,
                "elapsed": elapsed,
            }
            log.append(entry)
            print(
                f"  step {s:5d} ce={entry['train_ce']:.3f} ppl={ppl:7.2f} ({elapsed:.0f}s)"
            )

            if ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, f"step_{s:06d}")
                save_checkpoint(
                    ckpt_path, model, optimizer, s, metadata={"val_ppl": ppl}
                )

    final_ppl = evaluate_ppl(model, val_data, cfg, np_rng, False)
    return model, optimizer, final_ppl, log



def _latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.exists(ckpt_dir):
        return None
    dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("step_")]
    if not dirs:
        return None
    dirs.sort(key=lambda d: int(d.split("_")[1]))
    return os.path.join(ckpt_dir, dirs[-1])
