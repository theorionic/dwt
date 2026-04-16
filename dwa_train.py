"""
DWA training utilities — everything needed to run the training loop.

This file complements dwa_model.py with:
  1. Data pipeline  (tokenizer + dataset streaming + batching)
  2. Multi-device sharding  (JAX mesh for TPU/GPU data-parallel)
  3. Checkpoint save/load  (includes alpha_ema — DWA-specific)
  4. Training loop  (wires three-phase schedule + alpha_ema)

If you already have a training framework, focus on sections 3 and 4 —
those are DWA-specific and will break silently if alpha_ema is not
properly initialized, updated, and checkpointed.

Dependencies: jax, flax[nnx], optax, numpy, tiktoken, datasets
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
import flax.nnx as nnx

from dwa_model import (
    LMConfig,
    DWALanguageModel,
    DenseLanguageModel,
    make_optimizer,
    make_dwa_step,
    make_dense_step,
    _eval_dwa_batch,
    _eval_dense_batch,
    to_bf16,
    get_lambda_sharp,
    get_aux_scale,
    update_ema,
    count_params,
    generate,
)


# ===========================================================================
# 1. Data pipeline
# ===========================================================================

def get_tokenizer():
    """GPT-2 BPE tokenizer via tiktoken. Returns a simple wrapper."""
    import tiktoken

    class Tokenizer:
        def __init__(self):
            self._enc = tiktoken.get_encoding("gpt2")
            self.vocab_size   = self._enc.n_vocab
            self.eos_token_id = self._enc.eot_token

        def encode(self, text: str) -> list[int]:
            return self._enc.encode(text)

        def decode(self, ids: list[int]) -> str:
            return self._enc.decode(ids)

    return Tokenizer()


def load_and_chunk(
    dataset_name: str,
    text_field: str   = "text",
    split: str        = "train",
    seq_len: int      = 512,
    val_fraction: float = 0.05,
    max_docs: int | None = None,
    cache_dir: str    = "data/tokenized",
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Stream a HuggingFace dataset, tokenize, split train/val.

    Returns (train_tokens, val_tokens, vocab_size) as flat int32 arrays.
    Caches to disk so subsequent runs skip the download.

    Parameters
    ----------
    dataset_name : HuggingFace dataset id, e.g. "HuggingFaceFW/fineweb-edu"
    text_field   : field in each example containing the text
    split        : HF dataset split name
    max_docs     : cap documents (None = full dataset)
    cache_dir    : where to save tokenized arrays
    """
    from datasets import load_dataset

    tok = get_tokenizer()

    tag = f"{dataset_name.replace('/', '_')}_{split}_{seq_len}"
    if max_docs:
        tag += f"_max{max_docs}"
    train_path = os.path.join(cache_dir, f"{tag}_train.npy")
    val_path   = os.path.join(cache_dir, f"{tag}_val.npy")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Loading cached data from {cache_dir}")
        train = np.load(train_path)
        val   = np.load(val_path)
        print(f"  Train: {len(train):,} tokens | Val: {len(val):,} tokens")
        return train, val, tok.vocab_size

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Streaming {dataset_name} [{split}]...")

    ds = load_dataset(dataset_name, split=split, streaming=True)
    all_tokens: list[int] = []
    n = 0

    for ex in ds:
        text = ex.get(text_field, "")
        if not text:
            continue
        all_tokens.extend(tok.encode(text))
        all_tokens.append(tok.eos_token_id)
        n += 1
        if n % 10_000 == 0:
            print(f"  {n:,} docs → {len(all_tokens):,} tokens")
        if max_docs and n >= max_docs:
            break

    arr   = np.array(all_tokens, dtype=np.int32)
    split_idx = int((1.0 - val_fraction) * len(arr))
    train = arr[:split_idx]
    val   = arr[split_idx:]

    np.save(train_path, train)
    np.save(val_path,   val)
    print(f"Saved: train={len(train):,} val={len(val):,} tokens")
    return train, val, tok.vocab_size


def get_batch(
    data: np.ndarray,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random batch of (x, y) from flat token array."""
    starts  = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    indices = starts[:, None] + np.arange(seq_len)   # [B, T]
    return data[indices], data[indices + 1]


# ===========================================================================
# 2. Multi-device sharding (JAX data-parallel)
# ===========================================================================
#
# Layout: batch sharded across "data" axis, model replicated on all devices.
# XLA auto-inserts AllReduce for gradient aggregation — no manual pmean needed.
#
# For model-parallel (needed at 1B+ if VRAM is tight), replace _replicated
# with a 2D mesh and add PartitionSpec to W_base / pool vectors.

_mesh            = js.Mesh(np.array(jax.devices()), ("data",))
_data_sharding   = js.NamedSharding(_mesh, js.PartitionSpec("data", None))
_replicated      = js.NamedSharding(_mesh, js.PartitionSpec())
N_DEVICES        = len(jax.devices())


def shard_batch(x: np.ndarray, y: np.ndarray):
    """Move numpy batch to devices — batch dimension sharded, rest replicated."""
    return jax.device_put(x, _data_sharding), jax.device_put(y, _data_sharding)


def replicate(model_or_opt: nnx.Module) -> nnx.Module:
    """Replicate model or optimizer state across all devices."""
    graph, state = nnx.split(model_or_opt)
    state = jax.device_put(state, _replicated)
    return nnx.merge(graph, state)


# ===========================================================================
# 3. Checkpoint save / load
# ===========================================================================
#
# WHY NOT ORBAX: Orbax has asyncio issues inside Kaggle/Colab environments.
# We use plain numpy save/load with a flat key namespace — verbose but reliable.
#
# CRITICAL (DWA-specific): alpha_ema must be saved alongside model state.
# It tracks per-vector utilization history; if you resume without it the
# utilization loss spikes for hundreds of steps until the EMA re-warms.

def _flatten(d: dict, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    out = []
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten(v, key))
        else:
            out.append((key, np.array(v)))
    return out


def _unflatten(items: list[tuple[str, np.ndarray]]) -> dict:
    root: dict = {}
    for key, val in items:
        parts = key.split("/")
        d = root
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return root


def save_checkpoint(
    path: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int,
    alpha_ema: jax.Array | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Save model + optimizer state + alpha_ema to a directory.

    Each array is saved as an individual .npy file so partial restores
    and manual inspection are easy.
    """
    os.makedirs(path, exist_ok=True)

    _, m_state = nnx.split(model)
    for key, arr in _flatten(nnx.to_pure_dict(m_state)):
        np.save(os.path.join(path, f"m_{key.replace('/', '_')}.npy"), arr)

    _, o_state = nnx.split(optimizer)
    for key, arr in _flatten(nnx.to_pure_dict(o_state)):
        np.save(os.path.join(path, f"o_{key.replace('/', '_')}.npy"), arr)

    if alpha_ema is not None:
        np.save(os.path.join(path, "alpha_ema.npy"), np.array(alpha_ema))

    meta = {"step": step}
    if metadata:
        meta.update(metadata)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(
    path: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    N: int | None = None,          # cfg.N — needed to restore alpha_ema
) -> tuple[int, jax.Array | None]:
    """
    Restore model + optimizer + alpha_ema from a checkpoint directory.

    Returns (step, alpha_ema).  alpha_ema is None if N is not provided or
    the file doesn't exist (first-time resume guard).
    """
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    step = meta["step"]

    def _restore(obj: nnx.Module, prefix: str) -> None:
        _, state = nnx.split(obj)
        pure  = nnx.to_pure_dict(state)
        flat  = _flatten(pure)
        items = []
        for key, _ in flat:
            fname = f"{prefix}{key.replace('/', '_')}.npy"
            items.append((key, jnp.array(np.load(os.path.join(path, fname)))))
        nnx.replace_by_pure_dict(state, _unflatten(items))
        nnx.update(obj, state)

    _restore(model,     "m_")
    _restore(optimizer, "o_")

    alpha_ema = None
    ema_path  = os.path.join(path, "alpha_ema.npy")
    if N is not None and os.path.exists(ema_path):
        alpha_ema = jnp.array(np.load(ema_path))

    return step, alpha_ema


def latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return the path of the most recent step_XXXXXX checkpoint, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("step_")]
    if not dirs:
        return None
    dirs.sort(key=lambda d: int(d.split("_")[1]))
    return os.path.join(ckpt_dir, dirs[-1])


# ===========================================================================
# 4. Training loops
# ===========================================================================

def evaluate_ppl(
    model,
    val_data: np.ndarray,
    cfg: LMConfig,
    rng: np.random.Generator,
    is_dwa: bool,
) -> float:
    """Run eval_steps batches, return perplexity."""
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = get_batch(val_data, cfg.seq_len, cfg.batch_size, rng)
        x_s, y_s = shard_batch(x, y)
        loss = _eval_dwa_batch(model, x_s, y_s) if is_dwa else _eval_dense_batch(model, x_s, y_s)
        losses.append(float(loss))
    return float(np.exp(np.mean(losses)))


def train_dwa(
    cfg: LMConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    ckpt_dir: str | None = None,
    resume: bool = False,
    seed: int = 0,
) -> tuple[DWALanguageModel, nnx.Optimizer, jax.Array, float, list[dict]]:
    """
    Full DWA training loop.

    Three-phase schedule is handled automatically via get_lambda_sharp /
    get_aux_scale. alpha_ema is initialized, updated each step, and saved
    in checkpoints.

    Returns (model, optimizer, alpha_ema, final_ppl, log).
    """
    print(f"\n{'=' * 64}")
    print("  Training DWA model")
    print(f"{'=' * 64}")

    # Build model + optimizer
    model = DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(seed)))
    model = to_bf16(model)          # cast BEFORE optimizer so Adam states are BF16
    model = replicate(model)

    opt = make_optimizer(model, cfg)
    opt = replicate(opt)

    step_fn   = make_dwa_step(cfg)
    alpha_ema = jax.device_put(jnp.ones(cfg.N) / cfg.N, _replicated)
    start     = 0

    print(f"  {count_params(model):,} params | {N_DEVICES} devices | batch={cfg.batch_size}")

    # Resume from checkpoint
    if resume and ckpt_dir:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt:
            start, loaded_ema = load_checkpoint(ckpt, model, opt, N=cfg.N)
            model     = replicate(model)
            opt       = replicate(opt)
            if loaded_ema is not None:
                alpha_ema = jax.device_put(loaded_ema, _replicated)
            start += 1
            print(f"  Resumed from step {start}")

    np_rng = np.random.default_rng(start)
    log: list[dict] = []
    t0 = time.perf_counter()

    for s in range(start, cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x_s, y_s = shard_batch(x, y)

        ls  = jnp.array(get_lambda_sharp(s, cfg))
        aux = jnp.array(get_aux_scale(s, cfg))

        total, bd, _alpha, alpha_ema = step_fn(
            model, opt, x_s, y_s,
            alpha_ema, ls, jnp.array(1.0), aux,
        )

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl = evaluate_ppl(model, val_data, cfg, np_rng, is_dwa=True)
            phase = (
                "warmup"   if s < cfg.phase1_end else
                "gate_on"  if s < cfg.phase2_end else
                "sharpen"
            )
            elapsed = time.perf_counter() - t0
            entry = {
                "step": s, "phase": phase,
                "train_ce": float(bd["ce"]),
                "val_ppl":  ppl,
                "lambda":   float(ls),
                "elapsed":  elapsed,
            }
            log.append(entry)
            print(
                f"  step {s:5d} [{phase:8s}]  "
                f"ce={entry['train_ce']:.3f}  ppl={ppl:7.2f}  "
                f"lambda={float(ls):.2f}  ({elapsed:.0f}s)"
            )

            if ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, f"step_{s:06d}")
                save_checkpoint(ckpt_path, model, opt, s, alpha_ema,
                                metadata={"phase": phase, "val_ppl": ppl})

    final_ppl = evaluate_ppl(model, val_data, cfg, np_rng, is_dwa=True)
    return model, opt, alpha_ema, final_ppl, log


def train_dense(
    cfg: LMConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    ckpt_dir: str | None = None,
    resume: bool = False,
    seed: int = 1,
) -> tuple[DenseLanguageModel, nnx.Optimizer, float, list[dict]]:
    """
    Dense baseline training loop.

    Returns (model, optimizer, final_ppl, log).
    """
    print(f"\n{'=' * 64}")
    print("  Training Dense baseline")
    print(f"{'=' * 64}")

    model = DenseLanguageModel(cfg, nnx.Rngs(params=jax.random.key(seed)))
    model = to_bf16(model)
    model = replicate(model)

    opt     = make_optimizer(model, cfg)
    opt     = replicate(opt)
    step_fn = make_dense_step()
    start   = 0

    print(f"  {count_params(model):,} params | {N_DEVICES} devices | batch={cfg.batch_size}")

    if resume and ckpt_dir:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt:
            start, _ = load_checkpoint(ckpt, model, opt)
            model = replicate(model)
            opt   = replicate(opt)
            start += 1
            print(f"  Resumed from step {start}")

    np_rng = np.random.default_rng(start + 99999)
    log: list[dict] = []
    t0 = time.perf_counter()

    for s in range(start, cfg.max_steps):
        x, y = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x_s, y_s = shard_batch(x, y)
        loss = step_fn(model, opt, x_s, y_s)

        if s % cfg.eval_every == 0 or s == cfg.max_steps - 1:
            ppl     = evaluate_ppl(model, val_data, cfg, np_rng, is_dwa=False)
            elapsed = time.perf_counter() - t0
            entry   = {"step": s, "train_ce": float(loss), "val_ppl": ppl, "elapsed": elapsed}
            log.append(entry)
            print(
                f"  step {s:5d}  ce={entry['train_ce']:.3f}  "
                f"ppl={ppl:7.2f}  ({elapsed:.0f}s)"
            )

            if ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, f"step_{s:06d}")
                save_checkpoint(ckpt_path, model, opt, s, metadata={"val_ppl": ppl})

    final_ppl = evaluate_ppl(model, val_data, cfg, np_rng, is_dwa=False)
    return model, opt, final_ppl, log
