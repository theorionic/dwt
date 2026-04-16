"""Checkpoint save/load for Flax NNX models using pure-dict serialization.

Avoids Orbax asyncio issues by using numpy arrays directly.
Saves model state + optimizer state + training metadata for full resume.
"""

import os
import json
import pickle
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx


def _flatten_dict(d, prefix=""):
    items = []
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        else:
            items.append((key, np.array(v)))
    return items


def _unflatten_dict(items):
    result = {}
    for key, val in items:
        parts = key.split("/")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return result


def save_checkpoint(
    path: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int,
    alpha_ema: jax.Array | None = None,
    metadata: dict | None = None,
) -> None:
    os.makedirs(path, exist_ok=True)

    _, state = nnx.split(model)
    pure = nnx.to_pure_dict(state)
    flat = _flatten_dict(pure)
    for key, arr in flat:
        np.save(os.path.join(path, f"m_{key.replace('/', '_')}.npy"), arr)

    _, opt_state = nnx.split(optimizer)
    opt_pure = nnx.to_pure_dict(opt_state)
    opt_flat = _flatten_dict(opt_pure)
    for key, arr in opt_flat:
        np.save(os.path.join(path, f"o_{key.replace('/', '_')}.npy"), arr)

    if alpha_ema is not None:
        np.save(os.path.join(path, "alpha_ema.npy"), np.array(alpha_ema))

    meta = {
        "step": step,
        "num_model_arrays": len(flat),
        "num_opt_arrays": len(opt_flat),
    }
    if metadata:
        meta.update(metadata)
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f)


def load_checkpoint(
    path: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    alpha_ema_shape: tuple | None = None,
) -> tuple[int, jax.Array | None]:
    with open(os.path.join(path, "metadata.json")) as f:
        meta = json.load(f)
    step = meta["step"]

    _, state = nnx.split(model)
    model_pure = nnx.to_pure_dict(state)
    model_flat = _flatten_dict(model_pure)
    model_items = []
    for key, _ in model_flat:
        fname = f"m_{key.replace('/', '_')}.npy"
        model_items.append((key, jnp.array(np.load(os.path.join(path, fname)))))
    model_loaded = _unflatten_dict(model_items)
    nnx.replace_by_pure_dict(state, model_loaded)
    nnx.update(model, state)

    _, opt_state = nnx.split(optimizer)
    opt_pure = nnx.to_pure_dict(opt_state)
    opt_flat = _flatten_dict(opt_pure)
    opt_items = []
    for key, _ in opt_flat:
        fname = f"o_{key.replace('/', '_')}.npy"
        opt_items.append((key, jnp.array(np.load(os.path.join(path, fname)))))
    opt_loaded = _unflatten_dict(opt_items)
    nnx.replace_by_pure_dict(opt_state, opt_loaded)
    nnx.update(optimizer, opt_state)

    alpha_ema = None
    ema_path = os.path.join(path, "alpha_ema.npy")
    if alpha_ema_shape is not None and os.path.exists(ema_path):
        alpha_ema = jnp.array(np.load(ema_path))

    return step, alpha_ema
