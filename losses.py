"""
Auxiliary losses for DWA training.

L_total = L_task
        + λ_util   · L_util    (prevent dead vectors)
        + λ_div    · L_div     (prevent key collapse)
        + λ_norm   · L_norm    (prevent assembly explosion)
        + λ_sparse · L_sparse  (encourage sparse assembly weights)
"""

from typing import Tuple

import jax.numpy as jnp
import jax

from config import DWAConfig


def utilization_loss(alpha_ema: jax.Array, beta: float = 0.1) -> jax.Array:
    """
    Prevent dead vectors: L_util = -mean_i log(1 - exp(-β · EMA(α_i)))

    Vectors with low EMA usage contribute large loss, pushing the model to
    use them. EMA(α_i) is the exponential moving average of per-vector
    assembly weight, tracked externally during training.

    alpha_ema: [N]
    """
    eps = 1e-8
    ema = jnp.clip(alpha_ema, eps, None)
    return -jnp.mean(jnp.log(1.0 - jnp.exp(-beta * ema) + eps))


def diversity_loss(alpha: jax.Array, keys: jax.Array) -> jax.Array:
    """
    Prevent key collapse: penalize cosine similarity between retrieved keys.

    Uses *mean* utilization across all batch/sequence positions so memory
    is O(N²) regardless of batch size — safe for sequence models where
    alpha has shape [batch*seq_len, N].

    alpha: [any_batch, N]
    keys:  [N, S, d_k]
    """
    N, S, d_k = keys.shape

    # Mean utilisation across batch (and sequence) positions: [N]
    alpha_mean = jnp.mean(alpha, axis=0)

    # Flatten aspects and normalise: [N, S*d_k]
    keys_flat = keys.reshape((N, S * d_k))
    k_norm = keys_flat / (jnp.linalg.norm(keys_flat, axis=-1, keepdims=True) + 1e-8)

    # Pairwise cosine similarity: [N, N]
    sim = jnp.einsum("id,jd->ij", k_norm, k_norm)

    # Outer product of mean utilisation: [N, N]  — O(N²), batch-independent
    outer = jnp.outer(alpha_mean, alpha_mean)

    off_diag = 1.0 - jnp.eye(N)
    return jnp.sum(outer * sim * off_diag) / (N * (N - 1) + 1e-8)


def norm_loss(W_assembled: jax.Array, W_base: jax.Array) -> jax.Array:
    """
    Prevent assembly explosion: L_norm = mean_batch ||W_assembled - W_base||²_F

    W_assembled: [batch, d_B, d_A]
    W_base:      [d_B, d_A]
    """
    diff = W_assembled - W_base[None]
    return jnp.mean(jnp.sum(diff ** 2, axis=(-2, -1)))


def sparsity_loss(alpha: jax.Array) -> jax.Array:
    """
    Weight entropy: L_sparse = -mean_batch Σ_i α_i log(α_i)

    Encourages the model to concentrate assembly weight on few vectors
    rather than diffusing across the whole pool.

    alpha: [batch, N]
    """
    eps = 1e-8
    return jnp.mean(-jnp.sum(alpha * jnp.log(alpha + eps), axis=-1))


def compute_losses(
    task_loss: jax.Array,
    alpha: jax.Array,
    alpha_ema: jax.Array,
    W_assembled: jax.Array,
    W_base: jax.Array,
    keys: jax.Array,
    cfg: DWAConfig,
    aux_scale: float = 1.0,
) -> Tuple[jax.Array, dict]:
    """
    Combine task loss with all auxiliary losses.

    aux_scale controls the auxiliary loss contribution:
      0.0 → phase 1 (task loss only, warmup)
      1.0 → phase 2+ (full aux losses active)

    Returns:
        total: scalar loss for gradient computation
        breakdown: dict of individual loss values for logging
    """
    l_util   = utilization_loss(alpha_ema, cfg.beta_util)
    l_div    = diversity_loss(alpha, keys)
    l_norm   = norm_loss(W_assembled, W_base)
    l_sparse = sparsity_loss(alpha)

    aux = (
        cfg.lambda_util   * l_util
        + cfg.lambda_div  * l_div
        + cfg.lambda_norm * l_norm
        + cfg.lambda_sparse * l_sparse
    )

    total = task_loss + aux_scale * aux

    return total, {
        "task":   task_loss,
        "util":   l_util,
        "div":    l_div,
        "norm":   l_norm,
        "sparse": l_sparse,
        "aux":    aux,
        "total":  total,
    }
