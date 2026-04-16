"""
Dynamic Weight Assembly (DWA) — complete model in a single file.

Architecture overview
---------------------
  tokens → embed → Transformer Part A (n_layers_A blocks)
         → DWA Middle (pool retrieval → per-token weight assembly)
         → Transformer Part B (n_layers_B blocks) → lm_head → logits

DWA middle, per token
  h_A  →  MultiAspectRetrieval(pool)  →  alpha [B*T, N]
  alpha, pool  →  assemble()  →  W_assembled [B*T, d, d],  b_assembled
  h_A + gamma * (h_A @ W_assembled.T + b_assembled)  →  LayerNorm  →  h_mid

Pool vector layout  (each vector is D-dimensional)
  [  U_i : d*r  |  V_i : r*d  |  b_i : d  |  ... free for key projection ... ]
  W_i = U_i @ V_i  is a rank-r d×d matrix

Three-phase training schedule
  Phase 1  (0 → phase1_end)        : lambda_sharp=0  (pure softmax), no aux losses
  Phase 2  (phase1_end → phase2_end): lambda ramps 0→5, aux losses active
  Phase 3  (phase2_end → ∞)        : lambda ramps 5→10, cosine LR decay

Auxiliary losses
  L_util   — prevent dead pool vectors
  L_div    — prevent key collapse (pool vectors becoming identical)
  L_norm   — prevent assembly explosion (||W_delta||_F)
  L_sparse — encourage sparse alpha (entropy regularisation)

Disk inference
  VectorPoolStore  — mmap-backed pool + flat MIPS index
  DWAInferenceModel — full generation with per-token dynamic fetching

Dependencies: jax, flax[nnx], optax, numpy
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import optax


# ===========================================================================
# 1. Configuration
# ===========================================================================

@dataclass
class DWAConfig:
    """
    Core DWA hyperparameters (used by the pool / retrieval / middle components).

    Constraint: D >= d_model * r + r * d_model + d_model
      For d=256, r=4:   D >= 2304  → use D=4096
      For d=512, r=8:   D >= 8704  → use D=16384
      For d=1024, r=16: D >= 34816 → use D=65536
    """

    # --- Pool ---
    N: int = 512          # number of pool vectors
    D: int = 2048         # pool vector dimension

    # --- Layer dimensions (must be symmetric: d_A == d_B) ---
    d_A: int = 256        # Part A output / middle input dimension
    d_B: int = 256        # middle output / Part B input dimension
    r: int = 4            # assembly rank (polysemantic slots per vector)

    # --- Retrieval ---
    S: int = 2            # number of retrieval aspects
    d_k: int = 64         # key/query dimension per aspect
    k_max: int = 16       # top-k vectors used in assembly (training == inference)

    # --- Shared with LM ---
    d_model: int = 256
    n_layers_A: int = 4
    n_layers_B: int = 4

    # --- DWA hyperparameters ---
    gamma_init: float = 0.01    # LoRA-style residual scale, starts near-zero
    tau_init: float = 0.0       # sigmoid gate threshold
    T_temperature: float = 1.0

    # --- Sharpness schedule ---
    lambda_sharp_phase2_end: float = 5.0
    lambda_sharp_final: float = 10.0

    # --- Auxiliary loss weights ---
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.01
    beta_util: float = 0.1     # scale in utilization loss
    ema_decay: float = 0.99    # EMA decay for per-vector utilization tracking

    # --- Phase boundaries (steps) ---
    phase1_end: int = 1_000
    phase2_end: int = 10_000

    def __post_init__(self) -> None:
        assert self.d_A == self.d_B, "d_A must equal d_B for the residual connection"
        required = self.d_B * self.r + self.r * self.d_A + self.d_B
        assert self.D >= required, (
            f"D={self.D} < {required} (d_B*r + r*d_A + d_B). "
            f"Minimum D for d={self.d_A}, r={self.r}: {required}"
        )


@dataclass
class LMConfig:
    """
    Full language model config — extends DWAConfig with transformer + training knobs.

    Quick scale guide
    -----------------
    ~30M  params : d_model=256,  n_layers=4+4,  N=256,   D=4096,  r=4
    ~120M params : d_model=512,  n_layers=6+6,  N=1024,  D=16384, r=8
    ~300M params : d_model=768,  n_layers=8+8,  N=4096,  D=32768, r=8
    ~1B   params : d_model=1024, n_layers=12+12, N=65536, D=65536, r=16
      (pool alone is ~8GB at fp32; store on disk with VectorPoolStore)
    """

    # --- Tokenisation ---
    vocab_size: int = 50257      # GPT-2 BPE default

    # --- Transformer ---
    d_model: int = 256
    n_heads: int = 8
    n_layers_A: int = 4          # blocks *before* DWA middle
    n_layers_B: int = 4          # blocks *after*  DWA middle
    seq_len: int = 512

    # --- DWA pool ---
    N: int = 256
    D: int = 4096                # must satisfy D >= d_model*r + r*d_model + d_model
    r: int = 4
    S: int = 2
    d_k: int = 64
    k_max: int = 16

    # --- DWA forward-pass hyperparameters ---
    gamma_init: float = 0.01
    tau_init: float = 0.0
    T_temperature: float = 1.0
    lambda_sharp_phase2_end: float = 5.0
    lambda_sharp_final: float = 10.0

    # --- Auxiliary loss weights ---
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.01
    beta_util: float = 0.1
    ema_decay: float = 0.99

    # --- Training schedule ---
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 20_000
    eval_every: int = 500
    eval_steps: int = 50
    phase1_end: int = 1_000
    phase2_end: int = 10_000
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        required = self.d_model * self.r + self.r * self.d_model + self.d_model
        assert self.D >= required, (
            f"D={self.D} < {required} required "
            f"(d_model={self.d_model}, r={self.r}). "
            f"Set D >= {required}."
        )

    def to_dwa_config(self) -> DWAConfig:
        """Extract DWAConfig for pool/retrieval/middle submodules."""
        return DWAConfig(
            N=self.N, D=self.D,
            d_A=self.d_model, d_B=self.d_model,
            r=self.r, S=self.S, d_k=self.d_k, k_max=self.k_max,
            d_model=self.d_model,
            n_layers_A=self.n_layers_A, n_layers_B=self.n_layers_B,
            gamma_init=self.gamma_init, tau_init=self.tau_init,
            T_temperature=self.T_temperature,
            lambda_sharp_phase2_end=self.lambda_sharp_phase2_end,
            lambda_sharp_final=self.lambda_sharp_final,
            lambda_util=self.lambda_util, lambda_div=self.lambda_div,
            lambda_norm=self.lambda_norm, lambda_sparse=self.lambda_sparse,
            beta_util=self.beta_util, ema_decay=self.ema_decay,
            phase1_end=self.phase1_end, phase2_end=self.phase2_end,
        )


# ===========================================================================
# 2. Auxiliary losses
# ===========================================================================

def utilization_loss(alpha_ema: jax.Array, beta: float = 0.1) -> jax.Array:
    """
    Prevent dead pool vectors.

    L_util = -mean_i  log(1 - exp(-beta * EMA(alpha_i)))

    Vectors with low EMA usage (near-dead) contribute large loss,
    pushing gradients to use them. alpha_ema is updated externally
    each step using update_ema().

    alpha_ema : [N]
    """
    eps = 1e-8
    ema = jnp.clip(alpha_ema, eps, None)
    return -jnp.mean(jnp.log(1.0 - jnp.exp(-beta * ema) + eps))


def diversity_loss(alpha: jax.Array, keys: jax.Array) -> jax.Array:
    """
    Prevent key collapse — penalise cosine similarity between retrieved keys.

    Weighted by mean utilization across the batch so high-use pairs are
    penalised more. Memory is O(N²), independent of batch size.

    alpha : [batch_or_B*T, N]
    keys  : [N, S, d_k]
    """
    N, S, d_k = keys.shape
    alpha_mean = jnp.mean(alpha, axis=0)                          # [N]
    keys_flat  = keys.reshape(N, S * d_k)
    k_norm     = keys_flat / (jnp.linalg.norm(keys_flat, axis=-1, keepdims=True) + 1e-8)
    sim        = jnp.einsum("id,jd->ij", k_norm, k_norm)         # [N, N]
    outer      = jnp.outer(alpha_mean, alpha_mean)                # [N, N]
    off_diag   = 1.0 - jnp.eye(N)
    return jnp.sum(outer * sim * off_diag) / (N * (N - 1) + 1e-8)


def norm_loss(W_assembled: jax.Array, W_base: jax.Array) -> jax.Array:
    """
    Prevent assembly explosion.

    L_norm = mean_batch  ||W_assembled - W_base||²_F

    W_assembled : [batch, d_B, d_A]
    W_base      : [d_B, d_A]
    """
    diff = W_assembled - W_base[None]
    return jnp.mean(jnp.sum(diff ** 2, axis=(-2, -1)))


def sparsity_loss(alpha: jax.Array) -> jax.Array:
    """
    Encourage sparse assembly weights (concentrate on fewer vectors).

    L_sparse = -mean_batch  sum_i alpha_i log(alpha_i)   (negative entropy)

    Higher entropy → more uniform → larger loss. Forces the model to pick
    a small set of relevant vectors rather than diffusing over the whole pool.

    alpha : [batch, N]
    """
    eps = 1e-8
    return jnp.mean(-jnp.sum(alpha * jnp.log(alpha + eps), axis=-1))


def compute_losses(
    task_loss: jax.Array,
    alpha: jax.Array,        # [B*T, N]
    alpha_ema: jax.Array,    # [N]
    W_assembled: jax.Array,  # [B*T, d_B, d_A]  — or pass w_norm scalar directly
    W_base: jax.Array,       # [d_B, d_A]
    keys: jax.Array,         # [N, S, d_k]
    cfg: DWAConfig,
    aux_scale: jax.Array,    # 0.0 in phase 1, 1.0 phase 2+
) -> Tuple[jax.Array, dict]:
    """
    Combine task CE loss with all auxiliary losses.

    Returns total scalar and a dict of named components for logging.
    """
    l_util   = utilization_loss(alpha_ema, cfg.beta_util)
    l_div    = diversity_loss(alpha, keys)
    l_norm   = norm_loss(W_assembled, W_base)
    l_sparse = sparsity_loss(alpha)

    aux = (
          cfg.lambda_util   * l_util
        + cfg.lambda_div    * l_div
        + cfg.lambda_norm   * l_norm
        + cfg.lambda_sparse * l_sparse
    )
    total = task_loss + aux_scale * aux

    return total, {
        "ce":     task_loss,
        "util":   l_util,
        "div":    l_div,
        "norm":   l_norm,
        "sparse": l_sparse,
        "aux":    aux,
        "total":  total,
    }


# ===========================================================================
# 3. Training schedule helpers
# ===========================================================================

def get_lambda_sharp(step: int, cfg: DWAConfig) -> float:
    """
    Sigmoid sharpness lambda for the current training step.

    Phase 1: lambda=0       → sigmoid=0.5 everywhere → pure softmax over top-k
    Phase 2: lambda: 0→5    → gate gradually sharpens
    Phase 3: lambda: 5→10   → final sharpening
    """
    if step < cfg.phase1_end:
        return 0.0
    if step < cfg.phase2_end:
        t = (step - cfg.phase1_end) / (cfg.phase2_end - cfg.phase1_end)
        return t * cfg.lambda_sharp_phase2_end
    t = min(1.0, (step - cfg.phase2_end) / cfg.phase2_end)
    return cfg.lambda_sharp_phase2_end + t * (cfg.lambda_sharp_final - cfg.lambda_sharp_phase2_end)


def get_aux_scale(step: int, cfg: DWAConfig) -> float:
    """Auxiliary losses are disabled (scale=0) during phase 1 warmup."""
    return 0.0 if step < cfg.phase1_end else 1.0


def update_ema(
    alpha_ema: jax.Array,   # [N]
    alpha: jax.Array,       # [batch_or_B*T, N]
    decay: float,
) -> jax.Array:
    """Update per-vector utilization EMA with the current batch mean."""
    batch_mean = jnp.mean(alpha.reshape(-1, alpha.shape[-1]), axis=0)  # [N]
    return decay * alpha_ema + (1.0 - decay) * batch_mean


# ===========================================================================
# 4. Core DWA model components
# ===========================================================================

class VectorPool(nnx.Module):
    """
    Learnable pool of N D-dimensional vectors.

    Each vector encodes both:
      - Low-rank matrix factors  (first d_B*r + r*d_A + d_B elements)
      - Key information for retrieval  (all D elements projected by W_K)

    Small init (stddev=0.01) so the model starts near identity: the assembly
    contribution is near-zero and gradient flows cleanly from task loss.
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.N = cfg.N
        self.D = cfg.D
        self.value = nnx.Param(
            nnx.initializers.normal(stddev=0.01)(rngs.params(), (cfg.N, cfg.D))
        )


class MultiAspectRetrieval(nnx.Module):
    """
    S-aspect cosine-similarity retrieval with sigmoid gating.

    Key design: W_K projects the *full* pool vector (matrix factor elements
    included), coupling the retrieval gradient with the assembly gradient —
    both update the same v_i parameters, creating a self-reinforcing signal.

    Top-k masking:
      Scores are masked to k_max *before* gating so training exactly matches
      inference (which fetches only k_max vectors via MIPS). Without this,
      the model learns to rely on vectors ranked just outside k_max that
      MIPS will never return.

    Phase 1 (lambda_sharp=0):
      sigmoid(0 * anything) = 0.5 for every kept vector → cancels in
      normalisation → equivalent to pure softmax over top-k. No sparse bias.

    Phase 2+ (lambda_sharp > 0):
      Gate sharpens → vectors below tau get near-zero weight → sparse alpha.
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        lecun = nnx.initializers.lecun_normal()
        self.W_Q = nnx.Param(lecun(rngs.params(), (cfg.S, cfg.d_A, cfg.d_k)))  # [S, d_A, d_k]
        self.W_K = nnx.Param(lecun(rngs.params(), (cfg.S, cfg.D, cfg.d_k)))    # [S, D, d_k]
        self.aspect_logits = nnx.Param(jnp.zeros(cfg.S))                        # [S]
        self.tau = nnx.Param(jnp.array(cfg.tau_init))                           # scalar

    def __call__(
        self,
        z: jax.Array,            # [batch, d_A]  — query from Part A
        pool_vectors: jax.Array, # [N, D]
        lambda_sharp: jax.Array, # scalar (JAX array for JIT stability)
        temperature: jax.Array,  # scalar
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Returns
        -------
        alpha  : [batch, N]       normalised assembly weights
        scores : [batch, N]       raw aspect-weighted cosine similarities
        keys   : [N, S, d_k]     aspect keys (for diversity loss)
        """
        # Aspect queries: [batch, S, d_k]
        queries = jnp.einsum("bd,sdq->bsq", z, self.W_Q.value)

        # Aspect keys for all pool vectors: [N, S, d_k]
        keys = jnp.einsum("nd,sdq->nsq", pool_vectors, self.W_K.value)

        # Normalise for cosine similarity
        q_norm = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        k_norm = keys    / (jnp.linalg.norm(keys,    axis=-1, keepdims=True) + 1e-8)

        # Per-aspect cosine similarities: [batch, N, S]
        sims = jnp.einsum("bsq,nsq->bns", q_norm, k_norm)

        # Aspect-weighted scores: [batch, N]
        w      = jax.nn.softmax(self.aspect_logits.value)          # [S]
        scores = jnp.einsum("bns,s->bn", sims, w)

        # Top-k mask: restrict to k_max vectors so training == inference.
        # Use a large *finite* negative (not -inf) to avoid 0 * -inf = NaN
        # when lambda_sharp=0 in phase 1.
        if self.cfg.k_max < self.cfg.N:
            _, top_idx  = jax.lax.top_k(scores, self.cfg.k_max)   # [batch, k_max]
            batch_idx   = jnp.arange(scores.shape[0])[:, None]
            mask        = jnp.zeros(scores.shape, dtype=jnp.bool_)
            mask        = mask.at[batch_idx, top_idx].set(True)
            scores      = jnp.where(mask, scores, -1e9)

        # Sigmoid gate
        g         = jax.nn.sigmoid(lambda_sharp * (scores - self.tau.value))
        alpha_raw = g * jnp.exp(scores / temperature)
        alpha     = alpha_raw / (jnp.sum(alpha_raw, axis=-1, keepdims=True) + 1e-8)

        return alpha, scores, keys


class DWAMiddleLayer(nnx.Module):
    """
    Assemble a per-token weight matrix from pool vectors and apply it.

    Pool vector layout (first elements; rest is free for key projection):
      [ U_i : d_B*r | V_i : r*d_A | b_i : d_B | ... ]

    Assembly:
      W_assembled = W_base + Σ_i alpha_i  (U_i @ V_i)
      b_assembled = b_base + Σ_i alpha_i  b_i

    Forward:
      h_mid = LayerNorm( h_A + gamma * (h_A @ W_assembled.T + b_assembled) )

    gamma is a learnable scalar initialised near-zero (LoRA-style ramp-up).
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg   = cfg
        self._u_end = cfg.d_B * cfg.r
        self._v_end = self._u_end + cfg.r * cfg.d_A
        self._b_end = self._v_end + cfg.d_B

        self.W_base    = nnx.Param(nnx.initializers.normal(0.01)(rngs.params(), (cfg.d_B, cfg.d_A)))
        self.b_base    = nnx.Param(jnp.zeros(cfg.d_B))
        self.gamma     = nnx.Param(jnp.array(cfg.gamma_init))
        self.layer_norm = nnx.LayerNorm(cfg.d_A, rngs=rngs)

    def assemble(
        self,
        pool_vectors: jax.Array,  # [N, D]
        alpha: jax.Array,         # [batch, N]
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Returns
        -------
        W_assembled : [batch, d_B, d_A]
        b_assembled : [batch, d_B]
        """
        cfg = self.cfg
        U    = pool_vectors[:, :self._u_end].reshape(cfg.N, cfg.d_B, cfg.r)  # [N, d_B, r]
        V    = pool_vectors[:, self._u_end:self._v_end].reshape(cfg.N, cfg.r, cfg.d_A)  # [N, r, d_A]
        bias = pool_vectors[:, self._v_end:self._b_end]                                 # [N, d_B]

        UV          = jnp.einsum("ndr,nre->nde", U, V)               # [N, d_B, d_A]
        W_delta     = jnp.einsum("bn,nde->bde", alpha, UV)           # [batch, d_B, d_A]
        b_delta     = jnp.einsum("bn,nd->bd",   alpha, bias)         # [batch, d_B]
        W_assembled = self.W_base.value[None] + W_delta
        b_assembled = self.b_base.value[None] + b_delta
        return W_assembled, b_assembled

    def __call__(
        self,
        h_A: jax.Array,          # [batch, d_A]
        W_assembled: jax.Array,  # [batch, d_B, d_A]
        b_assembled: jax.Array,  # [batch, d_B]
    ) -> jax.Array:              # [batch, d_A]
        h_transformed = jnp.einsum("ba,bca->bc", h_A, W_assembled) + b_assembled
        return self.layer_norm(h_A + self.gamma.value * h_transformed)


# ===========================================================================
# 5. Transformer building blocks
# ===========================================================================

class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention (pre-norm style, no dropout)."""

    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs) -> None:
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_qkv   = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
        self.W_out   = nnx.Linear(d_model, d_model,     use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [B, T, d_model]
        B, T, D = x.shape
        H, dh   = self.n_heads, self.d_head

        qkv = self.W_qkv(x)                          # [B, T, 3*d_model]
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def to_heads(z: jax.Array) -> jax.Array:
            return z.reshape(B, T, H, dh).transpose((0, 2, 1, 3))  # [B, H, T, dh]

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        scores  = jnp.einsum("bhtd,bhsd->bhts", q, k) * (dh ** -0.5)
        causal  = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores  = jnp.where(causal[None, None], scores, -1e9)
        attn    = jax.nn.softmax(scores, axis=-1)     # [B, H, T, T]

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)  # [B, H, T, dh]
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, D)
        return self.W_out(out)


class FeedForward(nnx.Module):
    """4x expansion FFN with GELU."""

    def __init__(self, d_model: int, rngs: nnx.Rngs) -> None:
        self.fc1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x)))


class TransformerBlock(nnx.Module):
    """Pre-norm transformer block: x + attn(ln(x)),  x + ffn(ln(x))."""

    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs) -> None:
        self.ln1  = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = CausalSelfAttention(d_model, n_heads, rngs)
        self.ln2  = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn  = FeedForward(d_model, rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ===========================================================================
# 6. Language models
# ===========================================================================

class DWALanguageModel(nnx.Module):
    """
    Transformer LM with a DWA dynamic assembly middle layer.

    Forward pass
    ------------
    token ids → embed → blocks_A → ln_mid
              → [flatten B*T] → DWA middle → [unflatten B,T]
              → blocks_B → ln_f → head → logits

    Returns
    -------
    logits  : [B, T, vocab_size]
    alpha   : [B*T, N]           per-token assembly weights (for aux losses)
    keys    : [N, S, d_k]        retrieval aspect keys      (for diversity loss)
    w_norm  : scalar             mean ||W_delta||²_F         (for norm loss)
    """

    def __init__(self, cfg: LMConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        dwa = cfg.to_dwa_config()

        # Embeddings
        self.tok_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.pos_emb = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (cfg.seq_len, cfg.d_model))
        )

        # Part A
        self.blocks_A = nnx.List(
            [TransformerBlock(cfg.d_model, cfg.n_heads, rngs) for _ in range(cfg.n_layers_A)]
        )
        self.ln_mid = nnx.LayerNorm(cfg.d_model, rngs=rngs)

        # DWA middle
        self.pool      = VectorPool(dwa, rngs)
        self.retrieval = MultiAspectRetrieval(dwa, rngs)
        self.middle    = DWAMiddleLayer(dwa, rngs)

        # Part B
        self.blocks_B = nnx.List(
            [TransformerBlock(cfg.d_model, cfg.n_heads, rngs) for _ in range(cfg.n_layers_B)]
        )

        # Output head
        self.ln_f = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,             # [B, T]  token ids
        lambda_sharp: jax.Array = jnp.array(0.0),
        temperature: jax.Array  = jnp.array(1.0),
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        B, T = x.shape

        # Embeddings
        h = self.tok_emb(x) + self.pos_emb.value[:T]   # [B, T, d_model]

        # Part A
        for block in self.blocks_A:
            h = block(h)
        h = self.ln_mid(h)

        # DWA middle — operate on flattened token dimension
        h_flat    = h.reshape(B * T, self.cfg.d_model)
        pool_vecs = self.pool.value                      # [N, D]

        alpha, _scores, keys = self.retrieval(h_flat, pool_vecs, lambda_sharp, temperature)

        W_assembled, b_assembled = self.middle.assemble(pool_vecs, alpha)

        # Inline W_delta norm (scalar) — avoids returning the full [B*T, d, d] tensor
        W_delta = W_assembled - self.middle.W_base.value[None]
        w_norm  = jnp.mean(jnp.sum(W_delta ** 2, axis=(-2, -1)))

        h_flat = self.middle(h_flat, W_assembled, b_assembled)
        h      = h_flat.reshape(B, T, self.cfg.d_model)

        # Part B
        for block in self.blocks_B:
            h = block(h)

        logits = self.head(self.ln_f(h))                # [B, T, vocab_size]
        return logits, alpha, keys, w_norm


class DenseLanguageModel(nnx.Module):
    """
    Standard GPT-style transformer — dense baseline for comparison.

    Depth: n_layers_A + 1 + n_layers_B  (matches DWA total layer count).
    Parameter count is *lower* than DWA (no pool or key projections).
    """

    def __init__(self, cfg: LMConfig, rngs: nnx.Rngs) -> None:
        self.cfg    = cfg
        n_layers    = cfg.n_layers_A + 1 + cfg.n_layers_B

        self.tok_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.pos_emb = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (cfg.seq_len, cfg.d_model))
        )
        self.blocks = nnx.List(
            [TransformerBlock(cfg.d_model, cfg.n_heads, rngs) for _ in range(n_layers)]
        )
        self.ln_f = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Returns logits: [B, T, vocab_size]"""
        h = self.tok_emb(x) + self.pos_emb.value[:x.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


# ===========================================================================
# 7. Training utilities
# ===========================================================================

def count_params(model: nnx.Module) -> int:
    """Total number of learnable scalar parameters."""
    _, state = nnx.split(model)
    return sum(x.size for x in jax.tree_util.tree_leaves(state))


def make_optimizer(model: nnx.Module, cfg: LMConfig) -> nnx.Optimizer:
    """
    AdamW with linear warmup + cosine decay.

    Uses optax.chain so you can easily swap in a different schedule or
    add gradient clipping without touching the model.
    """
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


def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """
    logits  : [B, T, vocab_size]
    targets : [B, T]  integer token ids
    """
    B, T, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
    ).mean()


def make_dwa_step(cfg: LMConfig):
    """
    Returns a JIT-compiled DWA training step closed over cfg.

    lambda_sharp and aux_scale are passed as JAX scalars so JIT traces on
    dtype/shape only — no re-tracing on value changes between steps.

    Returns (total_loss, loss_breakdown_dict, alpha, new_alpha_ema).
    """
    @nnx.jit
    def step(
        model:        DWALanguageModel,
        opt:          nnx.Optimizer,
        x:            jax.Array,   # [B, T]
        y:            jax.Array,   # [B, T]
        alpha_ema:    jax.Array,   # [N]
        lambda_sharp: jax.Array,   # scalar
        temperature:  jax.Array,   # scalar
        aux_scale:    jax.Array,   # scalar (0 phase 1, 1 phase 2+)
    ) -> Tuple[jax.Array, dict, jax.Array, jax.Array]:

        def loss_fn(m: DWALanguageModel):
            logits, alpha, keys, w_norm = m(x, lambda_sharp, temperature)
            # Use w_norm directly as norm loss to avoid passing the full [B*T,d,d] tensor
            ce    = cross_entropy(logits, y)
            l_u   = utilization_loss(alpha_ema, cfg.beta_util)
            l_d   = diversity_loss(alpha, keys)
            l_s   = sparsity_loss(alpha)
            aux   = (
                  cfg.lambda_util   * l_u
                + cfg.lambda_div    * l_d
                + cfg.lambda_norm   * w_norm
                + cfg.lambda_sparse * l_s
            )
            total = ce + aux_scale * aux
            return total, ({"ce": ce, "util": l_u, "div": l_d, "sparse": l_s, "norm": w_norm}, alpha)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (total, (breakdown, alpha)), grads = grad_fn(model)
        opt.update(model, grads)

        # EMA update fused into the same JIT call
        batch_mean    = jnp.mean(alpha.reshape(-1, cfg.N), axis=0)
        new_alpha_ema = cfg.ema_decay * alpha_ema + (1.0 - cfg.ema_decay) * batch_mean

        return total, breakdown, alpha, new_alpha_ema

    return step


def make_dense_step():
    """Returns a JIT-compiled dense model training step."""
    @nnx.jit
    def step(
        model: DenseLanguageModel,
        opt:   nnx.Optimizer,
        x:     jax.Array,
        y:     jax.Array,
    ) -> jax.Array:
        def loss_fn(m: DenseLanguageModel):
            return cross_entropy(m(x), y)
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model)
        opt.update(model, grads)
        return loss

    return step


@nnx.jit
def _eval_dwa_batch(model: DWALanguageModel, x: jax.Array, y: jax.Array) -> jax.Array:
    logits, *_ = model(x, jnp.array(0.0))
    return cross_entropy(logits, y)


@nnx.jit
def _eval_dense_batch(model: DenseLanguageModel, x: jax.Array, y: jax.Array) -> jax.Array:
    return cross_entropy(model(x), y)


def generate(
    model,
    prompt_ids: list[int],
    max_new: int,
    cfg: LMConfig,
    is_dwa: bool,
    temperature: float = 0.8,
) -> list[int]:
    """
    Autoregressive generation with temperature sampling.

    Always feeds [1, seq_len] so JIT compiles exactly once (no recompilation
    per new token length). Positions before the prompt are zero-padded.
    """
    rng_key = jax.random.key(42)
    pad_id  = 0

    # Pre-compile: one call with full seq_len
    dummy = jnp.zeros((1, cfg.seq_len), dtype=jnp.int32)
    if is_dwa:
        model(dummy, jnp.array(0.0))
    else:
        model(dummy)

    ids = list(prompt_ids)
    for _ in range(max_new):
        cur    = ids[-cfg.seq_len:]
        padded = [pad_id] * (cfg.seq_len - len(cur)) + cur
        ctx    = jnp.array(padded, dtype=jnp.int32)[None]  # [1, seq_len]
        pos    = min(len(cur) - 1, cfg.seq_len - 1)

        if is_dwa:
            logits, *_ = model(ctx, jnp.array(0.0))
        else:
            logits = model(ctx)

        last_logits = logits[0, pos] / temperature
        rng_key, sub = jax.random.split(rng_key)
        next_id = int(jax.random.categorical(sub, last_logits))
        ids.append(next_id)

    return ids


# ===========================================================================
# 8. BFloat16 helper (critical for TPU v5e performance)
# ===========================================================================

def to_bf16(model: nnx.Module) -> nnx.Module:
    """
    Cast all float32 parameters to bfloat16.

    Call this BEFORE make_optimizer() so Adam's m1/m2 states also track
    bfloat16 parameters — ~4x faster on TPU MXU, acceptable for 1K–100K steps.

    Usage:
        model = DWALanguageModel(cfg, rngs)
        model = to_bf16(model)          # cast FIRST
        opt   = make_optimizer(model, cfg)
    """
    graph, state = nnx.split(model)
    state = jax.tree_util.tree_map(
        lambda v: v.astype(jnp.bfloat16) if v.dtype == jnp.float32 else v,
        state,
    )
    return nnx.merge(graph, state)


# ===========================================================================
# 9. Disk-backed inference (pool on disk, Part A/B in VRAM)
# ===========================================================================

class VectorPoolStore:
    """
    Stores the DWA vector pool on disk as a memory-mapped numpy array.

    Memory profile (N=65536, D=16384, fp16)
      Pool on disk:   ~2 GB
      Key index:      ~500 MB
      VRAM:           ~500 MB  (Part A/B + retrieval projections + k active vectors)
      Per-token read: k * D * 2 bytes  (e.g. k=16 → 512 KB)

    Usage:
        store = VectorPoolStore("pool_dir", N=cfg.N, D=cfg.D)
        store.save(trained_model)       # one-time after training
        # later:
        store = VectorPoolStore("pool_dir", N=cfg.N, D=cfg.D)
        inf   = DWAInferenceModel(trained_model, store, cfg)
    """

    def __init__(self, path: str, N: int, D: int, dtype: np.dtype = np.float16) -> None:
        self.path      = path
        self.N, self.D = N, D
        self.dtype     = dtype
        self.pool_path  = os.path.join(path, "pool.npy")
        self.index_path = os.path.join(path, "keys.npy")
        self.meta_path  = os.path.join(path, "meta.npz")
        self._pool_mmap  = None
        self._keys_mmap  = None

    def save(self, model: DWALanguageModel) -> None:
        """Save trained model's pool + retrieval projections to disk."""
        os.makedirs(self.path, exist_ok=True)

        pool_np = np.array(model.pool.value, dtype=self.dtype)
        np.save(self.pool_path, pool_np)
        print(f"  Pool saved: {pool_np.nbytes / 1e6:.1f} MB  shape={pool_np.shape}")

        # Pre-compute normalised keys for flat MIPS search
        W_K_np   = np.array(model.retrieval.W_K.value, dtype=np.float32)   # [S, D, d_k]
        pool_f32 = np.array(model.pool.value, dtype=np.float32)
        keys     = np.einsum("nd,sdq->nsq", pool_f32, W_K_np)              # [N, S, d_k]
        keys_norm = keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        np.save(self.index_path, keys_norm.astype(np.float16))
        print(f"  Key index saved: {keys_norm.nbytes / 1e6:.1f} MB")

        np.savez(
            self.meta_path,
            W_Q          = np.array(model.retrieval.W_Q.value,         dtype=np.float32),
            W_K          = np.array(model.retrieval.W_K.value,         dtype=np.float32),
            W_base       = np.array(model.middle.W_base.value,         dtype=np.float32),
            b_base       = np.array(model.middle.b_base.value,         dtype=np.float32),
            gamma        = np.array(model.middle.gamma.value,          dtype=np.float32),
            tau          = np.array(model.retrieval.tau.value,         dtype=np.float32),
            aspect_logits= np.array(model.retrieval.aspect_logits.value, dtype=np.float32),
        )
        print("  Metadata saved")

    def _ensure_loaded(self) -> None:
        if self._pool_mmap is None:
            self._pool_mmap = np.load(self.pool_path,  mmap_mode="r")
            self._keys_mmap = np.load(self.index_path, mmap_mode="r")

    @property
    def pool(self) -> np.ndarray:
        self._ensure_loaded()
        return self._pool_mmap

    @property
    def keys(self) -> np.ndarray:
        self._ensure_loaded()
        return self._keys_mmap

    def fetch_vectors(self, indices: np.ndarray) -> jax.Array:
        """Load only the specified vectors from disk → JAX array on device."""
        return jnp.array(self.pool[indices], dtype=jnp.float32)

    def search(
        self,
        query: np.ndarray,  # [S, d_k]  normalised
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flat cosine-similarity search over the key index on disk."""
        meta    = np.load(self.meta_path)
        w       = np.array(jax.nn.softmax(jnp.array(meta["aspect_logits"])), dtype=np.float32)
        sims    = np.einsum("nsq,sq->ns", np.array(self.keys, dtype=np.float32), query.astype(np.float32))
        scores  = np.einsum("ns,s->n", sims, w)
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx, scores[top_idx]


class DWAInferenceModel:
    """
    DWA inference with pool vectors fetched from disk per token.

    VRAM footprint: Part A/B + retrieval projections + k active vectors.
    Pool and key index live on disk (memory-mapped, no full load into RAM).

    Usage:
        store = VectorPoolStore("pool_dir", N=cfg.N, D=cfg.D)
        store.save(trained_model)
        inf = DWAInferenceModel(trained_model, store, cfg)
        ids, ftimes = inf.generate(prompt_ids, max_new=200, tok=tok, k=32)
    """

    def __init__(self, model: DWALanguageModel, store: VectorPoolStore, cfg: LMConfig) -> None:
        self.model  = model
        self.store  = store
        self.cfg    = cfg

        meta = np.load(store.meta_path)
        self.W_Q          = jnp.array(meta["W_Q"])
        self.W_base       = jnp.array(meta["W_base"])
        self.b_base       = jnp.array(meta["b_base"])
        self.gamma        = jnp.array(meta["gamma"])
        self.tau          = jnp.array(meta["tau"])
        self.aspect_logits= jnp.array(meta["aspect_logits"])

        self._u_end = cfg.d_model * cfg.r
        self._v_end = self._u_end + cfg.r * cfg.d_model
        self._b_end = self._v_end + cfg.d_model

    def _retrieve_and_assemble(
        self,
        h_A: jax.Array,  # [1, d_model]
        k: int,
    ) -> Tuple[jax.Array, float]:
        # Compute aspect queries in VRAM
        queries = jnp.einsum("bd,sdq->bsq", h_A, self.W_Q)
        q_norm  = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        q_np    = np.array(q_norm[0])                         # [S, d_k]

        # Search disk index
        t0 = time.perf_counter()
        indices, scores_np = self.store.search(q_np, top_k=k)

        # Fetch k vectors from disk → VRAM
        vectors    = self.store.fetch_vectors(indices)         # [k, D]
        fetch_time = time.perf_counter() - t0

        # Assembly weights over the k fetched vectors
        W_K    = self.model.retrieval.W_K.value               # [S, D, d_k]
        keys   = jnp.einsum("kd,sdq->ksq", vectors, W_K)
        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        sims   = jnp.einsum("bsq,ksq->bks", q_norm, k_norm)  # [1, k, S]
        w      = jax.nn.softmax(self.aspect_logits)
        tok_sc = jnp.einsum("bks,s->bk", sims, w)            # [1, k]

        g         = jax.nn.sigmoid(10.0 * (tok_sc - self.tau))
        alpha_raw = g * jnp.exp(tok_sc)
        alpha     = alpha_raw / (jnp.sum(alpha_raw, axis=-1, keepdims=True) + 1e-8)

        U    = vectors[:, :self._u_end].reshape(k, self.cfg.d_model, self.cfg.r)
        V    = vectors[:, self._u_end:self._v_end].reshape(k, self.cfg.r, self.cfg.d_model)
        bias = vectors[:, self._v_end:self._b_end]

        UV          = jnp.einsum("kdr,kre->kde", U, V)
        W_delta     = jnp.einsum("bk,kde->bde", alpha, UV)
        b_delta     = jnp.einsum("bk,kd->bd",   alpha, bias)
        W_assembled = self.W_base[None] + W_delta
        b_assembled = self.b_base[None] + b_delta

        h_t   = jnp.einsum("ba,bca->bc", h_A, W_assembled) + b_assembled
        h_mid = self.model.middle.layer_norm(h_A + self.gamma * h_t)
        return h_mid, fetch_time

    def generate(
        self,
        token_ids: list[int],
        max_new: int,
        tok,
        temperature: float = 0.8,
        k: int = 16,
    ) -> Tuple[str, list[float]]:
        """Autoregressive generation with per-token disk fetching."""
        rng_key    = jax.random.key(42)
        ids        = list(token_ids)
        fetch_times: list[float] = []

        for _ in range(max_new):
            cur    = ids[-self.cfg.seq_len:]
            padded = [0] * (self.cfg.seq_len - len(cur)) + cur
            ctx    = jnp.array(padded, dtype=jnp.int32)[None]
            pos    = min(len(cur) - 1, self.cfg.seq_len - 1)

            # Part A forward
            h = self.model.tok_emb(ctx) + self.model.pos_emb.value[:self.cfg.seq_len]
            for block in self.model.blocks_A:
                h = block(h)
            h = self.model.ln_mid(h)

            # DWA middle with disk fetch (last token only)
            h_A            = h[:, pos:pos+1, :].reshape(1, self.cfg.d_model)
            h_mid, ft      = self._retrieve_and_assemble(h_A, k)
            fetch_times.append(ft)
            h = h.at[:, pos:pos+1, :].set(h_mid.reshape(1, 1, self.cfg.d_model))

            # Part B forward
            for block in self.model.blocks_B:
                h = block(h)
            logits = self.model.head(self.model.ln_f(h))

            last_logits  = logits[0, pos] / temperature
            rng_key, sub = jax.random.split(rng_key)
            ids.append(int(jax.random.categorical(sub, last_logits)))

        return tok.decode(ids), fetch_times


# ===========================================================================
# 10. Preset configurations
# ===========================================================================

def small_config(vocab_size: int = 50257) -> LMConfig:
    """~30M params. Fast to iterate on, fits in 8GB VRAM."""
    return LMConfig(
        vocab_size=vocab_size,
        d_model=256, n_heads=8, n_layers_A=4, n_layers_B=4, seq_len=512,
        N=256, D=4096, r=4, S=2, d_k=64, k_max=16,
        batch_size=64, lr=3e-4, warmup_steps=500, max_steps=20_000,
        phase1_end=1_000, phase2_end=10_000,
    )


def medium_config(vocab_size: int = 50257) -> LMConfig:
    """~120M params. Competitive quality, needs 16–40GB VRAM."""
    return LMConfig(
        vocab_size=vocab_size,
        d_model=512, n_heads=8, n_layers_A=6, n_layers_B=6, seq_len=1024,
        N=1024, D=16384, r=8, S=4, d_k=64, k_max=32,
        batch_size=32, lr=2e-4, warmup_steps=1_000, max_steps=100_000,
        phase1_end=2_000, phase2_end=50_000,
    )


def large_config(vocab_size: int = 50257) -> LMConfig:
    """
    ~1B+ params. Pool (~8GB) goes to disk via VectorPoolStore.
    Transformer portion fits in ~24GB VRAM.
    """
    return LMConfig(
        vocab_size=vocab_size,
        d_model=1024, n_heads=16, n_layers_A=12, n_layers_B=12, seq_len=2048,
        N=65536, D=65536, r=16, S=4, d_k=128, k_max=64,
        batch_size=8, lr=1e-4, warmup_steps=2_000, max_steps=500_000,
        phase1_end=5_000, phase2_end=200_000,
    )
