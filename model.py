"""
Dynamic Weight Assembly (DWA) model — JAX / Flax NNX implementation.

Forward pass:
  x → Part A (MLP) → h_A
  h_A → MultiAspectRetrieval(pool) → alpha [batch, N]
  (pool, alpha) → DWAMiddleLayer.assemble → W_assembled, b_assembled
  h_A → DWAMiddleLayer.forward → h_mid
  h_mid → Part B (MLP) → output
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from config import DWAConfig


# ---------------------------------------------------------------------------
# Vector pool
# ---------------------------------------------------------------------------

class VectorPool(nnx.Module):
    """
    Learnable pool of N D-dimensional vectors.

    Each vector v_i encodes both:
      - Matrix factors (first d_B*r + r*d_A + d_B elements)
      - Key information for retrieval (all D elements via W_K projection)
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.N = cfg.N
        self.D = cfg.D
        # Small init so model starts near identity (zero assembly contribution)
        self.vectors = nnx.Param(
            nnx.initializers.normal(stddev=0.01)(rngs.params(), (cfg.N, cfg.D))
        )

    @property
    def value(self) -> jax.Array:
        return self.vectors.value  # [N, D]


# ---------------------------------------------------------------------------
# Multi-aspect sigmoid-gated retrieval
# ---------------------------------------------------------------------------

class MultiAspectRetrieval(nnx.Module):
    """
    S-aspect sigmoid-gated retrieval.

    KEY design: W_K projects the FULL pool vector (including matrix factor
    elements), coupling the retrieval gradient with the assembly gradient.
    Both paths update the same v_i parameters — self-reinforcing learning.

    lambda_sharp annealing:
      phase 1  → lambda_sharp=0  →  sigmoid=0.5 for all  →  pure softmax
      phase 2+ → lambda_sharp>0  →  sparse sigmoid gating
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        lecun = nnx.initializers.lecun_normal()
        # Query projections: d_A → d_k, one per aspect  [S, d_A, d_k]
        self.W_Q = nnx.Param(lecun(rngs.params(), (cfg.S, cfg.d_A, cfg.d_k)))
        # Key projections: D → d_k, one per aspect      [S, D, d_k]
        # Uses the FULL vector (including U_i, V_i elements)
        self.W_K = nnx.Param(lecun(rngs.params(), (cfg.S, cfg.D, cfg.d_k)))
        # Learnable aspect importance weights (initialized uniform)
        self.aspect_logits = nnx.Param(jnp.zeros(cfg.S))
        # Learnable gate threshold τ
        self.tau = nnx.Param(jnp.array(cfg.tau_init))

    def __call__(
        self,
        z: jax.Array,             # [batch, d_A] — query from Part A
        pool_vectors: jax.Array,  # [N, D]
        lambda_sharp: float,
        temperature: float,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Returns:
            alpha:  [batch, N]       normalized assembly weights
            scores: [batch, N]       raw similarity scores (for aux losses)
            keys:   [N, S, d_k]     aspect keys (for diversity loss)
        """
        # Aspect queries per sample:  [batch, S, d_k]
        queries = jnp.einsum("bd,adq->baq", z, self.W_Q.value)

        # Aspect keys for all pool vectors:  [N, S, d_k]
        keys = jnp.einsum("np,apq->naq", pool_vectors, self.W_K.value)

        # Cosine similarity (normalize along d_k axis)
        q_norm = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Per-aspect cosine similarities:  [batch, N, S]
        sims = jnp.einsum("baq,naq->bna", q_norm, k_norm)

        # Aspect-weighted scores:  [batch, N]
        w = jax.nn.softmax(self.aspect_logits.value)  # [S]
        scores = jnp.einsum("bna,a->bn", sims, w)

        # Sigmoid gate  (lambda_sharp=0 → g=0.5 for all → cancels → pure softmax)
        g = jax.nn.sigmoid(lambda_sharp * (scores - self.tau.value))

        # Normalized assembly weights
        alpha_raw = g * jnp.exp(scores / temperature)
        alpha = alpha_raw / (jnp.sum(alpha_raw, axis=-1, keepdims=True) + 1e-8)

        return alpha, scores, keys


# ---------------------------------------------------------------------------
# Factorized assembly + middle layer forward
# ---------------------------------------------------------------------------

class DWAMiddleLayer(nnx.Module):
    """
    Splits each pool vector into low-rank factors, assembles a per-sample
    weight matrix, and applies the dynamic linear transform with a residual.

    Pool vector layout (first elements used; rest available for keys):
      [  U_i: d_B×r  |  V_i: r×d_A  |  b_i: d_B  |  ... unused ... ]

    Assembly:
      W_assembled = W_base + Σ_i α_i (U_i @ V_i)
      b_assembled = b_base + Σ_i α_i b_i

    Forward:
      h_mid = LayerNorm(h_A + γ · (h_A @ W_assembled^T + b_assembled))
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg

        # Slice boundaries for splitting pool vectors
        self._u_end = cfg.d_B * cfg.r
        self._v_end = self._u_end + cfg.r * cfg.d_A
        self._b_end = self._v_end + cfg.d_B

        # Base weight + bias (small init → graceful zero-retrieval behavior)
        self.W_base = nnx.Param(
            nnx.initializers.normal(stddev=0.01)(rngs.params(), (cfg.d_B, cfg.d_A))
        )
        self.b_base = nnx.Param(jnp.zeros(cfg.d_B))

        # Residual scale γ — LoRA-style, initialized to near-zero
        self.gamma = nnx.Param(jnp.array(cfg.gamma_init))

        self.layer_norm = nnx.LayerNorm(cfg.d_A, rngs=rngs)

    def assemble(
        self,
        pool_vectors: jax.Array,  # [N, D]
        alpha: jax.Array,         # [batch, N]
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Returns:
            W_assembled: [batch, d_B, d_A]
            b_assembled: [batch, d_B]
        """
        cfg = self.cfg

        # Split pool vectors into factorized components
        U    = pool_vectors[:, :self._u_end].reshape((cfg.N, cfg.d_B, cfg.r))         # [N, d_B, r]
        V    = pool_vectors[:, self._u_end:self._v_end].reshape((cfg.N, cfg.r, cfg.d_A))  # [N, r, d_A]
        bias = pool_vectors[:, self._v_end:self._b_end]                                 # [N, d_B]

        # Low-rank products: U_i @ V_i  →  [N, d_B, d_A]
        UV = jnp.einsum("ndr,nre->nde", U, V)

        # Weighted combination over pool
        W_delta = jnp.einsum("bn,nde->bde", alpha, UV)   # [batch, d_B, d_A]
        b_delta = jnp.einsum("bn,nd->bd",   alpha, bias)  # [batch, d_B]

        W_assembled = self.W_base.value[None] + W_delta   # broadcast base over batch
        b_assembled = self.b_base.value[None] + b_delta

        return W_assembled, b_assembled

    def __call__(
        self,
        h_A: jax.Array,           # [batch, d_A]
        W_assembled: jax.Array,   # [batch, d_B, d_A]
        b_assembled: jax.Array,   # [batch, d_B]
    ) -> jax.Array:
        """Returns h_mid: [batch, d_A]  (d_A == d_B in symmetric config)"""
        # h_A @ W_assembled^T per sample:  einsum over d_A axis
        h_transformed = (
            jnp.einsum("ba,bca->bc", h_A, W_assembled) + b_assembled
        )  # [batch, d_B]

        # Residual with learnable scale + LayerNorm
        h_mid = self.layer_norm(h_A + self.gamma.value * h_transformed)
        return h_mid  # [batch, d_A]


# ---------------------------------------------------------------------------
# Part A / Part B halves
# ---------------------------------------------------------------------------

class MLP(nnx.Module):
    """Simple GELU MLP — stands in for the Part A / Part B model halves."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        rngs: nnx.Rngs,
    ) -> None:
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nnx.List([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)
        return x


# ---------------------------------------------------------------------------
# Full DWA model
# ---------------------------------------------------------------------------

class DWAModel(nnx.Module):
    """
    Dynamic Weight Assembly model.

    The middle layer weight matrix is assembled fresh for each input by
    retrieving relevant vectors from the pool and composing their low-rank
    factors — enabling input-conditioned weight adaptation without the cost
    of full hypernetworks or the rigidity of static LoRA.
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        self.pool      = VectorPool(cfg, rngs)
        self.part_A    = MLP(cfg.d_model, cfg.d_model * 2, cfg.d_A, cfg.n_layers_A, rngs)
        self.retrieval = MultiAspectRetrieval(cfg, rngs)
        self.middle    = DWAMiddleLayer(cfg, rngs)
        self.part_B    = MLP(cfg.d_A, cfg.d_model * 2, cfg.d_model, cfg.n_layers_B, rngs)

    def __call__(
        self,
        x: jax.Array,          # [batch, d_model]
        lambda_sharp: float = 1.0,
        temperature: float = 1.0,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Returns:
            output:      [batch, d_model]
            alpha:       [batch, N]         assembly weights
            W_assembled: [batch, d_B, d_A]  assembled weight matrix
            keys:        [N, S, d_k]        retrieval aspect keys
        """
        # Part A: raw input → hidden representation
        h_A = self.part_A(x)  # [batch, d_A]

        # Retrieval: compute per-sample assembly weights over the pool
        pool_vecs = self.pool.value  # [N, D]
        alpha, scores, keys = self.retrieval(h_A, pool_vecs, lambda_sharp, temperature)

        # Assembly: compose pool vectors into a weight matrix
        W_assembled, b_assembled = self.middle.assemble(pool_vecs, alpha)

        # Middle layer: apply assembled transform with residual
        h_mid = self.middle(h_A, W_assembled, b_assembled)  # [batch, d_A]

        # Part B: transformed hidden state → output
        output = self.part_B(h_mid)  # [batch, d_model]

        return output, alpha, W_assembled, keys
