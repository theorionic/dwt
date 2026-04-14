"""
Transformer language models for the DWA validation experiment.

DWALanguageModel  — transformer with a DWA dynamic-weight middle layer.
DenseLanguageModel — standard transformer baseline (same depth).

The DWA middle layer sits between Part A and Part B transformer stacks.
It processes every token position independently:
  [B, T, d] → reshape [B·T, d] → retrieve+assemble+forward → reshape [B, T, d]

Each token's hidden state acts as the retrieval query.  The assembled
weight matrix is unique per token, making the FFN input-conditioned.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from config import LMConfig
from model import VectorPool, MultiAspectRetrieval, DWAMiddleLayer


# ---------------------------------------------------------------------------
# Shared transformer building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention (pre-norm style)."""

    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs) -> None:
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
        self.W_out  = nnx.Linear(d_model, d_model,     use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [B, T, d_model]
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        qkv = self.W_qkv(x)                        # [B, T, 3·d_model]
        q, k, v = jnp.split(qkv, 3, axis=-1)       # each [B, T, d_model]

        # → [B, H, T, d_head]
        def to_heads(z: jax.Array) -> jax.Array:
            return z.reshape(B, T, H, dh).transpose((0, 2, 1, 3))

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # Scaled dot-product: [B, H, T_q, T_k]
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * (dh ** -0.5)

        # Causal mask — lower-triangular, float fill
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(causal[None, None], scores, -1e9)
        attn   = jax.nn.softmax(scores, axis=-1)    # [B, H, T, T]

        # Aggregate values and merge heads
        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)           # [B, H, T, d_head]
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, D)      # [B, T, d_model]
        return self.W_out(out)


class FeedForward(nnx.Module):
    """Position-wise 4× expansion FFN with GELU."""

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


# ---------------------------------------------------------------------------
# DWA language model
# ---------------------------------------------------------------------------

class DWALanguageModel(nnx.Module):
    """
    Transformer LM where the middle layer is a DWA dynamic weight assembly.

    Architecture:
      embed → blocks_A (Part A) → DWA middle → blocks_B (Part B) → lm_head

    The DWA middle operates token-by-token: each token's contextual
    representation (post-Part-A attention) is the retrieval query that
    selects which pool vectors to compose into that token's weight matrix.

    Returns
    -------
    logits    : [B, T, vocab_size]
    alpha     : [B·T, N]   per-token assembly weights
    keys      : [N, S, d_k] aspect keys (for diversity loss)
    w_norm    : scalar     mean ‖W_delta‖²_F (for norm loss)
    """

    def __init__(self, cfg: LMConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        dwa = cfg.to_dwa_config()

        # Embeddings
        self.tok_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.pos_emb = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (cfg.seq_len, cfg.d_model))
        )

        # Part A transformer stack
        self.blocks_A = nnx.List([
            TransformerBlock(cfg.d_model, cfg.n_heads, rngs)
            for _ in range(cfg.n_layers_A)
        ])
        self.ln_mid = nnx.LayerNorm(cfg.d_model, rngs=rngs)

        # DWA components
        self.pool      = VectorPool(dwa, rngs)
        self.retrieval = MultiAspectRetrieval(dwa, rngs)
        self.middle    = DWAMiddleLayer(dwa, rngs)

        # Part B transformer stack
        self.blocks_B = nnx.List([
            TransformerBlock(cfg.d_model, cfg.n_heads, rngs)
            for _ in range(cfg.n_layers_B)
        ])

        # Output head
        self.ln_f = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,           # [B, T]  token ids
        lambda_sharp: float = 0.0,
        temperature: float = 1.0,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        B, T = x.shape
        cfg  = self.cfg

        # Embeddings
        h = self.tok_emb(x) + self.pos_emb.value[:T]   # [B, T, d_model]

        # Part A
        for block in self.blocks_A:
            h = block(h)
        h = self.ln_mid(h)

        # DWA middle — flatten sequence into batch dimension
        h_flat    = h.reshape(B * T, cfg.d_model)
        pool_vecs = self.pool.value                     # [N, D]

        alpha, _scores, keys = self.retrieval(
            h_flat, pool_vecs, lambda_sharp, temperature
        )                                               # alpha: [B·T, N]

        W_assembled, b_assembled = self.middle.assemble(pool_vecs, alpha)
        # Inline norm: mean ‖W_delta‖²_F  — scalar, avoids returning [B·T,d,d]
        W_delta = W_assembled - self.middle.W_base.value[None]
        w_norm  = jnp.mean(jnp.sum(W_delta ** 2, axis=(-2, -1)))

        h_mid_flat = self.middle(h_flat, W_assembled, b_assembled)
        h = h_mid_flat.reshape(B, T, cfg.d_model)

        # Part B
        for block in self.blocks_B:
            h = block(h)

        logits = self.head(self.ln_f(h))                # [B, T, vocab_size]
        return logits, alpha, keys, w_norm


# ---------------------------------------------------------------------------
# Dense baseline
# ---------------------------------------------------------------------------

class DenseLanguageModel(nnx.Module):
    """
    Standard GPT-style transformer — baseline for comparison.

    Uses n_layers_A + 1 + n_layers_B blocks so total depth matches the
    DWA model (which has n_layers_A + DWA-middle + n_layers_B).
    Parameter count will be *lower* than DWA (no pool or key projections),
    which we report explicitly.
    """

    def __init__(self, cfg: LMConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        n_layers = cfg.n_layers_A + 1 + cfg.n_layers_B

        self.tok_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.pos_emb = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (cfg.seq_len, cfg.d_model))
        )
        self.blocks = nnx.List([
            TransformerBlock(cfg.d_model, cfg.n_heads, rngs)
            for _ in range(n_layers)
        ])
        self.ln_f = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Returns logits: [B, T, vocab_size]"""
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb.value[:T]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))
