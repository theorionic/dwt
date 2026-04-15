"""
Disk-backed vector pool with dynamic fetching for DWA inference.

The pool lives on disk as a memory-mapped numpy array. At inference time,
only the k vectors needed for the current token are loaded into VRAM.

Components:
  - VectorPoolStore: mmap-backed pool on disk + flat MIPS index
  - DynamicRetriever: query → top-k IDs → load vectors → JAX arrays
  - DWAInferenceModel: full inference pipeline using dynamic fetching

Memory profile (full config, N=65536, D=16384):
  - VRAM: ~500MB (parts A/B + retrieval proj + W_base + active vectors)
  - Disk: ~2GB (pool, bf16) + ~500MB (index)
  - Per-token disk read: 16 × 16384 × 2 = 512KB
"""

import os
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from config import LMConfig
from lm_model import DWALanguageModel


# ---------------------------------------------------------------------------
# Disk-backed vector pool store
# ---------------------------------------------------------------------------

class VectorPoolStore:
    """
    Stores the DWA vector pool on disk as a memory-mapped numpy array.

    Supports:
      - Saving a trained model's pool to disk
      - Memory-mapped read access (no full load into RAM)
      - Flat cosine-similarity index for retrieval
      - Fetching only k vectors by index into a JAX array
    """

    def __init__(self, path: str, N: int, D: int, dtype: np.dtype = np.float16):
        self.path = path
        self.N = N
        self.D = D
        self.dtype = dtype
        self.pool_path = os.path.join(path, "pool.npy")
        self.index_path = os.path.join(path, "keys.npy")
        self.meta_path = os.path.join(path, "meta.npz")
        self._pool_mmap = None
        self._keys_mmap = None

    def save(self, model: DWALanguageModel) -> None:
        """Save a trained model's pool and retrieval projections to disk."""
        os.makedirs(self.path, exist_ok=True)

        # Save pool vectors
        pool_np = np.array(model.pool.value, dtype=self.dtype)
        np.save(self.pool_path, pool_np)
        print(f"  Pool saved: {pool_np.nbytes / 1e6:.1f} MB  shape={pool_np.shape}  dtype={self.dtype}")

        # Pre-compute and save key projections for all pool vectors
        # keys[i, s, d_k] = W_K[s] @ pool[i]
        W_K_np = np.array(model.retrieval.W_K.value, dtype=np.float32)  # [S, D, d_k]
        pool_f32 = np.array(model.pool.value, dtype=np.float32)
        keys = np.einsum("np,apq->naq", pool_f32, W_K_np)  # [N, S, d_k]
        # Normalise keys for cosine similarity
        keys_norm = keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        np.save(self.index_path, keys_norm.astype(np.float16))
        print(f"  Key index saved: {keys_norm.nbytes / 1e6:.1f} MB")

        # Save metadata (W_Q, W_K, W_base, gamma, tau, aspect_logits, etc.)
        np.savez(
            self.meta_path,
            W_Q=np.array(model.retrieval.W_Q.value, dtype=np.float32),
            W_K=np.array(model.retrieval.W_K.value, dtype=np.float32),
            W_base=np.array(model.middle.W_base.value, dtype=np.float32),
            b_base=np.array(model.middle.b_base.value, dtype=np.float32),
            gamma=np.array(model.middle.gamma.value, dtype=np.float32),
            tau=np.array(model.retrieval.tau.value, dtype=np.float32),
            aspect_logits=np.array(model.retrieval.aspect_logits.value, dtype=np.float32),
        )
        print(f"  Metadata saved")

    def _ensure_loaded(self) -> None:
        """Lazily mmap the pool and index files."""
        if self._pool_mmap is None:
            self._pool_mmap = np.load(self.pool_path, mmap_mode="r")
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
        """Load only the specified vectors from disk → JAX array (on device)."""
        vecs = self.pool[indices]                    # [k, D]  from mmap
        return jnp.array(vecs, dtype=jnp.float32)   # move to device

    def search(
        self,
        query: np.ndarray,     # [S, d_k]  normalised aspect queries
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flat cosine-similarity search over the key index.

        Returns:
            indices: [top_k]   int array of pool vector indices
            scores:  [top_k]  float array of similarity scores
        """
        # keys: [N, S, d_k]  (pre-normalised, stored on disk)
        # query: [S, d_k]    (normalised)
        # Per-aspect cosine similarity:  [N, S]
        sims = np.einsum("naq,sq->ns", np.array(self.keys, dtype=np.float32), query.astype(np.float32))

        # Aspect-weighted scores
        # Load aspect weights from metadata
        meta = np.load(self.meta_path)
        w = jax.nn.softmax(meta["aspect_logits"])   # [S]
        scores = np.einsum("ns,s->n", sims, np.array(w, dtype=np.float32))  # [N]

        # Top-k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        top_scores = scores[top_indices]
        return top_indices, top_scores


# ---------------------------------------------------------------------------
# Dynamic-fetching DWA inference model
# ---------------------------------------------------------------------------

class DWAInferenceModel:
    """
    DWA model that dynamically fetches pool vectors from disk at inference.

    VRAM footprint: only parts A/B + retrieval projections + k active vectors.
    Pool and key index live on disk (memory-mapped).

    Usage:
        store = VectorPoolStore("pool_on_disk", N=1024, D=8192)
        store.save(trained_model)
        inf = DWAInferenceModel(trained_model, store, cfg)
        logits = inf.generate(token_ids, max_new=100)
    """

    def __init__(
        self,
        model: DWALanguageModel,
        store: VectorPoolStore,
        cfg: LMConfig,
    ) -> None:
        self.cfg = cfg
        self.store = store

        # Extract model components (keep in VRAM)
        self.model = model

        # Pre-compute metadata as JAX arrays
        meta = np.load(store.meta_path)
        self.W_Q = jnp.array(meta["W_Q"])                # [S, d_A, d_k]
        self.W_base = jnp.array(meta["W_base"])           # [d_B, d_A]
        self.b_base = jnp.array(meta["b_base"])           # [d_B]
        self.gamma = jnp.array(meta["gamma"])              # scalar
        self.tau = jnp.array(meta["tau"])                  # scalar
        self.aspect_logits = jnp.array(meta["aspect_logits"])  # [S]

        # Slice boundaries for splitting pool vectors
        self._u_end = cfg.d_model * cfg.r
        self._v_end = self._u_end + cfg.r * cfg.d_model
        self._b_end = self._v_end + cfg.d_model

    def _retrieve_and_assemble(
        self,
        h_A: jax.Array,        # [1, d_model]  single token
        k: int,                # number of vectors to fetch
    ) -> Tuple[jax.Array, float]:
        """
        Retrieve k pool vectors from disk and assemble the weight matrix.

        Returns:
            h_mid:  [1, d_model]   transformed hidden state
            fetch_time: float      seconds spent on disk read
        """
        cfg = self.cfg

        # 1. Compute aspect queries (in VRAM)
        queries = jnp.einsum("bd,adq->baq", h_A, self.W_Q)  # [1, S, d_k]
        q_norm = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        q_np = np.array(q_norm[0])  # [S, d_k] → numpy for disk search

        # 2. Search MIPS index (on disk)
        t0 = time.perf_counter()
        indices, scores = self.store.search(q_np, top_k=k)
        t_search = time.perf_counter() - t0

        # 3. Fetch only k vectors from disk
        t0 = time.perf_counter()
        vectors = self.store.fetch_vectors(indices)  # [k, D] → JAX on device
        t_fetch = time.perf_counter() - t0

        # 4. Compute assembly weights (softmax over fetched vectors only)
        W_K = self.model.retrieval.W_K.value  # [S, D, d_k]
        keys = jnp.einsum("kp,apq->kaq", vectors, W_K)  # [k, S, d_k]
        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Cosine similarity with queries
        sims = jnp.einsum("baq,kaq->bka", q_norm, k_norm)  # [1, k, S]
        w_aspects = jax.nn.softmax(self.aspect_logits)     # [S]
        token_scores = jnp.einsum("bka,a->bk", sims, w_aspects)  # [1, k]

        # Sigmoid gate (use fixed λ=10 for inference)
        g = jax.nn.sigmoid(10.0 * (token_scores - self.tau))
        alpha_raw = g * jnp.exp(token_scores)
        alpha = alpha_raw / (jnp.sum(alpha_raw, axis=-1, keepdims=True) + 1e-8)  # [1, k]

        # 5. Assemble weight matrix from k vectors
        U = vectors[:, :self._u_end].reshape(k, cfg.d_model, cfg.r)
        V = vectors[:, self._u_end:self._v_end].reshape(k, cfg.r, cfg.d_model)
        bias = vectors[:, self._v_end:self._b_end]  # [k, d_model]

        UV = jnp.einsum("kdr,kre->kde", U, V)               # [k, d, d]
        W_delta = jnp.einsum("bk,kde->bde", alpha, UV)      # [1, d, d]
        b_delta = jnp.einsum("bk,kd->bd", alpha, bias)      # [1, d]

        W_assembled = self.W_base[None] + W_delta
        b_assembled = self.b_base[None] + b_delta

        # 6. Forward through middle layer
        h_transformed = (
            jnp.einsum("ba,bca->bc", h_A, W_assembled) + b_assembled
        )  # [1, d]
        h_mid = self.model.middle.layer_norm(h_A + self.gamma * h_transformed)

        fetch_time = t_search + t_fetch
        return h_mid, fetch_time

    def forward_token(self, h: jax.Array, k: int = 16) -> Tuple[jax.Array, float]:
        """
        Full DWA forward pass for a single token through the middle layer.

        h: [1, d_model]  hidden state after Part A blocks
        k: number of pool vectors to fetch
        """
        return self._retrieve_and_assemble(h, k)

    def generate(
        self,
        token_ids: list[int],
        max_new: int,
        itos: dict,
        temperature: float = 0.8,
        k: int = 16,
    ) -> Tuple[str, list[float]]:
        """
        Autoregressive generation with dynamic fetching.

        Returns:
            generated_text: str
            fetch_times: list of per-token disk fetch times (seconds)
        """
        rng_key = jax.random.key(42)
        ids = list(token_ids)
        fetch_times = []

        for step in range(max_new):
            # Pad/truncate to seq_len
            cur = ids[-self.cfg.seq_len:]
            padded = [0] * (self.cfg.seq_len - len(cur)) + cur
            context = jnp.array(padded, dtype=jnp.int32)[None]  # [1, seq_len]
            pos = min(len(cur) - 1, self.cfg.seq_len - 1)

            # Part A
            h = self.model.tok_emb(context) + self.model.pos_emb.value[:self.cfg.seq_len]
            for block in self.model.blocks_A:
                h = block(h)
            h = self.model.ln_mid(h)
            h_last = h[:, pos:pos+1, :]  # [1, 1, d_model]

            # DWA middle with dynamic fetch
            h_A = h_last.reshape(1, self.cfg.d_model)
            h_mid, ft = self.forward_token(h_A, k=k)
            fetch_times.append(ft)

            # Replace the last position with transformed hidden state
            h = h.at[:, pos:pos+1, :].set(h_mid.reshape(1, 1, self.cfg.d_model))

            # Part B
            for block in self.model.blocks_B:
                h = block(h)
            logits = self.model.head(self.model.ln_f(h))

            # Sample
            last_logits = logits[0, pos, :] / temperature
            rng_key, sub = jax.random.split(rng_key)
            next_id = int(jax.random.categorical(sub, last_logits))
            ids.append(next_id)

        text = "".join(itos[i] for i in ids)
        return text, fetch_times