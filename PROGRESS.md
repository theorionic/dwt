# DWA Project Progress Report

## What Was Done

### 1. Fixed JAX JIT Crash in Top-k Quality Evaluation

Both `run_quick.py` and `run_dynamic.py` had a crash in the top-k masking evaluation that simulates dynamic fetching quality. The crash was:

```
IndexError: Slice entries must be static integers. Got slice(JitTracer(~int32[]), None, None)
```

**Root cause:** Two issues:

1. **Scatter with traced indices:** `mask.at[jnp.arange(B*T)[:,None], top_idx].set(1.0)` uses JAX scatter with dynamically-computed indices from `argsort`. JAX's scatter doesn't support traced index arrays in this pattern inside JIT.

2. **Dynamic slice with traced k_val:** `alpha[:, -k_val:]` inside `@nnx.jit` where `k_val` is a function argument — `nnx.jit` traces all positional arguments, making `k_val` a traced value that can't be used in Python slices.

**Fix applied to both files:**

- **Replaced masking approach with direct top-k assembly.** Instead of creating a mask over all N alphas and calling `model.middle.assemble(pool_vecs, alpha_masked)`, we now:
  1. Gather only the top-k vectors: `top_vecs = pool_vecs[top_idx]`  (gather operation, JIT-safe)
  2. Extract top-k alpha weights: `alpha_k = jax.nn.softmax(jnp.take_along_axis(alpha, top_idx, axis=-1), axis=-1)`
  3. Inline the assembly computation using only k vectors with batched einsum operations

  This is also a **more faithful simulation** of dynamic fetching — it computes exactly what the inference model does: load k vectors, compute assembly weights only over those k, and assemble.

- **Captured k_val as closure variable** via `make_eval_topk(k_val)` factory function. This makes `k_val` a static Python int captured in the closure, not a traced function argument. Each value of k gets its own JIT-compiled function with static shapes.

### 2. Files Modified

- `run_quick.py` — Lines 143-203: Rewrote `_eval_topk` function
- `run_dynamic.py` — Lines 167-216: Rewrote `_eval_dwa_topk` function

### 3. What Needs To Happen Next

**Run the validation experiment** to verify the fix works end-to-end:

```bash
python run_quick.py
```

This trains DWA + Dense for 3K steps on small config (d_model=128, N=256), then:
- Saves pool to disk and benchmarks dynamic fetching latency
- Runs top-k quality evaluation (k=4, 8, 16, 32, 64, 256) comparing perplexity vs full-VRAM
- Prints summary comparing DWA vs Dense quality

The key metric we need: **how does perplexity degrade as k decreases?** This validates whether dynamic fetching (loading only k vectors from disk per token) preserves model quality.

Expected results (based on previous successful runs at larger scale):
- k=256 (all vectors) should match full model quality
- k=16-32 should be close to full quality
- k=4-8 should show some degradation
- This demonstrates the feasibility of the disk-pool + dynamic-fetch inference architecture

### 4. Longer-term TODO

- [ ] Run the quick validation experiment
- [ ] If validation passes, update REPORT.md with top-k quality results
- [ ] Consider implementing proper MIPS index (FAISS or similar) for the disk-backed pool
- [ ] Test on TPU for training speedup
- [ ] Per-component learning rate tuning (pool=3e-5, retrieval=1e-4, gate=1e-3)