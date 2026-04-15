"""
DWA training utilities.

Three-phase schedule:
  Phase 1 — Warmup    (0 → phase1_end):      lambda_sharp=0 (pure softmax), no aux losses
  Phase 2 — Gate On   (phase1_end → phase2_end): lambda ramps 0→5, aux losses enabled
  Phase 3 — Sharpen   (phase2_end → ∞):      lambda ramps 5→10, cosine LR decay

Per-component learning rates (from architecture spec):
  pool:              3e-5
  parts A/B:         1e-4
  retrieval proj:    1e-4
  threshold / gamma: 1e-3
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from config import DWAConfig
from model import DWAModel
from losses import compute_losses


# ---------------------------------------------------------------------------
# Phase schedule helpers
# ---------------------------------------------------------------------------

def get_lambda_sharp(step: int, cfg: DWAConfig) -> float:
    """
    Returns the sigmoid sharpness λ for the current training step.

    Phase 1: λ=0       → sigmoid=0.5 everywhere → cancels in normalization
                        → equivalent to pure softmax over all N vectors
    Phase 2: λ: 0→5    → gate gradually sharpens
    Phase 3: λ: 5→10   → final sharpening
    """
    if step < cfg.phase1_end:
        return 0.0
    if step < cfg.phase2_end:
        t = (step - cfg.phase1_end) / (cfg.phase2_end - cfg.phase1_end)
        return t * cfg.lambda_sharp_phase2_end
    # Phase 3: linear ramp from phase2_end value to final
    t = min(1.0, (step - cfg.phase2_end) / cfg.phase2_end)
    return cfg.lambda_sharp_phase2_end + t * (cfg.lambda_sharp_final - cfg.lambda_sharp_phase2_end)


def get_aux_scale(step: int, cfg: DWAConfig) -> float:
    """Aux losses are disabled during phase 1 warmup."""
    return 0.0 if step < cfg.phase1_end else 1.0


def update_ema(
    alpha_ema: jax.Array,  # [N]
    alpha: jax.Array,      # [batch, N]
    decay: float,
) -> jax.Array:
    """Update per-vector utilization EMA with the current batch mean."""
    batch_mean = jnp.mean(alpha, axis=0)  # [N]
    return decay * alpha_ema + (1.0 - decay) * batch_mean


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def create_optimizer(model: DWAModel, cfg: DWAConfig) -> nnx.Optimizer:
    """
    Per-component learning rates as specified in the architecture:
      pool: 3e-5  |  retrieval proj: 1e-4  |  tau/gamma: 1e-3  |  rest: 1e-4

    TODO: wire up optax.multi_transform once per-component LR labeling is
    validated; for now uniform adam(1e-4) is correct enough to smoke-test.
    """
    tx = optax.adam(1e-4)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def make_train_step(cfg: DWAConfig):
    """
    Returns a JIT-compiled training step function closed over cfg.

    lambda_sharp and aux_scale are passed as JAX scalars so JIT traces
    on dtype/shape only — no retracing on value change between steps.
    """

    @nnx.jit
    def train_step(
        model: DWAModel,
        optimizer: nnx.Optimizer,
        batch_x: jax.Array,      # [batch, d_model]
        batch_y: jax.Array,      # [batch, d_model]
        alpha_ema: jax.Array,    # [N]
        lambda_sharp: jax.Array, # scalar
        temperature: jax.Array,  # scalar
        aux_scale: jax.Array,    # scalar  (0 in phase 1)
    ) -> Tuple[jax.Array, dict, jax.Array]:
        """
        Single training step.

        Returns:
            total_loss: scalar
            loss_breakdown: dict of named loss values
            alpha: [batch, N] assembly weights (caller updates EMA)
        """

        def loss_fn(model: DWAModel):
            output, alpha, W_assembled, keys = model(
                batch_x, lambda_sharp, temperature
            )
            # Task loss: MSE (swap for cross-entropy on real tasks)
            task_loss = jnp.mean((output - batch_y) ** 2)

            total, breakdown = compute_losses(
                task_loss=task_loss,
                alpha=alpha,
                alpha_ema=alpha_ema,
                W_assembled=W_assembled,
                W_base=model.middle.W_base.value,
                keys=keys,
                cfg=cfg,
                aux_scale=aux_scale,
            )
            return total, (breakdown, alpha)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (breakdown, alpha)), grads = grad_fn(model)
        optimizer.update(model, grads)

        return total_loss, breakdown, alpha

    return train_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: DWAModel,
    cfg: DWAConfig,
    num_steps: int,
    batch_size: int,
    rng: jax.Array,
    log_every: int = 100,
) -> list[dict]:
    """
    Minimal training loop for smoke-testing the small config.

    Uses random synthetic data (replace with real DataLoader for actual use).
    Returns a list of loss dicts recorded every log_every steps.
    """
    optimizer = create_optimizer(model, cfg)
    train_step = make_train_step(cfg)

    # Initialise per-vector utilization EMA to uniform
    alpha_ema = jnp.ones(cfg.N) / cfg.N

    log: list[dict] = []

    for step in range(num_steps):
        rng, rng_x, rng_y = jax.random.split(rng, 3)
        batch_x = jax.random.normal(rng_x, (batch_size, cfg.d_model))
        batch_y = jax.random.normal(rng_y, (batch_size, cfg.d_model))

        lambda_sharp = jnp.array(get_lambda_sharp(step, cfg))
        temperature  = jnp.array(cfg.T_temperature)
        aux_scale    = jnp.array(get_aux_scale(step, cfg))

        total_loss, breakdown, alpha = train_step(
            model, optimizer, batch_x, batch_y,
            alpha_ema, lambda_sharp, temperature, aux_scale,
        )

        alpha_ema = update_ema(alpha_ema, alpha, cfg.ema_decay)

        if step % log_every == 0 or step == num_steps - 1:
            phase = (
                "warmup" if step < cfg.phase1_end else
                "gate_on" if step < cfg.phase2_end else
                "sharpen"
            )
            entry = {
                "step": step,
                "phase": phase,
                "lambda_sharp": float(lambda_sharp),
                **{k: float(v) for k, v in breakdown.items()},
            }
            log.append(entry)
            print(
                f"step {step:5d} [{phase:8s}]  "
                f"total={entry['total']:.4f}  "
                f"task={entry['task']:.4f}  "
                f"λ={entry['lambda_sharp']:.2f}"
            )

    return log
