from dataclasses import dataclass


@dataclass
class DWAConfig:
    # Pool
    N: int = 512        # pool size
    D: int = 2048       # pool vector dimension (must be >= d_B*r + r*d_A + d_B)
    # Layer dimensions (symmetric: d_A == d_B required for residual)
    d_A: int = 64       # Part A output dim / middle layer input dim
    d_B: int = 64       # Part B input dim / middle layer output dim
    r: int = 4          # assembly rank (polysemantic meaning slots per vector)
    # Retrieval
    S: int = 2          # number of retrieval aspects
    d_k: int = 32       # key/query dimension per aspect
    k_max: int = 8      # top-k vectors used in assembly
    # Model
    d_model: int = 128  # input/output feature dimension
    n_layers_A: int = 2
    n_layers_B: int = 2
    # DWA hyperparameters
    gamma_init: float = 0.01   # LoRA-style residual scale (starts tiny)
    tau_init: float = 0.0      # sigmoid gate threshold
    T_temperature: float = 1.0
    # Sharpness schedule: phase1=0 (pure softmax), ramped in phase 2-3
    lambda_sharp_phase2_end: float = 5.0
    lambda_sharp_final: float = 10.0
    # Auxiliary loss weights
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.01
    beta_util: float = 0.1    # scale in utilization loss
    ema_decay: float = 0.99   # EMA decay for per-vector alpha tracking
    # Phase boundaries (training steps)
    phase1_end: int = 1_000
    phase2_end: int = 10_000

    def __post_init__(self) -> None:
        assert self.d_A == self.d_B, "d_A must equal d_B for the residual connection"
        required = self.d_B * self.r + self.r * self.d_A + self.d_B
        assert self.D >= required, (
            f"D={self.D} < {required} (d_B*r + r*d_A + d_B) — "
            "pool vectors too small to hold factorized components"
        )


# Small config for validation / unit tests
SMALL_CONFIG = DWAConfig(
    N=512,
    D=2048,
    d_A=64,
    d_B=64,
    r=4,
    S=2,
    d_k=32,
    k_max=8,
    d_model=128,
    n_layers_A=2,
    n_layers_B=2,
)

@dataclass
class LMConfig:
    """Config for the language modelling experiment (DWA vs dense baseline)."""

    # Tokenisation — set after loading data
    vocab_size: int = 65

    # Transformer architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers_A: int = 2   # transformer blocks *before* the DWA middle layer
    n_layers_B: int = 2   # transformer blocks *after*  the DWA middle layer
    seq_len: int = 128

    # DWA pool
    N: int = 256
    D: int = 2048   # must be >= d_model*r + r*d_model + d_model
    r: int = 4
    S: int = 2
    d_k: int = 32
    k_max: int = 16

    # DWA forward-pass hyperparams
    gamma_init: float = 0.01
    tau_init: float = 0.0
    T_temperature: float = 1.0
    lambda_sharp_phase2_end: float = 5.0
    lambda_sharp_final: float = 10.0

    # Auxiliary loss weights
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.01
    beta_util: float = 0.1
    ema_decay: float = 0.99

    # Training schedule
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 300
    max_steps: int = 5000
    eval_every: int = 250
    eval_steps: int = 50
    phase1_end: int = 500    # warmup: lambda_sharp=0, no aux losses
    phase2_end: int = 3000   # gate-on: lambda ramps 0→5
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        required = self.d_model * self.r + self.r * self.d_model + self.d_model
        assert self.D >= required, (
            f"D={self.D} < {required} required "
            f"(d_model={self.d_model}, r={self.r})"
        )

    def to_dwa_config(self) -> "DWAConfig":
        """Extract a DWAConfig for the pool/retrieval/middle building blocks."""
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


# Full config (~1B params in pool alone — use bfloat16)
FULL_CONFIG = DWAConfig(
    N=65_536,
    D=16_384,
    d_A=256,
    d_B=256,
    r=24,
    S=4,
    d_k=64,
    k_max=16,
    d_model=512,
    n_layers_A=4,
    n_layers_B=4,
    phase1_end=1_000,
    phase2_end=10_000,
)
