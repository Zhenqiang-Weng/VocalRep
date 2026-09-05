"""Shared, validated configuration for diffusion models and training wrappers."""

from dataclasses import dataclass, field
import math


@dataclass
class DiffusionConfig:
    """Configure the model, flow path, optimizer, STFT, and chunked inference."""

    model_type: str = "dit"
    model_kwargs: dict = field(default_factory=dict)
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_hidden_size: int = 1024
    dropout: float = 0.1
    sigma: float = 0.0
    n_train_start: int = 100000
    diffusion_loss_weight: float = 1.0
    optimizer_type: str = "adam"
    lr: float = 9e-5
    betas: list | None = None
    eps: float = 1e-8
    weight_decay: float = 0.0
    scheduler_type: str = "noam"
    n_warmup: int = 4000
    init_scale: float = 0.5
    grad_clip_thresh: float = 1.0
    grad_acc_step: int = 1
    T_max: int = 10
    eta_min: float = 0.0
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    window: str = "hann"
    center: bool = True
    normalized: bool = False
    use_stft: bool | None = None
    chunk_size_seconds: float = 8.0
    sample_rate: int = 44100
    num_overlap: int = 2
    inference_batch_size: int = 1

    def __post_init__(self) -> None:
        self.model_type = self.model_type.lower()
        if self.betas is None:
            self.betas = [0.9, 0.999]
        if self.use_stft is None:
            self.use_stft = self.model_type == "dit"
        if self.model_type == "dit" and not self.use_stft:
            raise ValueError("DiT requires STFT input.")
        for name in (
            "lr",
            "chunk_size_seconds",
            "sample_rate",
            "n_fft",
            "hop_length",
            "win_length",
            "grad_acc_step",
            "n_warmup",
            "T_max",
            "num_overlap",
            "inference_batch_size",
            "hidden_size",
            "num_heads",
            "num_layers",
            "ffn_hidden_size",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.sigma) or self.sigma < 0 or self.diffusion_loss_weight < 0:
            raise ValueError("Noise level and diffusion loss weight must be non-negative.")
        if self.n_train_start < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Require a non-negative start step and dropout in [0, 1).")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must not exceed n_fft.")
        if self.window not in {"hann", "hamming", "blackman"}:
            raise ValueError(f"Unsupported STFT window: {self.window}")
        if self.use_stft and not self.center:
            raise ValueError(
                "Centered STFT is required for exact reconstruction with tapered windows."
            )
