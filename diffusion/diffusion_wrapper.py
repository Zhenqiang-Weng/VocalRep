"""Flow-matching training and chunked enhancement for audio diffusion models."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .models.config import DiffusionConfig
from .models.flow_matching import ConditionalFlowMatcher
from .models.registry import get_diffusion_model
from .models.scheduler import get_scheduler
from utils.auxiliary_training import AuxiliaryTraining


class DiffusionWrapper(AuxiliaryTraining):
    """Train an auxiliary velocity model and integrate it with Euler or RK4."""

    def __init__(
        self,
        config: DiffusionConfig | dict,
        accelerator=None,
        device: Optional[torch.device] = None,
    ):
        self.config = DiffusionConfig(**config) if isinstance(config, dict) else config
        self.accelerator = accelerator
        self.device = torch.device(accelerator.device if accelerator else device or "cpu")
        self.model = get_diffusion_model(self.config.model_type, self.config).to(self.device)
        self.flow_matcher = ConditionalFlowMatcher(sigma=self.config.sigma)
        self.window = getattr(torch, f"{self.config.window}_window")(self.config.win_length)
        self._init_optimizer()
        self._prepare_training()

    def _init_optimizer(self) -> None:
        cfg = self.config
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        if cfg.optimizer_type not in optimizers:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer_type}")
        kwargs = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
        if cfg.optimizer_type == "sgd":
            kwargs["momentum"] = 0.9
        else:
            kwargs.update(betas=cfg.betas, eps=cfg.eps)
        self.optimizer = optimizers[cfg.optimizer_type](self.model.parameters(), **kwargs)
        if cfg.scheduler_type == "constant":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        else:
            kwargs = {"n_warmup": cfg.n_warmup}
            if cfg.scheduler_type == "noam":
                kwargs["init_scale"] = cfg.init_scale
            elif cfg.scheduler_type == "cosine":
                kwargs.update(T_max=cfg.T_max, eta_min=cfg.eta_min)
            self.scheduler = get_scheduler(cfg.scheduler_type, self.optimizer, **kwargs)

    @staticmethod
    def _validate_wave(wave: torch.Tensor) -> None:
        if wave.ndim not in (2, 3) or min(wave.shape) <= 0:
            raise ValueError("Expected a non-empty waveform shaped [B, T] or [B, C, T].")
        if not wave.is_floating_point() or not torch.isfinite(wave).all():
            raise ValueError("Waveforms must contain finite floating-point samples.")

    def stft(self, wave: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, T] or [B, T] to real/imaginary features [B*C, 2, F, T]."""
        self._validate_wave(wave)
        wave = wave.float()
        flattened = wave.reshape(-1, wave.shape[-1])
        spec = torch.stft(
            flattened,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(wave),
            center=self.config.center,
            pad_mode="constant",
            normalized=self.config.normalized,
            return_complex=True,
        )
        return torch.view_as_real(spec).permute(0, 3, 1, 2).contiguous()

    def istft(self, spec: torch.Tensor, length: int, num_channels: int = 2) -> torch.Tensor:
        """Reconstruct a waveform with the requested sample and channel counts."""
        if spec.ndim != 4 or spec.shape[1] != 2 or spec.shape[0] % num_channels:
            raise ValueError("Expected [B*C, 2, F, T] STFT features with valid channel grouping.")
        complex_spec = torch.view_as_complex(spec.float().permute(0, 2, 3, 1).contiguous())
        wave = torch.istft(
            complex_spec,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(spec.device),
            center=self.config.center,
            normalized=self.config.normalized,
            length=length,
        )
        return wave.reshape(-1, num_channels, length)

    def _predict_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Adapt time windows to a fixed-size DiT without hard-coded frame counts."""
        model = self._unwrapped_model()
        input_size = getattr(model, "input_size", None) if self.config.use_stft else None
        if input_size is None:
            prediction = self.model(x, t)
        else:
            frequencies, frames = input_size
            if x.shape[-2] != frequencies:
                raise ValueError(f"DiT expects {frequencies} frequency bins, got {x.shape[-2]}.")
            chunks = []
            for offset in range(0, x.shape[-1], frames):
                part = x[..., offset : offset + frames]
                actual = part.shape[-1]
                padded = F.pad(part, (0, frames - actual))
                chunks.append(self.model(padded, t)[..., :actual])
            prediction = torch.cat(chunks, dim=-1)
        if prediction.shape != x.shape:
            raise ValueError(f"Velocity shape {prediction.shape} does not match input {x.shape}.")
        return prediction

    def _train_pair(self, source, target, current_step, t, train_steps, loss_function) -> dict:
        self._validate_wave(source)
        self._validate_wave(target)
        if source.shape != target.shape:
            raise ValueError("Source and target waveforms must have identical shapes.")
        if train_steps < 1:
            raise ValueError("train_steps must be positive.")
        if current_step < self.config.n_train_start:
            zero = source.new_zeros(())
            return {"diffusion_loss": zero, "flow_matching_loss": zero, "train_steps": 0}
        self.model.train()
        source, target = source.detach(), target.detach()
        channels = source.shape[1] if source.ndim == 3 else 1
        original_batch = source.shape[0]
        if self.config.use_stft:
            source, target = self.stft(source), self.stft(target)
        if t is not None:
            t = torch.as_tensor(t, device=source.device, dtype=source.dtype).flatten()
            if t.numel() == 1:
                t = t.expand(source.shape[0])
            elif self.config.use_stft and t.numel() == original_batch:
                t = t.repeat_interleave(channels)
            if t.shape != (source.shape[0],):
                raise ValueError("t must be a scalar or one value per original batch item.")
            train_steps = 1
        terms = []
        for _ in range(train_steps):
            sampled_t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                source, target, t=t
            )
            predicted = self._predict_velocity(xt, sampled_t)
            terms.append(loss_function(predicted.float(), ut.float()))
        flow_loss = torch.stack(terms).mean()
        weighted_loss = flow_loss * self.config.diffusion_loss_weight
        self._backward_step(weighted_loss)
        return {
            "diffusion_loss": weighted_loss.detach(),
            "flow_matching_loss": flow_loss.detach(),
            "train_steps": train_steps,
        }

    def train_step_dual(
        self,
        wave_source: torch.Tensor,
        wave_target: torch.Tensor,
        current_step: int,
        t: Optional[torch.Tensor] = None,
        train_steps: int = 2,
    ) -> dict:
        """Train the velocity field from separated audio toward its reference waveform."""
        return self._train_pair(wave_source, wave_target, current_step, t, train_steps, F.l1_loss)

    def train_step_single(
        self, wave: torch.Tensor, current_step: int, t: Optional[torch.Tensor] = None
    ) -> dict:
        """Train a velocity field from Gaussian noise toward a target waveform."""
        return self._train_pair(torch.randn_like(wave), wave, current_step, t, 1, F.mse_loss)

    def _solve_ode(self, x: torch.Tensor, num_steps: int, method: str) -> torch.Tensor:
        if num_steps < 1 or method not in {"euler", "rk4"}:
            raise ValueError("Require positive num_steps and method 'euler' or 'rk4'.")
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
            k1 = self._predict_velocity(x, t)
            if method == "euler":
                x = x + dt * k1
            else:
                k2 = self._predict_velocity(x + dt * k1 / 2, t + dt / 2)
                k3 = self._predict_velocity(x + dt * k2 / 2, t + dt / 2)
                k4 = self._predict_velocity(x + dt * k3, t + dt)
                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    @torch.inference_mode()
    def inference(
        self,
        wave_source: torch.Tensor,
        num_steps: int = 10,
        method: str = "euler",
        chunk_size_seconds: Optional[float] = None,
        num_overlap: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Enhance audio with overlap-add, preserving input shape and model mode."""
        self._validate_wave(wave_source)
        seconds = (
            self.config.chunk_size_seconds if chunk_size_seconds is None else chunk_size_seconds
        )
        overlap = self.config.num_overlap if num_overlap is None else num_overlap
        batch_size = self.config.inference_batch_size if batch_size is None else batch_size
        if not math.isfinite(seconds) or seconds <= 0 or overlap < 1 or batch_size < 1:
            raise ValueError("Chunk duration, overlap, and batch size must be positive.")
        if num_steps < 1 or method not in {"euler", "rk4"}:
            raise ValueError("Require positive num_steps and method 'euler' or 'rk4'.")
        chunk = int(seconds * self.config.sample_rate)
        if self.config.use_stft:
            chunk = chunk // self.config.hop_length + 1
        if chunk < 1 or overlap > chunk:
            raise ValueError("Chunk size must be positive and at least as large as overlap.")
        step = chunk // overlap
        training = self.model.training
        self.model.eval()
        try:
            source = self.stft(wave_source) if self.config.use_stft else wave_source.float()
            total = source.shape[-1]
            result = torch.zeros_like(source)
            counter = torch.zeros(total, device=source.device)
            # A positive window keeps every sample covered, including single chunks.
            window = torch.hann_window(chunk, periodic=False, device=source.device).clamp_min(1e-3)
            starts = list(range(0, total, step))
            for offset in range(0, len(starts), batch_size):
                locations = starts[offset : offset + batch_size]
                pieces = [
                    F.pad(source[..., start : start + chunk], (0, max(0, start + chunk - total)))
                    for start in locations
                ]
                predictions = self._solve_ode(torch.cat(pieces, dim=0), num_steps, method)
                for start, prediction in zip(locations, predictions.split(source.shape[0])):
                    length = min(chunk, total - start)
                    result[..., start : start + length] += (
                        prediction[..., :length] * window[:length]
                    )
                    counter[start : start + length] += window[:length]
            result = result / counter
            if self.config.use_stft:
                channels = wave_source.shape[1] if wave_source.ndim == 3 else 1
                result = self.istft(result, wave_source.shape[-1], channels).reshape(
                    wave_source.shape
                )
            if not torch.isfinite(result).all():
                raise ValueError("Diffusion inference produced non-finite audio.")
            return result
        finally:
            self.model.train(training)
