"""Conditional flow matching with deterministic or Gaussian bridge paths."""

import math
from typing import Optional

import torch
from torch import nn


class ConditionalFlowMatcher:
    """Sample the path x_t=(1-t)x0+t*x1+sigma*sqrt(t*(1-t))*eps and its derivative."""

    def __init__(self, sigma: float = 0.0):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and non-negative.")
        self.sigma = sigma

    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        times = torch.rand(batch_size, device=device)
        return times.clamp(1e-5, 1 - 1e-5) if self.sigma else times

    @staticmethod
    def _expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t.numel() == 1:
            t = t.reshape(1).expand(x.shape[0])
        if t.numel() != x.shape[0] or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError("Time must contain one finite value in [0, 1] per sample.")
        return t.reshape(-1, *([1] * (x.ndim - 1)))

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample an intermediate state, optionally using fixed noise for reproducibility."""
        if x0.shape != x1.shape:
            raise ValueError("Flow endpoints must have matching shapes.")
        time = self._expand_time(t, x0)
        state = (1 - time) * x0 + time * x1
        if self.sigma:
            eps = self.sample_noise_like(x0) if eps is None else eps
            state = state + self.sigma * torch.sqrt(time * (1 - time)) * eps
        return state

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        xt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the time derivative, including the Gaussian bridge noise term."""
        velocity = x1 - x0
        if self.sigma:
            if t is None or xt is None:
                raise ValueError("Stochastic flow requires both t and xt.")
            time = self._expand_time(t, x0)
            if ((time <= 0) | (time >= 1)).any():
                raise ValueError("Stochastic bridge derivatives require 0 < t < 1.")
            mean = (1 - time) * x0 + time * x1
            velocity = velocity + (0.5 - time) / (time * (1 - time)) * (xt - mean)
        return velocity

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ):
        """Return batch times, sampled states, target velocities, and optional noise."""
        if x0.shape != x1.shape or x0.ndim < 2:
            raise ValueError("Flow endpoints must share a batched tensor shape.")
        t = self.sample_time(x0.shape[0], x0.device) if t is None else t
        t = self._expand_time(t, x0).reshape(x0.shape[0])
        eps = self.sample_noise_like(x0) if self.sigma else None
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return (t, xt, ut, eps) if return_noise else (t, xt, ut)

    def compute_loss(
        self, model: nn.Module, x0: torch.Tensor, x1: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the mean-squared velocity prediction error."""
        t, xt, ut = self.sample_location_and_conditional_flow(x0, x1, t)
        return nn.functional.mse_loss(model(xt, t), ut)


class OptimalTransportFlowMatcher(ConditionalFlowMatcher):
    """Use the deterministic straight-line path."""

    def __init__(self):
        super().__init__(sigma=0.0)


class StochasticFlowMatcher(ConditionalFlowMatcher):
    """Use a Gaussian bridge with a strictly positive noise level."""

    def __init__(self, sigma: float = 0.1):
        if sigma <= 0:
            raise ValueError("StochasticFlowMatcher requires sigma > 0.")
        super().__init__(sigma=sigma)


OTFlowMatcher = OptimalTransportFlowMatcher
