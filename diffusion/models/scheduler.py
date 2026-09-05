"""Learning rate schedules with side-effect-free queries and resumable state."""

import math

from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """Linearly warm up, then decay in proportion to the inverse square root."""

    def __init__(self, optimizer, n_warmup=4000, init_scale=1.0, last_epoch=-1):
        if n_warmup < 1 or init_scale <= 0:
            raise ValueError("Warmup and initial scale must be positive.")
        self.n_warmup = n_warmup
        self.init_scale = init_scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.init_scale * min(step / self.n_warmup, math.sqrt(self.n_warmup / step))
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmupScheduler(LRScheduler):
    """Warm up linearly, then hold the rate or apply exponential decay."""

    def __init__(
        self,
        optimizer,
        n_warmup=2000,
        warmup_start_lr=0.1,
        decay_after_warmup=False,
        decay_rate=0.9999,
        last_epoch=-1,
    ):
        if n_warmup < 1 or not 0 <= warmup_start_lr <= 1 or not 0 < decay_rate <= 1:
            raise ValueError("Invalid warmup or decay parameters.")
        self.n_warmup = n_warmup
        self.warmup_start_lr = warmup_start_lr
        self.decay_after_warmup = decay_after_warmup
        self.decay_rate = decay_rate
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.n_warmup:
            scale = self.warmup_start_lr + (1 - self.warmup_start_lr) * step / self.n_warmup
        elif self.decay_after_warmup:
            scale = self.decay_rate ** (step - self.n_warmup)
        else:
            scale = 1.0
        return [base_lr * scale for base_lr in self.base_lrs]


class CosineAnnealingWarmupScheduler(LRScheduler):
    """Warm up linearly, then anneal to eta_min over T_max steps."""

    def __init__(self, optimizer, n_warmup=2000, T_max=100000, eta_min=0, last_epoch=-1):
        if n_warmup < 1 or T_max < 1 or eta_min < 0:
            raise ValueError("Invalid cosine scheduler parameters.")
        self.n_warmup = n_warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.n_warmup:
            return [base_lr * step / self.n_warmup for base_lr in self.base_lrs]
        progress = min(1.0, (step - self.n_warmup) / self.T_max)
        scale = (1 + math.cos(math.pi * progress)) / 2
        return [self.eta_min + (base_lr - self.eta_min) * scale for base_lr in self.base_lrs]


def get_scheduler(name: str, optimizer, **kwargs):
    """Build a configured learning rate scheduler by name."""
    schedulers = {
        "noam": NoamScheduler,
        "warmup": WarmupScheduler,
        "cosine": CosineAnnealingWarmupScheduler,
    }
    if name.lower() not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")
    return schedulers[name.lower()](optimizer, **kwargs)
