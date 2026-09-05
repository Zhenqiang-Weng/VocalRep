"""
Example diffuser implementation template

Place new model implementations in separate files and register with @register_diffusion_model.
"""

import torch
import torch.nn as nn

from ..registry import register_diffusion_model


@register_diffusion_model("template")
class TemplateDiffuser(nn.Module):
    """Minimal example model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(1, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, 1)
        )

    def forward(self, x, t):
        # Adapt the input to [B, T, 1] -> flatten
        out = self.net(x.unsqueeze(-1))
        return out.squeeze(-1)
