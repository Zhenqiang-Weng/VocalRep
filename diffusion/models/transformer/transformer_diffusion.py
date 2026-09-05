"""
Transformer diffusion model implementation
"""

import torch
import torch.nn as nn
import math

from ..registry import register_diffusion_model


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding

    Encode time step t as a high-dimensional vector using Transformer positional encoding
    """

    def __init__(self, dim, max_period=1000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        Args:
            t: [B] Time step in the range[0, 1]

        Returns:
            embedding: [B, dim] Time embedding
        """
        device = t.device
        half_dim = self.dim // 2

        # Compute frequencies
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half_dim, dtype=torch.float32, device=device)
            / half_dim
        )

        # Compute phases
        args = t[:, None].float() * freqs[None, :]

        # Sine and cosine
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Pad one element if dim is odd
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


@register_diffusion_model("transformer")
class TransformerDiffusionModel(nn.Module):
    """
    Transformer diffusion model

    Suitable for sequential data such as audio waveforms
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_size)

        # Input projection, assuming 1D or 2D input
        # For audio: [B, T] or [B, C, T]
        self.input_proj = nn.Linear(1, config.hidden_size)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_hidden_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x, t):
        """
        Args:
            x: [B, T] or [B, C, T] Input waveform
            t: [B] Time step

        Returns:
            velocity: [B, T] or [B, C, T] Predicted velocity field
        """
        orig_shape = x.shape

        # Handle multichannel input
        if x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B * C, T)  # [B*C, T]
            t = t.repeat_interleave(C)  # [B*C]
        else:
            B, T = x.shape
            C = 1

        # Input projection: [B, T] -> [B, T, hidden_size]
        x = x.unsqueeze(-1)  # [B, T, 1]
        x = self.input_proj(x)  # [B, T, hidden_size]

        # Time embedding: [B] -> [B, 1, hidden_size]
        t_embed = self.time_embed(t).unsqueeze(1)

        # Add the time embedding
        x = x + t_embed

        # Transformer
        x = self.transformer(x)  # [B, T, hidden_size]

        # Output projection
        x = self.output_proj(x).squeeze(-1)  # [B, T]

        # Restore the original shape
        if len(orig_shape) == 3:
            x = x.reshape(B, C, T)  # [B, C, T]

        return x
