"""
Simple MLP diffusion model implementation
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


@register_diffusion_model("simple_mlp")
class SimpleMLP(nn.Module):
    """
    Simple MLP model for testing

    Input: (batch, channels, time) or (batch, time)
    Output: Same shape as the input
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_size)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, config.ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, config.hidden_size),
        )

    def forward(self, x, t):
        """
        Args:
            x: [B, ...] Input data
            t: [B] Time step

        Returns:
            velocity: [B, ...] Predicted velocity field
        """
        # Save the original shape
        orig_shape = x.shape
        batch_size = x.shape[0]

        # Flatten
        x_flat = x.reshape(batch_size, -1)

        # Time embedding
        t_embed = self.time_embed(t)  # [B, hidden_size]

        # Use global pooling for input features
        if x_flat.shape[1] > self.config.hidden_size:
            # Pool without creating parameters after optimizer initialization.
            x_feat = torch.nn.functional.adaptive_avg_pool1d(
                x_flat.unsqueeze(1), self.config.hidden_size
            ).squeeze(1)
        else:
            # Otherwise pad to hidden_size
            x_feat = torch.nn.functional.pad(x_flat, (0, self.config.hidden_size - x_flat.shape[1]))

        # Concatenate features
        feat = torch.cat([x_feat, t_embed], dim=-1)  # [B, hidden_size * 2]

        # Process with the MLP
        out = self.mlp(feat)  # [B, hidden_size]

        # Project back to the original dimension
        if out.shape[1] != x_flat.shape[1]:
            out = torch.nn.functional.interpolate(
                out.unsqueeze(1), size=x_flat.shape[1], mode="linear", align_corners=False
            ).squeeze(1)

        # Restore the original shape
        out = out.view(orig_shape)

        return out
