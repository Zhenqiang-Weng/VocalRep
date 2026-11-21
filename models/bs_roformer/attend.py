from collections import namedtuple
from functools import wraps
import inspect
import os
from packaging import version

import torch
from torch import einsum, nn
import torch.nn.functional as F

# Constants

FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)

_SDPA_SUPPORTS_SCALE = (
    hasattr(F, "scaled_dot_product_attention")
    and "scale" in inspect.signature(F.scaled_dot_product_attention).parameters
)

# Helpers


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# Main class


class Attend(nn.Module):
    """Scaled dot-product attention with PyTorch 2.x SDPA acceleration."""

    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse("2.0.0")), (
            "in order to use flash attention, you must be using pytorch 2.0 or above"
        )

        # Determine efficient attention capabilities for CUDA and CPU.
        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        device_version = version.parse(f"{device_properties.major}.{device_properties.minor}")

        if device_version >= version.parse("8.0"):
            if os.name == "nt":
                print_once(
                    "Windows OS detected; using math or memory-efficient SDPA on CUDA tensors."
                )
                self.cuda_config = FlashAttentionConfig(False, True, True)
            else:
                print_once(
                    "GPU compute capability >= 8.0; using FlashAttention / SDPA on CUDA tensors."
                )
                self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once(
                "GPU compute capability < 8.0; using math or memory-efficient SDPA on CUDA tensors."
            )
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        dropout_p = self.dropout if self.training else 0.0

        # Pass scale directly to SDPA when supported to avoid allocating a scaled copy of q.
        if exists(self.scale):
            if _SDPA_SUPPORTS_SCALE:
                return F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, scale=self.scale
                )
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Let PyTorch's native SDPA dispatcher pick the fastest available kernel
        # (FlashAttention-2, cuDNN, memory-efficient, or math fallback) without
        # deprecated context-manager overhead.
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    def forward(self, q, k, v):
        """Compute attention over tensors shaped (batch, heads, seq_len, dim_head)."""
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # Similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # Attention weights
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Aggregate values
        return einsum("b h i j, b h j d -> b h i d", attn, v)
