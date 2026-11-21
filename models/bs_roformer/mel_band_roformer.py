from functools import partial
import random

import torch
from torch import nn, einsum, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

import numpy as np

from models.bs_roformer.attend import Attend

try:
    from models.bs_roformer.attend_sage import Attend as AttendSage
except Exception:
    pass
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack, reduce, repeat
from einops.layers.torch import Rearrange

from librosa import filters
from pathlib import Path

from .conditioner import BandEmbedder

_HAS_F_RMS_NORM = hasattr(F, "rms_norm")

# Helper functions


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def _get_cached_window(cache: dict, key, factory) -> Tensor:
    window = cache.get(key)
    if window is None:
        window = factory()
        cache[key] = window
    return window


def _compute_multi_stft_loss(
    recon_audio: Tensor,
    target: Tensor,
    window_sizes: Tuple[int, ...],
    multi_stft_n_fft: int,
    multi_stft_kwargs: dict,
    multi_stft_window_fn: Callable,
    window_cache: dict,
) -> Tensor:
    recon_f32 = recon_audio.float()
    target_f32 = target.float()
    recon_flat = recon_f32.reshape(-1, recon_f32.shape[-1])
    target_flat = target_f32.reshape(-1, target_f32.shape[-1])
    device = recon_flat.device
    device_key = str(device)

    multi_loss = recon_f32.new_zeros(())
    for window_size in window_sizes:
        window = _get_cached_window(
            window_cache,
            ("multi", window_size, device_key),
            lambda ws=window_size: multi_stft_window_fn(ws, device=device, dtype=torch.float32),
        )
        res_stft_kwargs = dict(
            n_fft=max(window_size, multi_stft_n_fft),
            win_length=window_size,
            return_complex=True,
            window=window,
            **multi_stft_kwargs,
        )
        recon_Y = torch.stft(recon_flat, **res_stft_kwargs)
        with torch.no_grad():
            target_Y = torch.stft(target_flat, **res_stft_kwargs)
        multi_loss = multi_loss + F.l1_loss(recon_Y, target_Y)
        del recon_Y, target_Y

    return multi_loss


# Norm


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if _HAS_F_RMS_NORM and not torch.onnx.is_in_onnx_export():
            return F.rms_norm(x, (x.shape[-1],), self.gamma, eps=1e-12)
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# Attention


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        flash=True,
        sage_attention=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        if sage_attention:
            self.attend = AttendSage(flash=flash, dropout=dropout)
        else:
            self.attend = Attend(flash=flash, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape

        q, k, v = self.to_qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x).transpose(1, 2).unsqueeze(-1).sigmoid()
        out = (out * gates).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
        self, *, dim, dim_head=32, heads=8, scale=8, flash=False, dropout=0.0, sage_attention=False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        if sage_attention:
            self.attend = AttendSage(scale=scale, dropout=dropout, flash=flash)
        else:
            self.attend = Attend(scale=scale, dropout=dropout, flash=flash)

        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        linear_attn=False,
        sage_attention=False,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                )

            self.layers.append(
                ModuleList([attn, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)])
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# bandsplit module


class BandSplit(Module):
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class
class MelBandRoformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        flash_attn=True,
        dim_freqs_in=1025,
        sample_rate=44100,  # needed for mel filter bank from librosa
        stft_n_fft=2048,
        stft_hop_length=512,
        # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=1,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        sage_attention=False,
        debug_visualize=False,
        debug_visualize_dir=None,
        debug_visualize_max_items=10,
        debug_visualize_batch_limit=1,
        mask_highfreq_cutoff_enabled=False,
        mask_highfreq_cutoff_hz=17000.0,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        self.mask_highfreq_cutoff_enabled = mask_highfreq_cutoff_enabled
        self.mask_highfreq_cutoff_hz = mask_highfreq_cutoff_hz

        self.layers = ModuleList([])

        if sage_attention:
            print("Use Sage Attention")

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            sage_attention=sage_attention,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(
                    Transformer(
                        depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs
                    )
                )
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs,
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_n_fft),
            return_complex=True,
        ).shape[1]

        freq_bins_hz = torch.linspace(0.0, sample_rate / 2, steps=freqs)
        if stereo:
            freq_bins_hz = repeat(freq_bins_hz, "f -> f s", s=self.audio_channels)
            freq_bins_hz = rearrange(freq_bins_hz, "f s -> (f s)")
        self.register_buffer("freq_bins_hz", freq_bins_hz, persistent=False)

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.0

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.0

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), (
            "all frequencies need to be covered by all bands for now"
        )

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length
        self._stft_window_cache = {}

    def _get_stft_window(self, device: torch.device) -> Tensor:
        return _get_cached_window(
            self._stft_window_cache,
            ("main", str(device)),
            lambda: self.stft_window_fn(device=device, dtype=torch.float32),
        )

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        """Run MelBandRoformer forward pass and optional loss calculation."""
        device = raw_audio.device
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        # Compute STFT in float32 for cuFFT / BF16 autocast safety.
        raw_audio_flat = raw_audio.reshape(batch * channels, raw_audio_length).float()
        stft_window = self._get_stft_window(device)

        stft_repr = torch.stft(
            raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)
        freq_bins, t_stft = stft_repr.shape[1], stft_repr.shape[2]
        stft_repr = stft_repr.view(batch, channels, freq_bins, t_stft, 2)

        # Merge stereo / mono into frequency axis for band splitting.
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * channels, t_stft, 2)

        # Gather all overlapping mel-band frequency bins in one indexing op.
        x = stft_repr[:, self.freq_indices]
        x = x.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        b, t, f, d = x.shape

        # Axial / hierarchical attention
        skip_sum = None
        for transformer_block in self.layers:
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block
                x_lin = x.reshape(b, t * f, d)
                if use_ckpt:
                    x_lin = checkpoint(linear_transformer, x_lin, use_reentrant=False)
                else:
                    x_lin = linear_transformer(x_lin)
                x = x_lin.view(b, t, f, d)
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection and skip_sum is not None:
                x = x + skip_sum

            x = x.transpose(1, 2).reshape(b * f, t, d)
            if use_ckpt:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            if use_ckpt:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x = x.view(b, t, f, d)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        num_stems = len(self.mask_estimators)
        if use_ckpt:
            masks = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1
            )
        else:
            masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = masks.view(batch, num_stems, t_stft, -1, 2).permute(0, 1, 3, 2, 4)

        # Complex multiplication in float32 for iSTFT safety.
        stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1).float().contiguous())
        masks_complex = torch.view_as_complex(masks.float().contiguous()).type(stft_complex.dtype)

        # Average estimated masks across overlapped frequencies without copying stft_complex.
        scatter_indices = self.freq_indices.view(1, 1, -1, 1).expand(
            batch, num_stems, -1, stft_complex.shape[-1]
        )
        masks_summed = torch.zeros(
            (batch, num_stems, stft_complex.shape[2], stft_complex.shape[3]),
            device=stft_complex.device,
            dtype=stft_complex.dtype,
        ).scatter_add_(2, scatter_indices, masks_complex)

        denom = self.num_bands_per_freq.repeat_interleave(channels).view(1, 1, -1, 1)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        if self.mask_highfreq_cutoff_enabled:
            highfreq_filter = (self.freq_bins_hz <= self.mask_highfreq_cutoff_hz).to(
                masks_averaged.device, dtype=masks_averaged.dtype
            )
            masks_averaged = masks_averaged * highfreq_filter.view(1, 1, -1, 1)

        stft_masked = stft_complex * masks_averaged
        stft_masked = (
            stft_masked.view(batch, num_stems, freq_bins, self.audio_channels, t_stft)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * num_stems * self.audio_channels, freq_bins, t_stft)
        )

        recon_audio = torch.istft(
            stft_masked,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )

        recon_audio = recon_audio.view(batch, num_stems, self.audio_channels, -1)
        if num_stems == 1:
            recon_audio = recon_audio.squeeze(1)

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = target.unsqueeze(-2)

        target = target[..., : recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio.float(), target.float())
        multi_stft_resolution_loss = _compute_multi_stft_loss(
            recon_audio,
            target,
            self.multi_stft_resolutions_window_sizes,
            self.multi_stft_n_fft,
            self.multi_stft_kwargs,
            self.multi_stft_window_fn,
            self._stft_window_cache,
        )

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )
        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)

    def freeze_transformer_layers(self):
        """Freeze all transformer blocks (time/freq/linear) in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_transformer_layers(self):
        """Unfreeze previously frozen transformer blocks in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = True


class ConvBlock2D(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, D, T, F) -> (B, D, T, F)
            nn.Conv2d(
                dim, dim, kernel_size=kernel_size, padding="same", groups=dim
            ),  # Depthwise convolution
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),  # Pointwise convolution
        )

    def forward(self, x):
        """
        Input: x in shape (B, T, F, D) - B: batch, T: time, F: freq, D: dim
        """
        x_permuted = x.permute(0, 3, 1, 2)
        x_conv = self.conv(x_permuted)
        x_restored = x_conv.permute(0, 2, 3, 1)

        return x_restored + x


class TDMelBandRoformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        num_bands=60,
        dim_head=64,
        heads=8,
        conv_kernel_size=3,
        attn_dropout=0.1,
        ff_dropout=0.1,
        flash_attn=True,
        dim_freqs_in=1025,
        sample_rate=44100,  # needed for mel filter bank from librosa
        stft_n_fft=2048,
        stft_hop_length=512,
        # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        zero_dc=True,
        mask_estimator_depth=1,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        sage_attention=False,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        self.layers = ModuleList([])
        self.conv_blocks = ModuleList([])

        if sage_attention:
            print("Use Sage Attention")

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            sage_attention=sage_attention,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(
                    Transformer(
                        depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs
                    )
                )
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs,
                )
            )

            self.layers.append(nn.ModuleList(tran_modules))
            self.conv_blocks.append(ConvBlock2D(dim, kernel_size=conv_kernel_size))

        self.zero_init_conv_blocks()
        # self.freeze_transformer_layers()

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_n_fft),
            return_complex=True,
        ).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.0

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.0

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), (
            "all frequencies need to be covered by all bands for now"
        )

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        # whether to zero out dc

        self.zero_dc = zero_dc

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length
        self._stft_window_cache = {}

    def _get_stft_window(self, device: torch.device) -> Tensor:
        return _get_cached_window(
            self._stft_window_cache,
            ("main", str(device)),
            lambda: self.stft_window_fn(device=device, dtype=torch.float32),
        )

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        """Run TDMelBandRoformer forward pass and optional loss calculation."""
        device = raw_audio.device
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        raw_audio_flat = raw_audio.reshape(batch * channels, raw_audio_length).float()
        stft_window = self._get_stft_window(device)

        stft_repr = torch.stft(
            raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)
        freq_bins, t_stft = stft_repr.shape[1], stft_repr.shape[2]
        stft_repr = stft_repr.view(batch, channels, freq_bins, t_stft, 2)

        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * channels, t_stft, 2)

        x = stft_repr[:, self.freq_indices]
        x = x.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        b, t, f, d = x.shape

        skip_sum = None
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block
                x_lin = x.reshape(b, t * f, d)
                if use_ckpt:
                    x_lin = checkpoint(linear_transformer, x_lin, use_reentrant=False)
                else:
                    x_lin = linear_transformer(x_lin)
                x = x_lin.view(b, t, f, d)
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection and skip_sum is not None:
                x = x + skip_sum

            x = x.transpose(1, 2).reshape(b * f, t, d)
            if use_ckpt:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            if use_ckpt:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x = x.view(b, t, f, d)
            if use_ckpt:
                x = checkpoint(self.conv_blocks[i], x, use_reentrant=False)
            else:
                x = self.conv_blocks[i](x)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        num_stems = len(self.mask_estimators)
        if use_ckpt:
            masks = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1
            )
        else:
            masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = masks.view(batch, num_stems, t_stft, -1, 2).permute(0, 1, 3, 2, 4)

        stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1).float().contiguous())
        masks_complex = torch.view_as_complex(masks.float().contiguous()).type(stft_complex.dtype)

        scatter_indices = self.freq_indices.view(1, 1, -1, 1).expand(
            batch, num_stems, -1, stft_complex.shape[-1]
        )
        masks_summed = torch.zeros(
            (batch, num_stems, stft_complex.shape[2], stft_complex.shape[3]),
            device=stft_complex.device,
            dtype=stft_complex.dtype,
        ).scatter_add_(2, scatter_indices, masks_complex)

        denom = self.num_bands_per_freq.repeat_interleave(channels).view(1, 1, -1, 1)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_masked = stft_complex * masks_averaged
        stft_masked = (
            stft_masked.view(batch, num_stems, freq_bins, self.audio_channels, t_stft)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * num_stems * self.audio_channels, freq_bins, t_stft)
        )

        if self.zero_dc:
            stft_masked = stft_masked.index_fill(1, tensor(0, device=device), 0.0)

        recon_audio = torch.istft(
            stft_masked,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )

        recon_audio = recon_audio.view(batch, num_stems, self.audio_channels, -1)
        if num_stems == 1:
            recon_audio = recon_audio.squeeze(1)

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = target.unsqueeze(-2)

        target = target[..., : recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio.float(), target.float())
        multi_stft_resolution_loss = _compute_multi_stft_loss(
            recon_audio,
            target,
            self.multi_stft_resolutions_window_sizes,
            self.multi_stft_n_fft,
            self.multi_stft_kwargs,
            self.multi_stft_window_fn,
            self._stft_window_cache,
        )

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )
        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss, recon_audio

        return total_loss, (loss, multi_stft_resolution_loss), recon_audio

    def zero_init_conv_blocks(self):
        """Initialize all ConvBlock2D modules in self.conv_blocks."""
        for conv_block in self.conv_blocks:
            for m in conv_block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def freeze_transformer_layers(self):
        """Freeze all transformer blocks (time/freq/linear) in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_transformer_layers(self):
        """Unfreeze previously frozen transformer blocks in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = True

    def frozen_all(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_only_band_split_and_mask_estimators(self):
        """Unfreeze only band split and mask estimators parameters."""
        for param in self.band_split.parameters():
            param.requires_grad = True
        for mask_estimator in self.mask_estimators:
            for param in mask_estimator.parameters():
                param.requires_grad = True

    def unfreeze_which_estimates(self, stem_indices: list[int]):
        """Unfreeze only specified mask estimators parameters.

        Args:
            stem_indices (List[int]): List of stem indices to unfreeze.
        """
        for idx in stem_indices:
            if idx < 0 or idx >= len(self.mask_estimators):
                raise ValueError(f"Stem index {idx} is out of range.")
            mask_estimator = self.mask_estimators[idx]
            for param in mask_estimator.parameters():
                param.requires_grad = True

    def freeze_which_estimates(self, stem_indices: list[int]):
        """Freeze only specified mask estimators parameters.

        Args:
            stem_indices (List[int]): List of stem indices to freeze.
        """
        for idx in stem_indices:
            if idx < 0 or idx >= len(self.mask_estimators):
                raise ValueError(f"Stem index {idx} is out of range.")
            mask_estimator = self.mask_estimators[idx]
            for param in mask_estimator.parameters():
                param.requires_grad = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ScaleTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        linear_attn=False,
        sage_attention=False,
        drop_prob=0.5,
    ):
        super().__init__()
        self.layers = ModuleList([])
        self.adaLN_modulations = nn.ModuleList([])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()
        self.drop_prob = drop_prob

        for _ in range(depth):
            norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                )

            norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.layers.append(
                ModuleList(
                    [norm1, attn, norm2, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]
                )
            )

            ada = nn.Sequential(
                nn.SELU(),
                nn.Linear(dim, 6 * dim, bias=True),
            )
            self.adaLN_modulations.append(ada)
        self.init_weights()

    def forward(self, x, band_embedding=None):
        if band_embedding is None:
            band_embedding = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)

        if self.training and self.drop_prob > 0.0 and random.random() < self.drop_prob:
            band_embedding = torch.zeros_like(band_embedding)

        for (norm1, attn, norm2, ff), ada in zip(self.layers, self.adaLN_modulations):
            shift_scale = ada(band_embedding)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shift_scale.chunk(
                6, dim=-1
            )
            x = x + attn(modulate(norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
            x = x + ff(modulate(norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return self.norm(x)

    def init_weights(self):
        """Initialize only adaLN_modulations."""
        GATE_BIAS_INIT = (
            1.0  # sigmoid(1.0) is about 0.73, allowing residual gradients from initialization
        )

        for ada in self.adaLN_modulations:
            linear = ada[1]  # SELU -> Linear(dim, 6 * dim)

            # Zero weights and biases
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

            # Adjust the gate bias to keep the residual branch partially active
            with torch.no_grad():
                D = linear.bias.shape[0] // 6
                linear.bias[2 * D : 3 * D].fill_(GATE_BIAS_INIT)  # gate_msa
                linear.bias[5 * D : 6 * D].fill_(GATE_BIAS_INIT)  # gate_mlp


class SpeakerMelBandRoformerExportable(nn.Module):
    """Exportable Mel-Band Roformer core conditioned on speaker embeddings."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = True,
        num_stems: int = 3,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        num_bands: int = 60,
        sample_rate: int = 44100,
        stft_n_fft: int = 2048,
        dim_freqs_in: int = 1025,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        flash_attn: bool = True,
        mask_estimator_depth: int = 2,
        mlp_expansion_factor: int = 4,
        use_torch_checkpoint: bool = False,
        skip_connection: bool = False,
        sage_attention: bool = False,
        spk_embd_dim: int = 192,
        **kwargs,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.dim = dim

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            sage_attention=sage_attention,
        )

        time_rotary = RotaryEmbedding(dim=dim_head)
        freq_rotary = RotaryEmbedding(dim=dim_head)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ScaleTransformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary,
                            **transformer_kwargs,
                        ),
                        ScaleTransformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary,
                            **transformer_kwargs,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = RMSNorm(dim)

        # ================= Mel Band Partition =================

        freqs = dim_freqs_in
        mel_fb = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands).astype(np.float32)

        mel_fb = torch.from_numpy(mel_fb)
        mel_fb[0, 0] = 1.0
        mel_fb[-1, -1] = 1.0

        freqs_per_band = mel_fb > 0
        assert freqs_per_band.any(dim=0).all()

        repeated = torch.arange(freqs).unsqueeze(0).repeat(num_bands, 1)
        freq_indices_base = repeated[freqs_per_band]

        # Interleave stereo channels
        freq_indices_stereo = (
            freq_indices_base.unsqueeze(-1).repeat(1, self.audio_channels) * self.audio_channels
            + torch.arange(self.audio_channels)
        ).reshape(-1)

        self.register_buffer("freq_indices", freq_indices_stereo, persistent=False)

        num_bands_per_freq = freqs_per_band.sum(dim=0)
        num_bands_per_freq_stereo = num_bands_per_freq.repeat_interleave(self.audio_channels)

        self.register_buffer(
            "num_bands_per_freq_stereo", num_bands_per_freq_stereo, persistent=False
        )

        freqs_per_bands_with_complex = tuple(
            2 * int(f) * self.audio_channels for f in freqs_per_band.sum(dim=1).tolist()
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for _ in range(self.num_stems)
            ]
        )

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)

    # ================= Manual Complex Multiply =================

    def _complex_mul(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute elementwise complex multiplication on tensors shaped (..., 2)."""
        ar, ai = a[..., 0], a[..., 1]
        br, bi = b[..., 0], b[..., 1]
        real = ar * br - ai * bi
        imag = ar * bi + ai * br
        return torch.stack([real, imag], dim=-1)

    # ================= Forward =================

    def forward(self, stft_repr: Tensor, speaker_embedding: Tensor) -> Tensor:
        b, f_stereo, t_stft, _ = stft_repr.shape
        device = stft_repr.device
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        # Gather mel-band frequency bins
        freq_idx = self.freq_indices
        x = stft_repr[:, freq_idx]  # (b, f_sel, t, 2)

        # Fold complex dimension into band features
        x = x.permute(0, 2, 1, 3).reshape(b, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)  # (b, t, nb, d)

        _, _, nb, d = x.shape

        spk = self.spk_embd_to_dim(speaker_embedding)
        spk_t = spk.unsqueeze(1).expand(b, nb, -1).reshape(b * nb, -1)
        spk_f = spk.unsqueeze(1).expand(b, t_stft, -1).reshape(b * t_stft, -1)

        skip_sum = None
        for time_tr, freq_tr in self.layers:
            if self.skip_connection and skip_sum is not None:
                x = x + skip_sum

            x = x.transpose(1, 2).reshape(b * nb, t_stft, d)
            if use_ckpt:
                x = checkpoint(time_tr, x, spk_t, use_reentrant=False)
            else:
                x = time_tr(x, spk_t)

            x = x.view(b, nb, t_stft, d).transpose(1, 2).reshape(b * t_stft, nb, d)
            if use_ckpt:
                x = checkpoint(freq_tr, x, spk_f, use_reentrant=False)
            else:
                x = freq_tr(x, spk_f)

            x = x.view(b, t_stft, nb, d)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        x = self.final_norm(x)

        # Mask estimation
        if use_ckpt:
            masks = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1
            )
        else:
            masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = masks.view(b, self.num_stems, t_stft, -1, 2)
        masks = masks.permute(0, 1, 3, 2, 4).contiguous()  # (b, n, f_sel, t, 2)

        # Scatter-average overlapped mel bands
        scatter_idx = freq_idx.view(1, 1, -1, 1, 1).expand_as(masks)
        full_mask = torch.zeros(
            b, self.num_stems, f_stereo, t_stft, 2, device=device, dtype=masks.dtype
        )
        full_mask.scatter_add_(2, scatter_idx, masks)

        denom = self.num_bands_per_freq_stereo.view(1, 1, -1, 1, 1)
        full_mask = full_mask / denom.clamp(min=1e-8)

        return self._complex_mul(stft_repr.unsqueeze(1), full_mask)
