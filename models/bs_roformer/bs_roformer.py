from functools import partial
import random

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.bs_roformer.attend import Attend
from models.bs_roformer.conditioner import BandEmbedder

try:
    from models.bs_roformer.attend_sage import Attend as AttendSage
except Exception:
    pass
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

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


def _get_cached_window(cache: dict, key, factory) -> Tensor:
    """Return a cached STFT window tensor, creating it once per key."""
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
    """Compute multi-resolution STFT L1 loss in FP32 with cached windows and no_grad targets."""
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


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if _HAS_F_RMS_NORM and not torch.onnx.is_in_onnx_export():
            return F.rms_norm(x, (x.shape[-1],), self.gamma, eps=1e-12)
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention


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
        self,
        *,
        dim,
        dim_head=32,
        heads=8,
        scale=8,
        flash=False,
        dropout=0.0,
        sage_attention=False,
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
        # x: (b*f, t, d), band_embedding: (b*f, d)
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


# BandSplit module


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
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

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
            net = []

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

DEFAULT_FREQS_PER_BANDS = (
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    128,
    129,
)


class BandConditionalBSRoformer(Module):
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
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        mlp_expansion_factor=4,
        use_torch_checkpoint=True,
        skip_connection=False,
        sage_attention=False,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands

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
            norm_output=False,
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
                ScaleTransformer(
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

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_win_length),
            return_complex=True,
        ).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, (
            f"the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}"
        )

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        self.band_cond_embedding = BandEmbedder(len(freqs_per_bands), dim, dim)

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
        self.register_buffer(
            "band_indices",
            torch.arange(len(self.freqs_per_bands), dtype=torch.long),
            persistent=False,
        )
        self._stft_window_cache = {}

    def _get_stft_window(self, device: torch.device) -> Tensor:
        return _get_cached_window(
            self._stft_window_cache,
            ("main", str(device)),
            lambda: self.stft_window_fn(device=device, dtype=torch.float32),
        )

    def create_band_conditioning(self, batch_size, device):
        """Create band conditioning embeddings shaped (batch_size * num_bands, dim)."""
        band_cond = self.band_cond_embedding(self.band_indices.to(device))
        band_cond = band_cond.unsqueeze(0).expand(batch_size, -1, -1)
        return band_cond.reshape(batch_size * band_cond.shape[1], -1)

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        """Run BandConditionalBSRoformer forward pass and optional loss calculation."""
        batch = raw_audio.shape[0]
        device = raw_audio.device
        x_is_mps = device.type == "mps"
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        # Compute STFT in float32 for cuFFT / BF16 autocast safety.
        raw_audio_len = raw_audio.shape[-1]
        raw_audio_flat = raw_audio.reshape(batch * channels, raw_audio_len).float()
        stft_window = self._get_stft_window(device)

        # RuntimeError: FFT operations are only supported on macOS 14+.
        try:
            stft_repr = torch.stft(
                raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True
            )
        except Exception:
            stft_repr = torch.stft(
                raw_audio_flat.cpu() if x_is_mps else raw_audio_flat,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True,
            ).to(device)
        stft_repr = torch.view_as_real(stft_repr)
        freq_bins, t_stft = stft_repr.shape[1], stft_repr.shape[2]
        stft_repr = stft_repr.view(batch, channels, freq_bins, t_stft, 2)

        # Merge stereo / mono into the frequency axis for band splitting.
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * channels, t_stft, 2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        b, t, f, d = x.shape
        band_embedding = self.create_band_conditioning(b, device)

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
                x = checkpoint(time_transformer, x, band_embedding, use_reentrant=False)
            else:
                x = time_transformer(x, band_embedding)

            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            if use_ckpt:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x = x.view(b, t, f, d)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)
        if use_ckpt:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = mask.view(batch, num_stems, t_stft, freq_bins * channels, 2).permute(0, 1, 3, 2, 4)

        # Modulate frequency representation in float32 for complex iSTFT safety.
        stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1).float().contiguous())
        mask_complex = torch.view_as_complex(mask.float().contiguous())
        stft_masked = stft_complex * mask_complex

        stft_masked = (
            stft_masked.view(batch, num_stems, freq_bins, self.audio_channels, t_stft)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * num_stems * self.audio_channels, freq_bins, t_stft)
        )

        try:
            recon_audio = torch.istft(
                stft_masked,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=raw_audio_len,
            )
        except Exception:
            recon_audio = torch.istft(
                stft_masked.cpu() if x_is_mps else stft_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio_len,
            ).to(device)

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
        multi_stft_resolution_loss = 0.0
        total_loss = loss

        if not return_loss_breakdown:
            return total_loss, recon_audio

        return total_loss, (loss, multi_stft_resolution_loss), recon_audio

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


class BSRoformer(Module):
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
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
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

        if sage_attention:
            print("Use Sage Attention")

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

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_win_length),
            return_complex=True,
        ).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, (
            f"the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}"
        )

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

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
        self._stft_window_cache = {}

    def _get_stft_window(self, device: torch.device) -> Tensor:
        return _get_cached_window(
            self._stft_window_cache,
            ("main", str(device)),
            lambda: self.stft_window_fn(device=device, dtype=torch.float32),
        )

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        """Run BSRoformer forward pass and optional loss calculation."""
        batch = raw_audio.shape[0]
        device = raw_audio.device
        x_is_mps = device.type == "mps"
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        raw_audio_len = raw_audio.shape[-1]
        raw_audio_flat = raw_audio.reshape(batch * channels, raw_audio_len).float()
        stft_window = self._get_stft_window(device)

        try:
            stft_repr = torch.stft(
                raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True
            )
        except Exception:
            stft_repr = torch.stft(
                raw_audio_flat.cpu() if x_is_mps else raw_audio_flat,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True,
            ).to(device)
        stft_repr = torch.view_as_real(stft_repr)
        freq_bins, t_stft = stft_repr.shape[1], stft_repr.shape[2]
        stft_repr = stft_repr.view(batch, channels, freq_bins, t_stft, 2)

        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * channels, t_stft, 2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        b, t, f, d = x.shape

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

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)
        if use_ckpt:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = mask.view(batch, num_stems, t_stft, freq_bins * channels, 2).permute(0, 1, 3, 2, 4)

        stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1).float().contiguous())
        mask_complex = torch.view_as_complex(mask.float().contiguous())
        stft_masked = stft_complex * mask_complex

        stft_masked = (
            stft_masked.view(batch, num_stems, freq_bins, self.audio_channels, t_stft)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * num_stems * self.audio_channels, freq_bins, t_stft)
        )

        try:
            recon_audio = torch.istft(
                stft_masked,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=raw_audio_len,
            )
        except Exception:
            recon_audio = torch.istft(
                stft_masked.cpu() if x_is_mps else stft_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio_len,
            ).to(device)

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


class SpeakerGuideBSRoformer(Module):
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
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        sage_attention=False,
        spk_embd_dim=192,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

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
            norm_output=False,
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
                ScaleTransformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                ScaleTransformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs,
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_win_length),
            return_complex=True,
        ).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, (
            f"the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}"
        )

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

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

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)
        self._stft_window_cache = {}

    def _get_stft_window(self, device: torch.device) -> Tensor:
        return _get_cached_window(
            self._stft_window_cache,
            ("main", str(device)),
            lambda: self.stft_window_fn(device=device, dtype=torch.float32),
        )

    def create_speaker_conditioning_for_fre_roformer(self, speaker_embedding, num_time_steps):
        if speaker_embedding is None:
            return None
        # Construct (b * t, d) speaker conditioning for (b * t, f, d) input.
        spk_proj = self.spk_embd_to_dim(speaker_embedding)
        b = spk_proj.shape[0]
        return spk_proj.unsqueeze(1).expand(b, num_time_steps, -1).reshape(b * num_time_steps, -1)

    def create_speaker_conditioning_for_time_roformer(self, speaker_embedding, num_bands):
        if speaker_embedding is None:
            return None
        # Construct (b * f, d) speaker conditioning for (b * f, t, d) input.
        spk_proj = self.spk_embd_to_dim(speaker_embedding)
        b = spk_proj.shape[0]
        return spk_proj.unsqueeze(1).expand(b, num_bands, -1).reshape(b * num_bands, -1)

    def forward(self, raw_audio, speaker_embedding=None, target=None, return_loss_breakdown=False):
        """Run SpeakerGuideBSRoformer forward pass and optional loss calculation."""
        batch = raw_audio.shape[0]
        device = raw_audio.device
        x_is_mps = device.type == "mps"
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        raw_audio_len = raw_audio.shape[-1]
        raw_audio_flat = raw_audio.reshape(batch * channels, raw_audio_len).float()
        stft_window = self._get_stft_window(device)

        try:
            stft_repr = torch.stft(
                raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True
            )
        except Exception:
            stft_repr = torch.stft(
                raw_audio_flat.cpu() if x_is_mps else raw_audio_flat,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True,
            ).to(device)
        stft_repr = torch.view_as_real(stft_repr)
        freq_bins, t_stft = stft_repr.shape[1], stft_repr.shape[2]
        stft_repr = stft_repr.view(batch, channels, freq_bins, t_stft, 2)

        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * channels, t_stft, 2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        b, t, f, d = x.shape

        # Project speaker embedding once and reuse across all axial transformer layers.
        if speaker_embedding is not None:
            spk_proj = self.spk_embd_to_dim(speaker_embedding)
            spk_t_cond = spk_proj.unsqueeze(1).expand(b, f, -1).reshape(b * f, -1)
            spk_f_cond = spk_proj.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)
        else:
            spk_t_cond = None
            spk_f_cond = None

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
                x = checkpoint(time_transformer, x, spk_t_cond, use_reentrant=False)
            else:
                x = time_transformer(x, spk_t_cond)

            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            if use_ckpt:
                x = checkpoint(freq_transformer, x, spk_f_cond, use_reentrant=False)
            else:
                x = freq_transformer(x, spk_f_cond)

            x = x.view(b, t, f, d)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)
        if use_ckpt:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = mask.view(batch, num_stems, t_stft, freq_bins * channels, 2).permute(0, 1, 3, 2, 4)

        stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1).float().contiguous())
        mask_complex = torch.view_as_complex(mask.float().contiguous())
        stft_masked = stft_complex * mask_complex

        stft_masked = (
            stft_masked.view(batch, num_stems, freq_bins, self.audio_channels, t_stft)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * num_stems * self.audio_channels, freq_bins, t_stft)
        )

        try:
            recon_audio = torch.istft(
                stft_masked,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=raw_audio_len,
            )
        except Exception:
            recon_audio = torch.istft(
                stft_masked.cpu() if x_is_mps else stft_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio_len,
            ).to(device)

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


class BDCSGBSRoformer(Module):
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
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        sage_attention=False,
        spk_embd_dim=192,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
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
            norm_output=False,
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
                ScaleTransformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    drop_prob=0.0,
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                ScaleTransformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs,
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_win_length),
            return_complex=True,
        ).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, (
            f"the number of freqs in the bands must equal {freqs} based on the STFT settings, "
            f"but got {sum(freqs_per_bands)}"
        )

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

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

        # multi-resolution stft loss (training only)
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)

        self.band_cond_embedding = BandEmbedder(len(freqs_per_bands), dim, dim)

        # ============================
        # PERF: cache / buffers
        # ============================
        # 1) Cache band indices as a buffer to avoid arange on every forward pass
        self.register_buffer(
            "band_indices",
            torch.arange(len(self.freqs_per_bands), dtype=torch.long),
            persistent=False,
        )

        # 2) stft window cache: key = (device_str, dtype)
        #    Cache the inference window to avoid rebuilding hann_window each forward pass
        self._stft_window_cache = {}

    # ----------------------------
    # PERF: cached stft window
    # ----------------------------
    def _get_stft_window(self, device: torch.device, dtype: torch.dtype = torch.float32):
        return _get_cached_window(
            self._stft_window_cache,
            ("main", str(device), dtype),
            lambda: self.stft_window_fn(device=device).to(dtype=dtype),
        )

    def create_band_conditioning(self, batch_size, device):
        """Create band conditioning embeddings shaped (batch_size * num_bands, dim)."""
        band_cond = self.band_cond_embedding(self.band_indices.to(device))
        band_cond = band_cond.unsqueeze(0).expand(batch_size, -1, -1)
        return band_cond.reshape(batch_size * band_cond.shape[1], -1)

    def create_speaker_conditioning_for_fre_roformer(self, speaker_embedding, num_time_steps):
        if speaker_embedding is None:
            return None
        spk_proj = self.spk_embd_to_dim(speaker_embedding)
        b = spk_proj.shape[0]
        return spk_proj.unsqueeze(1).expand(b, num_time_steps, -1).reshape(b * num_time_steps, -1)

    def create_speaker_conditioning_for_time_roformer(self, speaker_embedding, num_bands):
        if speaker_embedding is None:
            return None
        spk_proj = self.spk_embd_to_dim(speaker_embedding)
        b = spk_proj.shape[0]
        return spk_proj.unsqueeze(1).expand(b, num_bands, -1).reshape(b * num_bands, -1)

    def forward(self, raw_audio, speaker_embedding=None, target=None, return_loss_breakdown=False):
        """Run BDCSGBSRoformer forward pass and optional loss calculation."""
        batch = raw_audio.shape[0]
        device = raw_audio.device
        x_is_mps = device.type == "mps"
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). "
            "also need to be False if mono (channel dimension of 1)"
        )

        raw_audio_len = raw_audio.shape[-1]
        raw_audio_flat = raw_audio.reshape(batch * channels, raw_audio_len).float()
        stft_window = self._get_stft_window(device=device, dtype=torch.float32)

        try:
            stft_repr = torch.stft(
                raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True
            )
        except Exception:
            stft_repr = torch.stft(
                raw_audio_flat.cpu() if x_is_mps else raw_audio_flat,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True,
            ).to(device)

        stft_repr = torch.view_as_real(stft_repr)
        freq_bins, t_stft = stft_repr.shape[1], stft_repr.shape[2]
        stft_repr = stft_repr.view(batch, channels, freq_bins, t_stft, 2)

        # Merge stereo / mono into frequency for band splitting.
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * channels, t_stft, 2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        b, t, f, d = x.shape

        # Compute band and speaker conditioning once and reuse across all layers.
        band_embedding = self.create_band_conditioning(b, device)
        spk_f_cond = self.create_speaker_conditioning_for_fre_roformer(speaker_embedding, t)

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
                x = checkpoint(time_transformer, x, band_embedding, use_reentrant=False)
            else:
                x = time_transformer(x, band_embedding)

            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            if use_ckpt:
                x = checkpoint(freq_transformer, x, spk_f_cond, use_reentrant=False)
            else:
                x = freq_transformer(x, spk_f_cond)

            x = x.view(b, t, f, d)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)
        if use_ckpt:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = mask.view(batch, num_stems, t_stft, freq_bins * channels, 2).permute(0, 1, 3, 2, 4)

        stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1).float().contiguous())
        mask_complex = torch.view_as_complex(mask.float().contiguous())
        stft_masked = stft_complex * mask_complex

        stft_masked = (
            stft_masked.view(batch, num_stems, freq_bins, self.audio_channels, t_stft)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * num_stems * self.audio_channels, freq_bins, t_stft)
        )

        try:
            recon_audio = torch.istft(
                stft_masked,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=raw_audio_len,
            )
        except Exception:
            recon_audio = torch.istft(
                stft_masked.cpu() if x_is_mps else stft_masked,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio_len,
            ).to(device)

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

    def freeze_transformer_layers(self):
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_transformer_layers(self):
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = True


# Re-export canonical exportable models for backwards-compatible imports.
from models.bs_roformer.bs_roformer_exportable import (  # noqa: E402
    BDCSGBSRoformerExportable,
    SpeakerBSRoformerExportable,
    SpeakerRoformerExportable,
)
