"""Exportable BS-Roformer core models without STFT/iSTFT for training and ONNX/TensorRT export.

Input: STFT real representation (batch, freq_bins * channels, time_frames, 2)
Output: Masked STFT (batch, num_stems, freq_bins * channels, time_frames, 2)
"""

from typing import Tuple

from beartype import beartype
from rotary_embedding_torch import RotaryEmbedding
import torch
from torch import Tensor, nn
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint

from .bs_roformer import (
    DEFAULT_FREQS_PER_BANDS,
    BandSplit,
    MaskEstimator,
    RMSNorm,
    ScaleTransformer,
)
from .conditioner import BandEmbedder


def _complex_mul(stft_repr: Tensor, mask: Tensor) -> Tensor:
    """Manual complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i."""
    s_real, s_imag = stft_repr[..., 0], stft_repr[..., 1]
    m_real, m_imag = mask[..., 0], mask[..., 1]
    res_real = s_real * m_real - s_imag * m_imag
    res_imag = s_real * m_imag + s_imag * m_real
    return torch.stack([res_real, res_imag], dim=-1)


def convert_full_to_exportable(full_model: Module, exportable_model: Module, verbose: bool = True):
    """Copy matching weights from a full BS-Roformer model into an exportable core model."""
    full_state = full_model.state_dict()
    exportable_state = exportable_model.state_dict()

    matched = 0
    mismatched = 0
    missing = 0

    for key in exportable_state.keys():
        if key in full_state:
            if full_state[key].shape == exportable_state[key].shape:
                exportable_state[key] = full_state[key]
                matched += 1
            else:
                if verbose:
                    print(
                        f"[Shape Mismatch] {key}: {full_state[key].shape} vs {exportable_state[key].shape}"
                    )
                mismatched += 1
        else:
            if verbose:
                print(f"[Missing] {key}")
            missing += 1

    if verbose:
        print("\n=== Weight Copy Summary ===")
        print(f"  Matched: {matched}")
        print(f"  Mismatched: {mismatched}")
        print(f"  Missing: {missing}")
        print(f"  Total in exportable: {len(exportable_state)}")

    exportable_model.load_state_dict(exportable_state)
    return exportable_model


class BDCSGBSRoformerExportable(Module):
    """Exportable BS-Roformer core with band and speaker conditioning (no STFT/iSTFT)."""

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo: bool = True,
        num_stems: int = 3,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        mask_estimator_depth=2,
        mlp_expansion_factor=4,
        use_torch_checkpoint: bool = False,
        skip_connection=False,
        sage_attention=False,
        spk_embd_dim=192,
        **kwargs,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
        self.dim = dim
        self.spk_embd_dim = spk_embd_dim

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

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ScaleTransformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            drop_prob=0.0,
                            **transformer_kwargs,
                        ),
                        ScaleTransformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for _ in range(num_stems)
            ]
        )

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)
        self.band_cond_embedding = BandEmbedder(len(freqs_per_bands), dim, dim)

        self.register_buffer(
            "band_indices",
            torch.arange(len(self.freqs_per_bands), dtype=torch.long),
            persistent=False,
        )

    def _complex_mul(self, stft_repr: Tensor, mask: Tensor) -> Tensor:
        return _complex_mul(stft_repr, mask)

    def forward(
        self,
        stft_repr: Tensor,
        speaker_embedding: Tensor,
    ) -> Tensor:
        """Estimate masked STFT tensors for all configured stems.

        Args:
            stft_repr: Real/imag STFT tensor shaped (batch, freq_bins * channels, time_frames, 2).
            speaker_embedding: Speaker embedding shaped (batch, spk_embd_dim).

        Returns:
            Masked STFT tensor shaped (batch, num_stems, freq_bins * channels, time_frames, 2).
        """
        batch, freq_bins_stereo, t_stft, _ = stft_repr.shape
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        # (b, f*c, t, 2) -> (b, t, f*c*2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)  # (b, t, num_bands, dim)

        b, t, f, d = x.shape

        # Precompute band and speaker conditioning once for all layers.
        band_cond = self.band_cond_embedding(self.band_indices)
        band_embedding = band_cond.unsqueeze(0).expand(b, -1, -1).reshape(b * f, -1)

        spk_emb = self.spk_embd_to_dim(speaker_embedding)
        spk_f_cond = spk_emb.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)

        skip_sum = None
        for time_transformer, freq_transformer in self.layers:
            if self.skip_connection and skip_sum is not None:
                x = x + skip_sum

            # Time transformer: (b, t, f, d) -> (b*f, t, d)
            x = x.transpose(1, 2).reshape(b * f, t, d)
            if use_ckpt:
                x = checkpoint(time_transformer, x, band_embedding, use_reentrant=False)
            else:
                x = time_transformer(x, band_embedding)

            # Freq transformer: (b*f, t, d) -> (b*t, f, d)
            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            if use_ckpt:
                x = checkpoint(freq_transformer, x, spk_f_cond, use_reentrant=False)
            else:
                x = freq_transformer(x, spk_f_cond)

            # Restore: (b*t, f, d) -> (b, t, f, d)
            x = x.view(b, t, f, d)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        x = self.final_norm(x)

        # Mask estimation
        if use_ckpt:
            masks = [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators]
        else:
            masks = [fn(x) for fn in self.mask_estimators]
        mask = torch.stack(masks, dim=1)  # (b, num_stems, t, f*c*2)

        mask = mask.view(batch, self.num_stems, t_stft, freq_bins_stereo, 2).permute(0, 1, 3, 2, 4)

        return self._complex_mul(stft_repr.unsqueeze(1), mask)


class SpeakerBSRoformerExportable(Module):
    """Exportable BS-Roformer core conditioned on speaker embeddings in time and frequency."""

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo: bool = True,
        num_stems: int = 3,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        mask_estimator_depth=2,
        mlp_expansion_factor=4,
        use_torch_checkpoint: bool = False,
        skip_connection=False,
        sage_attention=False,
        spk_embd_dim=192,
        **kwargs,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
        self.dim = dim
        self.spk_embd_dim = spk_embd_dim

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

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ScaleTransformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            **transformer_kwargs,
                        ),
                        ScaleTransformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for _ in range(num_stems)
            ]
        )

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)

    def _complex_mul(self, stft_repr: Tensor, mask: Tensor) -> Tensor:
        return _complex_mul(stft_repr, mask)

    def forward(
        self,
        stft_repr: Tensor,
        speaker_embedding: Tensor,
    ) -> Tensor:
        """Estimate masked STFT tensors for all configured stems.

        Args:
            stft_repr: Real/imag STFT tensor shaped (batch, freq_bins * channels, time_frames, 2).
            speaker_embedding: Speaker embedding shaped (batch, spk_embd_dim).

        Returns:
            Masked STFT tensor shaped (batch, num_stems, freq_bins * channels, time_frames, 2).
        """
        batch, freq_bins_stereo, t_stft, _ = stft_repr.shape
        use_ckpt = self.use_torch_checkpoint and self.training and torch.is_grad_enabled()

        # (b, f*c, t, 2) -> (b, t, f*c*2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        if use_ckpt:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)  # (b, t, num_bands, dim)

        b, t, f, d = x.shape

        # Project speaker conditioning once and reuse across all layers.
        spk_emb = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        spk_t_cond = spk_emb.unsqueeze(1).expand(b, f, -1).reshape(b * f, -1)
        spk_f_cond = spk_emb.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)

        skip_sum = None
        for time_transformer, freq_transformer in self.layers:
            if self.skip_connection and skip_sum is not None:
                x = x + skip_sum

            # Time transformer: (b, t, f, d) -> (b*f, t, d)
            x = x.transpose(1, 2).reshape(b * f, t, d)
            if use_ckpt:
                x = checkpoint(time_transformer, x, spk_t_cond, use_reentrant=False)
            else:
                x = time_transformer(x, spk_t_cond)

            # Freq transformer: (b*f, t, d) -> (b*t, f, d)
            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            if use_ckpt:
                x = checkpoint(freq_transformer, x, spk_f_cond, use_reentrant=False)
            else:
                x = freq_transformer(x, spk_f_cond)

            # Restore: (b*t, f, d) -> (b, t, f, d)
            x = x.view(b, t, f, d)

            if self.skip_connection:
                skip_sum = x if skip_sum is None else skip_sum + x

        x = self.final_norm(x)

        # Mask estimation
        if use_ckpt:
            masks = [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators]
        else:
            masks = [fn(x) for fn in self.mask_estimators]
        mask = torch.stack(masks, dim=1)  # (b, num_stems, t, f*c*2)

        mask = mask.view(batch, self.num_stems, t_stft, freq_bins_stereo, 2).permute(0, 1, 3, 2, 4)

        return self._complex_mul(stft_repr.unsqueeze(1), mask)


# Backwards-compatible alias used by utils/settings.py and models/bs_roformer/__init__.py.
SpeakerRoformerExportable = SpeakerBSRoformerExportable
