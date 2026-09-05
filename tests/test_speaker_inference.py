"""Regression coverage for chunk boundaries, speaker segments, and output formats."""

from types import SimpleNamespace

from ml_collections import ConfigDict
import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from inference_with_spk import _save_stems, load_audio_with_fallback, run_folder
from spk_extract import extract_dominant_speaker_embedding_with_clusters
from utils.audio_utils import denormalize_audio, normalize_audio
from utils.model_utils import demix, demix_with_spk
from utils.settings import parse_args_inference


class IdentitySeparator(nn.Module):
    def forward(self, audio, embedding=None):
        return torch.stack([audio, audio * 0.5, audio * 0.25], dim=1)


def separation_config(batch_size=1):
    return ConfigDict(
        {
            "audio": {"chunk_size": 100, "sample_rate": 16000, "num_channels": 2},
            "model": {"spk_embd_dim": 192},
            "inference": {"num_overlap": 3, "batch_size": batch_size},
            "training": {
                "instruments": ["vocals", "backing_vocal", "instrumental"],
                "use_amp": False,
            },
        }
    )


@pytest.mark.parametrize("length", [1, 75, 100, 401])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_chunk_boundaries_preserve_audio(length, batch_size):
    wave = np.random.default_rng(42).normal(size=(2, length)).astype(np.float32)
    config = separation_config(batch_size)
    for output in (
        demix(config, IdentitySeparator(), wave, "cpu", "spk_bs_roformer"),
        demix_with_spk(
            config, IdentitySeparator(), wave, torch.ones(192), "cpu", "spk_bs_roformer"
        ),
    ):
        np.testing.assert_allclose(output["vocals"], wave, atol=1e-6)


class SpeakerStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.batch_sizes = []

    def forward(self, features):
        self.batch_sizes.append(len(features))
        return torch.ones(len(features), 192) + features.square().mean((1, 2))[:, None] * 0.01


@pytest.mark.parametrize("length", [8000, 40000, 64000])
def test_short_segments_and_stereo_layouts(length):
    audio = np.random.default_rng(1).normal(0, 0.1, length).astype(np.float32)
    model = SpeakerStub()
    result = extract_dominant_speaker_embedding_with_clusters(
        model,
        np.stack([audio, audio], axis=1),
        source_sr=16000,
        energy_threshold=0.001,
        batch_size=1,
        max_segments=2,
    )
    assert result is not None and result.mean_embedding.shape == (192,)
    assert max(model.batch_sizes) <= 1
    assert result.segment_indices[-1][1] == length


def test_silence_is_not_used_as_a_speaker_embedding():
    assert (
        extract_dominant_speaker_embedding_with_clusters(
            SpeakerStub(), np.zeros(16000), source_sr=16000
        )
        is None
    )
    with pytest.raises(ValueError):
        extract_dominant_speaker_embedding_with_clusters(
            SpeakerStub(), np.full(16000, np.nan), source_sr=16000
        )


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
def test_flac_output_and_resampling(tmp_path, subtype):
    wave = np.ones((2, 1600), dtype=np.float32) * 0.1
    args = SimpleNamespace(flac_file=True, pcm_type=subtype, draw_spectro=0)
    _save_stems({"vocals": wave}, tmp_path, 16000, args)
    assert sf.info(tmp_path / "vocals.flac").subtype == subtype
    audio, sr = load_audio_with_fallback(tmp_path / "vocals.flac", 8000)
    assert audio.shape == (2, 800) and sr == 8000


@pytest.mark.parametrize("wave", [np.zeros((2, 100)), np.ones(100), np.arange(100, dtype=float)])
def test_normalization_is_finite_and_reversible(wave):
    normalized, params = normalize_audio(wave)
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(denormalize_audio(normalized, params), wave, atol=1e-10)


def test_empty_input_is_an_error(tmp_path):
    args = parse_args_inference(
        {
            "model_type": "spk_bs_roformer",
            "config_path": "unused",
            "input_folder": str(tmp_path),
            "store_dir": str(tmp_path / "out"),
        }
    )
    with pytest.raises(ValueError, match="No supported audio"):
        run_folder(IdentitySeparator(), args, separation_config(), "cpu")
