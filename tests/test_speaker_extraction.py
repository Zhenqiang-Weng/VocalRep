"""Exercise training embedding extraction without downloading model weights."""

from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from scripts import extract_speaker_embeddings as extraction
from scripts.validate_training_data import Reporter, check_embedding
from spk_extract import ClusteringResult


class SpeakerStub(nn.Module):
    """Supply deterministic, nonzero segment vectors for offline pipeline tests."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, features):
        self.batch_sizes.append(len(features))
        return torch.ones(len(features), 192, device=features.device)


def make_track(root, name="song", rate=16000, channels=1, suffix=".wav", silent=False):
    track = root / name
    track.mkdir(parents=True)
    time = np.arange(int(rate * 0.55)) / rate
    audio = np.zeros_like(time) if silent else 0.1 * np.sin(2 * np.pi * 220 * time)
    if channels == 2:
        # Put the voice in the right channel to catch accidental left-only extraction.
        audio = np.stack([np.zeros_like(audio), audio], axis=1)
    sf.write(track / f"vocals{suffix}", audio, rate)
    return track


def arguments(tmp_path, roots=None, extra=()):
    return extraction.parse_args(
        [
            "--data-path",
            *(str(root) for root in (roots or [tmp_path / "data"])),
            "--device",
            "cpu",
            "--segment-duration",
            "0.2",
            "--batch-size",
            "2",
            "--report",
            str(tmp_path / "report.json"),
            *extra,
        ]
    )


@pytest.mark.parametrize("rate,channels,suffix", [(8000, 1, ".wav"), (44100, 2, ".flac")])
def test_extract_resampled_segment_rows_and_training_compatibility(
    tmp_path, monkeypatch, rate, channels, suffix
):
    track = make_track(tmp_path / "data", rate=rate, channels=channels, suffix=suffix)
    model = SpeakerStub()
    loader = Mock(return_value=model)
    monkeypatch.setattr(extraction, "load_pretrained", loader)
    report = extraction.run(arguments(tmp_path))
    embeddings = np.load(track / "embeddings.npy", allow_pickle=False)
    assert embeddings.shape == (3, 192) and embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all() and np.any(embeddings)
    assert model.batch_sizes == [2, 1]
    assert report["counts"] == {"written": 1, "skipped": 0, "failed": 0}
    loader.assert_called_once_with(None, device="cpu")
    reporter = Reporter()
    check_embedding(track, required=True, reporter=reporter)
    assert not reporter.errors


def test_multiple_roots_are_deduplicated_and_reuse_one_encoder(tmp_path, monkeypatch):
    train, valid = tmp_path / "train", tmp_path / "valid"
    make_track(train)
    make_track(valid)
    (train / ".cache").mkdir()
    loader = Mock(return_value=SpeakerStub())
    monkeypatch.setattr(extraction, "load_pretrained", loader)
    report = extraction.run(arguments(tmp_path, [train, valid, train]))
    assert report["counts"] == {"written": 2, "skipped": 0, "failed": 0}
    loader.assert_called_once()


def test_rerun_validates_existing_arrays_without_loading_model(tmp_path, monkeypatch):
    track = make_track(tmp_path / "data")
    output = track / "embeddings.npy"
    np.save(output, np.full((2, 192), 3, dtype=np.float32))
    original = output.read_bytes()
    loader = Mock(side_effect=AssertionError("An existing valid file must not load CAM++."))
    monkeypatch.setattr(extraction, "load_pretrained", loader)
    report = extraction.run(arguments(tmp_path))
    assert report["counts"]["skipped"] == 1 and output.read_bytes() == original
    loader.assert_not_called()


@pytest.mark.parametrize("invalid", [np.zeros((1, 192)), np.ones(192), np.full((1, 192), np.nan)])
def test_invalid_existing_features_require_explicit_overwrite(tmp_path, monkeypatch, invalid):
    track = make_track(tmp_path / "data")
    output = track / "embeddings.npy"
    np.save(output, invalid)
    original = output.read_bytes()
    loader = Mock(return_value=SpeakerStub())
    monkeypatch.setattr(extraction, "load_pretrained", loader)
    report = extraction.run(arguments(tmp_path))
    assert report["counts"]["failed"] == 1 and output.read_bytes() == original
    loader.assert_not_called()
    report = extraction.run(arguments(tmp_path, extra=["--overwrite"]))
    assert report["counts"]["written"] == 1
    extraction.validate_embeddings(np.load(output, allow_pickle=False))


def test_missing_silent_and_ambiguous_vocals_fail_without_zero_fallback(tmp_path, monkeypatch):
    root = tmp_path / "data"
    silent = make_track(root, "silent", silent=True)
    ambiguous = make_track(root, "ambiguous")
    sf.write(ambiguous / "vocals.flac", np.ones(3200), 16000)
    missing = root / "missing"
    missing.mkdir()
    good = make_track(root, "good")
    monkeypatch.setattr(extraction, "load_pretrained", Mock(return_value=SpeakerStub()))
    args = arguments(tmp_path)
    report = extraction.run(args)
    assert report["counts"] == {"written": 1, "skipped": 0, "failed": 3}
    assert (good / "embeddings.npy").is_file()
    assert all(not (track / "embeddings.npy").exists() for track in [silent, ambiguous, missing])
    assert (
        extraction.main(["--data-path", str(root), "--device", "cpu", "--report", str(args.report)])
        == 1
    )


def test_failed_overwrite_preserves_original_and_atomic_save_does_not_clobber(
    tmp_path, monkeypatch
):
    track = make_track(tmp_path / "data", silent=True)
    output = track / "embeddings.npy"
    original = np.ones((1, 192), dtype=np.float32)
    extraction.atomic_save_embeddings(output, original, overwrite=False)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        extraction.atomic_save_embeddings(output, original * 2, overwrite=False)
    monkeypatch.setattr(extraction, "load_pretrained", Mock(return_value=SpeakerStub()))
    report = extraction.run(arguments(tmp_path, extra=["--overwrite"]))
    assert report["counts"]["failed"] == 1 and output.read_bytes() == before
    assert not list(track.glob("*.tmp"))


def test_optional_clustering_keeps_selected_rows_not_the_mean(tmp_path, monkeypatch):
    track = make_track(tmp_path / "data")
    embeddings = np.stack([np.full(192, value) for value in [1, 2, 9]]).astype(np.float32)
    result = ClusteringResult(
        torch.full((192,), 1.5),
        embeddings,
        np.array([0, 0, 1]),
        0,
        [(0, 3200), (3200, 6400), (6400, 8800)],
        2,
    )
    extract = Mock(return_value=result)
    monkeypatch.setattr(extraction, "extract_dominant_speaker_embedding_with_clusters", extract)
    args = arguments(tmp_path, extra=["--max-clusters", "3"])
    actual = extraction.extract_track(SpeakerStub(), track / "vocals.wav", args)
    np.testing.assert_array_equal(actual, embeddings[:2])
    assert extract.call_args.kwargs["max_clusters"] == 3


def test_model_load_failure_is_reported_without_repeated_downloads(tmp_path, monkeypatch):
    make_track(tmp_path / "data", "first")
    make_track(tmp_path / "data", "second")
    loader = Mock(side_effect=OSError("checkpoint is unavailable"))
    monkeypatch.setattr(extraction, "load_pretrained", loader)
    report = extraction.run(arguments(tmp_path))
    assert report["counts"]["failed"] == 2
    assert all("Cannot load" in record["error"] for record in report["tracks"])
    loader.assert_called_once()


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "0"],
        ["--segment-duration", "nan"],
        ["--energy-threshold", "-1"],
        ["--vocal-stem", "../vocals"],
        ["--report", "embeddings.npy"],
    ],
)
def test_invalid_cli_settings_are_rejected(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        arguments(tmp_path, extra=extra)
    assert error.value.code == 2


def test_empty_dataset_and_report_collisions_are_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No track directories"):
        extraction.discover_tracks([empty])
    track = make_track(tmp_path / "data")
    args = arguments(tmp_path, extra=["--report", str(track / "configuration.json")])
    with pytest.raises(ValueError, match="protect dataset files"):
        extraction.run(args)


def test_explicit_unavailable_cuda_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert extraction.resolve_device("auto") == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        extraction.resolve_device("cuda:0")
