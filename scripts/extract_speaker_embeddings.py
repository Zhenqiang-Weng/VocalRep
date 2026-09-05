"""Extract pretrained CAM++ embeddings for aligned training and validation tracks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

from spk_extract import extract_dominant_speaker_embedding_with_clusters
from spk_extract.pretrained import MODEL_ID, MODEL_REVISION, load_pretrained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse dataset roots and bounded pretrained extraction settings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        nargs="+",
        required=True,
        help="Training/validation roots containing one directory per track.",
    )
    parser.add_argument(
        "--vocal-stem", default="vocals", help="Clean target-vocal stem name, without extension."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Existing CAM++ directory; omit to use the verified official model cache.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--segment-duration", type=float, default=2.0)
    parser.add_argument("--energy-threshold", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-segments", type=int, default=64)
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=1,
        help="1 keeps all active target-vocal segments; larger values keep the largest cluster.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and atomically replace existing embeddings.npy files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/speaker_embeddings/report.json"),
        help="JSON run report; replaced on each invocation.",
    )
    args = parser.parse_args(argv)
    if (
        not args.vocal_stem
        or args.vocal_stem in {".", ".."}
        or any(separator in args.vocal_stem for separator in ("/", "\\"))
    ):
        parser.error("--vocal-stem must be a filename stem, not a path")
    if not np.isfinite(args.segment_duration) or args.segment_duration < 0.1:
        parser.error("--segment-duration must be finite and at least 0.1 seconds")
    if not np.isfinite(args.energy_threshold) or args.energy_threshold < 0:
        parser.error("--energy-threshold must be finite and non-negative")
    if min(args.batch_size, args.max_segments, args.max_clusters) < 1:
        parser.error("--batch-size, --max-segments, and --max-clusters must be positive")
    if args.report.suffix.lower() != ".json":
        parser.error("--report must be a JSON file outside the track directories")
    return args


def discover_tracks(roots: list[Path]) -> list[Path]:
    """Match the trainers' root/track layout and deduplicate overlapping roots."""
    tracks = set()
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {root}")
        children = [
            path.resolve()
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        if not children:
            raise ValueError(f"No track directories under {root}; pass the parent of each track.")
        tracks.update(children)
    return sorted(tracks)


def find_vocals(track: Path, stem: str) -> Path:
    """Require exactly one WAV/FLAC source instead of guessing between duplicates."""
    candidates = sorted(
        path
        for path in track.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in {".wav", ".flac"}
    )
    if not candidates:
        raise FileNotFoundError(f"Missing {stem}.wav or {stem}.flac in {track}")
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous vocal source in {track}: {[p.name for p in candidates]}")
    return candidates[0]


def validate_embeddings(embeddings: np.ndarray) -> None:
    """Reject incompatible, non-real, non-finite, or zero speaker features."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 1 or embeddings.shape[1] != 192:
        raise ValueError(f"Expected embeddings with shape (N, 192), got {embeddings.shape}")
    if embeddings.dtype.kind != "f" or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must contain finite real floating-point values.")
    if not np.all(np.any(embeddings != 0, axis=1)):
        raise ValueError("Embeddings contain a zero speaker vector.")


def atomic_save_embeddings(path: Path, embeddings: np.ndarray, overwrite: bool) -> None:
    """Publish complete NPY data without clobbering existing files by default."""
    validate_embeddings(embeddings)
    embeddings = embeddings.astype(np.float32, copy=False)
    validate_embeddings(embeddings)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, embeddings, allow_pickle=False)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Same-directory hard linking is atomic and fails if the destination exists.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_track(model: torch.nn.Module, source: Path, args: argparse.Namespace) -> np.ndarray:
    """Resample clean vocals to 16 kHz and return individual pretrained segment vectors."""
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] not in (1, 2):
        raise ValueError(f"Expected mono or stereo vocals, got {audio.shape[1]} channels.")
    # Pass mono explicitly to avoid ambiguous channel layouts for very short clips.
    if not np.isfinite(audio).all():
        raise ValueError(f"Vocal audio contains NaN or Inf: {source}")
    result = extract_dominant_speaker_embedding_with_clusters(
        model,
        audio.mean(axis=1),
        source_sr=sample_rate,
        segment_duration=args.segment_duration,
        energy_threshold=args.energy_threshold,
        max_clusters=args.max_clusters,
        device=args.device,
        batch_size=args.batch_size,
        max_segments=args.max_segments,
    )
    if result is None:
        raise ValueError("No active vocal segments of at least 0.1 seconds; no embedding written.")
    embeddings = result.all_embeddings[result.labels == result.largest_cluster_label]
    validate_embeddings(embeddings)
    return embeddings.astype(np.float32, copy=False)


def resolve_device(value: str) -> str:
    """Select CPU/CUDA explicitly and fail early for an unavailable requested GPU."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, and cuda:N.")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"Requested CUDA device is unavailable: {value}")
    return str(device)


def run(args: argparse.Namespace) -> dict:
    """Process every track, preserving existing outputs and recording individual failures."""
    tracks = discover_tracks(args.data_path)
    args.device = resolve_device(args.device)
    report_path = args.report.resolve()
    for track in tracks:
        if report_path == track / "embeddings.npy" or report_path.parent == track:
            raise ValueError("Keep --report outside track directories to protect dataset files.")
    model = None
    model_error = None
    records = []
    for track in tracks:
        destination = track / "embeddings.npy"
        record = {"track": str(track), "output": str(destination)}
        try:
            source = find_vocals(track, args.vocal_stem)
            record["source"] = str(source)
            if destination.exists() and not args.overwrite:
                embeddings = np.load(destination, allow_pickle=False)
                validate_embeddings(embeddings)
                record.update(status="skipped", shape=list(embeddings.shape))
            else:
                if model is None:
                    # Load only when needed, once for all training and validation roots.
                    if model_error is not None:
                        raise RuntimeError(model_error)
                    try:
                        model = load_pretrained(args.model_dir, device=args.device)
                    except (OSError, RuntimeError, ValueError) as error:
                        model_error = f"Cannot load the pretrained CAM++ model: {error}"
                        raise RuntimeError(model_error) from error
                embeddings = extract_track(model, source, args)
                atomic_save_embeddings(destination, embeddings, args.overwrite)
                record.update(status="written", shape=list(embeddings.shape))
        except (OSError, RuntimeError, ValueError, EOFError) as error:
            record.update(status="failed", error=str(error))
        records.append(record)
        details = record.get("error", str(record.get("shape", "")))
        print(f"[{record['status']}] {track}: {details}", flush=True)
    counts = Counter(record["status"] for record in records)
    report = {
        "model": (
            {"directory": str(args.model_dir.resolve())}
            if args.model_dir
            else {"id": MODEL_ID, "revision": MODEL_REVISION}
        ),
        "settings": {
            name: getattr(args, name)
            for name in (
                "device",
                "vocal_stem",
                "segment_duration",
                "energy_threshold",
                "batch_size",
                "max_segments",
                "max_clusters",
                "overwrite",
            )
        },
        "counts": {name: counts[name] for name in ("written", "skipped", "failed")},
        "tracks": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {report['counts']}; report: {args.report}")
    return report


def main(argv: list[str] | None = None) -> int:
    """Return a nonzero exit status if any dataset track could not be processed."""
    args = parse_args(argv)
    try:
        report = run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return int(report["counts"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
