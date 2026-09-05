#!/usr/bin/env python3
"""Validate MSS training and validation layouts without loading full audio."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import soundfile as sf


class Reporter:
    """Collect validation errors and warnings while continuing the scan."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.audio_files = 0
        self.tracks = 0

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check source-separation training/validation data before launch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--dataset-type", required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--data-path", required=True, nargs="+", type=Path)
    parser.add_argument("--valid-path", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--require-embeddings",
        action="store_true",
        help="Require an embeddings.npy array with shape (N, 192) for every track.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help=(
            "Maximum tracks/items checked per root (per instrument for type 3); "
            "0 checks everything."
        ),
    )
    return parser.parse_args()


def limited(items: Iterable[Path], maximum: int) -> list[Path]:
    values = sorted(items)
    return values if maximum <= 0 else values[:maximum]


def read_config(path: Path) -> tuple[list[str], int, int, bool]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    try:
        instruments = list(config["training"]["instruments"])
        sample_rate = int(config["audio"]["sample_rate"])
        chunk_size = int(config["audio"]["chunk_size"])
        other_fix = bool(config["training"].get("other_fix", False))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid training/audio configuration: {exc}") from exc

    if not instruments or not all(isinstance(item, str) and item for item in instruments):
        raise ValueError("training.instruments must be a non-empty list of names")
    return instruments, sample_rate, chunk_size, other_fix


def check_audio(path: Path, expected_rate: int, reporter: Reporter) -> sf.SoundFile | None:
    import soundfile as sf

    if not path.is_file():
        reporter.error(f"missing audio: {path}")
        return None
    try:
        info = sf.info(path)
    except Exception as exc:
        reporter.error(f"cannot read audio header {path}: {exc}")
        return None

    reporter.audio_files += 1
    if info.frames <= 0:
        reporter.error(f"empty audio file: {path}")
    if info.samplerate != expected_rate:
        reporter.error(
            f"sample-rate mismatch for {path}: {info.samplerate}, expected {expected_rate}"
        )
    if info.channels > 2:
        reporter.warning(f"{path} has {info.channels} channels; only the first two are used")
    return info


def find_stem(track_dir: Path, instrument: str) -> Path | None:
    for suffix in (".wav", ".flac"):
        candidate = track_dir / f"{instrument}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def check_embedding(track_dir: Path, required: bool, reporter: Reporter) -> None:
    import numpy as np

    path = track_dir / "embeddings.npy"
    if not path.is_file():
        if required:
            reporter.error(f"missing speaker embeddings: {path}")
        return
    try:
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        reporter.error(f"cannot read {path}: {exc}")
        return
    if embeddings.ndim != 2 or embeddings.shape[0] < 1 or embeddings.shape[1] != 192:
        reporter.error(f"invalid embedding shape for {path}: {embeddings.shape}, expected (N, 192)")
        return
    if not np.issubdtype(embeddings.dtype, np.number):
        reporter.error(f"non-numeric embedding dtype for {path}: {embeddings.dtype}")
        return
    if not np.isfinite(embeddings).all():
        reporter.error(f"embedding contains NaN or Inf values: {path}")


def check_aligned(infos: list[tuple[Path, sf.SoundFile]], reporter: Reporter) -> None:
    if len(infos) < 2:
        return
    reference_path, reference = infos[0]
    for path, info in infos[1:]:
        if info.frames != reference.frames:
            reporter.error(
                f"unaligned frame counts: {reference_path}={reference.frames}, {path}={info.frames}"
            )


def check_track_layout(
    roots: list[Path],
    instruments: list[str],
    expected_rate: int,
    aligned: bool,
    require_embeddings: bool,
    maximum: int,
    reporter: Reporter,
) -> None:
    for root in roots:
        if not root.is_dir():
            reporter.error(f"training root is not a directory: {root}")
            continue
        tracks = limited((path for path in root.iterdir() if path.is_dir()), maximum)
        if not tracks:
            reporter.error(f"no track directories found under: {root}")
            continue

        for track_dir in tracks:
            reporter.tracks += 1
            infos: list[tuple[Path, sf.SoundFile]] = []
            for instrument in instruments:
                stem_path = find_stem(track_dir, instrument)
                if stem_path is None:
                    reporter.error(f"missing {instrument}.wav|flac in training track: {track_dir}")
                    continue
                info = check_audio(stem_path, expected_rate, reporter)
                if info is not None:
                    infos.append((stem_path, info))
            if aligned:
                check_aligned(infos, reporter)
            check_embedding(track_dir, require_embeddings, reporter)


def check_pool_layout(
    roots: list[Path],
    instruments: list[str],
    expected_rate: int,
    maximum: int,
    reporter: Reporter,
) -> None:
    for root in roots:
        if not root.is_dir():
            reporter.error(f"training root is not a directory: {root}")
            continue
        for instrument in instruments:
            pool = root / instrument
            files = [*pool.glob("*.wav"), *pool.glob("*.flac")]
            files = limited(files, maximum)
            if not files:
                reporter.error(f"no wav/flac files found for instrument pool: {pool}")
            for path in files:
                reporter.tracks += 1
                check_audio(path, expected_rate, reporter)


def check_csv_layout(
    csv_paths: list[Path],
    instruments: list[str],
    expected_rate: int,
    maximum: int,
    reporter: Reporter,
) -> None:
    if len(csv_paths) != 1:
        reporter.error("dataset type 3 currently supports exactly one CSV file")
        return
    csv_path = csv_paths[0]
    if not csv_path.is_file():
        reporter.error(f"training CSV does not exist: {csv_path}")
        return

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"instrum", "path"}.issubset(reader.fieldnames):
            reporter.error(f"CSV must contain columns 'instrum' and 'path': {csv_path}")
            return
        rows = list(reader)

    rows_by_instrument: dict[str, list[tuple[int, dict[str, str]]]] = {
        instrument: [] for instrument in instruments
    }
    for row_number, row in enumerate(rows, start=2):
        instrument = row.get("instrum", "")
        if instrument not in rows_by_instrument:
            reporter.error(f"unknown instrument at {csv_path}:{row_number}: {instrument!r}")
            continue
        rows_by_instrument[instrument].append((row_number, row))

    for instrument, instrument_rows in rows_by_instrument.items():
        if not instrument_rows:
            reporter.error(f"CSV has no rows for instrument {instrument!r}: {csv_path}")
            continue
        rows_to_check = instrument_rows if maximum <= 0 else instrument_rows[:maximum]
        for row_number, row in rows_to_check:
            raw_path = row.get("path", "")
            if not raw_path:
                reporter.error(f"empty path at {csv_path}:{row_number}")
                continue
            reporter.tracks += 1
            check_audio(Path(raw_path), expected_rate, reporter)


def check_validation(
    roots: list[Path],
    instruments: list[str],
    expected_rate: int,
    other_fix: bool,
    require_embeddings: bool,
    maximum: int,
    reporter: Reporter,
) -> None:
    for root in roots:
        if not root.is_dir():
            reporter.error(f"validation root is not a directory: {root}")
            continue
        mixtures = limited(root.glob("*/mixture.wav"), maximum)
        if not mixtures:
            reporter.error(f"no <track>/mixture.wav files found under: {root}")
            continue

        for mixture in mixtures:
            reporter.tracks += 1
            track_dir = mixture.parent
            infos: list[tuple[Path, sf.SoundFile]] = []
            mixture_info = check_audio(mixture, expected_rate, reporter)
            if mixture_info is not None:
                infos.append((mixture, mixture_info))
            for instrument in instruments:
                filename = (
                    "vocals.wav" if instrument == "other" and other_fix else f"{instrument}.wav"
                )
                path = track_dir / filename
                info = check_audio(path, expected_rate, reporter)
                if info is not None:
                    infos.append((path, info))
            check_aligned(infos, reporter)
            check_embedding(track_dir, require_embeddings, reporter)


def main() -> int:
    args = parse_args()
    reporter = Reporter()
    try:
        instruments, sample_rate, chunk_size, other_fix = read_config(args.config_path)
    except Exception as exc:
        print(f"error: cannot load config {args.config_path}: {exc}", file=sys.stderr)
        return 2

    if args.dataset_type in (1, 4):
        check_track_layout(
            args.data_path,
            instruments,
            sample_rate,
            # Type 1 mixes tracks independently, but its loader still uses the
            # shortest stem length for every file in a track. Equal frame counts
            # avoid malformed chunks when one of those files is shorter than a
            # configured chunk.
            aligned=True,
            require_embeddings=args.require_embeddings and args.dataset_type == 4,
            maximum=args.max_tracks,
            reporter=reporter,
        )
    elif args.dataset_type == 2:
        check_pool_layout(args.data_path, instruments, sample_rate, args.max_tracks, reporter)
    else:
        check_csv_layout(args.data_path, instruments, sample_rate, args.max_tracks, reporter)

    check_validation(
        args.valid_path,
        instruments,
        sample_rate,
        other_fix,
        args.require_embeddings,
        args.max_tracks,
        reporter,
    )

    for message in reporter.warnings:
        print(f"warning: {message}", file=sys.stderr)
    for message in reporter.errors:
        print(f"error: {message}", file=sys.stderr)

    duration = chunk_size / sample_rate
    print(
        f"Checked {reporter.tracks} items and {reporter.audio_files} audio files; "
        f"config chunk={chunk_size} samples ({duration:.2f}s at {sample_rate} Hz)."
    )
    if reporter.errors:
        print(
            f"Validation failed with {len(reporter.errors)} error(s) and "
            f"{len(reporter.warnings)} warning(s).",
            file=sys.stderr,
        )
        return 1
    print(f"Validation passed with {len(reporter.warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
