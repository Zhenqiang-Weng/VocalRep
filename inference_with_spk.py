"""Separate music using a pretrained CAM++ speaker embedding."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from tqdm.auto import tqdm

from spk_extract import extract_dominant_speaker_embedding_with_clusters
from spk_extract.pretrained import load_pretrained
from utils.audio_utils import denormalize_audio, draw_spectrogram, normalize_audio
from utils.model_utils import demix, demix_with_spk, load_start_checkpoint
from utils.settings import get_model_from_config, parse_args_inference


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}


def load_audio_with_ffmpeg(path: str, sr: int, mono: bool = False):
    """Decode audio as float PCM with channels first."""
    channels = 1 if mono else 2
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg is required to decode this audio format.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode(errors="replace")) from error
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    return (audio if mono else audio.reshape(-1, channels).T), sr


def load_audio_with_fallback(path: str, sr: int, mono: bool = False):
    """Load and resample audio, falling back to FFmpeg for unsupported codecs."""
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    try:
        audio, original_sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.T
        if mono:
            audio = audio.mean(axis=0)
        if original_sr != sr:
            audio = scipy.signal.resample_poly(audio, sr, original_sr, axis=-1).astype(np.float32)
    except (sf.LibsndfileError, OSError):
        audio, _ = load_audio_with_ffmpeg(path, sr, mono)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Audio is empty or contains non-finite samples: {path}")
    return audio, sr


def _prepare_mix(path: Path, config):
    """Load a mixture with the number of channels required by the model."""
    audio, sr = load_audio_with_fallback(path, config.audio.sample_rate)
    channels = getattr(config.audio, "num_channels", 2)
    if audio.shape[0] == 1 and channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    elif channels == 1:
        audio = audio.mean(axis=0, keepdims=True)
    if audio.shape[0] != channels:
        raise ValueError(f"Expected {channels} audio channels, got {audio.shape[0]}: {path}")
    return audio, sr


def _save_stems(waveforms: dict, output_dir: Path, sr: int, args) -> None:
    """Write finite waveforms with a subtype supported by the selected format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "flac" if args.flac_file else "wav"
    subtype = args.pcm_type if args.flac_file else "FLOAT"
    for instrument, waveform in waveforms.items():
        if not np.isfinite(waveform).all():
            raise ValueError(f"Non-finite samples in the {instrument} output.")
        sf.write(output_dir / f"{instrument}.{extension}", waveform.T, sr, subtype=subtype)
        if args.draw_spectro > 0:
            draw_spectrogram(
                waveform.T, sr, args.draw_spectro, str(output_dir / f"{instrument}.jpg")
            )


def _load_diffusion(args, config, device) -> dict:
    """Load optional trained diffusion models only when enhancement is requested."""
    if not args.use_diffusion:
        return {}
    from diffusion.diffusion_wrapper import DiffusionConfig, DiffusionWrapper

    types = getattr(config.training, "diffusion_model", None)
    if types is None or len(types) != len(config.training.instruments):
        raise ValueError("Configure one diffusion_model entry per instrument.")
    wrappers = {}
    for instrument, model_type in zip(config.training.instruments, types):
        if model_type is None or model_type.lower() == "none":
            continue
        checkpoint = Path(args.diffusion_model_path) / f"{instrument}_diffusion.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Diffusion checkpoint is missing: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        diffusion_config = DiffusionConfig(**state["config"])
        if diffusion_config.model_type != model_type:
            raise ValueError(f"Diffusion model type does not match the checkpoint: {checkpoint}")
        wrapper = DiffusionWrapper(diffusion_config, device=device)
        wrapper.load_checkpoint_from_dict(state)
        wrapper.eval()
        wrappers[instrument] = wrapper
    return wrappers


def run_folder(model, args, config, device, verbose: bool = False) -> list[dict]:
    """Run blind separation, CAM++ extraction, and guided separation per mixture."""
    input_dir = Path(args.input_folder)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported audio files found in {input_dir}")
    if len({path.stem.casefold() for path in paths}) != len(paths):
        raise ValueError("Input filenames must have unique stems to avoid output collisions.")
    if "vocals" not in config.training.instruments:
        raise ValueError("Speaker-guided inference requires a 'vocals' output stem.")
    if args.use_tta:
        raise ValueError("Speaker-guided inference does not support --use_tta.")

    started = time.monotonic()
    model.eval()
    speaker_model = load_pretrained(args.spk_model_path, device=str(device))
    expected_dim = getattr(config.model, "spk_embd_dim", 192)
    actual_dim = speaker_model.xvector.dense.linear.out_channels
    if actual_dim != expected_dim:
        raise ValueError(f"Speaker embedding dimension {actual_dim} != expected {expected_dim}.")
    diffusion = _load_diffusion(args, config, device)
    output_root = Path(args.store_dir)
    reports = []
    for path in tqdm(paths, desc="Speaker-guided separation", disable=verbose):
        track_started = time.monotonic()
        mix, sr = _prepare_mix(path, config)
        original_mix = mix.copy()
        norm_params = None
        if getattr(config.inference, "normalize", False):
            mix, norm_params = normalize_audio(mix)

        if verbose:
            print(f"{path.name}: blind separation", flush=True)
        blind = demix(config, model, mix, device, model_type=args.model_type)
        blind_original = {
            name: denormalize_audio(waveform, norm_params) if norm_params else waveform
            for name, waveform in blind.items()
        }
        _save_stems(blind_original, output_root / "wo_spk" / path.stem, sr, args)
        clustering = extract_dominant_speaker_embedding_with_clusters(
            speaker_model,
            blind_original["vocals"],
            source_sr=sr,
            segment_duration=args.spk_segment_duration,
            energy_threshold=args.spk_energy_threshold,
            max_clusters=args.spk_max_clusters,
            batch_size=args.spk_batch_size,
            max_segments=args.spk_max_segments,
            device=device,
        )
        if clustering is None:
            raise ValueError(f"No voiced segments found for speaker extraction: {path}")
        embedding = clustering.mean_embedding
        if embedding.shape != (expected_dim,) or not torch.isfinite(embedding).all():
            raise ValueError(f"Invalid speaker embedding for {path}")
        embedding_dir = output_root / "embeddings" / path.stem
        embedding_dir.mkdir(parents=True, exist_ok=True)
        np.save(embedding_dir / "embedding.npy", embedding.numpy()[None, :])
        if verbose:
            print(
                f"{path.name}: CAM++ embedding {tuple(embedding.shape)}, guided separation",
                flush=True,
            )
        guided = demix_with_spk(
            config,
            model,
            mix,
            embedding,
            device,
            model_type=args.model_type,
            pbar=not args.disable_detailed_pbar,
        )
        for instrument, wrapper in diffusion.items():
            with torch.inference_mode():
                wave = torch.from_numpy(guided[instrument])[None].to(device)
                guided[instrument] = (
                    wrapper.inference(wave, num_steps=args.diffusion_steps, method="euler")[0]
                    .cpu()
                    .numpy()
                )
        if norm_params:
            guided = {name: denormalize_audio(wave, norm_params) for name, wave in guided.items()}
        if args.extract_instrumental:
            guided["instrumental"] = original_mix - guided["vocals"]
        if args.extract_other:
            guided["other"] = original_mix - sum(
                guided[name] for name in config.training.instruments if name != "other"
            )
        _save_stems(guided, output_root / "with_spk" / path.stem, sr, args)
        reports.append(
            {
                "input": str(path),
                "sample_rate": sr,
                "samples": original_mix.shape[-1],
                "stems": list(guided),
                "speaker_embedding_shape": list(embedding.shape),
                "speaker_segments": len(clustering.segment_indices),
                "speaker_clusters": clustering.n_clusters,
                "elapsed_seconds": round(time.monotonic() - track_started, 3),
            }
        )
    (output_root / "run_summary.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed {len(reports)} track(s) in {time.monotonic() - started:.2f} seconds.")
    return reports


def proc_folder(dict_args=None) -> list[dict]:
    """Load the trained separation model and process a directory of mixtures."""
    args = parse_args_inference(dict_args)
    checkpoint_path = Path(args.start_check_point)
    if not args.start_check_point or not checkpoint_path.is_file():
        raise FileNotFoundError("Provide a trained separation checkpoint with --start_check_point.")
    with checkpoint_path.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("The separation checkpoint is a Git LFS pointer. Run 'git lfs pull'.")
    device_ids = args.device_ids if isinstance(args.device_ids, list) else [args.device_ids]
    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0]}")
    elif not args.force_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.inference_batch_size is not None:
        config.inference.batch_size = args.inference_batch_size
    if args.inference_chunk_size is not None:
        config.inference.chunk_size = args.inference_chunk_size
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    load_start_checkpoint(args, model, checkpoint, type_="inference")
    del checkpoint
    model = model.to(device)
    if device.type == "cuda" and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder()
