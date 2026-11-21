"""Shared Accelerate training runtime optimized for throughput and GPU memory."""

from __future__ import annotations

from contextlib import nullcontext
import glob
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW, RAdam, RMSprop
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import wandb

from models.preprocess.feature_extract import STFTConfig, STFTProcessorBatch
from utils.metrics import get_metrics
from utils.model_utils import demix, demix_with_spk, prefer_target_instrument
from utils.muon import SingleDeviceMuonWithAuxAdam
from utils.settings import manual_seed
from utils.training_cli import build_wandb_config

EXPORTABLE_MODELS = (
    "bs_roformer_exportable",
    "spk_bs_roformer_exportable",
    "speaker_mel_band_roformer_exportable",
)

STANDARD_ROFORMER_MODELS = (
    "mel_band_roformer",
    "bs_roformer",
    "mel_band_roformer_disc",
    "td_mel_band_roformer",
    "band_conditioned_mel_band_roformer",
)

LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def configure_torch_runtime(
    seed: int,
    *,
    deterministic: bool = False,
    allow_tf32: bool = True,
) -> None:
    """Configure RNG seeds, cuDNN autotuning, and TF32 matrix acceleration."""
    manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    use_tf32 = bool(allow_tf32 and not deterministic)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def set_multiprocessing_spawn(accelerator=None) -> None:
    """Set PyTorch multiprocessing start method to spawn when supported."""
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as error:
        if accelerator is not None:
            accelerator.print(f"Warning: multiprocessing start method already set: {error}")


def resolve_gradient_accumulation_steps(
    args: Any,
    config: Any,
    *,
    auxiliary_active: bool = False,
) -> int:
    """Resolve micro-step accumulation count while enforcing auxiliary-wrapper invariants."""
    cli_steps = getattr(args, "gradient_accumulation_steps", None)
    cfg_steps = int(getattr(config.training, "gradient_accumulation_steps", 1))
    steps = int(cli_steps) if cli_steps is not None else cfg_steps
    if steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {steps}.")
    if auxiliary_active and steps != 1:
        raise ValueError(
            "Keep Accelerator gradient_accumulation_steps=1 when discriminator or diffusion "
            "wrappers are enabled; configure grad_acc_step on the auxiliary wrappers instead."
        )
    return steps


def enable_gradient_checkpointing(
    model: nn.Module,
    override: Optional[bool],
    config: Any = None,
    accelerator=None,
) -> bool:
    """Toggle activation checkpointing on the separation model to reduce VRAM."""
    if override is not None:
        target = bool(override)
    elif config is not None and hasattr(config.training, "gradient_checkpointing"):
        target = bool(config.training.gradient_checkpointing)
    else:
        target = bool(getattr(model, "use_torch_checkpoint", False))

    updated = False
    for module in model.modules():
        if hasattr(module, "use_torch_checkpoint"):
            module.use_torch_checkpoint = target
            updated = True

    if accelerator is not None and updated:
        accelerator.print(f"Gradient checkpointing: {'enabled' if target else 'disabled'}")
    return target


def maybe_compile_model(
    model: nn.Module,
    *,
    compile_enabled: bool = False,
    compile_mode: str = "default",
    accelerator=None,
) -> nn.Module:
    """Optionally wrap the separation model with torch.compile."""
    if not compile_enabled:
        return model
    if not hasattr(torch, "compile"):
        if accelerator is not None:
            accelerator.print("Warning: torch.compile is unavailable in this PyTorch build.")
        return model
    if accelerator is not None:
        accelerator.print(f"Compiling separation model with torch.compile(mode={compile_mode!r})")
    return torch.compile(model, mode=compile_mode)


def unwrap_model_state_dict(accelerator, model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a clean state_dict stripped of DDP and torch.compile wrappers."""
    unwrapped = model
    while True:
        next_mod = accelerator.unwrap_model(unwrapped)
        if hasattr(next_mod, "_orig_mod"):
            next_mod = next_mod._orig_mod
        if next_mod is unwrapped:
            break
        unwrapped = next_mod
    return unwrapped.state_dict()


def init_wandb(
    accelerator,
    args: Any,
    config: Any,
    device_ids: list[int],
    batch_size: int,
) -> None:
    """Initialize W&B on the main process and disable it on worker ranks."""
    run_config = build_wandb_config(args, config, device_ids, batch_size)
    if accelerator.is_main_process:
        if args.wandb_key is not None and args.wandb_key.strip() != "":
            wandb.login(key=args.wandb_key)
            wandb.init(project=args.wandb_project, name=args.wandb_name, config=run_config)
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                mode="offline",
                config=run_config,
            )
    else:
        wandb.init(mode="disabled")


def load_separator_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    accelerator,
    *,
    strict: bool = False,
    copy_first_mask_estimator: bool = False,
    unfreeze_mask_estimators: bool = False,
    estimator_unfreeze_indexes: Optional[Sequence[int]] = None,
) -> None:
    """Load initial weights into the separation model on CPU before distributed wrapping."""
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
    with path.open("rb") as stream:
        if stream.read(len(LFS_POINTER_HEADER)) == LFS_POINTER_HEADER:
            raise ValueError(f"Checkpoint is an unresolved Git LFS pointer: {path}")

    accelerator.print(f"Start from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "state" in checkpoint:
            state_dict = checkpoint["state"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)
    del checkpoint, state_dict

    if copy_first_mask_estimator and hasattr(model, "copy_first_mask_estimator_to_others"):
        model.copy_first_mask_estimator_to_others()
    if unfreeze_mask_estimators and (
        hasattr(model, "frozen_all") and hasattr(model, "unfreeze_mask_estimators_by_indexes")
    ):
        model.frozen_all()
        model.unfreeze_mask_estimators_by_indexes(list(estimator_unfreeze_indexes or [0, 1]))
    accelerator.print("Checkpoint loaded successfully")


def build_optimizer(
    config: Any,
    model: nn.Module,
    accelerator=None,
) -> torch.optim.Optimizer:
    """Build the configured optimizer over trainable parameters with fused CUDA support."""
    optim_params: dict[str, Any] = {}
    if "optimizer" in config:
        optim_params = dict(config["optimizer"])
        if accelerator is not None and config.training.optimizer != "muon":
            accelerator.print(f"Optimizer params from config:\n{optim_params}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in model.")

    # Use fused CUDA optimizer kernels automatically when parameters already reside on CUDA.
    all_cuda_float = torch.cuda.is_available() and all(
        param.is_cuda and param.is_floating_point() for param in trainable_params
    )

    name = config.training.optimizer
    lr = config.training.lr

    if name == "adam":
        if all_cuda_float and "fused" not in optim_params and "foreach" not in optim_params:
            optim_params["fused"] = True
        return Adam(trainable_params, lr=lr, **optim_params)
    if name == "adamw":
        if all_cuda_float and "fused" not in optim_params and "foreach" not in optim_params:
            optim_params["fused"] = True
        return AdamW(trainable_params, lr=lr, **optim_params)
    if name == "radam":
        return RAdam(trainable_params, lr=lr, **optim_params)
    if name == "rmsprop":
        return RMSprop(trainable_params, lr=lr, **optim_params)
    if name == "prodigy":
        from prodigyopt import Prodigy

        return Prodigy(trainable_params, lr=lr, **optim_params)
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(trainable_params, lr=lr, **optim_params)
    if name == "muon":
        muon_params = [param for param in trainable_params if param.ndim >= 2]
        adam_params = [param for param in trainable_params if param.ndim < 2]
        if (
            not hasattr(config, "optimizer")
            or "muon_group" not in config.optimizer
            or "adam_group" not in config.optimizer
        ):
            raise ValueError(
                "Muon optimizer requires config.optimizer.muon_group and config.optimizer.adam_group."
            )
        param_groups = [
            dict(params=muon_params, use_muon=True, **dict(config.optimizer.muon_group)),
            dict(params=adam_params, use_muon=False, **dict(config.optimizer.adam_group)),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)
    if name == "sgd":
        if accelerator is not None:
            accelerator.print("Use SGD optimizer")
        return SGD(trainable_params, lr=lr, **optim_params)
    raise ValueError(f"Unknown optimizer: {name}")


def build_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    drop_last: bool = True,
) -> DataLoader:
    """Create a throughput-tuned training DataLoader."""
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": bool(drop_last and len(dataset) >= batch_size),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def build_stft_processor(config: Any, device: torch.device) -> STFTProcessorBatch:
    """Build STFTProcessorBatch prioritizing model.stft_* settings over audio.* fallbacks."""
    n_fft = int(getattr(config.model, "stft_n_fft", getattr(config.audio, "n_fft", 2048)))
    hop_length = int(
        getattr(config.model, "stft_hop_length", getattr(config.audio, "hop_length", 512))
    )
    win_length = int(
        getattr(config.model, "stft_win_length", getattr(config.audio, "stft_win_length", n_fft))
    )
    stft_cfg = STFTConfig(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        normalized=getattr(config.model, "stft_normalized", False),
        center=True,
        istft_length=getattr(config.audio, "chunk_size", None),
        stereo=getattr(config.model, "stereo", True),
        num_stems=getattr(config.model, "num_stems", 1),
        use_multi_stft_loss=True,
        multi_weight=getattr(config.model, "multi_stft_resolution_loss_weight", 1.0),
        multi_window_sizes=tuple(
            getattr(
                config.model,
                "multi_stft_resolutions_window_sizes",
                (4096, 2048, 1024, 512, 256),
            )
        ),
        multi_hop_length=getattr(config.model, "multi_stft_hop_size", 147),
        multi_normalized=getattr(config.model, "multi_stft_normalized", False),
    )
    return STFTProcessorBatch(stft_cfg, device=device)


def _get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    window = torch.ones(window_size, dtype=torch.float32)
    if fade_size > 0:
        window[:fade_size] = torch.linspace(0, 1, fade_size)
        window[-fade_size:] = torch.linspace(1, 0, fade_size)
    return window


def demix_with_spk_exportable(
    config: Any,
    model: nn.Module,
    mix: np.ndarray | torch.Tensor,
    spk_emb: torch.Tensor,
    device: torch.device,
    stft_processor: STFTProcessorBatch,
) -> dict[str, np.ndarray]:
    """Run batched, mixed-precision chunked inference for exportable core models."""
    was_training = model.training
    model.eval()

    if hasattr(config, "inference") and hasattr(config.inference, "chunk_size"):
        chunk_size = config.inference.chunk_size
    else:
        chunk_size = getattr(config.audio, "chunk_size", 882000)

    num_overlap = getattr(config.inference, "num_overlap", 2)
    infer_batch_size = max(1, int(getattr(config.inference, "batch_size", 1)))
    step = chunk_size // num_overlap
    border = chunk_size - step
    fade_size = step // 10

    instruments = config.training.instruments
    num_stems = len(instruments)

    mix_tensor = torch.as_tensor(mix, dtype=torch.float32)
    if mix_tensor.ndim == 1:
        mix_tensor = mix_tensor.unsqueeze(0).repeat(2, 1)
    elif mix_tensor.shape[0] == 1:
        mix_tensor = mix_tensor.repeat(2, 1)

    channels, length_init = mix_tensor.shape
    if length_init > 2 * border and border > 0:
        mix_tensor = F.pad(mix_tensor, (border, border), mode="reflect")
    else:
        border = 0

    total_samples = mix_tensor.shape[1]
    target_device = torch.device(device)
    use_amp = bool(getattr(config.training, "use_amp", True) and target_device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (target_device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    spk_device = torch.as_tensor(spk_emb, dtype=torch.float32, device=target_device)
    if spk_device.ndim == 1:
        spk_device = spk_device.unsqueeze(0)

    # Accumulate overlap-add on CPU so validation never retains full-song buffers in VRAM.
    result = torch.zeros((num_stems, channels, total_samples), dtype=torch.float32)
    counter = torch.zeros((1, 1, total_samples), dtype=torch.float32)
    windowing_array = _get_windowing_array(chunk_size, fade_size)

    try:
        with torch.inference_mode():
            start = 0
            batch_chunks = []
            batch_spans = []
            while start < total_samples:
                end = min(start + chunk_size, total_samples)
                chunk = mix_tensor[:, start:end]
                actual_len = chunk.shape[1]
                if actual_len < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - actual_len), mode="constant", value=0.0)
                batch_chunks.append(chunk)
                batch_spans.append((start, end, actual_len))
                start += step

                if len(batch_chunks) >= infer_batch_size or start >= total_samples:
                    chunk_batch = torch.stack(batch_chunks, dim=0).to(
                        target_device, non_blocking=True
                    )
                    batch_spk = spk_device.expand(chunk_batch.shape[0], -1)
                    chunk_stft = stft_processor.stft_batch(chunk_batch)
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if use_amp
                        else nullcontext()
                    )
                    with autocast_ctx:
                        output_stft = model(chunk_stft, batch_spk)
                    output_audio = stft_processor.istft_batch(output_stft).float().cpu()

                    for idx, (seg_start, seg_end, seg_len) in enumerate(batch_spans):
                        at_start = seg_start == 0
                        at_end = seg_start + chunk_size >= total_samples
                        if (at_start or at_end) and fade_size > 0:
                            window = windowing_array.clone()
                            if at_start:
                                window[:fade_size] = 1.0
                            if at_end:
                                window[-fade_size:] = 1.0
                        else:
                            window = windowing_array
                        win_part = window[:seg_len]
                        result[:, :, seg_start:seg_end] += (
                            output_audio[idx, :, :, :seg_len] * win_part
                        )
                        counter[:, :, seg_start:seg_end] += win_part

                    batch_chunks.clear()
                    batch_spans.clear()
    finally:
        model.train(was_training)

    result = result / counter.clamp(min=1e-8)
    if border > 0:
        result = result[:, :, border:-border]

    result_np = result.numpy()
    return {instr: result_np[i] for i, instr in enumerate(instruments)}


def _discover_validation_mixtures(valid_paths: Sequence[str]) -> list[str]:
    mixtures: list[str] = []
    for valid_path in valid_paths:
        part = sorted(glob.glob(os.path.join(valid_path, "*", "mixture.wav")))
        if not part:
            part = sorted(glob.glob(os.path.join(valid_path, "*", "mixture.flac")))
        if not part:
            print(f"No validation data found in: {valid_path}")
        mixtures.extend(part)
    return mixtures


class MSSSpeakerValidationDataset(Dataset):
    """Validation dataset yielding mixture paths paired with mean speaker embeddings."""

    def __init__(self, args: Any):
        self.list_of_files = _discover_validation_mixtures(args.valid_path)

    def __len__(self) -> int:
        return len(self.list_of_files)

    def __getitem__(self, index: int):
        path = self.list_of_files[index]
        folder = os.path.dirname(path)
        embedding_path = os.path.join(folder, "embeddings.npy")
        embedding = torch.zeros(192, dtype=torch.float32)

        if os.path.exists(embedding_path):
            try:
                embeddings = np.load(embedding_path, allow_pickle=False)
                if embeddings.shape[0] > 0:
                    emb_mean = np.mean(embeddings, axis=0)
                    embedding = torch.from_numpy(np.ascontiguousarray(emb_mean, dtype=np.float32))
            except Exception as error:
                print(f"Error loading embedding for {path}: {error}")

        return path, embedding


class MSSValidationDataset(Dataset):
    """Validation dataset yielding mixture paths without speaker embeddings."""

    def __init__(self, args: Any):
        self.list_of_files = _discover_validation_mixtures(args.valid_path)

    def __len__(self) -> int:
        return len(self.list_of_files)

    def __getitem__(self, index: int):
        return self.list_of_files[index]


def _read_reference_stem(folder: str, stem_name: str) -> np.ndarray:
    for ext in ("wav", "flac"):
        candidate = os.path.join(folder, f"{stem_name}.{ext}")
        if os.path.isfile(candidate):
            track, _ = sf.read(candidate, dtype="float32")
            return track
    raise FileNotFoundError(f"Reference stem '{stem_name}' not found in {folder}")


def validate_separator(
    model: nn.Module,
    valid_loader: DataLoader,
    args: Any,
    config: Any,
    device: torch.device,
    metric_list: Sequence[str],
    *,
    verbose: bool = False,
    stft_processor: Optional[STFTProcessorBatch] = None,
    use_spk: bool = True,
    diffusion_wrappers: Optional[Sequence[Any]] = None,
    current_step: Optional[int] = None,
) -> dict[str, dict[str, list[torch.Tensor]]]:
    """Run validation under inference_mode and restore the model's training mode."""
    was_training = model.training
    model.eval()
    instruments = prefer_target_instrument(config)
    all_metrics: dict[str, dict[str, list[torch.Tensor]]] = {
        metric: {instr: [] for instr in instruments} for metric in metric_list
    }

    iterator = tqdm(valid_loader, ncols=200) if verbose else valid_loader
    pbar_dict: dict[str, float] = {}

    try:
        with torch.inference_mode():
            for batch in iterator:
                if use_spk:
                    path = batch[0][0]
                    spk_emb = batch[1].to(device, non_blocking=True)
                else:
                    path = batch[0]
                    spk_emb = None

                mix, _ = sf.read(path, dtype="float32")
                folder = os.path.dirname(path)

                if use_spk:
                    if args.model_type in EXPORTABLE_MODELS:
                        res = demix_with_spk_exportable(
                            config, model, mix.T, spk_emb, device, stft_processor
                        )
                    else:
                        res = demix_with_spk(
                            config, model, mix.T, spk_emb, device, model_type=args.model_type
                        )
                else:
                    res = demix(config, model, mix.T, device, model_type=args.model_type)

                for stem_idx, instr in enumerate(instruments):
                    if instr != "other" or config.training.other_fix is False:
                        track = _read_reference_stem(folder, instr)
                    else:
                        track = mix - _read_reference_stem(folder, "vocals")

                    if (
                        diffusion_wrappers is not None
                        and stem_idx < len(diffusion_wrappers)
                        and diffusion_wrappers[stem_idx] is not None
                        and current_step is not None
                        and current_step >= diffusion_wrappers[stem_idx].config.n_train_start
                    ):
                        diff_model = diffusion_wrappers[stem_idx]
                        source_wave = torch.from_numpy(res[instr]).unsqueeze(0).to(device)
                        enhanced_wave = diff_model.inference(
                            source_wave, num_steps=10, method="euler"
                        )
                        res[instr] = enhanced_wave.squeeze(0).cpu().numpy()

                    references = np.expand_dims(track, axis=0)
                    estimates = np.expand_dims(res[instr].T, axis=0)
                    results = get_metrics(list(metric_list), references, estimates, mix)
                    for metric_name, value in results:
                        all_metrics[metric_name][instr].append(
                            torch.tensor([value], device=device, dtype=torch.float32)
                        )
                        pbar_dict[f"{metric_name}_{instr}"] = value
                if verbose:
                    iterator.set_postfix(pbar_dict)
    finally:
        model.train(was_training)

    return all_metrics


def gather_metric_tensors(
    accelerator,
    metric_dict: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Gather per-instrument validation metric tensors across distributed processes."""
    gathered: dict[str, torch.Tensor] = {}
    for instr, values in metric_dict.items():
        dtype = values[0].dtype if values else torch.float32
        local_tensor = (
            torch.cat(values, dim=0)
            if values
            else torch.empty(0, device=accelerator.device, dtype=dtype)
        )
        gathered[instr] = accelerator.gather_for_metrics(local_tensor).cpu()
    return gathered


def save_last_checkpoint(accelerator, model: nn.Module, results_path: str, model_type: str) -> None:
    """Save the latest separation weights on rank 0."""
    store_path = os.path.join(results_path, f"last_{model_type}.ckpt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save(unwrap_model_state_dict(accelerator, model), store_path)


def run_pre_validation(
    *,
    accelerator,
    model: nn.Module,
    valid_loader: DataLoader,
    valid_dataset_length: int,
    args: Any,
    config: Any,
    device: torch.device,
    metric_list: Sequence[str],
    stft_processor: Optional[STFTProcessorBatch] = None,
    use_spk: bool = True,
) -> None:
    """Run a single validation pass before training starts and print aggregated metrics."""
    if valid_dataset_length == 0:
        return

    metrics_dict = validate_separator(
        model,
        valid_loader,
        args,
        config,
        device,
        metric_list,
        verbose=accelerator.is_main_process,
        stft_processor=stft_processor,
        use_spk=use_spk,
    )
    gathered_metrics = {
        metric_name: gather_metric_tensors(accelerator, metric_data)
        for metric_name, metric_data in metrics_dict.items()
    }
    accelerator.wait_for_everyone()

    instruments = prefer_target_instrument(config)
    for metric_name in metric_list:
        metric_avg = 0.0
        for instr in instruments:
            metric_data = gathered_metrics[metric_name][instr].numpy()
            metric_val = float(metric_data[:valid_dataset_length].mean())
            accelerator.print(
                f"Instr {instr} {metric_name.upper()}: {metric_val:.4f} Debug: {len(metric_data)}"
            )
            metric_avg += metric_val
        metric_avg /= len(instruments)
        if len(instruments) > 1:
            accelerator.print(f"{metric_name.upper()} Avg: {metric_avg:.4f}")


def evaluate_and_checkpoint(
    *,
    accelerator,
    model: nn.Module,
    valid_loader: DataLoader,
    valid_dataset_length: int,
    args: Any,
    config: Any,
    device: torch.device,
    metric_list: Sequence[str],
    epoch: int,
    best_sdr: float,
    best_si_sdr: float,
    stft_processor: Optional[STFTProcessorBatch] = None,
    use_spk: bool = True,
    diffusion_wrappers: Optional[Sequence[Any]] = None,
    current_step: Optional[int] = None,
    save_best_sdr: bool = True,
) -> tuple[float, float, float, float]:
    """Run validation, log metrics, save best checkpoints, and return averages and best scores."""
    if valid_dataset_length == 0:
        return best_sdr, best_si_sdr, best_sdr, best_si_sdr

    metrics_dict = validate_separator(
        model,
        valid_loader,
        args,
        config,
        device,
        metric_list,
        verbose=accelerator.is_main_process,
        stft_processor=stft_processor,
        use_spk=use_spk,
        diffusion_wrappers=diffusion_wrappers,
        current_step=current_step,
    )

    gathered_metrics = {
        metric_name: gather_metric_tensors(accelerator, metric_data)
        for metric_name, metric_data in metrics_dict.items()
    }
    accelerator.wait_for_everyone()

    instruments = prefer_target_instrument(config)
    sdr_avg = 0.0
    si_sdr_avg = 0.0
    metrics_to_log: dict[str, Any] = {}

    for metric_name in metric_list:
        metric_avg = 0.0
        for instr in instruments:
            metric_data = gathered_metrics[metric_name][instr].numpy()
            metric_val = float(metric_data[:valid_dataset_length].mean())
            if accelerator.is_main_process:
                print(
                    f"Instr {instr} {metric_name.upper()}: {metric_val:.4f} Debug: {len(metric_data)}"
                )
                metrics_to_log[f"{instr}_{metric_name}"] = metric_val
            metric_avg += metric_val
        metric_avg /= len(instruments)
        if metric_name == "sdr":
            sdr_avg = metric_avg
        elif metric_name == "si_sdr":
            si_sdr_avg = metric_avg
        if accelerator.is_main_process:
            metrics_to_log[f"{metric_name}_avg"] = metric_avg
            if len(instruments) > 1:
                print(f"{metric_name.upper()} Avg: {metric_avg:.4f}")

    if accelerator.is_main_process:
        state_dict = None
        if si_sdr_avg > best_si_sdr:
            best_si_sdr = si_sdr_avg
            store_path = os.path.join(
                args.results_path, f"model_{args.model_type}_ep_{epoch}_sisdr_{si_sdr_avg:.4f}.ckpt"
            )
            print(f"Store weights (Best SI-SDR): {store_path}")
            state_dict = unwrap_model_state_dict(accelerator, model)
            accelerator.save(state_dict, store_path)

        if sdr_avg > best_sdr:
            best_sdr = sdr_avg
            if save_best_sdr:
                store_path = os.path.join(
                    args.results_path, f"model_{args.model_type}_ep_{epoch}_sdr_{sdr_avg:.4f}.ckpt"
                )
                print(f"Store weights (Best SDR): {store_path}")
                if state_dict is None:
                    state_dict = unwrap_model_state_dict(accelerator, model)
                accelerator.save(state_dict, store_path)

        metrics_to_log["best_sdr"] = best_sdr
        metrics_to_log["best_si_sdr"] = best_si_sdr
        metrics_to_log["epoch"] = epoch
        wandb.log(metrics_to_log)

    return sdr_avg, si_sdr_avg, best_sdr, best_si_sdr


def setup_discriminators(
    config: Any,
    args: Any,
    accelerator,
    *,
    default_type: str = "wave",
) -> tuple[list[Any], list[Optional[str]]]:
    """Instantiate per-stem DiscriminatorWrapper modules without double-wrapping in Accelerate."""
    if not args.use_discriminator:
        return [], []

    from discriminator.discriminator_wrapper import DiscriminatorConfig, DiscriminatorWrapper

    gan_models_raw = getattr(config.training, "gan_model", None)
    num_stems = len(config.training.instruments)

    if gan_models_raw is None:
        accelerator.print(
            f"Warning: gan_model not found in config, using default '{default_type}' for all stems"
        )
        gan_models: list[Optional[str]] = [default_type] * num_stems
    elif len(gan_models_raw) != num_stems:
        raise ValueError(
            f"gan_model length ({len(gan_models_raw)}) must match number of instruments ({num_stems})"
        )
    else:
        gan_models = list(gan_models_raw)

    accelerator.print(f"Creating discriminators for {num_stems} stems with types: {gan_models}")

    discriminators: list[Any] = []
    for idx, (instr, gan_type) in enumerate(zip(config.training.instruments, gan_models)):
        if gan_type is None or str(gan_type).lower() == "none":
            discriminators.append(None)
            accelerator.print(f"Discriminator for stem {idx} ({instr}): DISABLED")
            continue

        disc_cfg = DiscriminatorConfig(input_type=gan_type)
        disc_wrapper = DiscriminatorWrapper(disc_cfg, accelerator)

        if args.discriminator_start_check_point:
            ckpt_path = os.path.join(args.discriminator_start_check_point, f"{instr}.pth")
            if os.path.exists(ckpt_path):
                disc_wrapper.load_checkpoint(ckpt_path)
                accelerator.print(f"Loaded discriminator checkpoint for {instr} from {ckpt_path}")
            else:
                accelerator.print(f"Warning: checkpoint not found for {instr} at {ckpt_path}")

        disc_wrapper.train()
        discriminators.append(disc_wrapper)
        accelerator.print(f"Discriminator for stem {idx} ({instr}): type '{gan_type}'")

    return discriminators, gan_models


def setup_diffusion_wrappers(
    config: Any,
    args: Any,
    accelerator,
) -> list[Any]:
    """Instantiate per-stem DiffusionWrapper modules without double-wrapping in Accelerate."""
    if not args.use_diffusion:
        return []

    from diffusion.diffusion_wrapper import DiffusionConfig, DiffusionWrapper

    diffusion_model_raw = getattr(config.training, "diffusion_model", None)
    num_stems = len(config.training.instruments)

    if diffusion_model_raw is None:
        accelerator.print(
            "Warning: diffusion_model not found in config, using default 'dit' for all stems"
        )
        diffusion_models: list[Optional[str]] = ["dit"] * num_stems
    elif len(diffusion_model_raw) != num_stems:
        raise ValueError(
            f"diffusion_model length ({len(diffusion_model_raw)}) must match number of instruments ({num_stems})"
        )
    else:
        diffusion_models = list(diffusion_model_raw)

    accelerator.print(
        f"Creating diffusion models for {num_stems} stems with types: {diffusion_models}"
    )

    wrappers: list[Any] = []
    for instr, diffusion_type in zip(config.training.instruments, diffusion_models):
        if diffusion_type is None or str(diffusion_type).lower() == "none":
            wrappers.append(None)
            accelerator.print(f"Stem '{instr}': No diffusion model (None)")
            continue

        diffusion_options = dict(getattr(config.training, "diffusion_options", {}))
        diffusion_options["model_type"] = diffusion_type
        diff_cfg = DiffusionConfig(**diffusion_options)
        diff_wrapper = DiffusionWrapper(diff_cfg, accelerator)

        if args.diffusion_model_path:
            ckpt_path = os.path.join(args.diffusion_model_path, f"{instr}_diffusion.pth")
            if os.path.exists(ckpt_path):
                diff_wrapper.load_checkpoint(ckpt_path)
                accelerator.print(f"Loaded diffusion checkpoint for {instr} from {ckpt_path}")
            else:
                accelerator.print(
                    f"Warning: diffusion checkpoint not found for {instr} at {ckpt_path}"
                )

        diff_wrapper.train()
        accelerator.print(f"Stem '{instr}': Diffusion model type '{diffusion_type}'")
        wrappers.append(diff_wrapper)

    return wrappers


class StepLossMeter:
    """Accumulate scalar loss tensors on-device and sync only on logging intervals."""

    def __init__(self, device: torch.device):
        self.device = device
        self.epoch_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.window_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.epoch_count = 0
        self.window_count = 0

    def update(self, loss: torch.Tensor) -> None:
        detached = loss.detach().float()
        self.epoch_sum += detached
        self.window_sum += detached
        self.epoch_count += 1
        self.window_count += 1

    def flush_window(self) -> tuple[float, float]:
        """Return (window_avg, epoch_avg) with a single host synchronization."""
        if self.window_count == 0 or self.epoch_count == 0:
            return 0.0, 0.0
        vals = torch.stack((self.window_sum / self.window_count, self.epoch_sum / self.epoch_count))
        window_avg, epoch_avg = vals.tolist()
        self.window_sum.zero_()
        self.window_count = 0
        return float(window_avg), float(epoch_avg)

    def epoch_average(self) -> float:
        if self.epoch_count == 0:
            return 0.0
        return float((self.epoch_sum / self.epoch_count).item())
