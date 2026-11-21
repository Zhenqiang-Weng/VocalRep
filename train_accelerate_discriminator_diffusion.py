# coding: utf-8
"""Accelerate training entry point with optional discriminator and flow-matching diffusion."""

from __future__ import annotations

import os
import warnings

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import auraloss
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchaudio
from tqdm.auto import tqdm
import wandb

from utils.dataset import MSSDataset
from utils.losses import masked_loss
from utils.settings import get_model_from_config
from utils.train_utils import (
    MSSValidationDataset,
    STANDARD_ROFORMER_MODELS,
    StepLossMeter,
    build_optimizer,
    build_train_dataloader,
    configure_torch_runtime,
    enable_gradient_checkpointing,
    evaluate_and_checkpoint,
    init_wandb,
    load_separator_checkpoint,
    maybe_compile_model,
    resolve_gradient_accumulation_steps,
    run_pre_validation,
    save_last_checkpoint,
    set_multiprocessing_spawn,
    setup_diffusion_wrappers,
    setup_discriminators,
    validate_separator,
)
from utils.training_cli import build_training_parser

__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"
__version__ = "1.1.0"

warnings.filterwarnings("ignore")

metric_list = ["sdr", "si_sdr", "l1_freq", "bleedless", "fullness"]


def valid(
    model,
    valid_loader,
    args,
    config,
    device,
    verbose=False,
    diffusion_wrapper=None,
    current_step=None,
):
    """Evaluate a separator (and optional diffusion enhancer) on the validation split."""
    return validate_separator(
        model,
        valid_loader,
        args,
        config,
        device,
        metric_list,
        verbose=verbose,
        use_spk=False,
        diffusion_wrappers=diffusion_wrapper,
        current_step=current_step,
    )


def _to_float(value) -> float:
    return float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)


def train_model(args):
    parser = build_training_parser("Train a separator with optional discriminator and diffusion.")
    parser.add_argument("--use-multistft-loss", "--use_multistft_loss", action="store_true")
    parser.add_argument("--use-mse-loss", "--use_mse_loss", action="store_true")
    parser.add_argument("--use-l1-loss", "--use_l1_loss", action="store_true")
    parser.add_argument("--use-discriminator", "--use_discriminator", action="store_true")
    parser.add_argument(
        "--discriminator-start-checkpoint",
        "--discriminator_start_check_point",
        dest="discriminator_start_check_point",
        default="",
    )
    parser.add_argument(
        "--use-frequency-weighted-loss",
        "--use_frequency_weighted_loss",
        action="store_true",
    )
    parser.add_argument(
        "--copy-first-mask-estimator",
        "--copy_first_mask_estimator",
        action="store_true",
    )
    parser.add_argument(
        "--unfreeze-mask-estimators",
        "--unfreeze_mask_estimators",
        action="store_true",
    )
    parser.add_argument(
        "--estimator-unfreeze-indexes",
        "--estimator_unfreeze_indexes",
        dest="estimator_unfreeze_indexes",
        nargs="+",
        type=int,
        default=[0, 1],
    )
    parser.add_argument("--use-diffusion", "--use_diffusion", action="store_true")
    parser.add_argument(
        "--diffusion-model-path",
        "--diffusion_model_path",
        dest="diffusion_model_path",
        default="",
    )
    args = parser.parse_args() if args is None else parser.parse_args(args)

    model, config = get_model_from_config(args.model_type, args.config_path)
    auxiliary_active = bool(args.use_discriminator or args.use_diffusion)
    grad_accum_steps = resolve_gradient_accumulation_steps(
        args, config, auxiliary_active=auxiliary_active
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    configure_torch_runtime(
        args.seed + accelerator.process_index,
        deterministic=args.deterministic,
        allow_tf32=args.allow_tf32,
    )
    set_multiprocessing_spawn(accelerator)

    accelerator.print(f"Instruments: {config.training.instruments}")
    accelerator.print(
        f"Processes: {accelerator.num_processes} | Device: {device} | "
        f"Grad accum: {grad_accum_steps} | Mixed precision: {accelerator.mixed_precision}"
    )

    os.makedirs(args.results_path, exist_ok=True)

    batch_size = config.training.batch_size
    init_wandb(accelerator, args, config, args.device_ids, batch_size)

    config.training.num_steps *= accelerator.num_processes

    # Build the metadata cache once on rank 0 before worker processes open it.
    with accelerator.main_process_first():
        trainset = MSSDataset(
            config,
            args.data_path,
            batch_size=batch_size,
            metadata_path=os.path.join(args.results_path, f"metadata_{args.dataset_type}.pkl"),
            dataset_type=args.dataset_type,
            verbose=accelerator.is_main_process,
        )

    train_loader = build_train_dataloader(
        trainset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )

    validset = MSSValidationDataset(args)
    valid_dataset_length = len(validset)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False)

    load_separator_checkpoint(
        model,
        args.start_check_point,
        accelerator,
        strict=False,
        copy_first_mask_estimator=args.copy_first_mask_estimator,
        unfreeze_mask_estimators=args.unfreeze_mask_estimators,
        estimator_unfreeze_indexes=args.estimator_unfreeze_indexes,
    )
    enable_gradient_checkpointing(
        model, args.gradient_checkpointing, config=config, accelerator=accelerator
    )

    model = model.to(device)
    optimizer = build_optimizer(config, model, accelerator)

    if accelerator.is_main_process:
        print(f"Processes GPU: {accelerator.num_processes}")
        print(
            f"Patience: {config.training.patience} Reduce factor: {config.training.reduce_factor} "
            f"Batch size: {batch_size} Optimizer: {config.training.optimizer}"
        )

    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=config.training.patience,
        factor=config.training.reduce_factor,
    )

    loss_multistft = None
    if args.use_multistft_loss:
        loss_options = dict(config.loss_multistft) if hasattr(config, "loss_multistft") else {}
        accelerator.print(f"Loss options: {loss_options}")
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)

    model = maybe_compile_model(
        model,
        compile_enabled=args.compile_model,
        compile_mode=args.compile_mode,
        accelerator=accelerator,
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    train_loader, scheduler = accelerator.prepare(train_loader, scheduler)
    valid_loader = accelerator.prepare(valid_loader)

    accelerator.print(f"Model prepared on device: {next(model.parameters()).device}")

    # Auxiliary wrappers prepare their own model/optimizer/scheduler with Accelerate internally.
    discriminators, gan_models = setup_discriminators(config, args, accelerator, default_type="mel")
    diffusion_wrappers = setup_diffusion_wrappers(config, args, accelerator)

    # Mel spectrogram extractor is only allocated when at least one mel discriminator is active.
    mel = None
    to_db = None
    mel_hop = 441
    if args.use_discriminator and any(m == "mel" for m in gan_models if m is not None):
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_fft=2048, hop_length=mel_hop, n_mels=128, power=2.0
        ).to(device)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=130).to(device)

    def wave_bc_to_mel(wave: torch.Tensor) -> torch.Tensor:
        # FFT-based mel features require FP32 inputs when running under BF16 autocast.
        spec = to_db(mel(wave.float()) + 1e-5).permute(0, 1, 3, 2)
        b_sz, ch, frames, n_mels = spec.shape
        return spec.reshape(b_sz * ch, frames, n_mels)

    def samp_to_frames(lengths: torch.Tensor, channels: int) -> torch.Tensor:
        return torch.ceil(lengths.float() / mel_hop).long().repeat_interleave(channels)

    if args.pre_valid:
        run_pre_validation(
            accelerator=accelerator,
            model=model,
            valid_loader=valid_loader,
            valid_dataset_length=valid_dataset_length,
            args=args,
            config=config,
            device=device,
            metric_list=metric_list,
            use_spk=False,
        )

    accelerator.print(f"Train for: {config.training.num_epochs}")
    best_sdr = -float("inf")
    best_si_sdr = -float("inf")
    current_step = 0

    log_interval = max(1, int(args.log_interval))
    valid_every = max(1, int(args.valid_every))
    empty_cache_interval = max(0, int(args.empty_cache_interval))
    save_interval = int(getattr(config.training, "save_interval", 1000))
    instruments = config.training.instruments
    num_stems = len(instruments)
    is_standard_roformer = args.model_type in STANDARD_ROFORMER_MODELS

    for epoch in range(config.training.num_epochs):
        model.train()
        accelerator.print(f"Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]['lr']}")
        loss_meter = StepLossMeter(device)
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, ncols=130)
        for i, (batch, mixes) in enumerate(pbar):
            should_log = accelerator.is_main_process and (
                ((i + 1) % log_interval == 0) or (i + 1 == num_batches)
            )
            total_gan_loss = 0.0
            total_fm_loss = 0.0
            disc_losses_dict: dict[str, object] = {}
            gen_losses_dict: dict[str, object] = {}
            diffusion_losses_dict: dict[str, object] = {}

            with accelerator.accumulate(model):
                if is_standard_roformer:
                    loss, y_ = model(mixes, batch)
                else:
                    y_ = model(mixes)
                    if args.use_multistft_loss:
                        y1_ = y_.reshape(y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3])
                        y1 = batch.reshape(
                            batch.shape[0], batch.shape[1] * batch.shape[2], batch.shape[3]
                        )
                        loss = loss_multistft(y1_, y1)
                        if args.use_mse_loss:
                            loss = loss + 1000.0 * F.mse_loss(y1_, y1)
                        if args.use_l1_loss:
                            loss = loss + 1000.0 * F.l1_loss(y1_, y1)
                    elif args.use_mse_loss:
                        loss = F.mse_loss(y_, batch)
                    elif args.use_l1_loss:
                        loss = F.l1_loss(y_, batch)
                    else:
                        loss = masked_loss(
                            y_, batch, q=config.training.q, coarse=config.training.coarse_loss_clip
                        )

                if args.use_discriminator:
                    b_size, stems, channels, samples = y_.shape
                    if stems != num_stems:
                        raise ValueError(f"Expected {num_stems} stems, got {stems}")

                    len_frames = (
                        samp_to_frames(
                            torch.full((b_size,), samples, device=y_.device, dtype=torch.long),
                            channels,
                        )
                        if mel is not None
                        else None
                    )

                    for stem_idx, (disc, input_type, instr) in enumerate(
                        zip(discriminators, gan_models, instruments)
                    ):
                        if disc is None or input_type is None or str(input_type).lower() == "none":
                            continue

                        fake_stem = y_[:, stem_idx, :, :]
                        real_stem = batch[:, stem_idx, :, :]

                        if input_type == "mel":
                            fake_input = wave_bc_to_mel(fake_stem)
                            real_input = wave_bc_to_mel(real_stem)
                            length_input = len_frames
                        elif input_type in {"wave", "music"}:
                            fake_input = fake_stem.reshape(b_size * channels, samples)
                            real_input = real_stem.reshape(b_size * channels, samples)
                            length_input = None
                        else:
                            raise ValueError(f"Unknown input_type: {input_type}")

                        d_losses = disc.train_step(
                            fake_input,
                            real_input,
                            length_input,
                            current_step=current_step,
                            compute_generator_losses=False,
                        )
                        g_losses = disc.get_generator_losses(
                            fake_input, real_input, length_input, current_step=current_step
                        )
                        gan_loss = g_losses.get("gan_loss", 0.0)
                        fm_loss = g_losses.get("hdn_loss", 0.0)

                        total_gan_loss = total_gan_loss + gan_loss
                        total_fm_loss = total_fm_loss + fm_loss

                        if should_log:
                            disc_losses_dict[f"disc_{instr}"] = d_losses.get("disc_loss", 0.0)
                            gen_losses_dict[f"gen_{instr}_gan"] = gan_loss
                            gen_losses_dict[f"gen_{instr}_fm"] = fm_loss

                    loss = loss + total_gan_loss + total_fm_loss

                if args.use_diffusion:
                    stems = y_.shape[1]
                    if stems != num_stems:
                        raise ValueError(f"Expected {num_stems} stems, got {stems}")

                    for stem_idx, (diff_model, instr) in enumerate(
                        zip(diffusion_wrappers, instruments)
                    ):
                        if diff_model is None:
                            continue

                        pred_stem = y_[:, stem_idx, :, :]
                        target_stem = batch[:, stem_idx, :, :]
                        diff_losses = diff_model.train_step_dual(
                            pred_stem, target_stem, current_step=current_step
                        )
                        if should_log:
                            diffusion_losses_dict[f"diff_{instr}_loss"] = diff_losses[
                                "diffusion_loss"
                            ]

                accelerator.backward(loss)
                if accelerator.sync_gradients and config.training.grad_clip:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process:
                loss_meter.update(loss)
                if should_log:
                    window_loss, avg_loss = loss_meter.flush_window()
                    log_dict: dict[str, object] = {
                        "loss": 100.0 * window_loss,
                        "avg_loss": 100.0 * avg_loss,
                        "total": loss_meter.epoch_count,
                        "i": i,
                    }
                    if args.use_discriminator:
                        for key, val in disc_losses_dict.items():
                            log_dict[key] = _to_float(val)
                        for key, val in gen_losses_dict.items():
                            log_dict[key] = _to_float(val)
                        log_dict["total_gan_loss"] = _to_float(total_gan_loss)
                        log_dict["total_fm_loss"] = _to_float(total_fm_loss)
                    if args.use_diffusion:
                        for key, val in diffusion_losses_dict.items():
                            log_dict[key] = _to_float(val)
                    wandb.log(log_dict)
                    pbar.set_postfix(
                        {
                            "loss": f"{100.0 * window_loss:.4f}",
                            "avg_loss": f"{100.0 * avg_loss:.4f}",
                        }
                    )

            if (
                auxiliary_active
                and save_interval > 0
                and (current_step + 1) % save_interval == 0
                and accelerator.is_main_process
            ):
                ckpt_dir = os.path.join(args.results_path, "last_additional_model_ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                if args.use_discriminator:
                    for disc, instr in zip(discriminators, instruments):
                        if disc is not None:
                            disc.save_checkpoint(os.path.join(ckpt_dir, f"{instr}.pth"))
                    accelerator.print(f"Discriminator checkpoints saved to {ckpt_dir}")
                if args.use_diffusion:
                    for diff_model, instr in zip(diffusion_wrappers, instruments):
                        if diff_model is not None:
                            diff_model.save_checkpoint(
                                os.path.join(ckpt_dir, f"{instr}_diffusion.pth")
                            )
                    accelerator.print(f"Diffusion model checkpoints saved to {ckpt_dir}")

            current_step += 1
            if (
                empty_cache_interval > 0
                and current_step % empty_cache_interval == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

        if num_batches > 0:
            del (
                batch,
                mixes,
                y_,
                loss,
                total_gan_loss,
                total_fm_loss,
                disc_losses_dict,
                gen_losses_dict,
                diffusion_losses_dict,
            )

        if accelerator.is_main_process:
            epoch_loss = loss_meter.epoch_average()
            print(f"Training loss: {epoch_loss:.6f}")
            wandb.log({"train_loss": epoch_loss, "epoch": epoch})

        save_last_checkpoint(accelerator, model, args.results_path, args.model_type)

        should_validate = (
            not args.skip_valid
            and valid_dataset_length > 0
            and (((epoch + 1) % valid_every == 0) or (epoch + 1 == config.training.num_epochs))
        )
        if should_validate:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _, si_sdr_avg, best_sdr, best_si_sdr = evaluate_and_checkpoint(
                accelerator=accelerator,
                model=model,
                valid_loader=valid_loader,
                valid_dataset_length=valid_dataset_length,
                args=args,
                config=config,
                device=device,
                metric_list=metric_list,
                epoch=epoch,
                best_sdr=best_sdr,
                best_si_sdr=best_si_sdr,
                use_spk=False,
                diffusion_wrappers=diffusion_wrappers if args.use_diffusion else None,
                current_step=current_step,
                save_best_sdr=False,
            )
            # Every distributed process must advance its scheduler identically.
            scheduler.step(si_sdr_avg)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train_model(None)
