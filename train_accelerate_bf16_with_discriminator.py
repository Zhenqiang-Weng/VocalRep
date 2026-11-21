# coding: utf-8
"""Accelerate training entry point for speaker-guided separation with GAN discriminators."""

from __future__ import annotations

import os
import warnings

from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from utils.dataset_with_spk import MSSDatasetWithSpk
from utils.settings import get_model_from_config
from utils.train_utils import (
    EXPORTABLE_MODELS,
    MSSSpeakerValidationDataset,
    StepLossMeter,
    build_optimizer,
    build_stft_processor,
    build_train_dataloader,
    configure_torch_runtime,
    demix_with_spk_exportable,
    enable_gradient_checkpointing,
    evaluate_and_checkpoint,
    init_wandb,
    load_separator_checkpoint,
    maybe_compile_model,
    resolve_gradient_accumulation_steps,
    run_pre_validation,
    save_last_checkpoint,
    set_multiprocessing_spawn,
    setup_discriminators,
    validate_separator,
)
from utils.training_cli import build_training_parser

__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"
__version__ = "1.1.0"

warnings.filterwarnings("ignore")

METRIC_LIST = ["sdr", "si_sdr"]
AVAILABLE_MODELS = list(EXPORTABLE_MODELS)
MSSValidationDataset = MSSSpeakerValidationDataset
__all__ = [
    "AVAILABLE_MODELS",
    "METRIC_LIST",
    "MSSValidationDataset",
    "demix_with_spk_exportable",
    "train_model",
    "valid",
]


def valid(model, valid_loader, args, config, device, verbose=False, stft_processor=None):
    """Evaluate a speaker-guided separation model on the validation split."""
    return validate_separator(
        model,
        valid_loader,
        args,
        config,
        device,
        METRIC_LIST,
        verbose=verbose,
        stft_processor=stft_processor,
        use_spk=True,
    )


def _to_float(value) -> float:
    return float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)


def train_model(args):
    parser = build_training_parser(
        "Train a speaker-guided separator with a discriminator.",
        include_mix_consistent_loss=True,
    )
    parser.add_argument("--use-discriminator", "--use_discriminator", action="store_true")
    parser.add_argument(
        "--discriminator-start-checkpoint",
        "--discriminator_start_check_point",
        dest="discriminator_start_check_point",
        default="",
        help="Optional discriminator checkpoint directory.",
    )
    args = parser.parse_args() if args is None else parser.parse_args(args)

    model, config = get_model_from_config(args.model_type, args.config_path)
    grad_accum_steps = resolve_gradient_accumulation_steps(
        args, config, auxiliary_active=bool(args.use_discriminator)
    )
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)
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
        trainset = MSSDatasetWithSpk(
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

    load_separator_checkpoint(model, args.start_check_point, accelerator, strict=False)
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

    is_exportable = args.model_type in EXPORTABLE_MODELS
    stft_processor = build_stft_processor(config, device) if is_exportable else None
    discriminators, _ = setup_discriminators(config, args, accelerator, default_type="wave")

    if args.pre_valid:
        run_pre_validation(
            accelerator=accelerator,
            model=model,
            valid_loader=valid_loader,
            valid_dataset_length=valid_dataset_length,
            args=args,
            config=config,
            device=device,
            metric_list=["sdr"],
            stft_processor=stft_processor,
            use_spk=True,
        )

    accelerator.print(f"Train for: {config.training.num_epochs}")
    best_sdr = -float("inf")
    best_si_sdr = -float("inf")
    current_step = 0

    log_interval = max(1, int(args.log_interval))
    valid_every = max(1, int(args.valid_every))
    empty_cache_interval = max(0, int(args.empty_cache_interval))
    save_interval = int(getattr(config.training, "save_interval", 1000))
    use_multi_stft = getattr(config.training, "use_multi_stft_loss", True)
    mix_consistent_weight = getattr(config.training, "total_mix_consistent_loss_weight", 0.5)
    instruments = config.training.instruments
    num_stems = len(instruments)

    for epoch in range(config.training.num_epochs):
        model.train()
        accelerator.print(f"Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]['lr']}")
        loss_meter = StepLossMeter(device)
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, ncols=130)
        for i, (batch, mixes, spk_emb) in enumerate(pbar):
            should_log = accelerator.is_main_process and (
                ((i + 1) % log_interval == 0) or (i + 1 == num_batches)
            )
            total_mix_loss = None
            total_gan_loss = 0.0
            total_fm_loss = 0.0
            disc_losses_dict: dict[str, object] = {}
            gen_losses_dict: dict[str, object] = {}

            with accelerator.accumulate(model):
                if is_exportable:
                    x_stft = stft_processor.stft_batch(mixes)
                    y_stft = model(x_stft, spk_emb)
                    y_ = stft_processor.istft_batch(y_stft)
                    loss = stft_processor.compute_loss(y_, batch, use_multi_stft=use_multi_stft)
                else:
                    loss, y_ = model(mixes, spk_emb, batch)

                if args.use_mix_consistent_loss:
                    total_mix_loss = F.l1_loss(y_.sum(dim=1), batch.sum(dim=1))
                    loss = loss + mix_consistent_weight * total_mix_loss

                if args.use_discriminator:
                    b_size, stems, channels, samples = y_.shape
                    if stems != num_stems:
                        raise ValueError(f"Expected {num_stems} stems, got {stems}")

                    for stem_idx, (disc, instr) in enumerate(zip(discriminators, instruments)):
                        if disc is None:
                            continue

                        fake_input = y_[:, stem_idx, :, :].reshape(b_size * channels, samples)
                        real_input = batch[:, stem_idx, :, :].reshape(b_size * channels, samples)

                        d_losses = disc.train_step(
                            fake_input,
                            real_input,
                            None,
                            current_step=current_step,
                            compute_generator_losses=False,
                        )
                        g_losses = disc.get_generator_losses(
                            fake_input, real_input, None, current_step=current_step
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
                    if total_mix_loss is not None:
                        log_dict["mix_consistent_loss"] = 100.0 * _to_float(total_mix_loss)
                    if args.use_discriminator:
                        for key, val in disc_losses_dict.items():
                            log_dict[key] = _to_float(val)
                        for key, val in gen_losses_dict.items():
                            log_dict[key] = _to_float(val)
                        log_dict["total_gan_loss"] = _to_float(total_gan_loss)
                        log_dict["total_fm_loss"] = _to_float(total_fm_loss)
                    wandb.log(log_dict)
                    pbar.set_postfix(
                        {
                            "loss": f"{100.0 * window_loss:.4f}",
                            "avg_loss": f"{100.0 * avg_loss:.4f}",
                        }
                    )

            if (
                args.use_discriminator
                and save_interval > 0
                and (current_step + 1) % save_interval == 0
                and accelerator.is_main_process
            ):
                ckpt_dir = os.path.join(args.results_path, "last_discriminator_ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                for disc, instr in zip(discriminators, instruments):
                    if disc is not None:
                        disc.save_checkpoint(os.path.join(ckpt_dir, f"{instr}.pth"))
                accelerator.print(f"Discriminator checkpoints saved to {ckpt_dir}")

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
                spk_emb,
                y_,
                loss,
                total_mix_loss,
                total_gan_loss,
                total_fm_loss,
                disc_losses_dict,
                gen_losses_dict,
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
            sdr_avg, _, best_sdr, best_si_sdr = evaluate_and_checkpoint(
                accelerator=accelerator,
                model=model,
                valid_loader=valid_loader,
                valid_dataset_length=valid_dataset_length,
                args=args,
                config=config,
                device=device,
                metric_list=METRIC_LIST,
                epoch=epoch,
                best_sdr=best_sdr,
                best_si_sdr=best_si_sdr,
                stft_processor=stft_processor,
                use_spk=True,
                save_best_sdr=True,
            )
            # Every distributed process must advance its scheduler identically.
            scheduler.step(sdr_avg)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train_model(None)
