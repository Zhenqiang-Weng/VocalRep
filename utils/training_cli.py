"""Shared command-line arguments for the Accelerate training entry points."""

from __future__ import annotations

import argparse
import os


def build_wandb_config(
    args: argparse.Namespace,
    config: object,
    device_ids: list[int],
    batch_size: int,
) -> dict[str, object]:
    """Build public W&B metadata without copying credentials into the run."""

    public_args = vars(args).copy()
    public_args.pop("wandb_key", None)
    return {
        "config": config,
        "args": public_args,
        "device_ids": device_ids,
        "batch_size": batch_size,
    }


def build_training_parser(
    description: str,
    *,
    include_mix_consistent_loss: bool = False,
) -> argparse.ArgumentParser:
    """Create the common parser used by all training scripts.

    Both kebab-case and the legacy underscore option names are accepted so that
    existing launch commands remain compatible.
    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        "--model_type",
        dest="model_type",
        required=True,
        help="Model identifier supported by utils.settings.get_model_from_config.",
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        required=True,
        help="YAML model/training configuration file.",
    )
    parser.add_argument(
        "--start-checkpoint",
        "--start_check_point",
        dest="start_check_point",
        default="",
        help="Optional separator checkpoint used to initialize model weights.",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        required=True,
        help="Directory for checkpoints, logs, and the dataset metadata cache.",
    )
    parser.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        nargs="+",
        required=True,
        help="Training roots; dataset type 3 currently accepts exactly one CSV.",
    )
    parser.add_argument(
        "--dataset-type",
        "--dataset_type",
        dest="dataset_type",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
        help="Dataset layout described in docs/TRAINING_DATA.md.",
    )
    parser.add_argument(
        "--valid-path",
        "--valid_path",
        dest="valid_path",
        nargs="+",
        required=True,
        help="One or more validation roots containing track subdirectories.",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes per training process.",
    )
    parser.add_argument(
        "--pin-memory",
        "--pin_memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pin DataLoader CPU memory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic PyTorch algorithms where available (slower).",
    )
    parser.add_argument(
        "--device-ids",
        "--device_ids",
        dest="device_ids",
        nargs="+",
        type=int,
        default=[0],
        help="Visible CUDA device IDs, mainly recorded in run metadata.",
    )

    if include_mix_consistent_loss:
        parser.add_argument(
            "--use-mix-consistent-loss",
            "--use_mix_consistent_loss",
            action="store_true",
        )
    parser.add_argument("--pre-valid", "--pre_valid", action="store_true")

    parser.add_argument(
        "--wandb-key",
        "--wandb_key",
        dest="wandb_key",
        default=os.environ.get("WANDB_API_KEY", ""),
        help="Prefer the WANDB_API_KEY environment variable instead of this option.",
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        dest="wandb_project",
        default="msst-accelerate",
    )
    parser.add_argument(
        "--wandb-name",
        "--wandb_name",
        dest="wandb_name",
        default="",
    )
    return parser
