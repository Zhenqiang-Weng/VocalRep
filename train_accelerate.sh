#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Required project settings
# -----------------------------------------------------------------------------
# Fill the placeholder training/validation paths before running this script.
TRAIN_SCRIPT="train_accelerate_bf16.py"
MODEL_TYPE="spk_bs_roformer_exportable"
CONFIG_PATH="ckpts/multi_stem/config.yaml"
RESULTS_PATH="results/multi_stem"
DATASET_TYPE=4

TRAIN_DATA_PATHS=(
    "/path/to/train"
)

VALID_DATA_PATHS=(
    "/path/to/valid"
)

# Leave empty to train from scratch. Do not point this at a Git LFS pointer.
START_CHECKPOINT=""

# -----------------------------------------------------------------------------
# Runtime settings
# -----------------------------------------------------------------------------
# Physical GPU IDs. Use (0) for one GPU or (0 1 2 3) for four GPUs.
GPU_IDS=(0)
MIXED_PRECISION="bf16"       # no, fp16, or bf16
NUM_WORKERS=4                 # per training process
PIN_MEMORY=true
SEED=42
DETERMINISTIC=false
MAIN_PROCESS_PORT=29500

# W&B runs offline unless WANDB_API_KEY is exported in the shell.
WANDB_PROJECT="msst-accelerate"
WANDB_NAME="spk-bs-roformer-$(date +%Y%m%d-%H%M%S)"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

# Optional flags supported by train_accelerate_bf16.py.
PRE_VALIDATE=false
USE_MIX_CONSISTENT_LOSS=false

# The discriminator variant uses the same speaker-guided inputs:
# TRAIN_SCRIPT="train_accelerate_bf16_with_discriminator.py"
# EXTRA_ARGS=(--use-discriminator)
#
# The discriminator+diffusion entry point requires a different matching model
# and config. Do not switch only TRAIN_SCRIPT/EXTRA_ARGS.
EXTRA_ARGS=()

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

is_placeholder() {
    [[ "$1" == /path/to/* ]]
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

validate_boolean() {
    local name="$1"
    local value="$2"
    [[ "$value" == true || "$value" == false ]] || \
        fail "$name must be true or false (got: $value)"
}

validate_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be a non-negative integer (got: $value)"
}

validate_input_paths() {
    local path
    for path in "${TRAIN_DATA_PATHS[@]}"; do
        is_placeholder "$path" && fail "fill TRAIN_DATA_PATHS in train_accelerate.sh"
        if [[ "$DATASET_TYPE" -eq 3 ]]; then
            [[ -f "$path" ]] || fail "training CSV does not exist: $path"
        else
            [[ -d "$path" ]] || fail "training directory does not exist: $path"
        fi
    done

    for path in "${VALID_DATA_PATHS[@]}"; do
        is_placeholder "$path" && fail "fill VALID_DATA_PATHS in train_accelerate.sh"
        [[ -d "$path" ]] || fail "validation directory does not exist: $path"
    done
}

require_command python
require_command accelerate
[[ -f "$TRAIN_SCRIPT" ]] || fail "training entry point does not exist: $TRAIN_SCRIPT"
[[ -f "$CONFIG_PATH" ]] || fail "configuration does not exist: $CONFIG_PATH"
[[ ${#GPU_IDS[@]} -gt 0 ]] || fail "GPU_IDS must contain at least one GPU"
[[ "$DATASET_TYPE" =~ ^[1-4]$ ]] || fail "DATASET_TYPE must be 1, 2, 3, or 4"
[[ "$MIXED_PRECISION" == no || "$MIXED_PRECISION" == fp16 || "$MIXED_PRECISION" == bf16 ]] || \
    fail "MIXED_PRECISION must be no, fp16, or bf16"
validate_nonnegative_integer NUM_WORKERS "$NUM_WORKERS"
validate_nonnegative_integer SEED "$SEED"
validate_nonnegative_integer MAIN_PROCESS_PORT "$MAIN_PROCESS_PORT"
validate_boolean PIN_MEMORY "$PIN_MEMORY"
validate_boolean DETERMINISTIC "$DETERMINISTIC"
validate_boolean PRE_VALIDATE "$PRE_VALIDATE"
validate_boolean USE_MIX_CONSISTENT_LOSS "$USE_MIX_CONSISTENT_LOSS"

declare -A seen_gpu_ids=()
for gpu_id in "${GPU_IDS[@]}"; do
    validate_nonnegative_integer GPU_ID "$gpu_id"
    [[ -z "${seen_gpu_ids[$gpu_id]:-}" ]] || fail "duplicate GPU ID: $gpu_id"
    seen_gpu_ids[$gpu_id]=1
done
validate_input_paths

if [[ -n "$START_CHECKPOINT" ]]; then
    [[ -f "$START_CHECKPOINT" ]] || fail "checkpoint does not exist: $START_CHECKPOINT"
    if [[ "$(head -n 1 "$START_CHECKPOINT")" == "version https://git-lfs.github.com/spec/v1" ]]; then
        fail "checkpoint is only a Git LFS pointer; run 'git lfs pull' first"
    fi
fi

mkdir -p "$RESULTS_PATH"

printf 'Validating training and validation data...\n'
validator_args=(
    python scripts/validate_training_data.py
    --config-path "$CONFIG_PATH"
    --dataset-type "$DATASET_TYPE"
    --data-path "${TRAIN_DATA_PATHS[@]}"
    --valid-path "${VALID_DATA_PATHS[@]}"
)
if [[ "$TRAIN_SCRIPT" == train_accelerate_bf16*.py && "$DATASET_TYPE" -eq 4 ]]; then
    validator_args+=(--require-embeddings)
fi
"${validator_args[@]}"

physical_gpu_csv="$(IFS=,; printf '%s' "${GPU_IDS[*]}")"
export CUDA_VISIBLE_DEVICES="$physical_gpu_csv"

logical_gpu_ids=()
for ((index = 0; index < ${#GPU_IDS[@]}; index++)); do
    logical_gpu_ids+=("$index")
done

launch_args=(
    accelerate launch
    --mixed_precision "$MIXED_PRECISION"
    --num_processes "${#GPU_IDS[@]}"
    --num_machines 1
    --main_process_port "$MAIN_PROCESS_PORT"
)
if (( ${#GPU_IDS[@]} > 1 )); then
    launch_args+=(--multi_gpu)
fi

training_args=(
    "$TRAIN_SCRIPT"
    --model-type "$MODEL_TYPE"
    --config-path "$CONFIG_PATH"
    --results-path "$RESULTS_PATH"
    --data-path "${TRAIN_DATA_PATHS[@]}"
    --valid-path "${VALID_DATA_PATHS[@]}"
    --dataset-type "$DATASET_TYPE"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --device-ids "${logical_gpu_ids[@]}"
    --wandb-project "$WANDB_PROJECT"
    --wandb-name "$WANDB_NAME"
)

[[ "$PIN_MEMORY" == true ]] && training_args+=(--pin-memory)
[[ "$DETERMINISTIC" == true ]] && training_args+=(--deterministic)
[[ "$PRE_VALIDATE" == true ]] && training_args+=(--pre-valid)
[[ "$USE_MIX_CONSISTENT_LOSS" == true ]] && training_args+=(--use-mix-consistent-loss)
[[ -n "$START_CHECKPOINT" ]] && training_args+=(--start-checkpoint "$START_CHECKPOINT")
training_args+=("${EXTRA_ARGS[@]}")

printf 'Launching command:'
printf ' %q' "${launch_args[@]}" "${training_args[@]}"
printf '\n'

exec "${launch_args[@]}" "${training_args[@]}"
