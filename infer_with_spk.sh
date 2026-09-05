#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Required settings
# -----------------------------------------------------------------------------
# Project-relative paths are resolved from this repository's root. Every value
# can also be overridden through an environment variable of the same name.
PYTHON_BIN="${PYTHON_BIN:-python}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-inference_with_spk.py}"
MODEL_TYPE="${MODEL_TYPE:-spk_bs_roformer}"
CONFIG_PATH="${CONFIG_PATH:-ckpts/multi_stem/config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-ckpts/multi_stem/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt}"
INPUT_FOLDER="${INPUT_FOLDER:-/path/to/input}"
STORE_DIR="${STORE_DIR:-results/spk_inference}"

# Leave empty to download and verify the official pretrained CAM++ model.
SPK_MODEL_PATH="${SPK_MODEL_PATH:-}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"

# -----------------------------------------------------------------------------
# Runtime settings
# -----------------------------------------------------------------------------
# Physical GPU IDs, separated by spaces: GPU_IDS="0" or GPU_IDS="0 1".
# inference_with_spk.py receives logical IDs after CUDA_VISIBLE_DEVICES is set.
GPU_IDS="${GPU_IDS:-0}"
FORCE_CPU="${FORCE_CPU:-false}"
EXTRACT_INSTRUMENTAL="${EXTRACT_INSTRUMENTAL:-false}"
EXTRACT_OTHER="${EXTRACT_OTHER:-false}"
DISABLE_DETAILED_PBAR="${DISABLE_DETAILED_PBAR:-false}"
OUTPUT_FLAC="${OUTPUT_FLAC:-false}"
PCM_TYPE="${PCM_TYPE:-PCM_24}"

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

is_placeholder() {
    [[ "$1" == /path/to/* ]]
}

require_executable() {
    local executable="$1"
    if [[ "$executable" == */* ]]; then
        [[ -x "$executable" ]] || fail "executable does not exist or is not executable: $executable"
    else
        command -v "$executable" >/dev/null 2>&1 || fail "required command not found: $executable"
    fi
}

validate_boolean() {
    local name="$1"
    local value="$2"
    [[ "$value" == true || "$value" == false ]] || fail "$name must be true or false (got: $value)"
}

is_git_lfs_pointer() {
    local path="$1"
    local lfs_header='version https://git-lfs.github.com/spec/v1'
    LC_ALL=C head -c "${#lfs_header}" "$path" 2>/dev/null |
        LC_ALL=C grep -aqF "$lfs_header"
}

require_executable "$PYTHON_BIN"
require_executable realpath
[[ -f "$INFERENCE_SCRIPT" ]] || fail "inference entry point does not exist: $INFERENCE_SCRIPT"
[[ -f "$CONFIG_PATH" ]] || fail "configuration does not exist: $CONFIG_PATH"
[[ -n "$CHECKPOINT_PATH" ]] || fail "CHECKPOINT_PATH must not be empty"
[[ -f "$CHECKPOINT_PATH" ]] || fail "checkpoint does not exist: $CHECKPOINT_PATH"
if is_git_lfs_pointer "$CHECKPOINT_PATH"; then
    fail "checkpoint is only a Git LFS pointer; install Git LFS and run 'git lfs pull' first"
fi

is_placeholder "$INPUT_FOLDER" && fail "fill INPUT_FOLDER in infer_with_spk.sh or export INPUT_FOLDER"
[[ -d "$INPUT_FOLDER" ]] || fail "input directory does not exist: $INPUT_FOLDER"
[[ -n "$STORE_DIR" ]] || fail "STORE_DIR must not be empty"
store_dir_abs="$(realpath -m -- "$STORE_DIR")"
[[ "$store_dir_abs" != / ]] || fail "STORE_DIR must not be the filesystem root"
[[ "$store_dir_abs" != "$SCRIPT_DIR" ]] || \
    fail "STORE_DIR must be a dedicated output directory, not the repository root"

if [[ -n "$SPK_MODEL_PATH" ]]; then
    [[ -d "$SPK_MODEL_PATH" ]] || fail "CAM++ model directory does not exist: $SPK_MODEL_PATH"
fi
[[ "$INFERENCE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "INFERENCE_BATCH_SIZE must be positive"

validate_boolean FORCE_CPU "$FORCE_CPU"
validate_boolean EXTRACT_INSTRUMENTAL "$EXTRACT_INSTRUMENTAL"
validate_boolean EXTRACT_OTHER "$EXTRACT_OTHER"
validate_boolean DISABLE_DETAILED_PBAR "$DISABLE_DETAILED_PBAR"
validate_boolean OUTPUT_FLAC "$OUTPUT_FLAC"
[[ "$PCM_TYPE" == PCM_16 || "$PCM_TYPE" == PCM_24 ]] || \
    fail "PCM_TYPE must be PCM_16 or PCM_24 (got: $PCM_TYPE)"

read -r -a physical_gpu_ids <<< "$GPU_IDS"
logical_gpu_ids=(0)
if [[ "$FORCE_CPU" == true ]]; then
    export CUDA_VISIBLE_DEVICES=""
else
    [[ ${#physical_gpu_ids[@]} -gt 0 ]] || fail "GPU_IDS must contain at least one GPU ID"

    declare -A seen_gpu_ids=()
    for gpu_id in "${physical_gpu_ids[@]}"; do
        [[ "$gpu_id" =~ ^[0-9]+$ ]] || fail "GPU IDs must be non-negative integers (got: $gpu_id)"
        [[ -z "${seen_gpu_ids[$gpu_id]:-}" ]] || fail "duplicate GPU ID: $gpu_id"
        seen_gpu_ids[$gpu_id]=1
    done

    physical_gpu_csv="$(IFS=,; printf '%s' "${physical_gpu_ids[*]}")"
    export CUDA_VISIBLE_DEVICES="$physical_gpu_csv"

    gpu_check_output=""
    if ! gpu_check_output="$(
        "$PYTHON_BIN" -c \
            'import sys, torch; expected = int(sys.argv[1]); count = torch.cuda.device_count(); sys.exit(0 if torch.cuda.is_available() and count >= expected else f"PyTorch sees {count} CUDA device(s), expected at least {expected}")' \
            "${#physical_gpu_ids[@]}" 2>&1
    )"; then
        fail "GPU environment validation failed: $gpu_check_output"
    fi

    logical_gpu_ids=()
    for ((index = 0; index < ${#physical_gpu_ids[@]}; index++)); do
        logical_gpu_ids+=("$index")
    done
fi

mkdir -p "$STORE_DIR"

inference_args=(
    "$PYTHON_BIN" "$INFERENCE_SCRIPT"
    --model_type "$MODEL_TYPE"
    --config_path "$CONFIG_PATH"
    --start_check_point "$CHECKPOINT_PATH"
    --input_folder "$INPUT_FOLDER"
    --store_dir "$STORE_DIR"
    --device_ids "${logical_gpu_ids[@]}"
    --pcm_type "$PCM_TYPE"
    --inference_batch_size "$INFERENCE_BATCH_SIZE"
)

[[ -n "$SPK_MODEL_PATH" ]] && inference_args+=(--spk_model_path "$SPK_MODEL_PATH")
[[ "$FORCE_CPU" == true ]] && inference_args+=(--force_cpu)
[[ "$EXTRACT_INSTRUMENTAL" == true ]] && inference_args+=(--extract_instrumental)
[[ "$EXTRACT_OTHER" == true ]] && inference_args+=(--extract_other)
[[ "$DISABLE_DETAILED_PBAR" == true ]] && inference_args+=(--disable_detailed_pbar)
[[ "$OUTPUT_FLAC" == true ]] && inference_args+=(--flac_file)

printf 'Launching command:'
printf ' %q' "${inference_args[@]}"
printf '\n'

exec "${inference_args[@]}"
