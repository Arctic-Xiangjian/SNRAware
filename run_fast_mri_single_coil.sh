#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

show_help() {
  cat <<'EOF'
run_fast_mri_single_coil.sh

This wrapper is kept for compatibility. The primary entrypoint is:
  uv run python train_fastmri_single_coil.py --train-root ... --val-root ...

Legacy compatibility:
  - Supports the old first positional CUDA device id.
  - Supports the main legacy environment variables.
  - Forwards any trailing KEY=VALUE items as --override KEY=VALUE.

Examples:
  TRAIN_ROOT=/data/fastmri/train VAL_ROOT=/data/fastmri/val DRY_RUN=true ./run_fast_mri_single_coil.sh
  CUDA_DEVICE=1 TRAIN_ROOT=/data/fastmri/train VAL_ROOT=/data/fastmri/val ./run_fast_mri_single_coil.sh
  ./run_fast_mri_single_coil.sh 0 lora.r=16 fastmri_finetune.batch_size=2
EOF
}

normalize_bool() {
  printf '%s' "${1}" | tr '[:upper:]' '[:lower:]'
}

append_if_set() {
  local env_name="${1}"
  local flag="${2}"
  if [[ -v "${env_name}" ]]; then
    local value="${!env_name}"
    if [[ -n "${value}" ]]; then
      CMD+=("${flag}" "${value}")
    fi
  fi
}

append_bool_if_set() {
  local env_name="${1}"
  local true_flag="${2:-}"
  local false_flag="${3:-}"
  if [[ ! -v "${env_name}" ]]; then
    return 0
  fi

  local normalized
  normalized="$(normalize_bool "${!env_name}")"
  case "${normalized}" in
    true|1|yes)
      if [[ -n "${true_flag}" ]]; then
        CMD+=("${true_flag}")
      fi
      ;;
    false|0|no)
      if [[ -n "${false_flag}" ]]; then
        CMD+=("${false_flag}")
      fi
      ;;
    *)
      echo "Invalid boolean value for ${env_name}: ${!env_name}" >&2
      exit 1
      ;;
  esac
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

CUDA_DEVICE_VALUE="${CUDA_DEVICE:-}"
if [[ $# -gt 0 && "${1}" != *=* && "${1}" != --* ]]; then
  CUDA_DEVICE_VALUE="${1}"
  shift
fi

CMD=(uv run python train_fastmri_single_coil.py)

if [[ -n "${CUDA_DEVICE_VALUE}" ]]; then
  CMD+=(--cuda-device "${CUDA_DEVICE_VALUE}")
fi

append_if_set TRAIN_ROOT --train-root
append_if_set VAL_ROOT --val-root
append_if_set TEST_ROOT --test-root

append_if_set MODEL_SIZE --model-size
append_if_set BASE_MODEL_CONFIG --base-model-config
append_if_set BASE_MODEL_CHECKPOINT --base-model-checkpoint

append_if_set MODE --mode
append_if_set TRAIN_PATCH_SIZE --train-patch-size
append_if_set CROP_SIZE --crop-size
append_if_set PATCH_OVERLAP --patch-overlap
append_if_set EVAL_PATCH_BATCH_SIZE --eval-patch-batch-size

append_if_set UNET_LR --unet-lr
append_if_set ADAPTER_LR --adapter-lr
append_if_set WEIGHT_DECAY --weight-decay
append_if_set MAX_EPOCHS --max-epochs
append_if_set WARMUP_EPOCHS --warmup-epochs
append_if_set BATCH_SIZE --batch-size
append_if_set NUM_WORKERS --num-workers
append_if_set ACC_FACTOR --acc-factor
append_if_set SCHEDULER_T_MAX --scheduler-t-max
append_if_set LOG_EVERY_N_STEPS --log-every-n-steps
append_if_set TRAIN_SAMPLE_RATE --train-sample-rate
append_if_set TRAIN_VOLUME_SAMPLE_RATE --train-volume-sample-rate
append_if_set SAMPLE_RATE --train-sample-rate
append_if_set VOLUME_SAMPLE_RATE --train-volume-sample-rate

if [[ -v EVAL_EVERY ]]; then
  CMD+=(--evaluate-every "${EVAL_EVERY}")
elif [[ -v EVALUATE_EVERY_N_EPOCHS ]]; then
  CMD+=(--evaluate-every "${EVALUATE_EVERY_N_EPOCHS}")
fi

append_if_set DEVICE --device
append_if_set SAVE_ROOT --save-root
append_if_set RUN_NAME --run-name
append_if_set PROJECT --project
append_if_set SEED --seed
append_if_set SAMPLE_SEED --sample-seed

append_bool_if_set USE_UNET --use-unet --no-use-unet
append_bool_if_set TRAIN_PRE_POST --train-pre-post --no-train-pre-post
append_bool_if_set GRADIENT_CHECKPOINT_FROZEN_BASE --gradient-checkpoint-frozen-base --no-gradient-checkpoint-frozen-base
append_bool_if_set USE_WANDB --wandb --no-wandb
append_bool_if_set DETERMINISTIC_MASK --deterministic-mask --no-deterministic-mask
append_bool_if_set PIN_MEMORY --pin-memory --no-pin-memory
append_bool_if_set PERSISTENT_WORKERS --persistent-workers --no-persistent-workers
append_bool_if_set SHUFFLE_TRAIN --shuffle-train --no-shuffle-train
append_bool_if_set DRY_RUN --dry-run

if [[ -v USE_BF16 ]]; then
  case "$(normalize_bool "${USE_BF16}")" in
    true|1|yes)
      CMD+=(--precision bf16)
      ;;
    false|0|no)
      CMD+=(--precision fp32)
      ;;
    *)
      echo "Invalid boolean value for USE_BF16: ${USE_BF16}" >&2
      exit 1
      ;;
  esac
fi

for override in "$@"; do
  CMD+=(--override "${override}")
done

echo "Primary FastMRI single-coil entrypoint: train_fastmri_single_coil.py" >&2
exec "${CMD[@]}"
