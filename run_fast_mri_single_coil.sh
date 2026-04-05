#!/usr/bin/env bash
set -euo pipefail

# FastMRI single-coil x8 fine-tuning launcher for SNRAware.
#
# Default behavior:
# - single-coil FastMRI bridge path
# - x8 acceleration
# - "warmup_then_both" mode
#   = train the g-factor U-Net first, then continue with LoRA + U-Net
# - short "fine-tune test" schedule by default
#
# Common usage:
#   ./run_fast_mri_single_coil.sh
#   ./run_fast_mri_single_coil.sh 1
#   CUDA_DEVICE=1 ./run_fast_mri_single_coil.sh
#   TRAIN_PATCH_SIZE=64 ./run_fast_mri_single_coil.sh
#   TRAIN_PATCH_SIZE=null ./run_fast_mri_single_coil.sh
#   TRAIN_ROOT=/data/fastmri/singlecoil_train \
#   VAL_ROOT=/data/fastmri/singlecoil_val \
#   TEST_ROOT=/data/fastmri/singlecoil_val \
#   ./run_fast_mri_single_coil.sh
#
# Extra Hydra overrides can be appended at the end:
#   ./run_fast_mri_single_coil.sh 0 fastmri_finetune.max_epochs=8 lora.r=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

show_help() {
  cat <<'EOF'
run_fast_mri_single_coil.sh

Environment variables you can change:
  CUDA_DEVICE            Physical GPU id to expose. Default: 0
  DEVICE                 Torch device passed to Hydra. Default: cuda:0

  TRAIN_ROOT             FastMRI single-coil train folder.
  VAL_ROOT               FastMRI single-coil val folder.
  TEST_ROOT              FastMRI single-coil test folder. Default: VAL_ROOT

  MODEL_SIZE             Base-model preset: small|large. Default: small
  BASE_MODEL_CONFIG      Explicit YAML override. Default: empty -> preset from MODEL_SIZE
  BASE_MODEL_CHECKPOINT  Explicit .pts override. Default: empty -> preset from MODEL_SIZE

  MODE                   Default: warmup_then_both
  ACC_FACTOR             Default: 8
  CROP_SIZE              Eval crop size. Accepts 320 or 320x320. Default: 320
  TRAIN_PATCH_SIZE       Train patch size. Accepts 64, 64x96, or null. Default: 64
  EVAL_PATCH_BATCH_SIZE  Sliding-window patch batch size for val/test. Default: 64
  PATCH_OVERLAP          Eval patch overlap. Accepts 16 or 16x24. Default: 16
  BATCH_SIZE             Default: 1
  NUM_WORKERS            Default: 4
  MAX_EPOCHS             Default: 4
  WARMUP_EPOCHS          Default: 2
  UNET_LR                Default: 1e-4
  ADAPTER_LR             Default: 1e-5
  WEIGHT_DECAY           Default: 0.0
  TRAIN_PRE_POST         Train SNRAware pre/post with adapters. Default: false
  GRADIENT_CHECKPOINT_FROZEN_BASE  Keep base-model checkpointing active in training. Default: true
  SAMPLE_RATE            Training-only sample rate. Default: 0.02
  VOLUME_SAMPLE_RATE     Training-only volume sample rate. Default: null
  RUN_NAME               Default: empty -> auto-generated
  SAVE_ROOT              Default: ./checkpoints/fine_tune
  USE_WANDB              Default: false
  PROJECT                Default: fastmri-snraware
  USE_BF16               Training autocast in bf16. Default: true
  DRY_RUN                Print resolved command and exit. Default: false
  SAMPLE_SEED            Default: 1234
  DETERMINISTIC_MASK     Default: true

Examples:
  CUDA_DEVICE=0 ./run_fast_mri_single_coil.sh
  MODEL_SIZE=large ./run_fast_mri_single_coil.sh
  TRAIN_PATCH_SIZE=64 ./run_fast_mri_single_coil.sh
  TRAIN_PATCH_SIZE=64x96 EVAL_PATCH_BATCH_SIZE=64 PATCH_OVERLAP=16x24 ./run_fast_mri_single_coil.sh
  TRAIN_PATCH_SIZE=null ./run_fast_mri_single_coil.sh
  USE_BF16=false ./run_fast_mri_single_coil.sh
  DRY_RUN=true ./run_fast_mri_single_coil.sh
  SAMPLE_RATE=null MAX_EPOCHS=20 WARMUP_EPOCHS=5 ./run_fast_mri_single_coil.sh
  ./run_fast_mri_single_coil.sh 1 lora.r=16 fastmri_finetune.batch_size=2
EOF
}

normalize_bool() {
  printf '%s' "${1}" | tr '[:upper:]' '[:lower:]'
}

parse_spatial_pair() {
  local raw="${1}"
  local label="${2}"
  local allow_null="${3:-false}"
  local allow_zero="${4:-false}"
  local normalized="${raw// /}"
  normalized="${normalized//[/}"
  normalized="${normalized//]/}"

  if [[ "$(printf '%s' "${normalized}" | tr '[:upper:]' '[:lower:]')" == "null" ]]; then
    if [[ "${allow_null}" == "true" ]]; then
      printf 'null'
      return 0
    fi
    echo "${label} cannot be null." >&2
    return 1
  fi

  local first=""
  local second=""
  if [[ "${normalized}" =~ ^([0-9]+)$ ]]; then
    first="${BASH_REMATCH[1]}"
    second="${BASH_REMATCH[1]}"
  elif [[ "${normalized}" =~ ^([0-9]+)[xX,]([0-9]+)$ ]]; then
    first="${BASH_REMATCH[1]}"
    second="${BASH_REMATCH[2]}"
  else
    echo "Invalid ${label}: ${raw}" >&2
    echo "Expected formats like 128, 128x160, or null (when supported)." >&2
    return 1
  fi

  if [[ "${allow_zero}" != "true" && ( "${first}" == "0" || "${second}" == "0" ) ]]; then
    echo "${label} must be positive, got ${raw}" >&2
    return 1
  fi

  printf '%s,%s' "${first}" "${second}"
}

pair_to_hydra_list() {
  local pair="${1}"
  if [[ "${pair}" == "null" ]]; then
    printf 'null'
    return 0
  fi

  local first
  local second
  IFS=, read -r first second <<<"${pair}"
  printf '[%s,%s]' "${first}" "${second}"
}

extract_native_patch_size() {
  local config_path="${1}"
  awk '
    $1 == "cutout_shape:" {in_cutout=1; next}
    in_cutout && $1 == "-" {
      values[count++] = $2
      if (count == 2) {
        printf "%sx%s", values[0], values[1]
        exit
      }
      next
    }
    in_cutout && $1 != "-" {in_cutout=0}
  ' "${config_path}" 2>/dev/null || true
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

CUDA_DEVICE_DEFAULT="${CUDA_DEVICE:-0}"
if [[ $# -gt 0 && "${1}" != *=* && "${1}" != --* ]]; then
  CUDA_DEVICE_DEFAULT="${1}"
  shift
fi

TRAIN_ROOT="${TRAIN_ROOT:-./data/fastmri/singlecoil_train}"
VAL_ROOT="${VAL_ROOT:-./data/fastmri/singlecoil_val}"
TEST_ROOT="${TEST_ROOT:-${VAL_ROOT}}"

MODEL_SIZE="${MODEL_SIZE:-small}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-}"
BASE_MODEL_CHECKPOINT="${BASE_MODEL_CHECKPOINT:-}"

MODE="${MODE:-warmup_then_both}"
ACC_FACTOR="${ACC_FACTOR:-8}"
CROP_SIZE="${CROP_SIZE:-320}"
TRAIN_PATCH_SIZE="${TRAIN_PATCH_SIZE:-64}"
EVAL_PATCH_BATCH_SIZE="${EVAL_PATCH_BATCH_SIZE:-64}"
PATCH_OVERLAP="${PATCH_OVERLAP:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
UNET_LR="${UNET_LR:-1e-4}"
ADAPTER_LR="${ADAPTER_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SAMPLE_RATE="${SAMPLE_RATE:-0.02}"
VOLUME_SAMPLE_RATE="${VOLUME_SAMPLE_RATE:-null}"
RUN_NAME="${RUN_NAME:-}"
SAVE_ROOT="${SAVE_ROOT:-./checkpoints/fine_tune}"
USE_WANDB="$(normalize_bool "${USE_WANDB:-false}")"
PROJECT="${PROJECT:-fastmri-snraware}"
USE_BF16="$(normalize_bool "${USE_BF16:-true}")"
SAMPLE_SEED="${SAMPLE_SEED:-1234}"
DETERMINISTIC_MASK="$(normalize_bool "${DETERMINISTIC_MASK:-true}")"
PIN_MEMORY="$(normalize_bool "${PIN_MEMORY:-true}")"
PERSISTENT_WORKERS="$(normalize_bool "${PERSISTENT_WORKERS:-false}")"
SHUFFLE_TRAIN="$(normalize_bool "${SHUFFLE_TRAIN:-true}")"
EVALUATE_EVERY_N_EPOCHS="${EVALUATE_EVERY_N_EPOCHS:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-20}"
SCHEDULER_T_MAX="${SCHEDULER_T_MAX:-0}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_PRE_POST="$(normalize_bool "${TRAIN_PRE_POST:-false}")"
GRADIENT_CHECKPOINT_FROZEN_BASE="$(normalize_bool "${GRADIENT_CHECKPOINT_FROZEN_BASE:-true}")"
DRY_RUN="$(normalize_bool "${DRY_RUN:-false}")"

case "${MODEL_SIZE}" in
  small|large) ;;
  *)
    echo "Unsupported MODEL_SIZE: ${MODEL_SIZE}" >&2
    echo "Expected MODEL_SIZE to be one of: small, large" >&2
    exit 1
    ;;
esac

case "${MODE}" in
  unet_only|unet_and_lora|lora_only|warmup_then_both) ;;
  *)
    echo "Unsupported MODE: ${MODE}" >&2
    echo "Expected MODE to be one of: unet_only, unet_and_lora, lora_only, warmup_then_both" >&2
    exit 1
    ;;
esac

RESOLVED_CROP_PAIR="$(parse_spatial_pair "${CROP_SIZE}" "CROP_SIZE" false false)"
RESOLVED_TRAIN_PATCH_PAIR="$(parse_spatial_pair "${TRAIN_PATCH_SIZE}" "TRAIN_PATCH_SIZE" true false)"
RESOLVED_PATCH_OVERLAP_PAIR="$(parse_spatial_pair "${PATCH_OVERLAP}" "PATCH_OVERLAP" false true)"

IFS=, read -r CROP_H CROP_W <<<"${RESOLVED_CROP_PAIR}"
IFS=, read -r OVERLAP_H OVERLAP_W <<<"${RESOLVED_PATCH_OVERLAP_PAIR}"

PATCH_INFERENCE_STATUS="disabled"
RESOLVED_TRAIN_PATCH_HYDRA="null"
TRAIN_INPUT_SIZE_LABEL="${CROP_H}x${CROP_W}"
PATCH_OVERLAP_LABEL="${OVERLAP_H}x${OVERLAP_W}"

if [[ "${RESOLVED_TRAIN_PATCH_PAIR}" != "null" ]]; then
  IFS=, read -r PATCH_H PATCH_W <<<"${RESOLVED_TRAIN_PATCH_PAIR}"
  if (( PATCH_H > CROP_H || PATCH_W > CROP_W )); then
    echo "TRAIN_PATCH_SIZE must fit inside CROP_SIZE, got TRAIN_PATCH_SIZE=${PATCH_H}x${PATCH_W} and CROP_SIZE=${CROP_H}x${CROP_W}" >&2
    exit 1
  fi
  if (( OVERLAP_H >= PATCH_H || OVERLAP_W >= PATCH_W )); then
    echo "PATCH_OVERLAP must be smaller than TRAIN_PATCH_SIZE, got PATCH_OVERLAP=${OVERLAP_H}x${OVERLAP_W} and TRAIN_PATCH_SIZE=${PATCH_H}x${PATCH_W}" >&2
    exit 1
  fi
  PATCH_INFERENCE_STATUS="enabled"
  RESOLVED_TRAIN_PATCH_HYDRA="$(pair_to_hydra_list "${RESOLVED_TRAIN_PATCH_PAIR}")"
  TRAIN_INPUT_SIZE_LABEL="${PATCH_H}x${PATCH_W}"
else
  PATCH_OVERLAP_LABEL="${OVERLAP_H}x${OVERLAP_W} (unused)"
fi

RESOLVED_CROP_HYDRA="$(pair_to_hydra_list "${RESOLVED_CROP_PAIR}")"
RESOLVED_PATCH_OVERLAP_HYDRA="[${OVERLAP_H},${OVERLAP_W},0]"

if ! command -v uv >/dev/null 2>&1; then
  echo "Could not find uv on PATH." >&2
  echo "Please install uv and retry." >&2
  exit 1
fi

if [[ -n "${BASE_MODEL_CONFIG}" || -n "${BASE_MODEL_CHECKPOINT}" ]]; then
  if [[ -z "${BASE_MODEL_CONFIG}" || -z "${BASE_MODEL_CHECKPOINT}" ]]; then
    echo "BASE_MODEL_CONFIG and BASE_MODEL_CHECKPOINT must either both be set or both be unset." >&2
    exit 1
  fi
  RESOLVED_BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG}"
  RESOLVED_BASE_MODEL_CHECKPOINT="${BASE_MODEL_CHECKPOINT}"
  USE_BASE_MODEL_PRESET=false
else
  case "${MODEL_SIZE}" in
    small)
      RESOLVED_BASE_MODEL_CONFIG="./checkpoints/small/snraware_small_model.yaml"
      RESOLVED_BASE_MODEL_CHECKPOINT="./checkpoints/small/snraware_small_model.pts"
      ;;
    large)
      RESOLVED_BASE_MODEL_CONFIG="./checkpoints/large/snraware_large_model.yaml"
      RESOLVED_BASE_MODEL_CHECKPOINT="./checkpoints/large/snraware_large_model.pts"
      ;;
  esac
  USE_BASE_MODEL_PRESET=true
fi

if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "TRAIN_ROOT does not exist: ${TRAIN_ROOT}" >&2
  echo "Set TRAIN_ROOT=/path/to/singlecoil_train and retry." >&2
  exit 1
fi

if [[ ! -d "${VAL_ROOT}" ]]; then
  echo "VAL_ROOT does not exist: ${VAL_ROOT}" >&2
  echo "Set VAL_ROOT=/path/to/singlecoil_val and retry." >&2
  exit 1
fi

if [[ ! -d "${TEST_ROOT}" ]]; then
  echo "TEST_ROOT does not exist: ${TEST_ROOT}" >&2
  echo "Set TEST_ROOT=/path/to/singlecoil_test (or reuse VAL_ROOT) and retry." >&2
  exit 1
fi

if [[ ! -f "${RESOLVED_BASE_MODEL_CONFIG}" ]]; then
  echo "Base model config not found: ${RESOLVED_BASE_MODEL_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${RESOLVED_BASE_MODEL_CHECKPOINT}" ]]; then
  echo "Base model checkpoint not found: ${RESOLVED_BASE_MODEL_CHECKPOINT}" >&2
  exit 1
fi

CHECKPOINT_NATIVE_SPATIAL_SIZE="$(extract_native_patch_size "${RESOLVED_BASE_MODEL_CONFIG}")"
if [[ -z "${CHECKPOINT_NATIVE_SPATIAL_SIZE}" ]]; then
  CHECKPOINT_NATIVE_SPATIAL_SIZE="unknown"
fi

if [[ "${DEVICE}" != cpu* ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE_DEFAULT}}"
fi

echo "Running FastMRI single-coil fine-tune with:"
echo "  UV_BIN=uv"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  DEVICE=${DEVICE}"
echo "  MODEL_SIZE=${MODEL_SIZE}"
echo "  TRAIN_ROOT=${TRAIN_ROOT}"
echo "  VAL_ROOT=${VAL_ROOT}"
echo "  TEST_ROOT=${TEST_ROOT}"
echo "  BASE_MODEL_CONFIG=${RESOLVED_BASE_MODEL_CONFIG}"
echo "  BASE_MODEL_CHECKPOINT=${RESOLVED_BASE_MODEL_CHECKPOINT}"
echo "  CHECKPOINT_NATIVE_SPATIAL_SIZE=${CHECKPOINT_NATIVE_SPATIAL_SIZE}"
echo "  MODE=${MODE}"
echo "  ACC_FACTOR=${ACC_FACTOR}"
echo "  TRAIN_INPUT_SIZE=${TRAIN_INPUT_SIZE_LABEL}"
echo "  EVAL_CROP_SIZE=${CROP_H}x${CROP_W}"
echo "  TRAIN_PATCH_SIZE=${RESOLVED_TRAIN_PATCH_PAIR}"
echo "  EVAL_PATCH_BATCH_SIZE=${EVAL_PATCH_BATCH_SIZE}"
echo "  PATCH_OVERLAP=${PATCH_OVERLAP_LABEL}"
echo "  PATCH_INFERENCE_FOR_VAL_TEST=${PATCH_INFERENCE_STATUS}"
echo "  MAX_EPOCHS=${MAX_EPOCHS}"
echo "  WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "  USE_BF16=${USE_BF16}"
echo "  TRAIN_PRE_POST=${TRAIN_PRE_POST}"
echo "  GRADIENT_CHECKPOINT_FROZEN_BASE=${GRADIENT_CHECKPOINT_FROZEN_BASE}"
echo "  TRAIN_SAMPLE_RATE=${SAMPLE_RATE}"
echo "  SAVE_ROOT=${SAVE_ROOT}"
echo "  DRY_RUN=${DRY_RUN}"
if [[ "${CHECKPOINT_NATIVE_SPATIAL_SIZE}" != "unknown" ]]; then
  echo "  NOTE=Native pretrained spatial size is ${CHECKPOINT_NATIVE_SPATIAL_SIZE}; larger TRAIN_PATCH_SIZE values may cause partial pretrained loading."
fi

CMD=(
  uv run python -m snraware.projects.mri.denoising.train
  "logging.use_wandb=${USE_WANDB}"
  "logging.project=${PROJECT}"
  "lora.enabled=true"
  "fastmri_finetune.train_root=${TRAIN_ROOT}"
  "fastmri_finetune.val_root=${VAL_ROOT}"
  "fastmri_finetune.test_root=${TEST_ROOT}"
  "fastmri_finetune.challenge=singlecoil"
  "fastmri_finetune.acc_factor=${ACC_FACTOR}"
  "fastmri_finetune.crop_size=${RESOLVED_CROP_HYDRA}"
  "fastmri_finetune.train_patch_size=${RESOLVED_TRAIN_PATCH_HYDRA}"
  "fastmri_finetune.eval_patch_batch_size=${EVAL_PATCH_BATCH_SIZE}"
  "fastmri_finetune.mode=${MODE}"
  "fastmri_finetune.batch_size=${BATCH_SIZE}"
  "fastmri_finetune.num_workers=${NUM_WORKERS}"
  "fastmri_finetune.pin_memory=${PIN_MEMORY}"
  "fastmri_finetune.persistent_workers=${PERSISTENT_WORKERS}"
  "fastmri_finetune.shuffle_train=${SHUFFLE_TRAIN}"
  "fastmri_finetune.max_epochs=${MAX_EPOCHS}"
  "fastmri_finetune.warmup_epochs=${WARMUP_EPOCHS}"
  "fastmri_finetune.unet_lr=${UNET_LR}"
  "fastmri_finetune.adapter_lr=${ADAPTER_LR}"
  "fastmri_finetune.weight_decay=${WEIGHT_DECAY}"
  "fastmri_finetune.train_pre_post=${TRAIN_PRE_POST}"
  "fastmri_finetune.gradient_checkpoint_frozen_base=${GRADIENT_CHECKPOINT_FROZEN_BASE}"
  "fastmri_finetune.train_sample_rate=${SAMPLE_RATE}"
  "fastmri_finetune.train_volume_sample_rate=${VOLUME_SAMPLE_RATE}"
  "fastmri_finetune.sample_seed=${SAMPLE_SEED}"
  "fastmri_finetune.deterministic_mask_from_name=${DETERMINISTIC_MASK}"
  "fastmri_finetune.evaluate_every_n_epochs=${EVALUATE_EVERY_N_EPOCHS}"
  "fastmri_finetune.log_every_n_steps=${LOG_EVERY_N_STEPS}"
  "fastmri_finetune.scheduler_t_max=${SCHEDULER_T_MAX}"
  "fastmri_finetune.save_root=${SAVE_ROOT}"
  "fastmri_finetune.device=${DEVICE}"
  "fastmri_finetune.use_bf16=${USE_BF16}"
  "overlap_for_inference=${RESOLVED_PATCH_OVERLAP_HYDRA}"
)

if [[ "${USE_BASE_MODEL_PRESET}" == true ]]; then
  CMD+=("base_model.variant=${MODEL_SIZE}")
else
  CMD+=("base_model.config_path=${RESOLVED_BASE_MODEL_CONFIG}")
  CMD+=("base_model.checkpoint_path=${RESOLVED_BASE_MODEL_CHECKPOINT}")
fi

if [[ -n "${RUN_NAME}" ]]; then
  CMD+=("fastmri_finetune.run_name=${RUN_NAME}")
fi

if [[ "${MODE}" == "unet_only" ]]; then
  CMD+=("lora.enabled=false")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Command:\n  '
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

exec "${CMD[@]}"
