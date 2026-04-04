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
  UV_BIN                 uv executable. Default: uv

  TRAIN_ROOT             FastMRI single-coil train folder.
  VAL_ROOT               FastMRI single-coil val folder.
  TEST_ROOT              FastMRI single-coil test folder. Default: VAL_ROOT

  BASE_MODEL_CONFIG      Default: ./checkpoints/snraware_large_model.yaml
  BASE_MODEL_CHECKPOINT  Default: ./checkpoints/snraware_large_model.pts

  MODE                   Default: warmup_then_both
  ACC_FACTOR             Default: 8
  BATCH_SIZE             Default: 1
  NUM_WORKERS            Default: 4
  MAX_EPOCHS             Default: 4
  WARMUP_EPOCHS          Default: 2
  UNET_LR                Default: 1e-4
  ADAPTER_LR             Default: 1e-5
  WEIGHT_DECAY           Default: 0.0
  SAMPLE_RATE            Training-only sample rate. Default: 0.02
  VOLUME_SAMPLE_RATE     Training-only volume sample rate. Default: null
  RUN_NAME               Default: empty -> auto-generated
  SAVE_ROOT              Default: ./checkpoints/fine_tune
  USE_WANDB              Default: false
  PROJECT                Default: fastmri-snraware
  USE_BF16               Training autocast in bf16. Default: true
  SAMPLE_SEED            Default: 1234
  DETERMINISTIC_MASK     Default: true

Examples:
  CUDA_DEVICE=0 ./run_fast_mri_single_coil.sh
  USE_BF16=false ./run_fast_mri_single_coil.sh
  SAMPLE_RATE=null MAX_EPOCHS=20 WARMUP_EPOCHS=5 ./run_fast_mri_single_coil.sh
  ./run_fast_mri_single_coil.sh 1 lora.r=16 fastmri_finetune.batch_size=2
EOF
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

UV_BIN="${UV_BIN:-uv}"
if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "Could not find uv executable: ${UV_BIN}" >&2
  echo "Install uv or set UV_BIN=/full/path/to/uv and retry." >&2
  exit 1
fi

TRAIN_ROOT="${TRAIN_ROOT:-./data/fastmri/singlecoil_train}"
VAL_ROOT="${VAL_ROOT:-./data/fastmri/singlecoil_val}"
TEST_ROOT="${TEST_ROOT:-${VAL_ROOT}}"

BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-./checkpoints/snraware_large_model.yaml}"
BASE_MODEL_CHECKPOINT="${BASE_MODEL_CHECKPOINT:-./checkpoints/snraware_large_model.pts}"

MODE="${MODE:-warmup_then_both}"
ACC_FACTOR="${ACC_FACTOR:-8}"
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
USE_WANDB="${USE_WANDB:-false}"
PROJECT="${PROJECT:-fastmri-snraware}"
USE_BF16="${USE_BF16:-true}"
SAMPLE_SEED="${SAMPLE_SEED:-1234}"
DETERMINISTIC_MASK="${DETERMINISTIC_MASK:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-true}"
EVALUATE_EVERY_N_EPOCHS="${EVALUATE_EVERY_N_EPOCHS:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-20}"
SCHEDULER_T_MAX="${SCHEDULER_T_MAX:-0}"
DEVICE="${DEVICE:-cuda:0}"

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

if [[ ! -f "${BASE_MODEL_CONFIG}" ]]; then
  echo "Base model config not found: ${BASE_MODEL_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${BASE_MODEL_CHECKPOINT}" ]]; then
  echo "Base model checkpoint not found: ${BASE_MODEL_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${DEVICE}" != cpu* ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE_DEFAULT}}"
fi

echo "Running FastMRI single-coil fine-tune with:"
echo "  UV_BIN=${UV_BIN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  DEVICE=${DEVICE}"
echo "  TRAIN_ROOT=${TRAIN_ROOT}"
echo "  VAL_ROOT=${VAL_ROOT}"
echo "  TEST_ROOT=${TEST_ROOT}"
echo "  MODE=${MODE}"
echo "  ACC_FACTOR=${ACC_FACTOR}"
echo "  MAX_EPOCHS=${MAX_EPOCHS}"
echo "  WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "  USE_BF16=${USE_BF16}"
echo "  TRAIN_SAMPLE_RATE=${SAMPLE_RATE}"
echo "  SAVE_ROOT=${SAVE_ROOT}"

CMD=(
  "${UV_BIN}" run python -m snraware.projects.mri.denoising.train
  "base_model.config_path=${BASE_MODEL_CONFIG}"
  "base_model.checkpoint_path=${BASE_MODEL_CHECKPOINT}"
  "logging.use_wandb=${USE_WANDB}"
  "logging.project=${PROJECT}"
  "lora.enabled=true"
  "fastmri_finetune.train_root=${TRAIN_ROOT}"
  "fastmri_finetune.val_root=${VAL_ROOT}"
  "fastmri_finetune.test_root=${TEST_ROOT}"
  "fastmri_finetune.challenge=singlecoil"
  "fastmri_finetune.acc_factor=${ACC_FACTOR}"
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
)

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

exec "${CMD[@]}"
