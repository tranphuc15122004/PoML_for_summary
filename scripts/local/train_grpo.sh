#!/bin/bash
# ==============================================================================
# NOTE: legacy local launcher with older defaults; canonical runs use the Qwen3-specific PBS scripts.
# PoML for Summary - GRPO Training Only (Local)
# ==============================================================================
# Usage:
#   ./train_grpo.sh
#   GPU_MODEL=v100 ./train_grpo.sh
#   RESUME_CHECKPOINT=models/grpo_checkpoints/checkpoint-100 ./train_grpo.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
SRC_PATH="${PROJECT_ROOT}/src"
GPU_MODEL="${GPU_MODEL:-auto}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Auto-detect GPU if not specified
if [ "${GPU_MODEL}" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "Detected GPU: ${GPU_NAME}"
        if [[ "${GPU_NAME}" == *"V100"* ]]; then
            GPU_MODEL="v100"
        elif [[ "${GPU_NAME}" == *"A100"* ]]; then
            GPU_MODEL="a100"
        elif [[ "${GPU_NAME}" == *"H100"* || "${GPU_NAME}" == *"H200"* ]]; then
            GPU_MODEL="h200"
        else
            GPU_MODEL="a100"
        fi
    else
        GPU_MODEL="a100"
        echo "No GPU detected, defaulting to A100 config"
    fi
fi

case "${GPU_MODEL}" in
    a100|h200)
        DTYPE="bf16"
        GRPO_BATCH=4
        ;;
    v100)
        DTYPE="fp16"
        GRPO_BATCH=2
        ;;
    *)
        DTYPE="bf16"
        GRPO_BATCH=4
        ;;
esac

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "=============================================================="
echo "GRPO Training"
echo "Project: ${PROJECT_ROOT}"
echo "GPU Model: ${GPU_MODEL}"
echo "Dtype: ${DTYPE}"
echo "Batch Size: ${GRPO_BATCH}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "Resume from: ${RESUME_CHECKPOINT}"
fi
echo "Start time: $(date)"
echo "=============================================================="

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME_ARG=""
if [ -n "${RESUME_CHECKPOINT}" ] && [ -d "${RESUME_CHECKPOINT}" ]; then
    RESUME_ARG="--resume ${RESUME_CHECKPOINT}"
fi

python src/SFT_GRPO/train_grpo.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --output_dir "models/grpo_checkpoints" \
    --lr 5e-7 \
    --num_generations 4 \
    --beta 0.04 \
    --total_steps 800 \
    --train_data "data/grpo_train.jsonl" \
    --val_data "data/grpo_val.jsonl" \
    --per_device_train_batch_size "${GRPO_BATCH}" \
    --bf16 "$([ "${DTYPE}" = "bf16" ] && echo True || echo False)" \
    --fp16 "$([ "${DTYPE}" = "fp16" ] && echo True || echo False)" \
    ${RESUME_ARG}

echo "=============================================================="
echo "GRPO complete. Model saved to models/grpo_checkpoints/"
echo "End time: $(date)"
echo "=============================================================="