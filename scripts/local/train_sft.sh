#!/bin/bash
# ==============================================================================
# PoML for Summary - SFT Training Only (Local)
# ==============================================================================
# Usage:
#   ./train_sft.sh
#   GPU_MODEL=v100 ./train_sft.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
SRC_PATH="${PROJECT_ROOT}/src"
GPU_MODEL="${GPU_MODEL:-auto}"

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
        MAX_SEQ_LEN=3072
        PACKING="False"
        SFT_BATCH=4
        ;;
    v100)
        DTYPE="fp16"
        MAX_SEQ_LEN=2048
        PACKING="False"
        SFT_BATCH=2
        ;;
    *)
        DTYPE="bf16"
        MAX_SEQ_LEN=3072
        PACKING="False"
        SFT_BATCH=4
        ;;
esac

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "=============================================================="
echo "SFT Training"
echo "Project: ${PROJECT_ROOT}"
echo "GPU Model: ${GPU_MODEL}"
echo "Dtype: ${DTYPE}"
echo "Max Seq Length: ${MAX_SEQ_LEN}"
echo "Batch Size: ${SFT_BATCH}"
echo "Start time: $(date)"
echo "=============================================================="

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/launch/sft.py \
    --max_seq_length "${MAX_SEQ_LEN}" \
    --packing "${PACKING}" \
    --per_device_train_batch_size "${SFT_BATCH}" \
    --bf16 "$([ "${DTYPE}" = "bf16" ] && echo True || echo False)" \
    --fp16 "$([ "${DTYPE}" = "fp16" ] && echo True || echo False)" \
    --output_dir "models/sft_lora"

echo "=============================================================="
echo "SFT complete. Model saved to models/sft_lora/"
echo "End time: $(date)"
echo "=============================================================="