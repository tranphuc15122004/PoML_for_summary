#!/bin/bash
# ==============================================================================
# PoML for Summary - Local Training Script (no scheduler)
# ==============================================================================
# Usage:
#   ./train.sh              # Run full pipeline
#   STAGE=sft ./train.sh    # Run only SFT
#   STAGE=grpo ./train.sh   # Run only GRPO
#   STAGE=data ./train.sh   # Run only data preparation
#   STAGE=eval ./train.sh   # Run only evaluation
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
SRC_PATH="${PROJECT_ROOT}/src"
STAGE="${STAGE:-full}"

# Auto-detect GPU
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

case "${GPU_MODEL}" in
    a100|h200)
        DTYPE="bf16"
        MAX_SEQ_LEN=3072
        PACKING="False"
        SFT_BATCH=4
        GRPO_BATCH=4
        ;;
    v100)
        DTYPE="fp16"
        MAX_SEQ_LEN=2048
        PACKING="False"
        SFT_BATCH=2
        GRPO_BATCH=2
        ;;
    *)
        DTYPE="bf16"
        MAX_SEQ_LEN=3072
        PACKING="False"
        SFT_BATCH=4
        GRPO_BATCH=4
        ;;
esac

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "=============================================================="
echo "Project: ${PROJECT_ROOT}"
echo "Stage: ${STAGE}"
echo "GPU Model: ${GPU_MODEL}"
echo "Dtype: ${DTYPE}"
echo "Max Seq Length: ${MAX_SEQ_LEN}"
echo "Start time: $(date)"
echo "=============================================================="

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_data_prep() {
    echo ">>> [1/4] Data Preparation"
    python src/dataset/length_profiler.py
    python src/dataset/augmenter.py
    echo "Data preparation complete. Outputs in data/"
    ls -la data/
}

run_sft() {
    echo ">>> [2/4] SFT Training"
    python launch_sft.py \
        --max_seq_length "${MAX_SEQ_LEN}" \
        --packing "${PACKING}" \
        --per_device_train_batch_size "${SFT_BATCH}" \
        --bf16 "$([ "${DTYPE}" = "bf16" ] && echo True || echo False)" \
        --fp16 "$([ "${DTYPE}" = "fp16" ] && echo True || echo False)" \
        --output_dir "models/sft_lora"
    echo "SFT complete. Model saved to models/sft_lora/"
}

run_grpo() {
    echo ">>> [3/4] GRPO Training"
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
        --fp16 "$([ "${DTYPE}" = "fp16" ] && echo True || echo False)"
    echo "GRPO complete. Model saved to models/grpo_checkpoints/"
}

run_eval() {
    echo ">>> [4/4] Evaluation"
    python src/SFT_GRPO/evaluate.py \
        --models base,sft=models/sft_lora/final,grpo=models/grpo_checkpoints/final \
        --test_data data/test.jsonl \
        --output_dir models/eval_results
    echo "Evaluation complete. Results in models/eval_results/"
}

case "${STAGE}" in
    data) run_data_prep ;;
    sft) run_sft ;;
    grpo) run_grpo ;;
    eval) run_eval ;;
    full)
        run_data_prep
        run_sft
        run_grpo
        run_eval
        ;;
    *) echo "Usage: STAGE={data|sft|grpo|eval|full} ./train.sh"; exit 1 ;;
esac

echo "=============================================================="
echo "End time: $(date)"
echo "Done!"
echo "=============================================================="