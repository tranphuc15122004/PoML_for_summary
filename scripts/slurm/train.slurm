#!/bin/bash
#SBATCH --job-name=poml_summary_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/jp09/dd9648/PoML_for_summary/logs/train_%j.out
#SBATCH --error=/scratch/jp09/dd9648/PoML_for_summary/logs/train_%j.err

# ==============================================================================
# PoML for Summary - Full Training Pipeline Slurm Script
# ==============================================================================
# Usage:
#   sbatch train.slurm                    # Run full pipeline
#   sbatch --export=STAGE=sft train.slurm # Run only SFT
#   sbatch --export=STAGE=grpo train.slurm # Run only GRPO
#   sbatch --export=STAGE=data train.slurm # Run only data preparation
#   sbatch --export=STAGE=eval train.slurm # Run only evaluation
# ==============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT="/scratch/jp09/dd9648/PoML_for_summary"
VENV_PATH="${PROJECT_ROOT}/.venv"
SRC_PATH="${PROJECT_ROOT}/src"
LOGS_DIR="${PROJECT_ROOT}/logs"
STAGE="${STAGE:-full}"  # full | data | sft | grpo | eval

# GPU-specific settings
GPU_MODEL="${GPU_MODEL:-a100}"  # a100 | h200 | v100
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

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
mkdir -p "${LOGS_DIR}"
cd "${PROJECT_ROOT}"

echo "=============================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Stage: ${STAGE}"
echo "GPU Model: ${GPU_MODEL}"
echo "Dtype: ${DTYPE}"
echo "Max Seq Length: ${MAX_SEQ_LEN}"
echo "Start time: $(date)"
echo "=============================================================="

# Activate venv
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH}"

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
run_data_prep() {
    echo ">>> [1/4] Data Preparation"
    echo "Profiling lengths..."
    python src/dataset/length_profiler.py
    
    echo "Augmenting data..."
    python src/dataset/augmenter.py
    
    echo "Data preparation complete. Outputs in data/"
    ls -la data/
}

run_sft() {
    echo ">>> [2/4] SFT Training"
    
    python scripts/launch/sft.py \
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

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
case "${STAGE}" in
    data)
        run_data_prep
        ;;
    sft)
        run_sft
        ;;
    grpo)
        run_grpo
        ;;
    eval)
        run_eval
        ;;
    full)
        run_data_prep
        run_sft
        run_grpo
        run_eval
        ;;
    *)
        echo "ERROR: Unknown STAGE='${STAGE}'. Use: data | sft | grpo | eval | full"
        exit 1
        ;;
esac

echo "=============================================================="
echo "End time: $(date)"
echo "Job completed successfully!"
echo "=============================================================="