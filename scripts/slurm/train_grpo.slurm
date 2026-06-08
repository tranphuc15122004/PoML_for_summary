#!/bin/bash
#SBATCH --job-name=poml_grpo_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jp09/dd9648/PoML_for_summary/logs/grpo_%j.out
#SBATCH --error=/scratch/jp09/dd9648/PoML_for_summary/logs/grpo_%j.err

# ==============================================================================
# PoML for Summary - GRPO Training Only (Slurm)
# ==============================================================================
# Usage:
#   sbatch train_grpo.slurm
#   sbatch --export=GPU_MODEL=v100 train_grpo.slurm
#   sbatch --export=RESUME_CHECKPOINT=models/grpo_checkpoints/checkpoint-100 train_grpo.slurm
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/jp09/dd9648/PoML_for_summary"
VENV_PATH="${PROJECT_ROOT}/.venv"
SRC_PATH="${PROJECT_ROOT}/src"
LOGS_DIR="${PROJECT_ROOT}/logs"
GPU_MODEL="${GPU_MODEL:-a100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

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

mkdir -p "${LOGS_DIR}"
cd "${PROJECT_ROOT}"

echo "=============================================================="
echo "GRPO Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
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