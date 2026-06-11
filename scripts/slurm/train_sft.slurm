#!/bin/bash
#SBATCH --job-name=poml_sft_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jp09/dd9648/PoML_for_summary/logs/sft_%j.out
#SBATCH --error=/scratch/jp09/dd9648/PoML_for_summary/logs/sft_%j.err

# ==============================================================================
# PoML for Summary - SFT Training Only (Slurm)
# ==============================================================================
# Usage:
#   sbatch train_sft.slurm
#   sbatch --export=GPU_MODEL=v100 train_sft.slurm
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/jp09/dd9648/PoML_for_summary"
VENV_PATH="${PROJECT_ROOT}/.venv"
SRC_PATH="${PROJECT_ROOT}/src"
LOGS_DIR="${PROJECT_ROOT}/logs"
GPU_MODEL="${GPU_MODEL:-a100}"

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

mkdir -p "${LOGS_DIR}"
cd "${PROJECT_ROOT}"

echo "=============================================================="
echo "SFT Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
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