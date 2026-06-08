# Training Scripts

## Directory Structure

```
scripts/
├── pbs/        # PBS/Torque job scripts
├── slurm/      # Slurm job scripts
└── local/      # Local execution shell scripts
```

## Scripts Overview

### Full Pipeline (Data → SFT → GRPO → Eval)

| Scheduler | Script | Usage |
|-----------|--------|-------|
| PBS | `pbs/train.pbs` | `qsub pbs/train.pbs` |
| Slurm | `slurm/train.slurm` | `sbatch slurm/train.slurm` |
| Local | `local/train.sh` | `./local/train.sh` |

**Stage control:** `STAGE=data|sft|grpo|eval|full` (default: `full`)

```bash
# PBS
qsub -v STAGE=sft pbs/train.pbs

# Slurm
sbatch --export=STAGE=grpo slurm/train.slurm

# Local
STAGE=sft ./local/train.sh
```

### SFT Only

| Scheduler | Script | Usage |
|-----------|--------|-------|
| PBS | `pbs/train_sft.pbs` | `qsub pbs/train_sft.pbs` |
| Slurm | `slurm/train_sft.slurm` | `sbatch slurm/train_sft.slurm` |
| Local | `local/train_sft.sh` | `./local/train_sft.sh` |

**GPU override:** `GPU_MODEL=a100|h200|v100`

```bash
# PBS
qsub -v GPU_MODEL=v100 pbs/train_sft.pbs

# Slurm
sbatch --export=GPU_MODEL=v100 slurm/train_sft.slurm

# Local
GPU_MODEL=v100 ./local/train_sft.sh
```

### GRPO Only

| Scheduler | Script | Usage |
|-----------|--------|-------|
| PBS | `pbs/train_grpo.pbs` | `qsub pbs/train_grpo.pbs` |
| Slurm | `slurm/train_grpo.slurm` | `sbatch slurm/train_grpo.slurm` |
| Local | `local/train_grpo.sh` | `./local/train_grpo.sh` |

**Options:**
- `GPU_MODEL=a100|h200|v100`
- `RESUME_CHECKPOINT=path/to/checkpoint`

```bash
# Resume from checkpoint
qsub -v RESUME_CHECKPOINT=models/grpo_checkpoints/checkpoint-100 pbs/train_grpo.pbs
sbatch --export=RESUME_CHECKPOINT=models/grpo_checkpoints/checkpoint-100 slurm/train_grpo.slurm
RESUME_CHECKPOINT=models/grpo_checkpoints/checkpoint-100 ./local/train_grpo.sh
```

## GPU Auto-Detection (Local Scripts)

Local scripts (`local/*.sh`) auto-detect GPU:
- A100/H200 → `bf16`, `max_seq_length=3072`, batch=4
- V100 → `fp16`, `max_seq_length=2048`, batch=2

Override with `GPU_MODEL` if needed.

## Prerequisites

1. **Data preparation** (run once before SFT/GRPO):
   ```bash
   python src/dataset/length_profiler.py
   python src/dataset/augmenter.py
   ```
   Generates: `data/sft_train.jsonl`, `data/sft_val.jsonl`, `data/grpo_train.jsonl`, `data/grpo_val.jsonl`, `data/test.jsonl`

2. **Environment**: Activate venv first
   ```bash
   source activate.sh
   ```

## Output Paths

- **Logs**: `logs/` (created automatically)
- **SFT checkpoints**: `models/sft_lora/`
- **GRPO checkpoints**: `models/grpo_checkpoints/`
- **Evaluation results**: `models/eval_results/`