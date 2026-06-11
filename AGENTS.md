# AGENTS.md — PoML for Summary

**Post-training LLMs for controllable Vietnamese text summarization** (Viettel AI).
Base model: `Qwen/Qwen2.5-3B-Instruct`. Pipeline: Data gen → SFT (LoRA) → GRPO → Eval.

## Setup

```bash
source activate.sh              # activates .venv + sets PYTHONPATH to src/
```

Python 3.12 venv at `.venv/`. All deps in `requirements.txt`. No extra build step.

## Source layout

```
docs/               # project documentation
  DATASETS.md
  pipeline_plan.md
  problem_statement.md
scripts/
  launch/           # convenience Python entry points
    sft.py, sft_aug.py, sft_no_aug.py, eval.py
  local/            # local shell scripts
  pbs/              # PBS/Torque job scripts
  slurm/            # Slurm job scripts
  tools/            # utility scripts (batch tests, config verification)
src/SFT_GRPO/       # training code
  train_sft.py      # Supervised Fine-Tuning
  train_grpo.py     # GRPO alignment (custom loop, not TRL's)
  rewards.py        # R_acc (ROUGE-L), R_len, R_style (LLM-as-Judge)
  evaluate.py       # Compare base / SFT / GRPO models
  config.py         # SFTConfig, GRPOConfig, EvalConfig, ModelConfig dataclasses
src/dataset/        # data pipeline
  dataset.py        # VietNews, WikiLingua, ViMs, VLSP loaders
  augmenter.py      # injects length/style instructions into raw data
  length_profiler.py
VDT_Textsum/        # raw data (gitignored)
data/               # generated JSONL splits (gitignored)
models/             # checkpoints (gitignored)
```

## Commands (in order)

```bash
# 1. Profile lengths & generate train/val/test JSONL
python src/dataset/length_profiler.py
python src/dataset/augmenter.py                   # outputs to data/

# 2. SFT
python scripts/launch/sft.py                      # stable defaults, ~20 GB VRAM (A100/H200)
python src/SFT_GRPO/train_sft.py --help           # CLI args override config defaults

# 3. GRPO (uses SFT output as starting point — loads same base model)
python src/SFT_GRPO/train_grpo.py                 # LoRA, 800 steps, bf16

# 4. Evaluate (base vs sft vs grpo)
python src/SFT_GRPO/evaluate.py
python src/SFT_GRPO/evaluate.py --models base,sft=models/sft_lora/final
```

## Per-GPU configuration

| GPU    | VRAM  | SFT dtype | Flash-attn | Batch (SFT) | Notes |
|--------|-------|-----------|-------------|-------------|-------|
| **H200** | 141 GB | bf16 | ✅ v2/v3 | 12–16 | Default config works as-is |
| **A100** | 40/80 GB | bf16 | ✅ v2 | 8–12 | Default config works as-is |
| **V100** | 16/32 GB | **fp16** | ❌ | 2–4 | Must override: `fp16=True, bf16=False, max_seq_length=2048` |

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set automatically in `train_sft.py` (helps all GPUs).
- `packing` requires flash-attention v2+ (A100/H200). Keep `False` on V100.
- Defaults are tuned for A100/H200. For V100, override in `scripts/launch/sft.py` or pass CLI flags.

## SFT metrics logging

Training produces persistent metric files under `{output_dir}/metrics/`:

- `train_metrics.csv` / `.jsonl` — step, loss, lr, epoch, grad_norm, GPU mem
- `eval_metrics.csv` / `.jsonl` — step, eval_loss
- `config.json` — training config snapshot

WandB logging via `--report_to wandb` still works independently.

## SFT gotchas

- `lora_alpha=r` (scaling=1) — critical for stability. Do not use α/r=2.
- `learning_rate=5e-5` — 2e-4 diverges with LoRA.
- `assistant_only_loss=True` — dataset must have `"messages"` column (chat format).
- `max_seq_length=3072` — covers 99.7% of samples. Lower to 2048 if OOM on V100.
- `warmup_ratio=0.1` — longer warmup prevents divergence.
- `report_to="none"` by default; set to `"wandb"` to log.

## GRPO gotchas

- `learning_rate=5e-7` — very low; GRPO is sensitive to large updates.
- Custom training loop (not `TRL.GRPOTrainer`). LoRA applied via `get_peft_model`.
- Reference model = frozen copy of policy base (same initial weights).
- Reward weights: accuracy 0.5, length 0.3, style 0.2 (`config.py:GRPOConfig`).
- Style reward uses LLM-as-Judge (reference model itself).
- Total steps: 800; validation every 2× `save_steps`.

## Data pipeline

- `augmenter.py` reads raw datasets from `VDT_Textsum/`, generates 5 JSONL splits under `data/`.
- SFT variants: 3× augmentation per raw sample (`num_variants=3`), random length/style.
- GRPO prompts: no assistant response — model generates K=4 completions during training.
- Meta fields (`length_requirement`, `style`) carried in every sample for reward functions.
- Length reward tolerances: "khoảng X từ" = ±20%, "trong khoảng lo-hi" = exact range, "không quá X" = max.

## Modifying configs

All configs are Python `dataclasses` in `src/SFT_GRPO/config.py`. Edit them directly or pass CLI args. `scripts/launch/sft.py` imports `SFTConfig` / `ModelConfig` directly — override by editing the file or using `--config path/to/json`.
