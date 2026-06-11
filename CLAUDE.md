# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Post-training pipeline for Vietnamese text summarization with constraint control. The project fine-tunes small LLMs (default: `Qwen/Qwen2.5-3B-Instruct`) to follow Vietnamese summarization instructions with length control, style/persona control, and structured output control. Training follows a two-stage pipeline: SFT → GRPO alignment.

**Research context:** Viettel AI R&D project. Datasets live in `VDT_Textsum/` (not committed). Models output to `models/`.

## Environment Setup

```bash
source activate.sh            # activate venv + set PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH   # if running manually
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation (run once before training)

```bash
python src/dataset/augmenter.py   # generates data/*.jsonl splits
```

Outputs: `data/sft_train.jsonl`, `data/sft_val.jsonl`, `data/grpo_train.jsonl`, `data/grpo_val.jsonl`, `data/test.jsonl`

Raw datasets expected at `VDT_Textsum/{VietNews,WikiLingua,VLSP,ViMs}/`.

## Training Commands

**SFT (Supervised Fine-Tuning):**
```bash
# Local (auto-detects GPU)
./scripts/local/train_sft.sh

# Manual
PYTHONPATH=src python scripts/launch/sft.py

# CLI with overrides
PYTHONPATH=src python src/SFT_GRPO/train_sft.py --model_name Qwen/Qwen2.5-3B-Instruct --output_dir models/sft_lora --epochs 1.0

# Resume from checkpoint
PYTHONPATH=src python src/SFT_GRPO/train_sft.py --resume models/sft_lora/checkpoint-500
```

**GRPO (Alignment Training):**
```bash
./scripts/local/train_grpo.sh

PYTHONPATH=src python src/SFT_GRPO/train_grpo.py --model_name models/sft_lora/final --output_dir models/grpo_checkpoints
```

**Full pipeline (data → SFT → GRPO → eval):**
```bash
STAGE=full ./scripts/local/train.sh
# or individual stages: STAGE=data|sft|grpo|eval
```

**HPC cluster:**
```bash
qsub scripts/pbs/train_sft.pbs          # PBS
sbatch scripts/slurm/train_sft.slurm   # Slurm
```

**GPU configs:** A100/H200 → `bf16`, seq_len=3072, batch=4. V100 → `fp16`, seq_len=2048, batch=2. Override: `GPU_MODEL=v100`.

## Code Architecture

```
docs/               # project documentation
  DATASETS.md
  pipeline_plan.md
  problem_statement.md
scripts/
  launch/           # convenience Python entry points
    sft.py, sft_aug.py, sft_no_aug.py, eval.py
  local/            # shell scripts
  pbs/              # PBS/Torque job scripts
  slurm/            # Slurm job scripts
  tools/            # utility scripts (batch tests, config verification)
src/
├── dataset/
│   ├── dataset.py          # BaseSummarizationDataset ABC + 4 concrete datasets
│   │                         (VietNewsDataset, WikiLinguaDataset, VLSPDataset, ViMsDataset)
│   ├── augmenter.py        # PromptAugmenter: converts raw {source, target} pairs
│   │                         into chat-format SFT samples and GRPO prompt dicts.
│   │                         build_all_splits() builds all 5 data splits at once.
│   └── length_profiler.py  # Word count distribution analysis for hyperparameter tuning
└── SFT_GRPO/
    ├── config.py           # Dataclasses: ModelConfig, SFTConfig, GRPOConfig, EvalConfig
    ├── train_sft.py        # SFTTrainer wrapper (TRL SFTTrainer + LoRA/QLoRA)
    ├── train_grpo.py       # Custom GRPOTrainer: rollout → reward → advantage → policy gradient
    ├── rewards.py          # Three reward functions: R_acc (ROUGE-L), R_len (word count adherence),
    │                         R_style (LLM-as-Judge 1–5 scale). Composite: w_acc·R_acc + w_len·R_len + w_style·R_style
    ├── metrics_logger.py   # MetricsTracker + MetricsCallback (CSV logging to output_dir/metrics/)
    └── evaluate.py         # Evaluation pipeline
```
```

### Key design decisions

- **Data format:** SFT data uses `{"messages": [...], "meta": {...}}` JSONL. GRPO data uses `{"prompt": [...], "reference": "...", "meta": {...}}` JSONL. The `meta` field carries `length_requirement` and `style` for reward computation.
- **Augmentation strategy:** Each raw sample generates 3 SFT variants with different length templates (`khoảng X từ`, `trong khoảng lo-hi từ`, `không quá X từ`) and randomly sampled styles. GRPO uses harder/more diverse styles than SFT.
- **GRPO implementation:** Custom training loop (not TRL's GRPOTrainer). Maintains a frozen reference model alongside the trainable policy. Loss = clipped policy gradient + β·KL.
- **LoRA config:** Default rank=32, alpha=32 (scaling=1), targeting all attention + MLP projections.
- **Config override:** Both `train_sft.py` and `train_grpo.py` accept `--config path/to.json` for full config override, or individual CLI flags for common hyperparameters.

## Output Paths

- SFT checkpoints: `models/sft_lora/`
- GRPO checkpoints: `models/grpo_checkpoints/`
- Training metrics (CSV): `models/{sft_lora,grpo_checkpoints}/metrics/`
- Eval results: `models/eval_results/`

---

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.