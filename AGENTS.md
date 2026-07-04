# AGENTS.md — PoML for Summary

**Post-training LLMs for controllable Vietnamese text summarization** (Viettel AI).
Base models: `Qwen3-4B-Base` (pretrained) and `Qwen3-4B` (instruction-tuned).
Pipeline: Data gen → SFT (LoRA) → GRPO alignment → Eval.

## Setup

```bash
source activate.sh              # activates .venv + sets PYTHONPATH to src/
```

Python 3.12 venv at `.venv/`. All deps in `requirements.txt`. No extra build step.

## Source layout

```
docs/                       # project documentation (see key docs linked below)
scripts/
  launch/                   # Python entry points
    eval.py                 #   eval launcher with Qwen3 model family discovery
  local/                    # shell scripts
  pbs/                      # PBS/Torque job scripts
    train_sft_qwen3_4b.pbs  #   SFT for Base/Instruct
    train_grpo_qwen3_4b.pbs #   GRPO for Base/Instruct (fresh / SFT-init / auto-resume)
    eval.pbs                #   eval all Qwen3 variants
    ablate_no_sent_qwen3_*.pbs  # sentence-reward ablation experiments
    smoke_test_qwen3_4b.pbs #   end-to-end smoke test
  slurm/                    # Slurm job scripts
  tools/                    # utility scripts (diagnose_grpo_rollout, grpo_decoding_matrix)
src/SFT_GRPO/               # training code
  train_sft.py              #   SFTTrainer + LoRA/QLoRA, auto-batch-calibration
  train_grpo.py             #   custom GRPO loop (rollout → reward → advantage → PG)
  rewards.py                #   R_acc (ROUGE-L), R_len, R_sent, degenerate detector
  evaluate.py               #   compare base / SFT / GRPO models
  config.py                 #   SFTConfig, GRPOConfig, EvalConfig, ModelConfig
  metrics_logger.py         #   CSV/metrics logging to output_dir
src/dataset/                # data pipeline
  dataset.py                #   VietNews, WikiLingua, ViMs, VLSP loaders
  augmenter.py              #   injects length/style instructions into raw data
  length_profiler.py        #   word-count distribution analysis
VDT_Textsum/                # raw data (gitignored)
data/                       # generated JSONL splits (gitignored)
models/                     # checkpoints (gitignored)
configs/                    # experiment-specific JSON overrides
  ablate_no_sentence_reward.json
```

## Model variants (Qwen3-4B)

Two model families, each with its own checkpoint tree:

| Family | Base weights | SFT checkpoint | GRPO checkpoints |
|--------|--------------|----------------|------------------|
| **Base** | `Qwen3-4B-Base` (pretrained) | `sft_qwen3_4b_base/final` | `grpo_qwen3_4b_base_{fresh,sft}_v{3,4,5}` |
| **Instruct** | `Qwen3-4B` (chat-tuned) | `sft_qwen3_4b_instruct/final` | `grpo_qwen3_4b_instruct_{fresh,sft}_v{3,4,5}` |

GRPO has two initialization modes:
- **fresh** — LoRA on top of pretrained base weights (no SFT)
- **sft** — warm-start from SFT checkpoint (recommended; GRPO ineffective without SFT init)

Three config versions:
- **v3** — initial configuration (baseline for reward-hacking analysis)
- **v4** — K=4, LR=5e-7, beta=0.15 (configuration-bundle ablation)
- **v5** — K=8, LR=2e-6, beta=0.04 (**main paper config**; best results)

## Key commands

```bash
# 1. Generate data splits
python src/dataset/augmenter.py                   # outputs to data/

# 2. SFT (choose variant via MODEL_VARIANT=base|instruct)
qsub -v MODEL_VARIANT=base scripts/pbs/train_sft_qwen3_4b.pbs
qsub -v MODEL_VARIANT=instruct scripts/pbs/train_sft_qwen3_4b.pbs

# 3. GRPO (choose variant, init mode, config version)
qsub -v MODEL_VARIANT=base,INIT=sft,OUTPUT_SUFFIX=_sft,LR=2e-6,NUM_GEN=8,BETA=0.04 \
     scripts/pbs/train_grpo_qwen3_4b.pbs          # Base SFT-init v5
qsub -v MODEL_VARIANT=instruct,INIT=fresh,OUTPUT_SUFFIX=_fresh \
     scripts/pbs/train_grpo_qwen3_4b.pbs          # Instruct fresh (default LR)

# 4. Evaluate all Qwen3 variants
qsub scripts/pbs/eval.pbs                         # full eval
# Or specific families:
PYTHONPATH=src python scripts/launch/eval.py --families qwen3_base,qwen3_instruct

# 5. Manual (local test)
PYTHONPATH=src python src/SFT_GRPO/train_sft.py --help
PYTHONPATH=src python src/SFT_GRPO/train_grpo.py --help
PYTHONPATH=src python src/SFT_GRPO/evaluate.py --models base,sft=models/sft_qwen3_4b_base/final
```

## Per-GPU configuration (Qwen3-4B)

| GPU    | VRAM  | SFT dtype | SFT batch | GRPO batch | Notes |
|--------|-------|-----------|-----------|------------|-------|
| **H200** | 141 GB | bf16 | 10×2=20 eff | 4×4=16 eff | Flash-attn ✅, packing ✅ |
| **A100** | 40/80 GB | bf16 | 6×3=18 eff | 2×8=16 eff | Flash-attn ✅ |
| **V100** | 16/32 GB | fp16 | 2×4=8 eff | 1×16=16 eff | Flash-attn ❌, packing ❌ |

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set automatically in `train_sft.py`.
- `packing` requires flash-attention v2+. Keep `False` on V100.
- Qwen3-4B vocab size 152064 (larger than Qwen2.5 → higher VRAM per sample).
- Auto-calibration (`auto_calibrate_batch=True`) probes VRAM to adjust batch size.
- GRPO always uses `gradient_checkpointing=True` (policy + reference model = ~80 GB on H200).

## Thinking mode handling

Qwen3 models have built-in `<think>` blocks. The codebase suppresses them:

```python
# In configs: disable_thinking=True
# In tokenizer: apply_chat_template(..., enable_thinking=False)
# Fallback: regex strip of <think>...</think> in evaluate.py
```

Always use `disable_thinking=True` for all Qwen3 experiments. Set via config, PBS override JSON, or `--disable_thinking` CLI flag.

## Reward system

Three components, computed at the word level (Vietnamese):

| Reward | Definition | Weight |
|--------|-----------|--------|
| **R_acc** | 0.5×ROUGE-1_F1 + 0.5×ROUGE-L_F1 (word-level LCS) | 0.5 |
| **R_len** | Length adherence — `khoảng X từ` (±20%), `trong khoảng lo-hi từ` (exact), `không quá X từ` (max) | 0.3 |
| **R_sent** | Sentence count adherence (same tolerance patterns) | 0.2 |

Composite (gated product, not linear sum):
```
R_total = R_acc × (1 + 0.3×R_len + 0.2×R_sent)
```

Degenerate outputs detected by `_is_degenerate()`: repetition loops, blobs, near-empty content. These get R_acc=0.

**Sentence reward ablation** (`configs/ablate_no_sentence_reward.json`): setting `reward_weight_sentence=0.0` did not hurt performance (ROUGE-2 and LenErr% slightly improved vs v4). Accuracy + length rewards may be sufficient.

## Key experimental findings

1. **SFT is the biggest improvement**: Base ROUGE-2 0.1056 → 0.2632 after SFT
2. **GRPO needs SFT warm-start**: Fresh GRPO (v5) ROUGE-2=0.1866 (Base) vs SFT+GRPO v5=0.2638 (Base)
3. **Best model**: `QWEN3_INSTRUCT_grpo_sft_v5` — ROUGE-2=0.2765 combined, 0.2962 on VietNews
4. **v5 > v4**: Across all branches, K=8, LR=2e-6, beta=0.04 outperforms K=4, LR=5e-7, beta=0.15
5. **Reward-hacking mitigation** (v3→v5): degenerate rate dropped from 23.81% → 0.32% on Base fresh

## SFT gotchas

- `lora_alpha=r` (scaling=1) — critical for stability. Do not use α/r=2.
- `learning_rate=5e-5` — 2e-4 diverges with LoRA.
- `assistant_only_loss=True` — dataset must have `"messages"` column (chat format).
- `max_seq_length=3072` — covers 99.7% of samples. Lower to 2048 if OOM on V100.
- `warmup_ratio=0.1` — longer warmup prevents divergence.
- SFT on Qwen3-4B: batch starts conservatively (10×2 on H200), auto-calibration scales up.

## GRPO gotchas

- `learning_rate=5e-7` (default) or 2e-6 (v5) — very low; GRPO is sensitive to large updates.
- Custom training loop (not `TRL.GRPOTrainer`). LoRA applied via `get_peft_model`.
- Reference model = frozen copy of policy base (same initial weights).
- **`repetition_penalty` must stay 1.0** — HF applies it over full input_ids including the source article. >1 penalises reusing article vocabulary, collapsing R_acc to ≈0.
- **`no_repeat_ngram_size` must stay 0** — >0 forbids n-grams from the source article, corrupting short Vietnamese summaries.
- Total steps: 800 (v3/v4) or configured via `TOTAL_STEPS` (v5).
- Ablation checkpoints saved every `save_steps`; `best/` symlink for best checkpoint.
- v5 requires `NUM_GEN=8` (K=8), which doubles VRAM per prompt vs K=4.

## Data pipeline

- `augmenter.py` reads raw datasets from `VDT_Textsum/`, generates 5 JSONL splits under `data/`.
- Data sources: VietNews (~105K), WikiLingua (~14K), ViMs (300), VLSP (285).
- **Leakage warning**: ViMs/VLSP test overlaps with train/val — only VietNews and WikiLingua support generalization claims.
- SFT: 3× augmentation per raw sample (`num_variants=3`), random length/style.
- GRPO prompts: no assistant response — model generates K completions during training.
- Meta fields (`length_requirement`, `style`) carried in every sample for reward functions.
- Length reward tolerances: `khoảng X từ` = ±20%, `trong khoảng lo-hi` = exact range, `không quá X` = max.

## Eval results (canonical artefacts)

| Artefact | Path | Role |
|----------|------|------|
| Main Qwen3 eval (v4/v5) | `models/eval_results/20260622_103706` | Primary results |
| v3 comparison | `models/eval_results/20260627_043351` | Reward-hacking analysis baseline |
| No-sentence ablation | `models/eval_results/20260630_020126` | Appendix-only, inconclusive |
| Frozen eval protocol | `temperature=0.3, top_p=0.9, do_sample=True, max_new_tokens=256` | No seed saved |

## Key documentation

| Doc | Content |
|-----|---------|
| `docs/PROJECT_AUDIT.md` | Provenance, claim boundaries, frozen evaluation protocol |
| `docs/report.md` | Full paper draft |
| `docs/REPORT_OUTLINE.md` | ACL 8-page structure |
| `docs/DATASETS.md` | Data sources, splits, leakage analysis |
| `docs/problem_statement.md` | Research questions (Base vs Instruct, reward hacking, v4/v5) |
| `docs/pipeline_plan.md` | Full pipeline description |
| `docs/qwen3_training_report.md` | Historical snapshot (22/06/2026) — Qwen3 training details |
| `docs/PROJECT_AUDIT.md` | (canonical source — overrides historical docs on conflict) |

## Modifying configs

All configs are Python `dataclasses` in `src/SFT_GRPO/config.py`. Edit them directly or pass CLI args. For PBS runs, override via `-v KEY=VAL` on `qsub` or by writing a JSON override config file. `ModelConfig.model_name_or_path` defaults to `Qwen2.5-3B-Instruct` — always override to a Qwen3-4B path for Qwen3 experiments.

## Output paths

| Stage | Checkpoints | Metrics |
|-------|-------------|---------|
| SFT | `models/sft_qwen3_4b_{base,instruct}/` | `metrics/train_metrics.csv`, `eval_metrics.csv` |
| GRPO | `models/grpo_qwen3_4b_{base,instruct}_{fresh,sft}_v{3,4,5}/` | same structure |
| Eval | `models/eval_results/{run_id}/` | Per-model JSON + aggregated tables |
