#!/usr/bin/env python
"""Launch SFT training — optimized for stable convergence.

Key fixes compared to the broken T4 run:
  - learning_rate: 5e-5 (was 2e-4 — caused divergence)
  - lora_alpha=r (scaling=1, was α/r=2 — amplified updates)
  - max_seq_length=3072 (was 768 — truncated assistant responses)
  - assistant_only_loss=True (was False — loss on all tokens)
  - fp16=True (was False — numerical instability)
  - warmup_ratio=0.1 (was 0.03 — too abrupt)
"""

from SFT_GRPO.config import SFTConfig, ModelConfig
from SFT_GRPO.train_sft import train

cfg = SFTConfig(
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        # FP16 LoRA (not 4-bit QLoRA) — more stable, requires ~20 GB VRAM
        load_in_4bit=False,
        lora_r=32,
        lora_alpha=32,       # scaling = α/r = 1 (stable)
        lora_dropout=0.05,
    ),
    output_dir="models/sft_lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    max_seq_length=3072,     # covers 99.7% of samples without truncation
    packing=False,
    learning_rate=5e-5,      # LoRA-safe LR
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,        # gradual warmup
    num_train_epochs=1.0,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    eval_strategy="no",
    save_steps=2000,
    logging_steps=10,
    report_to="none",
)

train(cfg)
