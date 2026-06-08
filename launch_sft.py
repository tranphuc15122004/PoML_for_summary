#!/usr/bin/env python
"""Launch SFT training — tuned for H200 141GB VRAM + Flash Attention 2.

Key choices:
  - batch=20, grad_accum=1 → effective_batch=20, ~40-60 GB VRAM
  - flash_attention_2 + packing → ~1.5x throughput vs standard attention
  - gradient_checkpointing=False — 141GB VRAM eliminates need for it (~35% faster steps)
  - lr=5e-5, lora_alpha=r (scaling=1) — prevents divergence
  - bf16=True (native on H200/A100; V100 → switch to fp16)
  - max_seq_length=3072 — covers 99.7% of samples

For A100 80GB:  batch=8,  grad_accum=2, packing=True,  gradient_checkpointing=False
For A100 40GB:  batch=4,  grad_accum=4, packing=True,  gradient_checkpointing=True
For V100 32GB:  batch=2,  grad_accum=4, packing=False, gradient_checkpointing=True, fp16=True
"""

from SFT_GRPO.config import SFTConfig, ModelConfig
from SFT_GRPO.train_sft import train

cfg = SFTConfig(
    model=ModelConfig(
        model_name_or_path="/g/data/hn98/dd9648/models/Qwen2.5-3B-Instruct",
        load_in_4bit=False,
        lora_r=32,
        lora_alpha=32,       # scaling = α/r = 1 (stable)
        lora_dropout=0.05,
    ),
    output_dir="models/sft_lora",
    per_device_train_batch_size=20,   # H200 141GB + flash_attn → 20
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    max_seq_length=3072,
    packing=True,                     # flash_attn installed → enable for ~1.5x throughput
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=2.0,
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,     # 141GB VRAM → not needed, saves ~35% step time
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    logging_steps=5,
    report_to="none",
)

train(cfg)
