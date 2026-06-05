#!/usr/bin/env python
"""Launch SFT training with T4-optimized config."""

from SFT_GRPO.config import SFTConfig, ModelConfig
from SFT_GRPO.train_sft import train

cfg = SFTConfig(
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        bnb_4bit_compute_dtype="float16",
    ),
    output_dir="models/sft_lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    max_seq_length=1024,
    packing=False,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    eval_strategy="no",
    save_steps=2000,
    logging_steps=10,
    report_to="none",
)

train(cfg)
