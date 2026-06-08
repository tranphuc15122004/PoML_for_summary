#!/usr/bin/env python
"""Launch SFT — AUGMENTED data (length + style constraints, 3 variants/sample).

Output: models/sft_aug/
"""

from SFT_GRPO.config import SFTConfig, ModelConfig
from SFT_GRPO.train_sft import train

cfg = SFTConfig(
    model=ModelConfig(
        model_name_or_path="/g/data/hn98/dd9648/models/Qwen2.5-3B-Instruct",
        load_in_4bit=False,
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.05,
    ),
    train_data_path="data/sft_train.jsonl",
    val_data_path="data/sft_val.jsonl",
    output_dir="models/sft_aug",
    run_name="sft_aug_qwen3b_2ep",
    per_device_train_batch_size=20,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    max_seq_length=3072,
    packing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=2.0,
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    logging_steps=5,
    report_to="none",
)

train(cfg)
