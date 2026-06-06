#!/usr/bin/env python
"""
SFT (Supervised Fine-Tuning) for Vietnamese text summarization.

Trains Qwen2.5-3B-Instruct with QLoRA on length/style-augmented instruction data.

Usage:
    python src/SFT+GRPO/train_sft.py                         # train from scratch
    python src/SFT+GRPO/train_sft.py --resume models/sft_lora/checkpoint-500
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

# Reduce CUDA memory fragmentation on Tesla T4
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset as HFDataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from trl import SFTConfig as TRLSFTConfig, SFTTrainer
from peft import LoraConfig

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFT_GRPO.config import SFTConfig, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data loading
# ==============================================================================

def load_jsonl(path: str) -> HFDataset:
    """Load SFT data from JSONL (each line = chat-format dict with 'messages' key)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return load_dataset("json", data_files=path, split="train")


# ==============================================================================
# Model loading
# ==============================================================================

def load_tokenizer(model_cfg: ModelConfig):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    # Set pad token for causal LMs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        logger.warning("No chat template found — using default Qwen2 format.")
    return tokenizer


def load_model(model_cfg: ModelConfig, device_map: str = "auto"):
    """Load model with optional QLoRA 4-bit or pure FP16."""
    if model_cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        )
    else:
        bnb_config = None

    torch_dtype = (
        getattr(torch, model_cfg.bnb_4bit_compute_dtype) if model_cfg.load_in_4bit
        else torch.float16  # native FP16 on T4
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1
    return model


def get_lora_config(model_cfg: ModelConfig) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        target_modules=model_cfg.lora_target_modules,
        lora_dropout=model_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ==============================================================================
# Training
# ==============================================================================

def train(cfg: SFTConfig, resume_from: Optional[str] = None):
    """Run SFT training loop."""
    # 1. Load data
    logger.info(f"Loading training data from {cfg.train_data_path}")
    train_dataset = load_jsonl(cfg.train_data_path)

    val_dataset = None
    if os.path.isfile(cfg.val_data_path):
        logger.info(f"Loading validation data from {cfg.val_data_path}")
        val_dataset = load_jsonl(cfg.val_data_path)
    else:
        logger.warning(f"Validation file not found: {cfg.val_data_path} — skipping eval")

    logger.info(f"Train samples: {len(train_dataset):,}")
    if val_dataset:
        logger.info(f"Val samples:   {len(val_dataset):,}")

    # 2. Load tokenizer & model
    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model)
    lora_config = get_lora_config(cfg.model)

    # 4. Configure TRL SFT
    # Note: assistant_only_loss=True enables prompt masking — SFTTrainer
    # automatically applies chat template and masks non-assistant tokens.
    # The dataset must have a "messages" column (conversational format).
    training_args = TRLSFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps if cfg.eval_strategy == "steps" else None,
        max_length=cfg.max_seq_length,
        packing=cfg.packing,
        dataset_text_field=None,  # Uses "messages" column from dataset
        report_to=cfg.report_to,
        run_name=cfg.run_name or f"sft_qwen3b_{cfg.num_train_epochs}ep",
        remove_unused_columns=False,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        assistant_only_loss=True,  # Mask prompt tokens — only compute loss on assistant responses
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 6. Optionally resume
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # 7. Save final
    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    logger.info(f"Model saved to {final_path}")

    # Save config alongside
    with open(os.path.join(final_path, "sft_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, default=str, indent=2)

    logger.info("SFT training complete!")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT for Vietnamese summarization")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--data_root", type=str, default="VDT_Textsum")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/sft_lora")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fp16", type=str, default=None, help="'true' or 'false' to override")
    parser.add_argument("--bf16", type=str, default=None, help="'true' or 'false' to override")
    args = parser.parse_args()

    cfg = SFTConfig(
        model=ModelConfig(model_name_or_path=args.model_name),
        data_root=args.data_root,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    if args.fp16 is not None:
        cfg.fp16 = args.fp16.lower() == "true"
        cfg.bf16 = not cfg.fp16
        cfg.model.bnb_4bit_compute_dtype = "float16" if cfg.fp16 else "bfloat16"
    if args.bf16 is not None:
        cfg.bf16 = args.bf16.lower() == "true"
        cfg.fp16 = not cfg.bf16
        cfg.model.bnb_4bit_compute_dtype = "bfloat16" if cfg.bf16 else "float16"

    # Override from JSON config if provided
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    logger.info(f"SFT Config: {cfg}")
    train(cfg, resume_from=args.resume)
