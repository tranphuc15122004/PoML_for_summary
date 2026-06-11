#!/usr/bin/env python
"""Launch SFT — AUGMENTED data (length + style constraints, 3 variants/sample).

Supports any model. Auto-derives output_dir and run_name from model name.

Usage:
    python scripts/launch/sft_aug.py
    python scripts/launch/sft_aug.py --model /path/to/model --epochs 5
    python scripts/launch/sft_aug.py --resume
    python scripts/launch/sft_aug.py --resume --checkpoint models/sft_aug_xxx/checkpoint-1000
"""

import argparse
import os
import sys
import types

# bitsandbytes 0.44.x references triton.ops which was removed in triton 2.x.
# PEFT imports bitsandbytes unconditionally during get_peft_model(); stub the
# missing submodule so the import succeeds without GPU-quantization support.
if "triton.ops" not in sys.modules:
    _triton_ops = types.ModuleType("triton.ops")
    _triton_perf = types.ModuleType("triton.ops.matmul_perf_model")
    _triton_perf.early_config_prune = lambda *a, **kw: None
    _triton_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    sys.modules["triton.ops"] = _triton_ops
    sys.modules["triton.ops.matmul_perf_model"] = _triton_perf

from SFT_GRPO.config import SFTConfig, ModelConfig, detect_gpu_config
from SFT_GRPO.train_sft import train

DEFAULT_CHECKPOINT = None
DEFAULT_MODEL = "/g/data/hn98/dd9648/models/Qwen3.5-4B"


def _model_short_name(path: str) -> str:
    """Extract short model identifier from a path or HF name.
    e.g. "/g/data/.../Qwen3.5-4B" → "Qwen3.5-4B"
         "Qwen/Qwen2.5-7B-Instruct" → "Qwen2.5-7B-Instruct"
    """
    return os.path.basename(path.rstrip("/"))


def build_config(
    epochs: float = 2.0,
    output_dir: str | None = None,
    model_path: str = DEFAULT_MODEL,
    train_data: str = "data/sft_train.jsonl",
    val_data: str = "data/sft_val.jsonl",
) -> SFTConfig:
    gpu = detect_gpu_config()
    model_short = _model_short_name(model_path)
    if output_dir is None:
        output_dir = f"models/sft_aug_{model_short}"
    return SFTConfig(
        model=ModelConfig(
            model_name_or_path=model_path,
            load_in_4bit=False,
            lora_r=32,
            lora_alpha=32,
            lora_dropout=0.05,
        ),
        train_data_path=train_data,
        val_data_path=val_data,
        output_dir=output_dir,
        run_name=f"sft_aug_{model_short}_{int(epochs)}ep",
        per_device_train_batch_size=gpu["per_device_train_batch_size"],
        per_device_eval_batch_size=gpu["per_device_train_batch_size"] * 2,
        gradient_accumulation_steps=gpu["gradient_accumulation_steps"],
        max_seq_length=gpu["max_seq_length"],
        packing=gpu["packing"],
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=epochs,
        bf16=gpu["bf16"],
        fp16=gpu["fp16"],
        gradient_checkpointing=gpu["gradient_checkpointing"],
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=5,
        report_to="none",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT augmented launcher")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--train_data", type=str, default="data/sft_train.jsonl",
                        help="Training data path")
    parser.add_argument("--val_data", type=str, default="data/sft_val.jsonl",
                        help="Validation data path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint to resume from (default: auto-detect)")
    parser.add_argument("--epochs", type=float, default=2.0,
                        help="Total epochs (default: 2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: auto-derived from model name)")
    args = parser.parse_args()

    # Derive default output_dir from model name if not specified
    if args.output_dir is None:
        model_short = _model_short_name(args.model)
        args.output_dir = f"models/sft_aug_{model_short}"

    if args.resume:
        checkpoint = args.checkpoint
        if checkpoint is None:
            if os.path.isdir(args.output_dir):
                found = sorted(
                    [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                    key=lambda x: int(x.split("-")[-1]),
                )
                checkpoint = os.path.join(args.output_dir, found[-1]) if found else None
        if not checkpoint or not os.path.isdir(checkpoint):
            print(f"ERROR: Checkpoint not found: {checkpoint}")
            print("Available checkpoints:")
            if os.path.isdir(args.output_dir):
                for d in sorted(os.listdir(args.output_dir)):
                    if d.startswith("checkpoint-"):
                        print(f"  {args.output_dir}/{d}")
            sys.exit(1)
        print(f"Resuming from: {checkpoint}")
        print(f"Total epochs:  {args.epochs}")
        print(f"Output dir:    {args.output_dir}")
        cfg = build_config(
            epochs=args.epochs,
            output_dir=args.output_dir,
            model_path=args.model,
            train_data=args.train_data,
            val_data=args.val_data,
        )
        train(cfg, resume_from=checkpoint)
    else:
        cfg = build_config(
            epochs=args.epochs,
            output_dir=args.output_dir,
            model_path=args.model,
            train_data=args.train_data,
            val_data=args.val_data,
        )
        train(cfg)
