#!/usr/bin/env python
"""
SFT (Supervised Fine-Tuning) for Vietnamese text summarization.

Trains a Qwen3-4B family model with LoRA on instruction-conditioned Vietnamese summarization data.

Usage:
    python src/SFT_GRPO/train_sft.py                         # train from scratch
    python src/SFT_GRPO/train_sft.py --resume models/sft_lora/checkpoint-500
"""

from __future__ import annotations

import sys
import types

# bitsandbytes 0.44.x references triton.ops which was removed in triton 3.x.
# PEFT imports bitsandbytes unconditionally during get_peft_model(); stub the
# missing submodule so the import succeeds without GPU-quantization support.
if "triton.ops" not in sys.modules:
    _triton_ops = types.ModuleType("triton.ops")
    _triton_perf = types.ModuleType("triton.ops.matmul_perf_model")
    _triton_perf.early_config_prune = lambda *a, **kw: None
    _triton_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    sys.modules["triton.ops"] = _triton_ops
    sys.modules["triton.ops.matmul_perf_model"] = _triton_perf

import json
import logging
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Logging: configure BEFORE importing 3rd-party libs so format takes effect.
# Use force=True to override any handlers that libraries might have installed.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Reduce CUDA memory fragmentation (helps all NVIDIA GPUs)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset as HFDataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig as TRLSFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFT_GRPO.config import SFTConfig, ModelConfig
from SFT_GRPO.metrics_logger import MetricsTracker, MetricsCallback


# ==============================================================================
# Data loading
# ==============================================================================

def load_jsonl(path: str) -> HFDataset:
    """Load SFT data from JSONL (each line = chat-format dict with 'messages' key)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    import json as _json
    with open(path, "r", encoding="utf-8") as _f:
        _lines = [l for l in _f if l.strip()]
    if not _lines:
        raise ValueError(
            f"Data file is empty: {path}\n"
            "Run data preparation first: PYTHONPATH=src python src/dataset/prepare_no_aug.py\n"
            "  or: PYTHONPATH=src python src/dataset/augmenter.py"
        )
    return load_dataset("json", data_files=path, split="train")


# ==============================================================================
# Model loading
# ==============================================================================

def load_tokenizer(model_cfg: ModelConfig, disable_thinking: bool = False):
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
        logger.warning("No chat template found - using the tokenizer fallback format.")
    # Qwen3 thinking mode: patch apply_chat_template to disable when configured
    # <think> blocks. TRL's SFTTrainer calls this internally, so patching the
    # tokenizer is the only way to control it without forking TRL.
    if disable_thinking:
        _orig_tpl = tokenizer.apply_chat_template
        def _no_thinking(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_tpl(*args, **kwargs)
        tokenizer.apply_chat_template = _no_thinking
        logger.info("Thinking mode disabled for Qwen3-family model.")
    return tokenizer


def load_model(model_cfg: ModelConfig, device_map: str = "auto"):
    """Load model with optional QLoRA 4-bit or pure FP16/bf16."""
    if model_cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        )
    else:
        bnb_config = None

    # Use model_cfg.bnb_4bit_compute_dtype as the overall compute dtype hint
    torch_dtype = getattr(torch, model_cfg.bnb_4bit_compute_dtype)

    # Probe flash-attn by actually importing it — find_spec() misses ABI errors
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
        logger.warning("flash-attn import failed — falling back to sdpa attention.")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    if hasattr(model.config, "pretraining_tp"):
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
# Batch size calibration
# ==============================================================================

def calibrate_batch_size(
    model,
    max_seq_length: int,
    target_fraction: float = 0.90,
    effective_batch: int = 16,
    gradient_checkpointing: bool = False,
) -> tuple:
    """Probe VRAM usage with dummy forward+backward passes to find the largest
    per-device batch size that fits within target_fraction of GPU memory.

    Uses two probes (batch=1, batch=2) and linear extrapolation:
        per_sample_mem  = mem(batch=2) - mem(batch=1)
        fixed_overhead  = mem(batch=1) - per_sample_mem   # model weights + framework
        optimizer_mem   = 2 * trainable_params * bytes    # AdamW m + v (not in probe)
        optimal_batch   = (target_bytes - fixed_overhead - optimizer_mem) / per_sample_mem

    gradient_accumulation_steps is then adjusted so that
        optimal_batch * grad_accum ≈ effective_batch.

    Returns:
        (per_device_batch_size, gradient_accumulation_steps)
    """
    import gc

    if not torch.cuda.is_available():
        logger.warning("[Calibration] No CUDA device — skipping")
        return 1, effective_batch

    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    target_bytes = int(total_vram * target_fraction)

    # AdamW keeps two fp32 state tensors per trainable parameter.
    # These are allocated on the first optimizer.step(), so they don't
    # appear in the probe peak. Subtract their estimated size upfront.
    optimizer_mem = sum(
        2 * p.numel() * 4  # always fp32 states regardless of model dtype
        for p in model.parameters() if p.requires_grad
    )

    logger.info(
        f"[Calibration] {props.name} ({total_vram/1e9:.1f} GB) | "
        f"target={target_fraction*100:.0f}% ({target_bytes/1e9:.1f} GB) | "
        f"optimizer_states≈{optimizer_mem/1e9:.2f} GB"
    )

    was_training = model.training

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable({"use_reentrant": False})
        except Exception:
            pass

    def _probe(batch_size: int) -> int:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(100, 2000, (batch_size, max_seq_length), device="cuda")
        model.train()
        loss = model(input_ids=ids, labels=ids.clone()).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated()
        del ids, loss
        torch.cuda.empty_cache()
        gc.collect()
        return peak

    try:
        mem_1 = _probe(1)
        mem_2 = _probe(2)
    except RuntimeError as exc:
        logger.warning(f"[Calibration] Probe failed ({exc}) — keeping initial batch size")
        if not was_training:
            model.eval()
        return 1, effective_batch
    finally:
        if not was_training:
            model.eval()
        if gradient_checkpointing:
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass

    per_sample = max(mem_2 - mem_1, 1)
    overhead = mem_1 - per_sample

    available = target_bytes - overhead - optimizer_mem
    optimal_batch = max(1, int(available / per_sample))
    grad_accum = max(1, round(effective_batch / optimal_batch))

    logger.info(
        f"[Calibration] model_overhead={overhead/1e9:.2f} GB | "
        f"per_sample={per_sample/1e9:.3f} GB | "
        f"available={available/1e9:.2f} GB | "
        f"→ batch={optimal_batch} × grad_accum={grad_accum} "
        f"(eff={optimal_batch * grad_accum}, target_eff={effective_batch})"
    )

    return optimal_batch, grad_accum


# ==============================================================================
# Training
# ==============================================================================

def _setup_file_logging(output_dir: str) -> str:
    """Add a FileHandler to the root logger → output_dir/training.log."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    return log_path


def train(cfg: SFTConfig, resume_from: Optional[str] = None):
    """Run SFT training loop."""
    import platform, time

    # 0. Set up file logging — captures everything from this point on
    log_path = _setup_file_logging(cfg.output_dir)

    t_start = time.time()
    sep = "=" * 70

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
    tokenizer = load_tokenizer(cfg.model, disable_thinking=cfg.disable_thinking)
    model = load_model(cfg.model)
    lora_config = get_lora_config(cfg.model)

    # Apply LoRA before calibration so the probe sees the correct memory
    # footprint: base model weights are frozen, only LoRA adapters (≈14 MB
    # for rank=32) have requires_grad=True. Without this, optimizer_mem
    # would be computed over all 4B parameters, underestimating available
    # VRAM by ~32 GB and producing a needlessly small batch size.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 2.5. Auto-calibrate batch size to target ~90% VRAM utilisation.
    # Probes VRAM with dummy forward+backward passes before the trainer is
    # created, so the batch size can be adjusted without wasting training time.
    if cfg.auto_calibrate_batch and torch.cuda.is_available():
        eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        cal_batch, cal_accum = calibrate_batch_size(
            model,
            max_seq_length=cfg.max_seq_length,
            target_fraction=cfg.calibrate_target_vram_fraction,
            effective_batch=eff_batch,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        if cal_batch != cfg.per_device_train_batch_size or cal_accum != cfg.gradient_accumulation_steps:
            logger.info(
                f"[Calibration] batch: {cfg.per_device_train_batch_size} → {cal_batch} | "
                f"grad_accum: {cfg.gradient_accumulation_steps} → {cal_accum}"
            )
            cfg.per_device_train_batch_size = cal_batch
            cfg.gradient_accumulation_steps = cal_accum
            cfg.per_device_eval_batch_size = max(1, cal_batch * 2)

    # --- Training header (printed after calibration so values are final) ---
    header_lines = [
        sep,
        "SFT TRAINING START",
        sep,
        f"  Output dir    : {cfg.output_dir}",
        f"  Log file      : {log_path}",
        f"  Model         : {cfg.model.model_name_or_path}",
        f"  Train data    : {cfg.train_data_path}",
        f"  Val data      : {cfg.val_data_path}",
        f"  Epochs        : {cfg.num_train_epochs}",
        f"  Batch size    : {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps}"
        f" (eff={cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps})",
        f"  Max seq len   : {cfg.max_seq_length}",
        f"  LR            : {cfg.learning_rate}  scheduler={cfg.lr_scheduler_type}",
        f"  LoRA r/alpha  : {cfg.model.lora_r}/{cfg.model.lora_alpha}",
        f"  bf16/fp16     : {cfg.bf16}/{cfg.fp16}  packing={cfg.packing}",
        f"  Node          : {platform.node()}",
        f"  PyTorch       : {torch.__version__}",
    ]
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        header_lines.append(f"  GPU           : {gpu_name}  ({gpu_mem:.1f} GB)")
        header_lines.append(f"  CUDA          : {torch.version.cuda}")
    header_lines.append(sep)
    for line in header_lines:
        logger.info(line)

    # 3. Configure TRL SFT
    # TRL auto-detects the "messages" column and applies the chat template.
    # With pinned TRL 0.13.0, assistant-only loss masking is not guaranteed.
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
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        dataset_text_field=None,  # auto-detect "messages" column
        report_to=cfg.report_to,
        run_name=cfg.run_name or f"sft_qwen3b_{cfg.num_train_epochs}ep",
        remove_unused_columns=False,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
    )

    # 5. Metrics tracker (logs to output_dir/metrics/)
    metrics_tracker = MetricsTracker(output_dir=cfg.output_dir)

    # 6. Trainer with metrics callback.
    # LoRA already applied above — do NOT pass peft_config again or TRL
    # will raise an error about double-wrapping a PeftModel.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(MetricsCallback(metrics_tracker))

    # 7. Save config snapshot
    metrics_tracker.save_config({k: str(v) for k, v in cfg.__dict__.items()})

    # 8. Optionally resume
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # 9. Save final
    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    logger.info(f"Model saved to {final_path}")

    # Save config alongside
    with open(os.path.join(final_path, "sft_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, default=str, indent=2)

    # Close metrics tracker
    metrics_tracker.close()

    # --- Training footer ---
    t_elapsed = time.time() - t_start
    h, rem = divmod(int(t_elapsed), 3600)
    m, s = divmod(rem, 60)
    footer_lines = [
        sep,
        "SFT TRAINING COMPLETE",
        sep,
        f"  Wall time     : {h}h {m}m {s}s",
        f"  Model saved   : {final_path}",
        f"  Train metrics : {metrics_tracker._train_file_csv}",
        f"  Eval metrics  : {metrics_tracker._eval_file_csv}",
        f"  Full log      : {log_path}",
    ]
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        footer_lines.append(f"  Peak GPU mem  : {peak_mem:.2f} GB")
    footer_lines.append(sep)
    for line in footer_lines:
        logger.info(line)


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
    # Legacy default; canonical Qwen3 runs pass --model_name explicitly.
    parser.add_argument("--output_dir", type=str, default="models/sft_lora")
    parser.add_argument("--lr", type=float, default=5e-5)
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
