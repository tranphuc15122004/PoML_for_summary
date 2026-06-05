"""Configuration dataclasses for SFT and GRPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ==============================================================================
# Shared
# ==============================================================================

@dataclass
class ModelConfig:
    """Model and quantization configuration."""

    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    """HuggingFace model ID or local path."""

    load_in_4bit: bool = False
    """Use FP16 LoRA instead of 4-bit QLoRA. Much faster on T4.
    Set True only if VRAM < 12 GB."""

    bnb_4bit_quant_type: str = "nf4"
    """4-bit quantization type: "nf4" or "fp4"."""

    bnb_4bit_compute_dtype: str = "float16"
    """Compute dtype for 4-bit base model. Use 'float16' for T4 (native),
    'bfloat16' for Ampere+ GPUs (A100, RTX 3090/4090)."""

    bnb_4bit_use_double_quant: bool = True
    """Double quantization for memory efficiency."""

    lora_r: int = 32
    """LoRA rank."""

    lora_alpha: int = 64
    """LoRA alpha scaling."""

    lora_dropout: float = 0.05
    """LoRA dropout."""

    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    """Which modules to apply LoRA to."""


# ==============================================================================
# SFT
# ==============================================================================

@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    model: ModelConfig = field(default_factory=ModelConfig)

    # Data
    train_data_path: str = "data/sft_train.jsonl"
    val_data_path: str = "data/sft_val.jsonl"

    # Training
    per_device_train_batch_size: int = 1
    """Reduced from 2 to fit Tesla T4 16 GB memory."""
    per_device_eval_batch_size: int = 2
    """Reduced from 4 to fit eval on T4."""
    gradient_accumulation_steps: int = 16
    """Effective batch size = 1 * 16 = 16 with 1 GPU."""

    max_seq_length: int = 2048
    """Reduced from 3072 to fit T4 memory. Covers 8000-char source ~2000 tokens."""

    packing: bool = False
    """Disabled because flash-attention is not available on this system
    (GLIBC 2.31). Packing without flash attention may cause cross-contamination
    between samples."""

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0

    # Optimizations
    bf16: bool = False
    fp16: bool = True
    # T4 has native FP16 Tensor Cores; bf16 is emulated and 3-5x slower
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )

    # LoRA specifics
    neftune_noise_alpha: Optional[float] = None
    """Optional — NEFTune noise for better generalization."""

    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 200
    eval_strategy: str = "steps"
    output_dir: str = "models/sft_lora"
    run_name: Optional[str] = None
    report_to: str = "none"
    """Set to 'wandb' for Weights & Biases logging."""

    # Dataset configs (passed to PromptAugmenter)
    data_root: str = "VDT_Textsum"
    max_source_chars: int = 8000
    max_summary_chars: int = 1500

    # Dataset splits indices
    sft_val_vn_size: int = 2000
    """First N VietNews val samples used for SFT val."""
    sft_val_wl_size: int = 500
    """First N WikiLingua val samples used for SFT val."""

    def __post_init__(self):
        self.model = ModelConfig(**self.model) if isinstance(self.model, dict) else self.model


# ==============================================================================
# GRPO
# ==============================================================================

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    model: ModelConfig = field(default_factory=ModelConfig)

    # Data
    train_data_path: str = "data/grpo_train.jsonl"
    val_data_path: str = "data/grpo_val.jsonl"

    # GRPO specifics
    num_generations: int = 4
    """Group size K: number of completions to sample per prompt."""

    temperature: float = 0.7
    """Temperature for rollout generation."""

    max_new_tokens: int = 256
    """Max tokens per generated completion."""

    top_p: float = 0.9
    """Top-p sampling for rollout."""

    # Loss hyperparameters
    epsilon: float = 0.2
    """PPO-style clipping parameter."""

    beta: float = 0.04
    """KL penalty coefficient."""

    # Training
    per_device_train_batch_size: int = 2
    """Number of prompts per device. Each generates K=4 completions."""

    gradient_accumulation_steps: int = 4
    """Effective prompts per update = 2 * 4 = 8."""

    learning_rate: float = 5e-7
    """Low LR — GRPO is sensitive to large updates."""

    lr_scheduler_type: str = "constant"
    warmup_steps: int = 20

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )

    # Reward weights
    reward_weight_accuracy: float = 0.5
    reward_weight_length: float = 0.3
    reward_weight_style: float = 0.2

    # Style reward (LLM-as-Judge)
    judge_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    """Model used for style evaluation. Use 'gpt-4o-mini' if API available."""

    # Logging & saving
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 5
    output_dir: str = "models/grpo_checkpoints"
    run_name: Optional[str] = None
    report_to: str = "none"

    total_steps: int = 800

    def __post_init__(self):
        self.model = ModelConfig(**self.model) if isinstance(self.model, dict) else self.model


# ==============================================================================
# Evaluation
# ==============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    test_data_path: str = "data/test.jsonl"

    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.3
    """Lower temperature for deterministic eval generation."""

    batch_size: int = 8
    """Generation batch size."""

    output_dir: str = "models/eval_results"

    # Models to compare
    model_paths: dict = field(
        default_factory=lambda: {
            "base": "Qwen/Qwen2.5-3B-Instruct",
            "sft": "models/sft_lora",
            "grpo": "models/grpo_checkpoints",
        }
    )
