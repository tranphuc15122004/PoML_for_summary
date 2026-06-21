"""Configuration dataclasses for SFT and GRPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def detect_gpu_config() -> Dict:
    """Return optimal SFTConfig overrides based on GPU VRAM.

    Tiers target effective batch size = 16 across all GPUs.
    Falls back to conservative defaults if torch/CUDA unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return dict(per_device_train_batch_size=1, gradient_accumulation_steps=16,
                        max_seq_length=1024, gradient_checkpointing=True,
                        bf16=False, fp16=False, packing=False)

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)

        if vram_gb >= 100:      # H200 141 GB, H100 SXM 80 GB (reported as ~94 GB usable)
            tier, params = "H200/H100", dict(
                per_device_train_batch_size=16, gradient_accumulation_steps=1,
                max_seq_length=3072, gradient_checkpointing=False,
                bf16=True, fp16=False, packing=True,
            )
        elif vram_gb >= 60:     # A100 80 GB
            tier, params = "A100-80G", dict(
                per_device_train_batch_size=8, gradient_accumulation_steps=2,
                max_seq_length=3072, gradient_checkpointing=False,
                bf16=True, fp16=False, packing=True,
            )
        elif vram_gb >= 35:     # A100 40 GB
            tier, params = "A100-40G", dict(
                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                max_seq_length=3072, gradient_checkpointing=True,
                bf16=True, fp16=False, packing=True,
            )
        elif vram_gb >= 20:     # RTX 3090/4090, A30 (24 GB)
            tier, params = "24GB", dict(
                per_device_train_batch_size=2, gradient_accumulation_steps=8,
                max_seq_length=2048, gradient_checkpointing=True,
                bf16=True, fp16=False, packing=True,
            )
        else:                   # V100 32 GB reported ~31 GB; 16 GB cards
            tier, params = "V100/≤32GB", dict(
                per_device_train_batch_size=1, gradient_accumulation_steps=16,
                max_seq_length=2048, gradient_checkpointing=True,
                bf16=False, fp16=True, packing=False,
            )

        eff = params["per_device_train_batch_size"] * params["gradient_accumulation_steps"]
        print(
            f"[GPU] {gpu_name} ({vram_gb:.0f} GB) → tier={tier} | "
            f"batch={params['per_device_train_batch_size']} × "
            f"{params['gradient_accumulation_steps']} (eff={eff}) | "
            f"seq={params['max_seq_length']} | "
            f"gc={params['gradient_checkpointing']} | "
            f"packing={params['packing']}"
        )
        return params

    except Exception as exc:
        print(f"[GPU] Detection failed ({exc}), using conservative defaults")
        return dict(per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    max_seq_length=2048, gradient_checkpointing=True,
                    bf16=True, fp16=False, packing=False)


# ==============================================================================
# Shared
# ==============================================================================

@dataclass
class ModelConfig:
    """Model and quantization configuration."""

    model_name_or_path: str = "/g/data/hn98/dd9648/models/Qwen2.5-3B-Instruct"
    """Local path to model weights (offline cluster — no HuggingFace downloads)."""

    load_in_4bit: bool = False
    """Use FP16/bf16 LoRA instead of 4-bit QLoRA. Set True only if VRAM < 12 GB."""

    bnb_4bit_quant_type: str = "nf4"
    """4-bit quantization type: "nf4" or "fp4"."""

    bnb_4bit_compute_dtype: str = "bfloat16"
    """Compute dtype. Use 'bfloat16' for A100/H200 (native), 'float16' for V100/T4."""

    bnb_4bit_use_double_quant: bool = True
    """Double quantization for memory efficiency."""

    lora_r: int = 32
    """LoRA rank. 32 = good capacity for summarization."""

    lora_alpha: int = 32
    """LoRA alpha scaling. Set = r for stable training (scaling factor = 1)."""

    lora_dropout: float = 0.05
    """LoRA dropout. 0.05 provides light regularization."""


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
    per_device_train_batch_size: int = 20
    """Per-device batch size. H200 141GB + flash_attn → 20; A100 80GB → 8; V100 32GB → 2."""
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    """Effective batch size = per_device_train_batch_size * grad_accum.
    H200: 20 × 1 = 20. A100-80G: 8 × 2 = 16. V100: 2 × 4 = 8."""

    max_seq_length: int = 3072
    """Covers 99.7% of samples without truncation. Lower to 2048 for V100."""

    packing: bool = True
    """Requires flash-attention v2+. Enabled for H200 (improves throughput ~1.5x)."""

    learning_rate: float = 5e-5
    """Lower LR for LoRA stability. QLoRA/FP16 LoRA needs 5e-5 or lower."""
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    """Longer warmup — let model adapt gradually to avoid divergence."""
    num_train_epochs: float = 2.0
    """H200 24h: 2 epochs ≈ 10–16h for 358K samples. A100 24h: 1 epoch."""

    # Optimizations
    bf16: bool = True
    fp16: bool = False
    # A100/H200: use bf16 (native). V100/T4: switch to fp16 (native).
    gradient_checkpointing: bool = False
    # H200 141GB: 3B model + LoRA + batch=20 uses ~40-60GB — no need for checkpointing.
    # Set True for A100 40GB or V100 to save memory at cost of ~35% step time.
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    dataloader_num_workers: int = 4
    """Parallel data loading workers. H200 nodes have large CPU — 4 is safe."""

    # LoRA specifics
    neftune_noise_alpha: Optional[float] = None
    """Optional — NEFTune noise for better generalization."""

    # Batch calibration
    auto_calibrate_batch: bool = True
    """Run a 2-step VRAM probe before training to find the largest batch size
    that fits within calibrate_target_vram_fraction of GPU memory."""
    calibrate_target_vram_fraction: float = 0.90
    """Target VRAM utilisation for calibration (0–1). 0.90 leaves 10% headroom."""

    # Logging & saving
    logging_steps: int = 5
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 100
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

    disable_thinking: bool = False
    """Set True for Qwen3/Qwen3.5 models to suppress <think> blocks during SFT.
    Passed as enable_thinking=False to the tokenizer's apply_chat_template."""

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

    max_new_tokens: int = 80
    """Max tokens per generated completion.
    References are median 17 words (~25 tokens). 80 tokens covers p99 (~47 words)
    with buffer and avoids ROUGE-L precision dilution from overly long generations."""

    max_seq_length: int = 3072
    """Total context length. H200: 3072. A100-40G: 2048. max_new_tokens must be < max_seq_length."""

    do_sample: bool = True
    """Use sampling for rollout generation. Set False for greedy (useful for smoke tests)."""

    top_p: float = 0.9
    """Top-p sampling for rollout."""

    # Loss hyperparameters
    epsilon: float = 0.2
    """PPO-style clipping parameter."""

    beta: float = 0.15
    """KL penalty coefficient. Higher value keeps policy closer to SFT reference.
    0.04 was too low — KL exploded to 4.6+ causing policy collapse within 200 steps.
    0.15 provides strong enough anchor while still allowing meaningful updates."""

    # Training
    per_device_train_batch_size: int = 4
    """Number of prompts per device. Each generates K completions.
    Calibration will probe and adjust automatically. With gradient_checkpointing,
    H200 can fit batch=4–8; defaults start conservatively and calibration scales up."""

    gradient_accumulation_steps: int = 1
    """Effective prompts per update = batch_size * grad_accum.
    H200: 12 × 1 = 12. A100-80G: 4 × 2 = 8."""

    learning_rate: float = 5e-7
    """Low LR — GRPO is sensitive to large updates."""

    lr_scheduler_type: str = "constant"
    warmup_steps: int = 20

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    # H200: policy (4B) + ref (4B) + rollouts + vocab 152K needs ~80 GB with
    # gradient checkpointing vs ~140 GB without. Always enable for GRPO.
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    dataloader_num_workers: int = 4
    """Parallel data loading workers."""

    # Auto-calibration (tương tự SFT)
    auto_calibrate_batch: bool = True
    """Run VRAM probe before training to find the largest batch size
    that fits within calibrate_target_vram_fraction of GPU memory."""
    calibrate_target_vram_fraction: float = 0.90
    """Target VRAM utilisation for calibration (0–1). 0.90 leaves 10% headroom."""
    calibrate_effective_batch: int = 16
    """Target effective batch size (batch × grad_accum) for calibration."""

    # Reward weights
    reward_weight_accuracy: float = 0.5
    reward_weight_length: float = 0.3
    reward_weight_sentence: float = 0.2

    # Length-scaled advantage
    length_advantage_alpha: float = 0.0
    """Amplification factor for length-correct completions.
    Disabled (0.0): when R_acc≈0 for all completions, this amplified length-only
    reward hacking — model learned to produce right-length garbage instead of
    content. Re-enable only after R_acc is reliably non-zero."""

    # Logging & saving
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 5
    output_dir: str = "models/grpo_checkpoints"
    run_name: Optional[str] = None
    report_to: str = "none"

    total_steps: int = 800

    repetition_penalty: float = 1.0
    """Rollout repetition penalty. MUST stay 1.0 (off). HF applies it over the FULL
    input_ids — i.e. the ~2000-token source article in the prompt — so >1 penalises
    the model for reusing the article's own vocabulary, which is exactly what a faithful
    summary must do. A value of 1.3 collapsed R_acc to ≈0 by pushing the decoder to
    out-of-distribution tokens (foreign scripts, symbols). Matches TRL's GRPO default
    (1.0). Degenerate outputs are handled reward-side by rewards._is_degenerate()."""

    no_repeat_ngram_size: int = 0
    """Block exact n-gram repetition during generation. 0 = disabled. A value of 3
    forbade reusing ANY trigram from the source article and corrupted short Vietnamese
    summaries; TRL's GRPO trainer does not use it."""

    disable_thinking: bool = False
    """Set True for Qwen3/Qwen3.5 models to suppress <think> blocks in rollouts.
    Passed as enable_thinking=False to all apply_chat_template calls in GRPO."""

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

    # Judge/backbone model for BARTScore and G-Eval
    judge_model_path: str = "/g/data/hn98/dd9648/models/Qwen3.5-4B"

    # Feature flags — disable for fast eval without LLM-based metrics
    enable_bart_score: bool = True
    enable_geval: bool = True

    # Base model path used when loading LoRA adapters
    base_model_path: str = "/g/data/hn98/dd9648/models/Qwen3.5-4B"

    # Models to compare
    model_paths: dict = field(
        default_factory=lambda: {
            "base": "/g/data/hn98/dd9648/models/Qwen3.5-4B",
            "sft_aug": "models/sft_aug_Qwen3.5-4B/final",
            "sft_no_aug": "models/sft_no_aug_Qwen3.5-4B/final",
        }
    )
