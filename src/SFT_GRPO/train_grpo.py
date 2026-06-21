#!/usr/bin/env python
"""
GRPO (Group Relative Policy Optimization) for Vietnamese summarization.

Optimizes the SFT model using multi-objective rewards:
    - Accuracy (ROUGE-L F1)
    - Length adherence

Usage:
    python src/SFT_GRPO/train_grpo.py
    python src/SFT_GRPO/train_grpo.py --resume models/grpo_checkpoints/checkpoint-100
"""

from __future__ import annotations

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

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Enable expandable memory segments to reduce CUDA fragmentation on H200/A100.
# This allows PyTorch's allocator to grow segments on demand instead of
# fragmenting the CUDA arena. Must be set before any torch.cuda call.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SFT_GRPO.config import GRPOConfig, ModelConfig
from SFT_GRPO.metrics_logger import MetricsTracker
from SFT_GRPO.rewards import compute_all_rewards

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Dataset
# ==============================================================================

class GRPODataset(Dataset):
    """Dataset for GRPO training: returns prompts with metadata."""

    def __init__(self, data_path: str):
        self.samples: List[Dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        logger.info(f"Loaded {len(self.samples)} GRPO prompts from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """Collate batch of prompts."""
    result: Dict[str, List] = defaultdict(list)
    for item in batch:
        result["prompt"].append(item["prompt"])
        result["reference"].append(item["reference"])
        result["meta"].append(item["meta"])
    return dict(result)


# ==============================================================================
# Utils
# ==============================================================================

def build_device_map(accelerator: Accelerator) -> str:
    """Determine device map based on available hardware."""
    if torch.cuda.device_count() > 0:
        return {"": accelerator.local_process_index}
    return "auto"


# ==============================================================================
# Batch size calibration (tự động dò VRAM)
# ==============================================================================

def calibrate_grpo_batch_size(
    policy_model,
    ref_model,
    tokenizer,
    num_generations: int = 4,
    max_seq_length: int = 3072,
    max_new_tokens: int = 256,
    target_fraction: float = 0.90,
    effective_batch: int = 16,
) -> tuple:
    """Dò VRAM để tìm batch size tối ưu cho GRPO.

    GRPO cần memory cho:
      - Policy model (trainable, forward + backward)
      - Reference model (frozen, forward only)
      - Rollout: K× batch sinh tokens (generate)
      - Optimizer states (AdamW: 2 × fp32 per trainable param)

    Chiến lược: probe với 1 prompt, đo peak memory, ngoại suy tuyến tính.

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

    # Optimizer states (AdamW): 2 × fp32 states per trainable param
    optimizer_mem = sum(
        2 * p.numel() * 4
        for p in policy_model.parameters() if p.requires_grad
    )

    logger.info(
        f"[Calibration] {props.name} ({total_vram/1e9:.1f} GB) | "
        f"target={target_fraction*100:.0f}% ({target_bytes/1e9:.1f} GB) | "
        f"optimizer≈{optimizer_mem/1e9:.2f} GB | "
        f"num_gen={num_generations}"
    )

    was_training = policy_model.training
    policy_model.train()
    ref_model.eval()

    def _probe_prompt(batch_size: int) -> int:
        """Probe peak VRAM for forward passes only (no backward).

        Running backward on the raw model (before accelerator.prepare) can leave
        grad state that interferes with training. We measure forward-only memory
        and apply a backward_factor in the extrapolation instead.
        """
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        # Use full intended lengths: logit tensor scales as B*K × seq_len × vocab.
        prompt_len = max_seq_length - max_new_tokens
        gen_len = max_new_tokens

        prompt_ids = torch.randint(100, 2000, (batch_size, prompt_len), device="cuda")
        dummy_gen = torch.randint(100, 2000, (batch_size * num_generations, gen_len), device="cuda")
        full_ids = torch.cat([prompt_ids.repeat_interleave(num_generations, dim=0), dummy_gen], dim=-1)
        full_mask = torch.ones_like(full_ids)

        with torch.no_grad():
            ref_out = ref_model(input_ids=full_ids, attention_mask=full_mask)
            del ref_out
            pol_out = policy_model(input_ids=full_ids, attention_mask=full_mask)
            gen_logits = pol_out.logits[:, prompt_len - 1: -1]
            log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
            del pol_out, gen_logits, log_probs

        peak = torch.cuda.max_memory_allocated()

        del prompt_ids, dummy_gen, full_ids, full_mask
        torch.cuda.empty_cache()
        gc.collect()
        return peak

    try:
        mem_1 = _probe_prompt(1)
        mem_2 = _probe_prompt(2)
    except RuntimeError as exc:
        logger.warning(f"[Calibration] Probe failed ({exc}) — keeping default batch size")
        return 1, effective_batch
    finally:
        if not was_training:
            policy_model.eval()

    # Two-point probe: mem_1 = peak for 1 prompt (K gens), mem_2 = peak for 2 prompts.
    # incremental = true marginal VRAM cost per extra prompt — avoids over-estimating
    # by treating model weights (16 GB fixed) as "variable" the way mem_1*0.6 did.
    # The dominant bottleneck is the logit tensor [B*K, seq, vocab] which scales
    # linearly with B; two-point measurement captures this directly.

    # ----------------------------------------------------------------------
    # fp32 logits overhead: accelerate's model wrapper converts ALL model
    # outputs from bf16 → fp32 for loss compatibility.  For Qwen3-4B this
    # doubles the logits tensor from ~3.7 GB/prompt (bf16) to ~7.5 GB/prompt
    # (fp32).  The probe runs BEFORE accelerator.prepare(), so this extra
    # copy is invisible to it — we must add it analytically.
    # ----------------------------------------------------------------------
    vocab_size = getattr(policy_model.config, "vocab_size", 152064)
    _gen_len = max_seq_length  # logits cover full sequence, not just gen portion
    # Extra bytes per prompt when bf16 logits → fp32 (4−2=2 bytes/element extra)
    fp32_logits_extra_per_prompt = num_generations * _gen_len * vocab_size * 2
    # For Qwen3-4B (vocab=152064): 4×3072×152064×2 ≈ 3.74 GB/prompt extra.

    # overhead corrected for fp32 logits copy on the first batch unit as well
    overhead = mem_1 + fp32_logits_extra_per_prompt
    incremental_fwd = max(mem_2 - mem_1, 100_000_000)  # floor at 100 MB

    # KV cache during generate(): K gens × full_seq × layers × kv_heads × head_dim × 4 bytes
    # For Qwen3-4B: ~36 layers, 8 KV heads, head_dim=128, per prompt ≈ 1.8 GB
    kv_cache_per_prompt = num_generations * max_seq_length * 36 * 8 * 128 * 4

    # With GC=True, backward recomputes the logit tensor (same size as forward peak)
    # plus gradient tensors for the gen portion — factor ≈ 1.5 is appropriate.
    backward_factor = 1.5
    safety = 0.85

    per_prompt = (incremental_fwd + fp32_logits_extra_per_prompt) * backward_factor + kv_cache_per_prompt

    available = target_bytes - overhead - optimizer_mem
    # overhead already covers 1 batch unit; remaining fits max_extra more
    max_extra = max(0, int(available / max(per_prompt, 1) * safety))
    # Hard-cap at 6 to prevent runaway batch from fp32 logits blow-up
    optimal_batch = max(1, min(6, 1 + max_extra))
    grad_accum = max(1, round(effective_batch / optimal_batch))

    logger.info(
        f"[Calibration] overhead≈{overhead/1e9:.2f} GB | "
        f"incremental≈{incremental_fwd/1e9:.2f} GB/prompt | "
        f"per_prompt≈{per_prompt/1e9:.3f} GB | "
        f"available={available/1e9:.2f} GB | "
        f"→ batch={optimal_batch} × grad_accum={grad_accum} "
        f"(eff={optimal_batch * grad_accum}, target={effective_batch})"
    )

    return optimal_batch, grad_accum


# ==============================================================================
# GRPO Trainer
# ==============================================================================

class GRPOTrainer:
    """Custom GRPO training loop for QLoRA-based summarization.

    Implements the GRPO algorithm:
        1. Rollout: sample K completions per prompt from π_θ
        2. Reward: compute R_total = w_acc·R_acc + w_len·R_len
        3. Advantage: A = (R − μ_group) / σ_group
        4. Policy gradient: L = -min(ρ·A, clip(ρ)·A) + β·KL
    """

    def __init__(self, cfg: GRPOConfig, resume_from_checkpoint: Optional[str] = None):
        self.cfg = cfg
        self.accelerator = Accelerator(
            mixed_precision="bf16" if cfg.bf16 else "fp16" if cfg.fp16 else "no",
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models need left padding for generation
        if self.tokenizer.padding_side != 'left':
            logger.info(f"Setting tokenizer.padding_side from '{self.tokenizer.padding_side}' to 'left'")
            self.tokenizer.padding_side = 'left'
        # Qwen3/Qwen3.5: disable thinking mode to prevent <think> blocks in rollouts
        self._disable_thinking = cfg.disable_thinking
        if self._disable_thinking:
            logger.info("Thinking mode disabled for Qwen3-family model (GRPO rollouts).")

        # Load policy model (trainable) and reference model (frozen).
        # The reference is the GRPO *initial policy* (base + SFT adapter), not the raw
        # base model, so the KL term anchors to the SFT start point rather than dragging
        # the policy back toward base (which caused initial KL of 66–86).
        self.policy_model = self._load_policy_model(resume_checkpoint=resume_from_checkpoint)
        self.ref_model = self._load_reference_model(init_adapter=resume_from_checkpoint)

        # Data
        self.train_dataset = GRPODataset(cfg.train_data_path)
        self.val_dataset = None
        if cfg.val_data_path and os.path.isfile(cfg.val_data_path):
            self.val_dataset = GRPODataset(cfg.val_data_path)

        # Auto-calibrate batch size dựa trên VRAM (tương tự SFT)
        if self.accelerator.is_main_process and cfg.auto_calibrate_batch and torch.cuda.is_available():
            cal_batch, cal_accum = calibrate_grpo_batch_size(
                self.policy_model,
                self.ref_model,
                self.tokenizer,
                num_generations=cfg.num_generations,
                max_seq_length=cfg.max_seq_length,
                max_new_tokens=cfg.max_new_tokens,
                target_fraction=cfg.calibrate_target_vram_fraction,
                effective_batch=cfg.calibrate_effective_batch,
            )
            logger.info(
                f"[Calibration] batch: {cfg.per_device_train_batch_size} → {cal_batch} | "
                f"grad_accum: {cfg.gradient_accumulation_steps} → {cal_accum}"
            )
            cfg.per_device_train_batch_size = cal_batch
            cfg.gradient_accumulation_steps = cal_accum

        # Optimizer (LoRA params only)
        self.optimizer = torch.optim.AdamW(
            [p for n, p in self.policy_model.named_parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )

        # LR scheduler
        self.lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.total_steps,
        )

        # Prepare with accelerate
        self.policy_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.policy_model, self.optimizer, self.lr_scheduler
        )
        # reference model is not prepared (frozen, no training)

        # Recover global_step from checkpoint dir name (e.g. "checkpoint-500" → 500)
        self.global_step = 0
        if resume_from_checkpoint:
            dir_name = os.path.basename(resume_from_checkpoint.rstrip("/"))
            if dir_name.startswith("checkpoint-"):
                try:
                    self.global_step = int(dir_name.split("-")[1])
                    logger.info(f"Resuming from step {self.global_step}")
                except (IndexError, ValueError):
                    pass
        self.best_val_reward = -float("inf")

        # Persistent metrics logging + file log
        if self.accelerator.is_main_process:
            self.metrics_tracker = MetricsTracker(output_dir=cfg.output_dir)
            self.metrics_tracker.save_config(cfg.__dict__)
            # Mirror all logger output to output_dir/train.log so it's readable
            # with `tail -f` during training (PBS stdout is only flushed at job end).
            _log_path = os.path.join(cfg.output_dir, "train.log")
            _fh = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
            _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logging.getLogger().addHandler(_fh)
            logger.info(f"File logging enabled: {_log_path}")
        else:
            self.metrics_tracker = None

    # ------------------------------------------------------------------
    # Chat template helper
    # ------------------------------------------------------------------

    def _fmt_prompt(self, messages) -> str:
        """Apply chat template, disabling Qwen3 thinking if configured."""
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._disable_thinking:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_quantized_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load model in bf16 (or 4-bit QLoRA when load_in_4bit=True)."""
        # Probe flash-attn availability — same pattern as train_sft.py
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"
            logger.warning("flash-attn import failed — falling back to sdpa attention.")

        torch_dtype = getattr(torch, self.cfg.model.bnb_4bit_compute_dtype)

        if self.cfg.model.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.model.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=self.cfg.model.bnb_4bit_use_double_quant,
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        model.config.use_cache = False
        if hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1
        return model

    def _load_policy_model(self, resume_checkpoint: Optional[str] = None) -> PeftModel:
        """Load policy model: base + LoRA (trainable). Optionally resume from checkpoint."""
        base = self._load_quantized_model(self.cfg.model.model_name_or_path)
        if resume_checkpoint and os.path.isdir(resume_checkpoint):
            model = PeftModel.from_pretrained(base, resume_checkpoint, is_trainable=True)
            logger.info(f"Resumed policy model from {resume_checkpoint}")
        else:
            lora_config = LoraConfig(
                r=self.cfg.model.lora_r,
                lora_alpha=self.cfg.model.lora_alpha,
                target_modules=self.cfg.model.lora_target_modules,
                lora_dropout=self.cfg.model.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, lora_config)

        # Gradient checkpointing: saves ~50 GB VRAM by discarding activations
        # during forward and recomputing them during backward, at ~20% speed cost.
        # Essential for GRPO which holds 2 models + rollouts on GPU.
        if self.cfg.gradient_checkpointing:
            gc_kwargs = self.cfg.gradient_checkpointing_kwargs or {}
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            except TypeError:
                model.gradient_checkpointing_enable()
            model.config.use_cache = False
            logger.info(f"Gradient checkpointing enabled on policy model (kwargs={gc_kwargs}).")

        model.print_trainable_parameters()
        return model

    def _load_reference_model(self, init_adapter: Optional[str] = None) -> AutoModelForCausalLM:
        """Load the frozen reference model = the GRPO *initial policy*.

        For SFT-warm-started runs the reference must be base+SFT (the policy's start
        point), not the raw base model — otherwise the KL penalty pulls the policy away
        from the SFT solution. When init_adapter points to a LoRA adapter dir, load it on
        top of the base and merge so the reference sits exactly at the policy init. For
        fresh runs (no adapter) the reference is the base model, which is correct.
        """
        model = self._load_quantized_model(self.cfg.model.model_name_or_path)
        if init_adapter and os.path.isdir(init_adapter):
            model = PeftModel.from_pretrained(model, init_adapter)
            model = model.merge_and_unload()
            logger.info(f"Reference model = base + adapter ({init_adapter}), merged & frozen.")
        else:
            logger.info("Reference model = base model (no init adapter — fresh run).")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    # ------------------------------------------------------------------
    # Rollout: generate K completions per prompt
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_completions(
        self, prompts_text: List[str], num_return_sequences: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """Generate completions for a batch of prompts.

        Returns only gen_ids and decoded texts.  Log-probability computation
        is handled separately in train_step via a forward pass at T=1.0, which
        keeps old_logprobs and new_logprobs numerically consistent (avoiding
        overflow from temperature-scaled scores).
        """
        prompt_encodings = self.tokenizer(
            prompts_text,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_seq_length - self.cfg.max_new_tokens,
            return_tensors="pt",
        ).to(self.accelerator.device)

        # Generate in eval mode so LoRA dropout is OFF during rollout. With dropout on
        # (the main loop keeps the policy in train mode), completions are drawn from a
        # perturbed distribution that no longer matches the π_θ used for the log-prob/KL
        # forward passes — an off-policy mismatch that also measurably degrades quality
        # (R_acc 0.36→0.25 in the decoding-matrix probe). Restore the prior mode after.
        was_training = self.policy_model.training
        self.policy_model.eval()
        try:
            output_ids = self.policy_model.generate(
                **prompt_encodings,
                max_new_tokens=self.cfg.max_new_tokens,
                num_return_sequences=num_return_sequences,
                temperature=self.cfg.temperature if self.cfg.do_sample else 1.0,
                top_p=self.cfg.top_p if self.cfg.do_sample else 1.0,
                do_sample=self.cfg.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.cfg.repetition_penalty,
                no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            )
        finally:
            if was_training:
                self.policy_model.train()

        prompt_len = prompt_encodings.input_ids.shape[1]
        gen_ids = output_ids[:, prompt_len:]  # [B*K, gen_len]

        generated_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        return gen_ids, generated_texts

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------

    def _compute_grpo_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        token_ref_logp: torch.Tensor,
        advantages: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Vectorized GRPO loss: clipped policy gradient + per-token KL.

        Args:
            old_logprobs:  [B*K, gen_len] detached, computed at T=1.0 before update.
            new_logprobs:  [B*K, gen_len] with grad, current policy T=1.0.
            token_ref_logp: [B*K, gen_len] detached, reference model.
            advantages:    [B*K] group-normalised.
            gen_mask:      [B*K, gen_len] bool, 1 for real tokens up to first EOS.

        Returns:
            (loss_tensor, loss_dict)
        """
        # Log importance ratio — clamped to bf16-safe range (exp(10)≈22026 < 65504).
        log_ratio = (new_logprobs - old_logprobs).clamp(-10.0, 10.0)
        rho = torch.exp(log_ratio)  # [B*K, gen_len]

        adv = advantages.unsqueeze(-1)  # [B*K, 1]
        surr1 = rho * adv
        surr2 = rho.clamp(1 - self.cfg.epsilon, 1 + self.cfg.epsilon) * adv
        token_loss = -torch.min(surr1, surr2)  # [B*K, gen_len]

        # Per-token KL — clamped to same bf16-safe range.
        r = (token_ref_logp - new_logprobs).clamp(-10.0, 10.0)
        kl_token = torch.exp(r) - r - 1  # [B*K, gen_len]

        n_valid = gen_mask.sum().clamp(min=1)
        policy_loss = (token_loss * gen_mask).sum() / n_valid
        kl_penalty = (kl_token * gen_mask).sum() / n_valid

        total_loss = policy_loss + self.cfg.beta * kl_penalty

        return total_loss, {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl_penalty.item(),
        }

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_rewards(
        self,
        generated_texts: List[str],
        references: List[str],
        meta_list: List[Dict],
        num_gen: int,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Compute rewards for all generated completions.

        Args:
            generated_texts: List of generated texts [B*K].
            references: List of reference summaries [B].
            meta_list: List of metadata dicts [B].
            num_gen: K (completions per prompt).

        Returns:
            Tuple of (rewards_tensor [B*K], reward_details_list).
        """
        batch_size = len(references)
        rewards = torch.zeros(batch_size * num_gen, device=self.accelerator.device)
        details_list: List[Dict] = []

        for i in range(batch_size):
            ref = references[i]
            meta = meta_list[i]
            length_req = meta.get("length_requirement", "khoảng 50 từ")
            sent_req = meta.get("sentence_requirement", None)

            for k in range(num_gen):
                idx = i * num_gen + k
                gen = generated_texts[idx] if idx < len(generated_texts) else ""

                reward_dict = compute_all_rewards(
                    generated=gen,
                    reference=ref if ref else gen,  # fallback if no reference
                    length_requirement=length_req,
                    sentence_requirement=sent_req,
                    w_acc=self.cfg.reward_weight_accuracy,
                    w_len=self.cfg.reward_weight_length,
                    w_sent=self.cfg.reward_weight_sentence,
                )
                rewards[idx] = reward_dict["total"]
                details_list.append(reward_dict)

        return rewards, details_list

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single GRPO training step.

        Args:
            batch: Collated batch with keys: prompt, reference, meta.

        Returns:
            Dict of metrics.
        """
        num_gen = self.cfg.num_generations
        prompts = batch["prompt"]
        references = batch["reference"]
        meta_list = batch["meta"]
        batch_size = len(prompts)

        # Format prompts using chat template
        prompt_texts = [self._fmt_prompt(msg_list) for msg_list in prompts]

        # 1. ROLLOUT — generate K completions per prompt
        expanded_prompts = [pt for pt in prompt_texts for _ in range(num_gen)]
        with torch.no_grad():
            gen_ids, gen_texts = self._generate_completions(expanded_prompts, num_return_sequences=1)

        # 2. REWARD
        rewards, reward_details = self._compute_rewards(gen_texts, references, meta_list, num_gen)
        rewards_2d = rewards.view(batch_size, num_gen)

        # 3. ADVANTAGE: group normalization + optional length scaling
        mean_rewards = rewards_2d.mean(dim=-1, keepdim=True)
        # Clamp std to avoid dividing by near-zero when all rewards are identical.
        std_rewards = rewards_2d.std(dim=-1, keepdim=True).clamp(min=0.01)
        advantages_2d = (rewards_2d - mean_rewards) / std_rewards

        if self.cfg.length_advantage_alpha > 0.0:
            len_rewards = torch.tensor(
                [d["length"] for d in reward_details], device=self.accelerator.device,
            ).view(batch_size, num_gen)
            advantages_2d = advantages_2d * (1.0 + self.cfg.length_advantage_alpha * len_rewards)

        advantages = advantages_2d.flatten()  # [B*K]

        # 4. Build full sequence tensors (shared by all three forward passes)
        prompt_encodings = self.tokenizer(
            expanded_prompts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_seq_length - self.cfg.max_new_tokens,
            return_tensors="pt",
        ).to(self.accelerator.device)

        prompt_len = prompt_encodings.input_ids.shape[1]
        gen_len = gen_ids.shape[1]
        attn_gen = torch.ones(gen_ids.shape, dtype=torch.long, device=self.accelerator.device)
        full_ids = torch.cat([prompt_encodings.input_ids, gen_ids], dim=-1)
        full_mask = torch.cat([prompt_encodings.attention_mask, attn_gen], dim=-1)

        # EOS mask: 1 for real tokens (up to and including first EOS), 0 for padding after EOS.
        eos_id = self.tokenizer.eos_token_id
        gen_mask = torch.zeros(gen_ids.shape, dtype=torch.float, device=self.accelerator.device)
        for i in range(gen_ids.shape[0]):
            eos_pos = (gen_ids[i] == eos_id).nonzero(as_tuple=True)[0]
            end = eos_pos[0].item() + 1 if len(eos_pos) > 0 else gen_len
            gen_mask[i, :end] = 1.0

        # 5. REF LOGPROBS — reference model forward (no grad, frozen).
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=full_ids, attention_mask=full_mask)
            ref_gen_logits = ref_out.logits[:, prompt_len - 1: -1]
            ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
            token_ref_logp = ref_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).detach()
            del ref_out, ref_gen_logits, ref_log_probs
            torch.cuda.empty_cache()

        # 6. NEW LOGPROBS — policy forward with grad (for policy gradient and KL).
        # Use unwrap_model to bypass accelerate's bf16→fp32 output cast, which would
        # allocate an extra ~84 GB fp32 logit tensor at batch=12. The PEFT model's weights
        # and activations are already bf16, so the computation is identical; only the
        # output cast is skipped. Consistent with the calibration probe (which also calls
        # the model directly, before accelerator.prepare).
        new_out = self.accelerator.unwrap_model(self.policy_model)(
            input_ids=full_ids, attention_mask=full_mask
        )
        # .contiguous() creates a 3.7 GB copy of the gen slice instead of a view into the
        # full 44 GB logit tensor. This lets `del new_out` immediately free the 44 GB buffer;
        # without it, the autograd graph keeps the full tensor alive for SliceBackward,
        # causing ~41 GB extra VRAM during the backward pass.
        new_gen_logits = new_out.logits[:, prompt_len - 1: -1].contiguous()
        del new_out
        new_log_probs = F.log_softmax(new_gen_logits, dim=-1)
        new_logprobs = new_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # has grad
        del new_gen_logits, new_log_probs
        torch.cuda.empty_cache()

        # OLD LOGPROBS — on-policy training: generation and optimization use the same
        # weights (no optimizer step in between), so π_old = π_θ and rho = 1.
        # Using new_logprobs.detach() as old_lp gives rho = exp(new_lp - const) with
        # value 1 but gradient = advantages — the correct policy gradient signal.
        # This avoids an extra forward pass and eliminates the bf16 overflow risk from
        # temperature-scaled generation scores.
        old_logprobs = new_logprobs.detach()

        # 7. GRPO LOSS
        loss, loss_dict = self._compute_grpo_loss(
            old_logprobs, new_logprobs, token_ref_logp, advantages, gen_mask,
        )

        # 9. BACKWARD — divide by grad_accum so accumulated gradient = mean (not sum).
        grad_accum = max(1, self.cfg.gradient_accumulation_steps)
        self.accelerator.backward(loss / grad_accum)

        mean_len_reward = sum(d["length"] for d in reward_details) / max(len(reward_details), 1)
        return {
            **loss_dict,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_acc": sum(d["accuracy"] for d in reward_details) / max(len(reward_details), 1),
            "reward_len": mean_len_reward,
            "reward_sent": sum(d.get("sentence", 0.0) for d in reward_details) / max(len(reward_details), 1),
            "advantage_mean": advantages.mean().item(),
            "len_scale_mean": 1.0 + self.cfg.length_advantage_alpha * mean_len_reward,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_dataset: GRPODataset) -> Dict[str, float]:
        """Compute average reward on validation set (no training)."""
        self.policy_model.eval()
        total_rewards: List[float] = []

        for batch in DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn):
            # Generate K=2 for validation (faster)
            num_gen = min(2, self.cfg.num_generations)
            prompts = batch["prompt"]
            references = batch["reference"]
            meta_list = batch["meta"]
            batch_size = len(prompts)

            prompt_texts = [self._fmt_prompt(msg_list) for msg_list in prompts]
            expanded_prompts = [pt for pt in prompt_texts for _ in range(num_gen)]

            _, gen_texts = self._generate_completions(expanded_prompts, num_return_sequences=1)

            for i in range(batch_size):
                ref = references[i]
                meta = meta_list[i]
                for k in range(num_gen):
                    idx = i * num_gen + k
                    gen = gen_texts[idx] if idx < len(gen_texts) else ""
                    rd = compute_all_rewards(
                        generated=gen,
                        reference=ref if ref else gen,
                        length_requirement=meta.get("length_requirement", "khoảng 50 từ"),
                        sentence_requirement=meta.get("sentence_requirement", None),
                        w_acc=self.cfg.reward_weight_accuracy,
                        w_len=self.cfg.reward_weight_length,
                        w_sent=self.cfg.reward_weight_sentence,
                    )
                    total_rewards.append(rd["total"])

        self.policy_model.train()

        mean_reward = sum(total_rewards) / max(len(total_rewards), 1)
        return {"val_reward": mean_reward, "val_samples": len(total_rewards)}

    # ------------------------------------------------------------------
    # Checkpoint saving (model + optimizer + scheduler + metadata)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> str:
        """Save full training state to a checkpoint directory."""
        checkpoint_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Model weights (LoRA adapter)
        self.accelerator.unwrap_model(self.policy_model).save_pretrained(checkpoint_dir)

        # Optimizer & scheduler state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": step,
            "best_val_reward": self.best_val_reward,
        }, os.path.join(checkpoint_dir, "training_state.pt"))

        # Config snapshot
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(self.cfg.__dict__, f, default=str, indent=2)

        # Training metrics snapshot
        if self.metrics_tracker is not None:
            import shutil
            metrics_dir = self.metrics_tracker.metrics_dir
            if os.path.isdir(metrics_dir):
                shutil.copytree(metrics_dir, os.path.join(checkpoint_dir, "metrics"),
                                dirs_exist_ok=True)

        logger.info(f"Checkpoint saved: {checkpoint_dir} (step {step})")
        return checkpoint_dir

    # ------------------------------------------------------------------
    # Sample generation logging (log một vài summary mẫu ra file)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_sample_generations(self, val_metrics: Dict) -> None:
        """Generate và log một vài mẫu tóm tắt để kiểm tra chất lượng."""
        if not self.val_dataset or len(self.val_dataset) == 0:
            return

        # Lấy 3 mẫu đầu tiên từ val set
        val_samples = [self.val_dataset[i] for i in range(min(3, len(self.val_dataset)))]

        # eval mode → LoRA dropout off (same rationale as the rollout path).
        was_training = self.policy_model.training
        self.policy_model.eval()

        log_entries = []
        for sample in val_samples:
            prompt_text = self._fmt_prompt(sample["prompt"])
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                    max_length=self.cfg.max_seq_length - self.cfg.max_new_tokens).to(
                self.accelerator.device
            )
            output_ids = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.cfg.repetition_penalty,
                no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            )
            prompt_len = inputs.input_ids.shape[1]
            generated = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

            log_entries.append({
                "reference": sample["reference"][:200],
                "generated": generated[:200],
                "length_req": sample["meta"].get("length_requirement", ""),
                "sent_req": sample["meta"].get("sentence_requirement", ""),
            })

        # Ghi vào file JSONL
        samples_log_dir = os.path.join(self.cfg.output_dir, "sample_generations")
        os.makedirs(samples_log_dir, exist_ok=True)
        log_file = os.path.join(samples_log_dir, f"step_{self.global_step}.jsonl")
        with open(log_file, "w", encoding="utf-8") as f:
            for entry in log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if was_training:
            self.policy_model.train()

        logger.info(f"Sample generations saved: {log_file}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full GRPO training loop."""
        logger.info("Starting GRPO training...")
        self.policy_model.train()

        total_steps = self.cfg.total_steps
        # When resuming from a checkpoint that already passed total_steps, extend by
        # total_steps more so the job actually trains rather than exiting immediately.
        if self.global_step >= total_steps:
            extended = self.global_step + total_steps
            logger.warning(
                f"Resumed from step {self.global_step} which is >= total_steps {total_steps}. "
                f"Extending total_steps to {extended} (adding {total_steps} more steps)."
            )
            total_steps = extended
            self.cfg.total_steps = extended

        grad_accum = max(1, self.cfg.gradient_accumulation_steps)
        progress_bar = tqdm(total=total_steps, desc="GRPO", disable=not self.accelerator.is_main_process,
                            initial=self.global_step)

        # Accumulators for gradient accumulation
        _accum_step = 0
        _accum_metrics: List[Dict] = []
        _step_start = time.time()

        while self.global_step < total_steps:
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.cfg.dataloader_num_workers,
                pin_memory=True,
            )

            for batch in dataloader:
                if self.global_step >= total_steps:
                    break

                # Accumulate gradients
                metrics = self.train_step(batch)
                _accum_metrics.append(metrics)
                _accum_step += 1

                if _accum_step % grad_accum != 0:
                    continue  # accumulate more gradients before stepping

                # NaN guard: skip update if any gradient is non-finite.
                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.policy_model.parameters() if p.requires_grad
                )
                if has_nan_grad:
                    logger.warning(
                        f"Step {self.global_step + 1}: NaN/Inf gradients — skipping optimizer step"
                    )
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    _accum_step = 0
                    _accum_metrics = []
                    continue

                # Gradient clipping + optimizer step
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy_model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.update(1)

                # Average metrics across accumulation sub-steps
                avg = {k: sum(m[k] for m in _accum_metrics) / len(_accum_metrics)
                       for k in _accum_metrics[0]}
                avg["grad_norm"] = total_norm.item() if torch.is_tensor(total_norm) else float(total_norm)
                avg["lr"] = self.lr_scheduler.get_last_lr()[0]

                # Step time tracking
                step_time = time.time() - _step_start
                avg["step_time_s"] = step_time
                remaining = (total_steps - self.global_step) * step_time
                avg["eta_s"] = remaining
                _accum_metrics = []

                # Logging
                if self.global_step % self.cfg.logging_steps == 0 and self.accelerator.is_main_process:
                    log_msg = (
                        f"Step {self.global_step}/{total_steps} | "
                        f"Loss: {avg['loss']:.4f} | "
                        f"R_mean: {avg['reward_mean']:.4f} | "
                        f"R_acc: {avg['reward_acc']:.4f} | "
                        f"R_len: {avg['reward_len']:.4f} | "
                        f"R_sent: {avg['reward_sent']:.4f} | "
                        f"LenScale: {avg['len_scale_mean']:.3f} | "
                        f"KL: {avg['kl']:.4f} | "
                        f"Grad: {avg['grad_norm']:.4f} | "
                        f"LR: {avg['lr']:.2e} | "
                        f"{step_time:.1f}s/step"
                    )
                    logger.info(log_msg)
                    self.metrics_tracker.log_train(
                        step=self.global_step,
                        total_steps=total_steps,
                        loss=avg["loss"],
                        policy_loss=avg["policy_loss"],
                        kl=avg["kl"],
                        reward_mean=avg["reward_mean"],
                        reward_std=avg["reward_std"],
                        reward_acc=avg["reward_acc"],
                        reward_len=avg["reward_len"],
                        reward_sent=avg["reward_sent"],
                        advantage_mean=avg["advantage_mean"],
                        len_scale_mean=avg["len_scale_mean"],
                        grad_norm=avg["grad_norm"],
                        lr=avg["lr"],
                        step_time_s=step_time,
                        eta_s=remaining,
                    )

                    # WandB logging (nếu được bật)
                    if self.cfg.report_to == "wandb":
                        try:
                            import wandb
                            wandb.log({
                                "train/loss": avg["loss"],
                                "train/policy_loss": avg["policy_loss"],
                                "train/kl": avg["kl"],
                                "train/reward_mean": avg["reward_mean"],
                                "train/reward_acc": avg["reward_acc"],
                                "train/reward_len": avg["reward_len"],
                                "train/reward_sent": avg["reward_sent"],
                                "train/advantage_mean": avg["advantage_mean"],
                                "train/grad_norm": avg["grad_norm"],
                                "train/lr": avg["lr"],
                                "train/step_time_s": step_time,
                                "system/gpu_mem_gb": MetricsTracker._get_gpu_mem(),
                            }, step=self.global_step)
                        except Exception:
                            pass

                # Save checkpoint (model + optimizer + scheduler)
                if self.global_step % self.cfg.save_steps == 0 and self.accelerator.is_main_process:
                    self._save_checkpoint(self.global_step)

                # Validation
                if (
                    self.global_step % (self.cfg.save_steps * 2) == 0
                    and self.val_dataset is not None
                    and self.accelerator.is_main_process
                ):
                    val_metrics = self.validate(self.val_dataset)
                    logger.info(f"Val reward: {val_metrics['val_reward']:.4f} (n={val_metrics['val_samples']})")
                    self.metrics_tracker.log_eval(
                        step=self.global_step,
                        val_reward=val_metrics["val_reward"],
                        val_samples=val_metrics["val_samples"],
                    )

                    # Log generated summaries mẫu
                    self._log_sample_generations(val_metrics)

                    if val_metrics["val_reward"] > self.best_val_reward:
                        self.best_val_reward = val_metrics["val_reward"]
                        best_dir = os.path.join(self.cfg.output_dir, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        self.accelerator.unwrap_model(self.policy_model).save_pretrained(best_dir)
                        # Save optimizer/scheduler state for best model
                        torch.save({
                            "optimizer": self.optimizer.state_dict(),
                            "lr_scheduler": self.lr_scheduler.state_dict(),
                            "global_step": self.global_step,
                            "best_val_reward": self.best_val_reward,
                        }, os.path.join(best_dir, "training_state.pt"))
                        logger.info(f"New best model (reward={self.best_val_reward:.4f})")

                    # WandB eval logging
                    if self.cfg.report_to == "wandb":
                        try:
                            import wandb
                            wandb.log({
                                "eval/val_reward": val_metrics["val_reward"],
                                "eval/val_samples": val_metrics["val_samples"],
                            }, step=self.global_step)
                        except Exception:
                            pass

        progress_bar.close()

        # Save final
        if self.accelerator.is_main_process:
            self._save_checkpoint(self.global_step)
            # Copy checkpoint cuối cùng vào thư mục "final" cho dễ truy cập
            final_dir = os.path.join(self.cfg.output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            self.accelerator.unwrap_model(self.policy_model).save_pretrained(final_dir)
            # Copy training state
            last_ckpt = os.path.join(self.cfg.output_dir, f"checkpoint-{self.global_step}")
            if os.path.isdir(last_ckpt):
                state_file = os.path.join(last_ckpt, "training_state.pt")
                if os.path.isfile(state_file):
                    import shutil
                    shutil.copy2(state_file, os.path.join(final_dir, "training_state.pt"))

            # Ghi training summary
            summary = {
                "status": "completed",
                "total_steps": total_steps,
                "completed_steps": self.global_step,
                "best_val_reward": self.best_val_reward,
                "output_dir": self.cfg.output_dir,
                "config": self.cfg.__dict__,
            }
            with open(os.path.join(self.cfg.output_dir, "training_summary.json"), "w") as f:
                json.dump(summary, f, default=str, indent=2)

            self.metrics_tracker.close()
            logger.info(f"Final model saved: {final_dir}")
            logger.info(f"Metrics saved to: {self.metrics_tracker.metrics_dir}")
            logger.info("GRPO training complete!")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    def _bool(v):
        return str(v).lower() in ("true", "1", "yes")

    parser = argparse.ArgumentParser(description="GRPO for Vietnamese summarization")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/grpo_checkpoints")
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--total_steps", type=int, default=800)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_data", type=str, default="data/grpo_train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/grpo_val.jsonl")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    # Hardware flags (used by local/pbs shell scripts)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--bf16", type=_bool, default=None)
    parser.add_argument("--fp16", type=_bool, default=None)
    args = parser.parse_args()

    cfg = GRPOConfig(
        model=ModelConfig(model_name_or_path=args.model_name),
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        beta=args.beta,
        total_steps=args.total_steps,
        run_name=args.run_name,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )

    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
            # Apply top-level config overrides
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            # Apply nested model config overrides (prefixed with 'model_' or direct model attrs)
            for k, v in overrides.items():
                if k.startswith("model_") and hasattr(cfg.model, k[6:]):
                    setattr(cfg.model, k[6:], v)
                elif hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)

    # Apply hardware CLI flags (override JSON config if provided)
    if args.per_device_train_batch_size is not None:
        cfg.per_device_train_batch_size = args.per_device_train_batch_size
    if args.bf16 is not None:
        cfg.bf16 = args.bf16
    if args.fp16 is not None:
        cfg.fp16 = args.fp16

    trainer = GRPOTrainer(cfg, resume_from_checkpoint=args.resume)
    trainer.train()
