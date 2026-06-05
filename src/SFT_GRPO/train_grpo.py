#!/usr/bin/env python
"""
GRPO (Group Relative Policy Optimization) for Vietnamese summarization.

Optimizes the SFT model using multi-objective rewards:
    - Accuracy (ROUGE-L F1)
    - Length adherence
    - Style adherence (LLM-as-Judge)

Usage:
    python src/SFT_GRPO/train_grpo.py
    python src/SFT_GRPO/train_grpo.py --resume models/grpo_checkpoints/checkpoint-100
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset as HFDataset, load_dataset
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
# GRPO Trainer
# ==============================================================================

class GRPOTrainer:
    """Custom GRPO training loop for QLoRA-based summarization.

    Implements the GRPO algorithm:
        1. Rollout: sample K completions per prompt from π_θ
        2. Reward: compute R_total = w_acc·R_acc + w_len·R_len + w_style·R_style
        3. Advantage: A = (R − μ_group) / σ_group
        4. Policy gradient: L = -min(ρ·A, clip(ρ)·A) + β·KL
    """

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
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

        # Load policy model (trainable) and reference model (frozen)
        self.policy_model = self._load_policy_model()
        self.ref_model = self._load_reference_model()

        # Data
        self.train_dataset = GRPODataset(cfg.train_data_path)
        self.val_dataset = None
        if cfg.val_data_path and os.path.isfile(cfg.val_data_path):
            self.val_dataset = GRPODataset(cfg.val_data_path)

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

        self.global_step = 0
        self.best_val_reward = -float("inf")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_quantized_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load a model with 4-bit quantization for GRPO."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.model.load_in_4bit,
            bnb_4bit_quant_type=self.cfg.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.cfg.model.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.cfg.model.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, self.cfg.model.bnb_4bit_compute_dtype),
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        return model

    def _load_policy_model(self) -> PeftModel:
        """Load policy model: base + LoRA (trainable)."""
        base = self._load_quantized_model(self.cfg.model.model_name_or_path)
        lora_config = LoraConfig(
            r=self.cfg.model.lora_r,
            lora_alpha=self.cfg.model.lora_alpha,
            target_modules=self.cfg.model.lora_target_modules,
            lora_dropout=self.cfg.model.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_config)
        model.print_trainable_parameters()
        return model

    def _load_reference_model(self) -> AutoModelForCausalLM:
        """Load reference model (frozen, no gradients)."""
        model = self._load_quantized_model(self.cfg.model.model_name_or_path)
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
    ) -> Tuple[List[torch.Tensor], List[str], List[List[float]]]:
        """Generate completions for a batch of prompts.

        Args:
            prompts_text: List of prompt strings (already formatted with chat template).
            num_return_sequences: K completions per prompt.

        Returns:
            Tuple of (prompt_logprobs, generated_texts, all_token_logprobs_2d)
        """
        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts_text,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_seq_length - self.cfg.max_new_tokens,
            return_tensors="pt",
        ).to(self.accelerator.device)

        # Generate
        output_ids = self.policy_model.generate(
            **prompt_encodings,
            max_new_tokens=self.cfg.max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Extract generated token ids (excluding prompt)
        prompt_len = prompt_encodings.input_ids.shape[1]
        gen_ids = output_ids.sequences[:, prompt_len:]  # [B*K, gen_len]

        # Decode
        generated_texts = self.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )

        # Compute log probabilities of generated tokens
        # logprobs for each generated token
        logprobs_list = []
        for seq_idx in range(gen_ids.shape[0]):
            seq = gen_ids[seq_idx]
            seq_logprobs = []
            for step_idx in range(len(seq)):
                # scores[step_idx] shape: [B*K, vocab_size]
                logits = output_ids.scores[step_idx][seq_idx]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_prob = log_probs[seq[step_idx]].item()
                seq_logprobs.append(token_log_prob)
            logprobs_list.append(seq_logprobs)

        return output_ids.sequences, generated_texts, logprobs_list

    # ------------------------------------------------------------------
    # KL divergence computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_kl(
        self,
        prompt_ids: torch.Tensor,
        gen_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unbiased KL between policy and reference model.

        KL = exp(log π_ref − log π_θ) - (log π_ref − log π_θ) - 1

        Args:
            prompt_ids: Tokenized prompts [B, prompt_len]
            gen_ids: Generated completion token ids [B*K, gen_len]

        Returns:
            Mean KL per batch.
        """
        full_input_ids = torch.cat([prompt_ids, gen_ids], dim=-1)
        attention_mask = torch.ones_like(full_input_ids)

        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits  # [B, seq_len, vocab]

        # Policy logits (with gradients)
        policy_outputs = self.policy_model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
        )
        policy_logits = policy_outputs.logits

        # Only compute KL on generated tokens
        prompt_len = prompt_ids.shape[1]
        gen_logprobs_ref = F.log_softmax(ref_logits[:, prompt_len - 1 : -1], dim=-1)
        gen_logprobs_policy = F.log_softmax(policy_logits[:, prompt_len - 1 : -1], dim=-1)

        gen_tokens = full_input_ids[:, prompt_len:]
        logp_ref = gen_logprobs_ref.gather(dim=-1, index=gen_tokens.unsqueeze(-1)).squeeze(-1)
        logp_policy = gen_logprobs_policy.gather(dim=-1, index=gen_tokens.unsqueeze(-1)).squeeze(-1)

        # Unbiased KL estimator: exp(r) - r - 1 where r = logp_ref - logp_policy
        r = logp_ref - logp_policy.detach()  # detach policy for stable KL
        kl = torch.exp(r) - r - 1
        return kl.mean()

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------

    def _compute_grpo_loss(
        self,
        old_logprobs: List[List[float]],
        new_logprobs: List[List[float]],
        advantages: torch.Tensor,
        kl_penalty: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss (clipped policy gradient + KL).

        Args:
            old_logprobs: Log probs from rollout policy π_old.
            new_logprobs: Log probs from current policy π_θ.
            advantages: Group-normalized advantages [B*K].
            kl_penalty: KL divergence penalty value.

        Returns:
            Tuple of (loss_tensor, loss_dict).
        """
        device = advantages.device

        # Compute importance ratio: ρ = exp(log π_θ − log π_old)
        policy_loss_sum = 0.0
        n_tokens = 0

        for i in range(len(old_logprobs)):
            old_lp = torch.tensor(old_logprobs[i], device=device)
            new_lp = torch.tensor(new_logprobs[i], device=device)
            rho = torch.exp(new_lp - old_lp)  # importance ratio per token
            adv = advantages[i]

            # Clipped surrogate objective (per-token)
            surr1 = rho * adv
            surr2 = torch.clamp(rho, 1 - self.cfg.epsilon, 1 + self.cfg.epsilon) * adv

            token_loss = -torch.min(surr1, surr2).sum()
            policy_loss_sum += token_loss
            n_tokens += len(old_logprobs[i])

        policy_loss = policy_loss_sum / max(n_tokens, 1)

        # Total loss = policy gradient + KL penalty
        total_loss = policy_loss + self.cfg.beta * kl_penalty

        loss_dict = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl_penalty.item(),
        }

        return total_loss, loss_dict

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
            style = meta.get("style", "báo chí")

            for k in range(num_gen):
                idx = i * num_gen + k
                gen = generated_texts[idx] if idx < len(generated_texts) else ""

                reward_dict = compute_all_rewards(
                    generated=gen,
                    reference=ref if ref else gen,  # fallback if no reference
                    length_requirement=length_req,
                    style=style,
                    judge_pipeline=self._style_judge if self.cfg.reward_weight_style > 0 else None,
                    w_acc=self.cfg.reward_weight_accuracy,
                    w_len=self.cfg.reward_weight_length,
                    w_style=self.cfg.reward_weight_style,
                )
                rewards[idx] = reward_dict["total"]
                details_list.append(reward_dict)

        return rewards, details_list

    def _style_judge(self, prompt: str, max_tokens: int = 5) -> List[Dict]:
        """LLM-as-Judge for style evaluation.

        Uses the reference model (frozen) to score style adherence.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(
            self.accelerator.device
        )
        with torch.no_grad():
            outputs = self.ref_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [{"generated_text": text}]

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
        prompt_texts = []
        for msg_list in prompts:
            text = self.tokenizer.apply_chat_template(
                msg_list, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)

        # 1. ROLLOUT: generate K completions per prompt with π_old
        with torch.no_grad():
            # Replicate each prompt K times for generation
            expanded_prompts = []
            for pt in prompt_texts:
                expanded_prompts.extend([pt] * num_gen)

            gen_ids, gen_texts, old_logprobs = self._generate_completions(
                expanded_prompts, num_return_sequences=1
            )

        # 2. REWARD: compute R_total for each completion
        rewards, reward_details = self._compute_rewards(
            gen_texts, references, meta_list, num_gen
        )
        # Reshape: [B, K]
        rewards_2d = rewards.view(batch_size, num_gen)

        # 3. ADVANTAGE: group normalization
        mean_rewards = rewards_2d.mean(dim=-1, keepdim=True)  # [B, 1]
        std_rewards = rewards_2d.std(dim=-1, keepdim=True) + 1e-8  # [B, 1]
        advantages_2d = (rewards_2d - mean_rewards) / std_rewards  # [B, K]
        advantages = advantages_2d.flatten()  # [B*K]

        # 4. Compute new logprobs (with gradients) and KL
        prompt_encodings = self.tokenizer(
            expanded_prompts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_seq_length - self.cfg.max_new_tokens,
            return_tensors="pt",
        ).to(self.accelerator.device)

        # Compute new logprobs for generated tokens
        full_ids = torch.cat([prompt_encodings.input_ids, gen_ids], dim=-1)
        attention_mask = torch.ones_like(full_ids)

        outputs = self.policy_model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        prompt_len = prompt_encodings.input_ids.shape[1]
        gen_logits = logits[:, prompt_len - 1: -1]  # align with generated tokens
        gen_tokens = gen_ids

        log_probs = F.log_softmax(gen_logits, dim=-1)
        new_logprobs_list = []
        for seq_idx in range(gen_tokens.shape[0]):
            seq = gen_tokens[seq_idx]
            seq_logprobs = []
            for step_idx in range(len(seq)):
                lp = log_probs[seq_idx, step_idx, seq[step_idx]].item()
                seq_logprobs.append(lp)
            new_logprobs_list.append(seq_logprobs)

        # KL divergence (reference vs policy)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=full_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits
            ref_gen_logits = ref_logits[:, prompt_len - 1: -1]
            ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)

            kl_sum = 0.0
            kl_count = 0
            for seq_idx in range(gen_tokens.shape[0]):
                for step_idx in range(len(gen_tokens[seq_idx])):
                    r = ref_log_probs[seq_idx, step_idx, gen_tokens[seq_idx, step_idx]] - \
                        log_probs[seq_idx, step_idx, gen_tokens[seq_idx, step_idx]]
                    kl_sum += (torch.exp(r) - r - 1).item()
                    kl_count += 1
            kl_value = kl_sum / max(kl_count, 1)

        # 5. GRPO LOSS
        loss, loss_dict = self._compute_grpo_loss(
            old_logprobs,
            new_logprobs_list,
            advantages,
            torch.tensor(kl_value, device=self.accelerator.device),
        )

        # 6. BACKWARD + UPDATE
        self.accelerator.backward(loss)

        # Gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        # Metrics
        metrics = {
            **loss_dict,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_acc": sum(d["accuracy"] for d in reward_details) / max(len(reward_details), 1),
            "reward_len": sum(d["length"] for d in reward_details) / max(len(reward_details), 1),
            "reward_style": sum(d["style"] for d in reward_details) / max(len(reward_details), 1),
            "advantage_mean": advantages.mean().item(),
            "grad_norm": total_norm,
            "lr": self.lr_scheduler.get_last_lr()[0],
        }
        return metrics

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

            prompt_texts = [
                self.tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
                for msg_list in prompts
            ]
            expanded_prompts = [pt for pt in prompt_texts for _ in range(num_gen)]

            _, gen_texts, _ = self._generate_completions(expanded_prompts, num_return_sequences=1)

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
                        style=meta.get("style", "báo chí"),
                        judge_pipeline=self._style_judge if self.cfg.reward_weight_style > 0 else None,
                        w_acc=self.cfg.reward_weight_accuracy,
                        w_len=self.cfg.reward_weight_length,
                        w_style=self.cfg.reward_weight_style,
                    )
                    total_rewards.append(rd["total"])

        self.policy_model.train()

        mean_reward = sum(total_rewards) / max(len(total_rewards), 1)
        return {"val_reward": mean_reward, "val_samples": len(total_rewards)}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full GRPO training loop."""
        logger.info("Starting GRPO training...")
        self.policy_model.train()

        # Progress bar
        total_steps = self.cfg.total_steps
        progress_bar = tqdm(total=total_steps, desc="GRPO", disable=not self.accelerator.is_main_process)

        while self.global_step < total_steps:
            # Create dataloader each epoch (shuffled)
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.per_device_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

            for batch in dataloader:
                if self.global_step >= total_steps:
                    break

                # Training step
                metrics = self.train_step(batch)
                self.global_step += 1
                progress_bar.update(1)

                # Logging
                if self.global_step % self.cfg.logging_steps == 0 and self.accelerator.is_main_process:
                    log_msg = (
                        f"Step {self.global_step}/{total_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"R_mean: {metrics['reward_mean']:.4f} | "
                        f"R_acc: {metrics['reward_acc']:.4f} | "
                        f"R_len: {metrics['reward_len']:.4f} | "
                        f"R_style: {metrics['reward_style']:.4f} | "
                        f"KL: {metrics['kl']:.4f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )
                    logger.info(log_msg)

                # Save checkpoint
                if self.global_step % self.cfg.save_steps == 0 and self.accelerator.is_main_process:
                    checkpoint_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{self.global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.accelerator.unwrap_model(self.policy_model).save_pretrained(checkpoint_dir)
                    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                        json.dump(self.cfg.__dict__, f, default=str, indent=2)
                    logger.info(f"Checkpoint saved: {checkpoint_dir}")

                # Validation
                if (
                    self.global_step % (self.cfg.save_steps * 2) == 0
                    and self.val_dataset is not None
                    and self.accelerator.is_main_process
                ):
                    val_metrics = self.validate(self.val_dataset)
                    logger.info(f"Val reward: {val_metrics['val_reward']:.4f} (n={val_metrics['val_samples']})")
                    if val_metrics["val_reward"] > self.best_val_reward:
                        self.best_val_reward = val_metrics["val_reward"]
                        best_dir = os.path.join(self.cfg.output_dir, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        self.accelerator.unwrap_model(self.policy_model).save_pretrained(best_dir)
                        logger.info(f"New best model (reward={self.best_val_reward:.4f})")

        progress_bar.close()

        # Save final
        if self.accelerator.is_main_process:
            final_dir = os.path.join(self.cfg.output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            self.accelerator.unwrap_model(self.policy_model).save_pretrained(final_dir)
            logger.info(f"Final model saved: {final_dir}")
            logger.info("GRPO training complete!")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

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
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    trainer = GRPOTrainer(cfg)
    trainer.train()
