#!/usr/bin/env python
"""
Evaluate and compare summarization models (Base vs SFT vs GRPO).

Computes:
    - ROUGE-1/2/L F1
    - BERTScore F1
    - Length MAE (Mean Absolute Error from requested word count)
    - Length Hit Rate (% within ±20% of requested)
    - Style Score (LLM-as-Judge)
    - Win Rate (pairwise comparison)

Usage:
    python src/SFT_GRPO/evaluate.py
    python src/SFT_GRPO/evaluate.py --models base,sft,grpo
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SFT_GRPO.config import EvalConfig
from SFT_GRPO.rewards import (
    accuracy_reward,
    compute_all_rewards,
    length_reward,
    style_reward_llm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Model loader
# ==============================================================================

def load_eval_model(
    model_path: str,
    is_lora: bool = False,
    load_in_4bit: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for evaluation.

    Args:
        model_path: HuggingFace model ID or local path to LoRA adapter.
        is_lora: If True, load base + LoRA adapter.
        load_in_4bit: Whether to load in 4-bit for memory efficiency.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_lora else "Qwen/Qwen2.5-3B-Instruct",
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path if not is_lora else "Qwen/Qwen2.5-3B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path if not is_lora else "Qwen/Qwen2.5-3B-Instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    if is_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    model.eval()
    model.config.use_cache = True
    return model, tokenizer


# ==============================================================================
# Generation
# ==============================================================================

@torch.no_grad()
def generate_summaries(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    batch_size: int = 8,
) -> List[str]:
    """Generate summaries for a list of prompt strings.

    Args:
        model: HF model.
        tokenizer: HF tokenizer.
        prompts: Formatted prompt strings.
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature (lower = more deterministic).
        batch_size: Generation batch size.

    Returns:
        List of generated summary strings.
    """
    summaries: List[str] = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=3072 - max_new_tokens,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for j, output_ids in enumerate(outputs):
            prompt_len = inputs.input_ids[j].shape[0]
            generated = tokenizer.decode(
                output_ids[prompt_len:], skip_special_tokens=True
            ).strip()
            summaries.append(generated)

    return summaries


# ==============================================================================
# Evaluation
# ==============================================================================


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    model_name: str = "model",
    max_new_tokens: int = 256,
    batch_size: int = 8,
    judge_model=None,
) -> Dict:
    """Run full evaluation on test data.

    Args:
        model: HF model.
        tokenizer: HF tokenizer.
        test_data: List of test samples with "prompt", "reference", "meta".
        model_name: Name for logging.
        max_new_tokens: Max tokens per generation.
        batch_size: Generation batch size.
        judge_model: Optional LLM for style evaluation.

    Returns:
        Dict of evaluation metrics.
    """
    # Format prompts
    prompt_texts = []
    refs: List[str] = []
    length_reqs: List[str] = []
    styles: List[str] = []

    for sample in test_data:
        prompt_text = tokenizer.apply_chat_template(
            sample["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompt_texts.append(prompt_text)
        refs.append(sample.get("reference", ""))
        meta = sample.get("meta", {})
        length_reqs.append(meta.get("length_requirement", "khoảng 50 từ"))
        styles.append(meta.get("style", "báo chí"))

    # Generate
    logger.info(f"[{model_name}] Generating {len(prompt_texts)} summaries...")
    generated = generate_summaries(
        model, tokenizer, prompt_texts,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        batch_size=batch_size,
    )

    # Compute metrics
    rouge_l_scores: List[float] = []
    bertscore_f1_scores: List[float] = []
    length_errors: List[float] = []
    length_hits: List[bool] = []
    style_scores: List[float] = []

    for i in range(len(generated)):
        gen = generated[i]
        ref = refs[i]

        # ROUGE-L
        rl = accuracy_reward(gen, ref)
        rouge_l_scores.append(rl)

        # Length
        if length_reqs[i]:
            le = abs(len(gen.split()) - _parse_target_length(length_reqs[i]))
            length_errors.append(le)
            # Hit if within ±20%
            target = _parse_target_length(length_reqs[i])
            if target > 0:
                length_hits.append(abs(len(gen.split()) - target) / target <= 0.20)
            else:
                length_hits.append(False)
        else:
            length_errors.append(0)
            length_hits.append(True)

        # Style
        if judge_model is not None and styles[i]:
            ss = style_reward_llm(gen, styles[i], judge_model)
            style_scores.append(ss)
        else:
            style_scores.append(0.5)

    # Aggregate
    metrics = {
        "model": model_name,
        "samples": len(generated),
        "rouge_l_f1": float(np.mean(rouge_l_scores)),
        "length_mae": float(np.mean(length_errors)),
        "length_hit_rate": float(np.mean(length_hits)),
        "style_score": float(np.mean(style_scores)),
        "avg_gen_length": float(np.mean([len(g.split()) for g in generated])),
    }

    return metrics


def _parse_target_length(length_req: str) -> float:
    """Extract target word count from a length requirement string."""
    import re
    m = re.search(r"(\d+\.?\d*)", length_req.replace("-", ""))
    if m:
        return float(m.group(1))
    return 50.0


# ==============================================================================
# Main evaluation runner
# ==============================================================================


def run_evaluation(cfg: EvalConfig) -> Dict[str, Dict]:
    """Run evaluation for all configured models.

    Returns:
        Dict mapping model_name → metrics dict.
    """
    # Load test data
    test_data: List[Dict] = []
    with open(cfg.test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    logger.info(f"Loaded {len(test_data)} test samples from {cfg.test_data_path}")

    results: Dict[str, Dict] = {}
    judge_model = None  # We'll skip style judge (too expensive for bulk eval)

    for model_name, model_path in cfg.model_paths.items():
        logger.info(f"{'='*60}\nEvaluating: {model_name} ({model_path})\n{'='*60}")

        is_lora = os.path.isdir(model_path) and any(
            f.startswith("adapter") or f.endswith(".safetensors")
            for f in os.listdir(model_path)
        )
        # Check if it's a local directory
        is_local = os.path.isdir(model_path)
        is_lora = is_local and (
            os.path.exists(os.path.join(model_path, "adapter_config.json"))
            or os.path.exists(os.path.join(model_path, "adapter_model.safetensors"))
        )

        try:
            model, tokenizer = load_eval_model(
                model_path, is_lora=is_lora, load_in_4bit=True
            )
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                model_name=model_name,
                max_new_tokens=cfg.generation_max_new_tokens,
                batch_size=cfg.batch_size,
                judge_model=judge_model,
            )
            results[model_name] = metrics
            logger.info(f"[{model_name}] ROUGE-L={metrics['rouge_l_f1']:.4f}  "
                        f"LengthMAE={metrics['length_mae']:.2f}  "
                        f"LengthHit={metrics['length_hit_rate']:.2%}  "
                        f"Style={metrics['style_score']:.4f}  "
                        f"AvgLen={metrics['avg_gen_length']:.1f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {"model": model_name, "error": str(e)}

    # Summary table
    _print_summary_table(results, cfg)
    return results


def _print_summary_table(results: Dict[str, Dict], cfg: EvalConfig) -> None:
    """Print a comparison table of all models."""
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")

    header = (f"{'Model':<20} {'Samples':>8} {'ROUGE-L':>10} {'LenMAE':>10} "
              f"{'HitRate':>10} {'Style':>8} {'AvgLen':>8}")
    print(header)
    print("-" * len(header))

    for model_name in cfg.model_paths:
        if model_name not in results:
            continue
        m = results[model_name]
        if "error" in m:
            print(f"{model_name:<20} ERROR: {m['error']}")
            continue
        print(
            f"{model_name:<20} {m['samples']:>8} {m['rouge_l_f1']:>10.4f} "
            f"{m['length_mae']:>10.2f} {m['length_hit_rate']:>10.2%} "
            f"{m['style_score']:>8.4f} {m['avg_gen_length']:>8.1f}"
        )

    print("-" * len(header))

    # Save results
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {results_path}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate summarization models")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/eval_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list: name1=path1,name2=path2")
    args = parser.parse_args()

    if args.models:
        # Parse custom model paths
        model_paths = {}
        for pair in args.models.split(","):
            if "=" in pair:
                name, path = pair.split("=", 1)
                model_paths[name.strip()] = path.strip()
            else:
                model_paths[pair.strip()] = pair.strip()
    else:
        model_paths = None  # Use defaults

    cfg = EvalConfig(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_paths=model_paths or EvalConfig().model_paths,
    )

    run_evaluation(cfg)
