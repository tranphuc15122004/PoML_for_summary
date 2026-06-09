#!/usr/bin/env python
"""Evaluate all trained summarization models on the test set.

Auto-discovers available models (Qwen2.5-3B, Qwen3.5-4B, aug/no-aug).
Computes ROUGE-1/2/L, length error, length hit rate, BARTScore, G-Eval.

Usage:
    # Quick test — 20 samples, no LLM-based metrics
    PYTHONPATH=src python launch_eval.py --quick

    # Full eval (all discovered models)
    PYTHONPATH=src python launch_eval.py

    # Specific family
    PYTHONPATH=src python launch_eval.py --families qwen2.5
    PYTHONPATH=src python launch_eval.py --families qwen3.5

    # Custom model paths
    PYTHONPATH=src python launch_eval.py --models "base=/path/to/model,sft_aug=/path/to/adapter/final"

    # With PBS
    qsub scripts/pbs/eval.pbs

Results saved to models/eval_results/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch

from SFT_GRPO.evaluate import (
    load_eval_model,
    load_judge_model,
    evaluate_model,
    _print_summary_table,
)
from SFT_GRPO.config import EvalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Model path definitions
# ==============================================================================

BASE_DIR = "/g/data/hn98/dd9648/models"
MODELS_DIR = "models"

MODEL_FAMILIES = {
    "qwen2.5": {
        "base": os.path.join(BASE_DIR, "Qwen2.5-3B-Instruct"),
        "sft_aug": os.path.join(MODELS_DIR, "sft_aug_Qwen25_3B_2e", "final"),
        "sft_no_aug": os.path.join(MODELS_DIR, "sft_no_aug_Qwen25_3B_2e", "final"),
    },
    "qwen3.5": {
        "base": os.path.join(BASE_DIR, "Qwen3.5-4B"),
        "sft_aug": os.path.join(MODELS_DIR, "sft_aug_Qwen3.5-4B", "final"),
        "sft_no_aug": os.path.join(MODELS_DIR, "sft_no_aug_Qwen3.5-4B", "final"),
    },
}

JUDGE_MODEL_PATH = os.path.join(BASE_DIR, "Qwen3.5-4B")


def discover_models(families: str | None = None) -> Dict[str, str]:
    """Discover available model checkpoints.

    Args:
        families: Filter to 'qwen2.5', 'qwen3.5', or None for all.

    Returns:
        Dict of {display_name: model_path} for all available models.
    """
    models = {}

    family_keys = ["qwen2.5", "qwen3.5"]
    if families and families != "all":
        family_keys = [f for f in family_keys if families.lower() in f]

    for fkey in family_keys:
        family = MODEL_FAMILIES[fkey]
        # Short label prefix
        label = fkey.replace(".", "").upper()  # QWEN25 or QWEN35

        if os.path.isdir(family["base"]):
            models[f"{label}_base"] = family["base"]

        for variant in ["sft_aug", "sft_no_aug"]:
            p = family[variant]
            if os.path.isdir(p):
                vlabel = variant.replace("sft_", "")
                models[f"{label}_{vlabel}"] = p

    return models


def load_test_data(path: str, max_samples: int | None = None) -> List[Dict]:
    """Load test JSONL data."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break
    return data


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained summarization models"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated 'name=path' pairs (overrides auto-discovery)"
    )
    parser.add_argument(
        "--families", type=str, default=None,
        help="Model families: 'qwen2.5', 'qwen3.5', or 'all' (default)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 20 samples, no BARTScore/G-Eval"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit test samples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--test_data", type=str, default="data/test.jsonl",
        help="Test data path"
    )
    parser.add_argument(
        "--judge_model", type=str, default=JUDGE_MODEL_PATH,
        help="Judge model for BARTScore/G-Eval"
    )
    parser.add_argument(
        "--enable_bart_score", type=str, default=None,
        choices=["true", "false"],
        help="Enable BARTScore (default: true, false in --quick)"
    )
    parser.add_argument(
        "--enable_geval", type=str, default=None,
        choices=["true", "false"],
        help="Enable G-Eval (default: true, false in --quick)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Generation batch size"
    )
    args = parser.parse_args()

    # ── Resolve model paths ──────────────────────────────────────────────
    if args.models:
        model_paths = {}
        for pair in args.models.split(","):
            if "=" in pair:
                name, path = pair.split("=", 1)
                model_paths[name.strip()] = path.strip()
            else:
                model_paths[pair.strip()] = pair.strip()
    else:
        model_paths = discover_models(families=args.families)

    if not model_paths:
        logger.error("No models found to evaluate. Check model paths.")
        sys.exit(1)

    # Filter out non-existent paths
    valid_models = {k: v for k, v in model_paths.items() if os.path.isdir(v)}
    missing = {k: v for k, v in model_paths.items() if not os.path.isdir(v)}
    if missing:
        logger.warning(f"Skipping {len(missing)} model(s) with missing paths:")
        for k, v in missing.items():
            logger.warning(f"  {k}: {v}")
    model_paths = valid_models

    if not model_paths:
        logger.error("No valid models to evaluate.")
        sys.exit(1)

    logger.info(f"Models to evaluate ({len(model_paths)}):")
    for name, path in model_paths.items():
        logger.info(f"  ✅ {name}: {path}")

    # ── Resolve flags ────────────────────────────────────────────────────
    enable_bart = args.enable_bart_score
    enable_geval = args.enable_geval
    max_samples = args.max_samples

    if args.quick:
        max_samples = max_samples or 20
        if enable_bart is None:
            enable_bart = "false"
        if enable_geval is None:
            enable_geval = "false"

    enable_bart = enable_bart is None or enable_bart.lower() == "true"
    enable_geval = enable_geval is None or enable_geval.lower() == "true"

    # ── Load test data ───────────────────────────────────────────────────
    test_data = load_test_data(args.test_data, max_samples=max_samples)
    if max_samples:
        logger.info(f"Using {len(test_data)}/{max_samples} test samples")
    else:
        logger.info(f"Loaded {len(test_data)} test samples")

    # ── Load judge model (shared) ────────────────────────────────────────
    judge_model, judge_tokenizer = None, None
    if enable_bart or enable_geval:
        logger.info(f"Loading judge model: {args.judge_model}")
        try:
            judge_model, judge_tokenizer = load_judge_model(args.judge_model)
        except Exception as e:
            logger.warning(f"Failed to load judge model: {e}")
            logger.warning("BARTScore and G-Eval will be skipped.")
            enable_bart = False
            enable_geval = False

    # ── Evaluate each model ──────────────────────────────────────────────
    results: Dict[str, Dict] = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name, model_path in model_paths.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"  Path: {model_path}")
        logger.info(f"{'=' * 60}")

        try:
            # Auto-detect base model for LoRA adapters (base_model_path=None)
            model, tokenizer = load_eval_model(
                model_path,
                base_model_path=None,
                load_in_4bit=True,
            )

            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                model_name=model_name,
                max_new_tokens=256,
                batch_size=args.batch_size,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                enable_bart_score=enable_bart,
                enable_geval=enable_geval,
            )
            results[model_name] = metrics

            logger.info(
                f"[{model_name}] R1={metrics['rouge_1']:.4f}  "
                f"R2={metrics['rouge_2']:.4f}  "
                f"RL={metrics['rouge_l']:.4f}  "
                f"LenErr={metrics['length_error']:.1f}  "
                f"HitRate={metrics['length_hit_rate']:.2%}  "
                f"BART={metrics['bart_score']:.3f}  "
                f"GEval={metrics['geval_avg']:.4f}  "
                f"AvgLen={metrics['avg_gen_length']:.1f}"
            )

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}", exc_info=True)
            results[model_name] = {"model": model_name, "error": str(e)}

    # ── Print summary ────────────────────────────────────────────────────
    print_summary(results, args.output_dir)

    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {results_path}")


def print_summary(results: Dict[str, Dict], output_dir: str):
    """Print comparison table and save summary."""
    print(f"\n{'=' * 100}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 100}")

    header = (
        f"{'Model':<26} {'N':>6} {'R-1':>7} {'R-2':>7} {'R-L':>7} "
        f"{'LenErr':>8} {'HitRate':>9} {'BART':>8} {'GEval':>7} {'AvgLen':>8}"
    )
    print(header)
    print("-" * len(header))

    for model_name in results:
        m = results[model_name]
        if "error" in m:
            print(f"{model_name:<26} ERROR: {m['error']}")
            continue

        def fmt(v):
            if isinstance(v, float) and np.isnan(v):
                return f"{'N/A':>7}"
            return f"{v:>7.4f}"

        print(
            f"{model_name:<26} {m['samples']:>6} {fmt(m['rouge_1'])} {fmt(m['rouge_2'])} "
            f"{fmt(m['rouge_l'])} {m['length_error']:>8.1f} {m['length_hit_rate']:>9.2%} "
            f"{m['bart_score']:>8.3f} {fmt(m['geval_avg'])} {m['avg_gen_length']:>8.1f}"
        )

    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
