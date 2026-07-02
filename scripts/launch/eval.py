#!/usr/bin/env python
"""Evaluate all trained summarization models on the test set.

Auto-discovers available models (Qwen2.5-3B, Qwen3.5-4B, aug/no-aug).
Computes ROUGE-2, length error, BARTScore.

Usage:
    # Quick test — 20 samples, no BARTScore
    PYTHONPATH=src python scripts/launch/eval.py --quick

    # Full eval (all discovered models)
    PYTHONPATH=src python scripts/launch/eval.py

    # Specific family
    PYTHONPATH=src python scripts/launch/eval.py --families qwen2.5

    # Custom model paths
    PYTHONPATH=src python scripts/launch/eval.py --models "base=/path/to/model,sft=/path/to/adapter/final"

    # With PBS
    qsub scripts/pbs/eval.pbs

Results saved to models/eval_results/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from typing import Dict, List, Optional

# bitsandbytes 0.44.x references triton.ops which was removed in triton 2.x.
if "triton.ops" not in sys.modules:
    _triton_ops = types.ModuleType("triton.ops")
    _triton_perf = types.ModuleType("triton.ops.matmul_perf_model")
    _triton_perf.early_config_prune = lambda *a, **kw: None
    _triton_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    sys.modules["triton.ops"] = _triton_ops
    sys.modules["triton.ops.matmul_perf_model"] = _triton_perf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from evaluation.evaluate import (
    EvalConfig,
    ModelEntry,
    DatasetEntry,
    load_eval_model,
    evaluate_model_on_dataset,
    aggregate,
    save_results,
    _load_test_data,
)

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
        "base":       os.path.join(BASE_DIR, "Qwen2.5-3B-Instruct"),
        "sft_aug":    os.path.join(MODELS_DIR, "sft_aug_Qwen25_3B_2e", "final"),
        "sft_no_aug": os.path.join(MODELS_DIR, "sft_no_aug_Qwen25_3B_2e", "final"),
    },
    "qwen3.5": {
        "base":       os.path.join(BASE_DIR, "Qwen3.5-4B"),
        "sft_aug":    os.path.join(MODELS_DIR, "sft_aug_Qwen3.5-4B", "final"),
        "sft_no_aug": os.path.join(MODELS_DIR, "sft_no_aug_Qwen3.5-4B", "final"),
    },
    "qwen3_base": {
        "base":  os.path.join(BASE_DIR, "Qwen3-4B-Base"),
        "sft":   os.path.join(MODELS_DIR, "sft_qwen3_4b_base", "final"),
        # GRPO from base (fresh)
        "grpo_fresh_v3": os.path.join(MODELS_DIR, "grpo_qwen3_4b_base_fresh_v3", "best"),
        "grpo_fresh_v4": os.path.join(MODELS_DIR, "grpo_qwen3_4b_base_fresh_v4", "final"),
        "grpo_fresh_v5": os.path.join(MODELS_DIR, "grpo_qwen3_4b_base_fresh_v5", "final"),
        # GRPO from SFT checkpoint
        "grpo_sft_v3":   os.path.join(MODELS_DIR, "grpo_qwen3_4b_base_sft_v3", "best"),
        "grpo_sft_v4":   os.path.join(MODELS_DIR, "grpo_qwen3_4b_base_sft_v4", "final"),
        "grpo_sft_v5":   os.path.join(MODELS_DIR, "grpo_qwen3_4b_base_sft_v5", "final"),
    },
    "qwen3_instruct": {
        "base":  os.path.join(BASE_DIR, "Qwen3-4B"),
        "sft":   os.path.join(MODELS_DIR, "sft_qwen3_4b_instruct", "final"),
        # GRPO from base (fresh)
        "grpo_fresh_v3": os.path.join(MODELS_DIR, "grpo_qwen3_4b_instruct_fresh_v3", "best"),
        "grpo_fresh_v4": os.path.join(MODELS_DIR, "grpo_qwen3_4b_instruct_fresh_v4", "final"),
        "grpo_fresh_v5": os.path.join(MODELS_DIR, "grpo_qwen3_4b_instruct_fresh_v5", "final"),
        # GRPO from SFT checkpoint
        "grpo_sft_v3":   os.path.join(MODELS_DIR, "grpo_qwen3_4b_instruct_sft_v3", "best"),
        "grpo_sft_v4":   os.path.join(MODELS_DIR, "grpo_qwen3_4b_instruct_sft_v4", "final"),
        "grpo_sft_v5":   os.path.join(MODELS_DIR, "grpo_qwen3_4b_instruct_sft_v5", "final"),
    },
}


def discover_models(families: Optional[str] = None) -> Dict[str, str]:
    """Discover available model checkpoints.

    families: comma-separated family names or 'all'.
      Prefix match: 'qwen3' matches 'qwen3_base' and 'qwen3_instruct'.
      Exact match:  'qwen3.5' matches only 'qwen3.5'.
    """
    all_keys = list(MODEL_FAMILIES.keys())
    if families and families != "all":
        requested = {f.strip().lower() for f in families.split(",")}
        all_keys = [
            k for k in all_keys
            if any(k == r or k.startswith(r + "_") for r in requested)
        ]

    models = {}
    for fkey in all_keys:
        family = MODEL_FAMILIES[fkey]
        # Label: qwen3_base → QWEN3_BASE, qwen2.5 → QWEN25
        label = fkey.replace(".", "").upper()
        for variant, path in family.items():
            if os.path.isdir(path):
                models[f"{label}_{variant}"] = path

    return models


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
        help=(
            "Comma-separated family names or 'all'. "
            "Options: qwen2.5, qwen3.5, qwen3_base, qwen3_instruct, qwen3 (matches both qwen3_* families). "
            "Default: all discovered families."
        )
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 20 samples, no BARTScore"
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
        "--enable_bart_score", type=str, default=None,
        choices=["true", "false"],
        help="Enable BARTScore (default: true, false in --quick)"
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

    valid_models = {k: v for k, v in model_paths.items() if os.path.isdir(v)}
    missing = {k: v for k, v in model_paths.items() if not os.path.isdir(v)}
    if missing:
        logger.warning(f"Skipping {len(missing)} model(s) with missing paths:")
        for k, v in missing.items():
            logger.warning(f"  {k}: {v}")

    if not valid_models:
        logger.error("No valid models to evaluate. Check model paths.")
        sys.exit(1)

    logger.info(f"Models to evaluate ({len(valid_models)}):")
    for name, path in valid_models.items():
        logger.info(f"  {name}: {path}")

    # ── Resolve flags ────────────────────────────────────────────────────
    max_samples = args.max_samples
    enable_bart = args.enable_bart_score
    if args.quick:
        max_samples = max_samples or 20
        if enable_bart is None:
            enable_bart = "false"
    enable_bart = enable_bart is None or enable_bart.lower() == "true"

    # ── Build config ─────────────────────────────────────────────────────
    dataset_name = os.path.splitext(os.path.basename(args.test_data))[0]
    cfg = EvalConfig(
        models=[ModelEntry(n, p) for n, p in valid_models.items()],
        test_datasets=[DatasetEntry(dataset_name, args.test_data)],
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_samples=max_samples,
        enable_bart_score=enable_bart,
    )

    # ── Load test data ───────────────────────────────────────────────────
    if not os.path.exists(args.test_data):
        logger.error(f"Test data not found: {args.test_data}")
        sys.exit(1)
    samples = _load_test_data(args.test_data, max_samples)
    logger.info(f"Loaded {len(samples)} test samples from '{dataset_name}'")

    # ── Load BARTScorer once (shared across all models) ──────────────────
    bart_scorer = None
    if enable_bart:
        try:
            import sys as _sys
            _sys.path.insert(
                0,
                os.path.join(os.path.dirname(__file__), "..", "..", "external", "BARTScore"),
            )
            from bart_score import BARTScorer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            bart_scorer = BARTScorer(
                device=device, max_length=1024, checkpoint=cfg.bart_checkpoint
            )
            logger.info(f"BARTScorer loaded: {cfg.bart_checkpoint} on {device}")
        except ImportError:
            logger.warning("BARTScore library not found — disabling.")
            cfg.enable_bart_score = False

    # ── Evaluate each model ──────────────────────────────────────────────
    all_per_sample: List[Dict] = []

    for model_entry in cfg.models:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating: {model_entry.name}  ←  {model_entry.path}")
        logger.info(f"{'=' * 60}")
        try:
            model, tokenizer = load_eval_model(model_entry.path)
            per_sample = evaluate_model_on_dataset(
                model_entry.name, model, tokenizer,
                dataset_name, samples, cfg, bart_scorer,
            )
            all_per_sample.extend(per_sample)

            agg = aggregate(per_sample)
            logger.info(
                f"[{model_entry.name}] "
                f"ROUGE-2={agg['rouge2']:.4f}  "
                f"LenErr={agg['length_error_pct']:.1f}%  "
                f"LenDist={agg['length_distance']:.1f}  "
                f"BART={agg['bart_score']:.3f}  "
                f"AvgLen={agg['avg_gen_length']:.1f}"
            )

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to evaluate {model_entry.name}: {e}", exc_info=True)

    # ── Save results ─────────────────────────────────────────────────────
    if all_per_sample:
        save_results(all_per_sample, cfg.output_dir, cfg)
    else:
        logger.error("No results collected — check model paths and dataset paths.")


if __name__ == "__main__":
    main()
