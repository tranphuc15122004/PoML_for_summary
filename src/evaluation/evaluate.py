#!/usr/bin/env python
"""Unified evaluation script for Vietnamese summarization models.

Metrics: ROUGE-2 F1, Length Error (%), BARTScore (log P(gen|article))

Usage:
    # CLI — single test file, multiple models
    PYTHONPATH=src python src/evaluation/evaluate.py \\
        --models "base=/g/data/hn98/dd9648/models/Qwen3-4B-Base,sft=models/sft_qwen3_4b_base/final" \\
        --test_data data/test.jsonl

    # Config file
    PYTHONPATH=src python src/evaluation/evaluate.py --config eval_config.json

    # Quick smoke test (no BARTScore, few samples)
    PYTHONPATH=src python src/evaluation/evaluate.py \\
        --models "base=/g/data/hn98/dd9648/models/Qwen3-4B-Base" \\
        --test_data data/test.jsonl \\
        --max_samples 20 --no_bart_score
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# External metric libraries
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "external", "BARTScore"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "external", "HeterSumGraph"))

try:
    from bart_score import BARTScorer
    _BARTSCORE_AVAILABLE = True
except ImportError:
    _BARTSCORE_AVAILABLE = False

try:
    from tools.utils import rouge_all as _rouge_all
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


# ==============================================================================
# Logging setup
# ==============================================================================

logger = logging.getLogger(__name__)

def _normalise_source(text: str) -> str:
    """Canonicalize source text before hashing for split-provenance checks."""
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.split())


def _source_hash(text: str) -> str:
    return hashlib.sha256(_normalise_source(text).encode("utf-8")).hexdigest()


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval.log")
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


# ==============================================================================
# Config
# ==============================================================================


@dataclass
class ModelEntry:
    name: str
    path: str


@dataclass
class DatasetEntry:
    name: str
    path: str


@dataclass
class EvalConfig:
    models: List[ModelEntry] = field(default_factory=list)
    test_datasets: List[DatasetEntry] = field(default_factory=list)
    batch_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True
    seed: Optional[int] = None
    bart_checkpoint: str = "facebook/bart-large-cnn"
    bart_batch_size: int = 4
    bart_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "models/eval_results"
    max_samples: Optional[int] = None
    enable_bart_score: bool = True


def _parse_config(args: argparse.Namespace) -> EvalConfig:
    """Build EvalConfig from CLI args or JSON config file."""
    cfg = EvalConfig()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            data = json.load(f)
        cfg.models = [ModelEntry(**m) for m in data.get("models", [])]
        cfg.test_datasets = [DatasetEntry(**d) for d in data.get("test_datasets", [])]
        cfg.batch_size = data.get("batch_size", cfg.batch_size)
        cfg.max_new_tokens = data.get("max_new_tokens", cfg.max_new_tokens)
        cfg.temperature = data.get("temperature", cfg.temperature)
        cfg.top_p = data.get("top_p", cfg.top_p)
        cfg.do_sample = data.get("do_sample", cfg.do_sample)
        cfg.seed = data.get("seed", cfg.seed)
        cfg.bart_checkpoint = data.get("bart_checkpoint", cfg.bart_checkpoint)
        cfg.bart_batch_size = data.get("bart_batch_size", cfg.bart_batch_size)
        cfg.bart_device = data.get("bart_device", cfg.bart_device)
        cfg.output_dir = data.get("output_dir", cfg.output_dir)
        cfg.max_samples = data.get("max_samples", cfg.max_samples)
        cfg.enable_bart_score = data.get("enable_bart_score", cfg.enable_bart_score)

    # CLI overrides
    if args.models:
        cfg.models = []
        for pair in args.models.split(","):
            if "=" in pair:
                name, path = pair.split("=", 1)
                cfg.models.append(ModelEntry(name.strip(), path.strip()))
            else:
                p = pair.strip()
                cfg.models.append(ModelEntry(os.path.basename(p), p))

    if args.test_data:
        cfg.test_datasets = []
        for pair in args.test_data.split(","):
            if "=" in pair:
                name, path = pair.split("=", 1)
                cfg.test_datasets.append(DatasetEntry(name.strip(), path.strip()))
            else:
                p = pair.strip()
                cfg.test_datasets.append(DatasetEntry(os.path.splitext(os.path.basename(p))[0], p))

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.no_bart_score:
        cfg.enable_bart_score = False
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.deterministic:
        cfg.temperature = 0.0
        cfg.do_sample = False
    if args.seed is not None:
        cfg.seed = args.seed

    return cfg


# ==============================================================================
# Data loading
# ==============================================================================


def _load_test_data(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
    return samples


# ==============================================================================
# Model loading
# ==============================================================================


def load_eval_model(
    model_path: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer. Auto-detects LoRA adapters via adapter_config.json."""
    is_lora = os.path.isdir(model_path) and os.path.exists(
        os.path.join(model_path, "adapter_config.json")
    )

    base_model_path = model_path
    if is_lora:
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path", model_path)
        logger.info(f"LoRA adapter detected — base model: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
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
    top_p: float = 0.9,
    do_sample: bool = True,
    batch_size: int = 8,
) -> List[str]:
    summaries: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=3072 - max_new_tokens,
            return_tensors="pt",
        ).to(model.device)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Avoid sampling-only knobs in greedy mode; some Transformers versions
        # warn or reject temperature=0 when sampling is disabled.
        if do_sample and temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
        else:
            generation_kwargs["do_sample"] = False
        outputs = model.generate(**generation_kwargs)

        for j, output_ids in enumerate(outputs):
            prompt_len = inputs.input_ids[j].shape[0]
            text = tokenizer.decode(
                output_ids[prompt_len:], skip_special_tokens=True
            ).strip()
            # Strip Qwen3 thinking blocks if present
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
            summaries.append(text)

    return summaries


# ==============================================================================
# Metric helpers
# ==============================================================================


def _extract_article(user_msg: str) -> str:
    parts = user_msg.split("Văn bản:\n", 1)
    return parts[1].strip() if len(parts) > 1 else user_msg.strip()


def _compute_rouge2(hyp: str, ref: str) -> float:
    if not _ROUGE_AVAILABLE:
        # Fallback: simple bigram F1
        def bigrams(text):
            tokens = text.split()
            return set(zip(tokens, tokens[1:]))
        h, r = bigrams(hyp), bigrams(ref)
        if not h or not r:
            return 0.0
        overlap = len(h & r)
        p = overlap / len(h)
        rec = overlap / len(r)
        return 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
    try:
        result = _rouge_all(hyp, ref)
        return float(result["rouge-2"]["f"])
    except Exception:
        return 0.0


def _compute_length_error(gen: str, target_length: float) -> float:
    """Relative length error: |words(gen) - target| / target * 100 (%)."""
    if target_length <= 0:
        return 0.0
    gen_len = len(gen.split())
    return abs(gen_len - target_length) / target_length * 100.0


def _parse_target_length(meta: Dict) -> float:
    """Get target word count from meta. Falls back to parsing length_requirement string."""
    if "target_length" in meta:
        return float(meta["target_length"])
    m = re.search(r"(\d+)", meta.get("length_requirement", "50"))
    return float(m.group(1)) if m else 50.0


def compute_metrics_batch(
    generated: List[str],
    references: List[str],
    articles: List[str],
    meta_list: List[Dict],
    bart_scorer: Optional["BARTScorer"],
    bart_batch_size: int = 4,
) -> List[Dict]:
    """Compute per-sample metrics for a batch of generated summaries."""
    # ROUGE-2 and length error (fast, per-sample)
    per_sample = []
    for gen, ref, meta in zip(generated, references, meta_list):
        target_len = _parse_target_length(meta)
        gen_len = len(gen.split())
        ref_len = len(ref.split())
        per_sample.append({
            "generated": gen,
            "reference": ref,
            "rouge2": _compute_rouge2(gen, ref),
            "length_error_pct": _compute_length_error(gen, target_len),
            "length_distance": abs(gen_len - ref_len),
            "target_length": target_len,
            "gen_length": gen_len,
            "ref_length": ref_len,
            "bart_score": float("nan"),
        })

    # BARTScore (batched)
    if bart_scorer is not None and articles:
        try:
            scores = bart_scorer.score(articles, generated, batch_size=bart_batch_size)
            for s, score in zip(per_sample, scores):
                s["bart_score"] = float(score)
        except Exception as e:
            logger.warning(f"BARTScore failed: {e}")

    return per_sample


# ==============================================================================
# Evaluation orchestration
# ==============================================================================


def evaluate_model_on_dataset(
    model_name: str,
    model,
    tokenizer,
    dataset_name: str,
    samples: List[Dict],
    cfg: EvalConfig,
    bart_scorer: Optional["BARTScorer"],
) -> List[Dict]:
    """Generate + compute metrics for one model on one dataset."""
    prompt_texts, references, articles, meta_list = [], [], [], []

    for sample in samples:
        try:
            prompt_text = tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_texts.append(prompt_text)
        references.append(sample.get("reference", ""))

        user_msg = next(
            (m.get("content", "") for m in sample.get("prompt", []) if m.get("role") == "user"),
            "",
        )
        articles.append(_extract_article(user_msg))
        meta_list.append(sample.get("meta", {}))

    logger.info(f"[{model_name} / {dataset_name}] Generating {len(prompt_texts)} summaries...")
    generated = generate_summaries(
        model, tokenizer, prompt_texts,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        batch_size=cfg.batch_size,
    )

    logger.info(f"[{model_name} / {dataset_name}] Computing metrics...")
    per_sample = compute_metrics_batch(
        generated, references, articles, meta_list,
        bart_scorer=bart_scorer if cfg.enable_bart_score else None,
        bart_batch_size=cfg.bart_batch_size,
    )

    for s, meta, article in zip(per_sample, meta_list, articles):
        s["model"] = model_name
        s["dataset"] = dataset_name
        s["meta_dataset"] = meta.get("dataset", "")
        s["source_hash"] = _source_hash(article)

    return per_sample


# ==============================================================================
# Aggregation
# ==============================================================================


def _safe_mean(vals: List[float]) -> float:
    clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(clean)) if clean else float("nan")


def aggregate(per_sample: List[Dict]) -> Dict:
    return {
        "n": len(per_sample),
        "rouge2": _safe_mean([s["rouge2"] for s in per_sample]),
        "length_error_pct": _safe_mean([s["length_error_pct"] for s in per_sample]),
        "length_distance": _safe_mean([s["length_distance"] for s in per_sample]),
        "bart_score": _safe_mean([s["bart_score"] for s in per_sample]),
        "avg_gen_length": _safe_mean([s["gen_length"] for s in per_sample]),
    }


# ==============================================================================
# Results saving
# ==============================================================================

_CSV_FIELDS = ["model", "dataset", "n", "rouge2", "length_error_pct", "length_distance", "bart_score", "avg_gen_length"]


def save_results(
    all_per_sample: List[Dict],
    output_dir: str,
    cfg: EvalConfig,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, ts)
    per_sample_dir = os.path.join(run_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    # Group by (model, dataset)
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for s in all_per_sample:
        key = (s["model"], s["dataset"])
        groups.setdefault(key, []).append(s)

    # Per-sample JSONL files
    for (model_name, dataset_name), samples in groups.items():
        fname = f"{model_name}_{dataset_name}.jsonl".replace("/", "_")
        fpath = os.path.join(per_sample_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Aggregate rows
    rows = []
    for (model_name, dataset_name), samples in sorted(groups.items()):
        agg = aggregate(samples)
        rows.append({"model": model_name, "dataset": dataset_name, **agg})

        # Per-meta_dataset sub-rows when a single file contains multiple source datasets
        sub_groups: Dict[str, List[Dict]] = {}
        for s in samples:
            md = s.get("meta_dataset", "")
            if md:
                sub_groups.setdefault(md, []).append(s)
        if len(sub_groups) > 1:
            for md in sorted(sub_groups):
                sub_agg = aggregate(sub_groups[md])
                rows.append({"model": model_name, "dataset": f"  {md}", **sub_agg})

    # Per-model overall average (across all datasets)
    models_seen = sorted({r["model"] for r in rows if not r["dataset"].startswith("  ")})
    for model_name in models_seen:
        model_samples = [s for s in all_per_sample if s["model"] == model_name]
        agg = aggregate(model_samples)
        rows.append({"model": model_name, "dataset": "ALL", **agg})

    # summary.json
    summary = {
        "run": ts,
        "config": {
            "models": [{"name": m.name, "path": m.path} for m in cfg.models],
            "test_datasets": [{"name": d.name, "path": d.path} for d in cfg.test_datasets],
            "max_samples": cfg.max_samples,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "do_sample": cfg.do_sample,
            "seed": cfg.seed,
            "bart_checkpoint": cfg.bart_checkpoint if cfg.enable_bart_score else None,
        },
        "manifest_hashes": {},
        "results": rows,
    }
    # Hash the ordered, de-duplicated source list for each evaluated dataset.
    for dataset_name in sorted({s["dataset"] for s in all_per_sample}):
        seen = set()
        hashes = []
        for sample in all_per_sample:
            if sample["dataset"] == dataset_name and sample["source_hash"] not in seen:
                seen.add(sample["source_hash"])
                hashes.append(sample["source_hash"])
        summary["manifest_hashes"][dataset_name] = hashlib.sha256(
            "\n".join(hashes).encode("ascii")
        ).hexdigest()
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # summary.csv
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})

    # summary.txt — pretty table
    txt_path = os.path.join(run_dir, "summary.txt")
    table = _format_table(rows, cfg.enable_bart_score)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(table)

    logger.info(f"Results saved to {run_dir}/")


def _fmt(v, precision=4):
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _format_table(rows: List[Dict], show_bart: bool) -> str:
    sep = "=" * (22 + 14 + 8 + 10 + 12 + 12 + (12 if show_bart else 0) + 10)
    header = f"{'Model':<22} {'Dataset':<14} {'N':>6}  {'ROUGE-2':>8}  {'LenErr%':>9}  {'LenDist':>8}"
    if show_bart:
        header += f"  {'BARTScore':>10}"
    header += f"  {'AvgLen':>8}"

    lines = [sep, "EVALUATION SUMMARY", sep, header, "-" * len(sep)]
    prev_model = None
    for row in rows:
        is_sub = row["dataset"].startswith("  ")
        if not is_sub and row["model"] != prev_model and prev_model is not None:
            lines.append("")
        if not is_sub:
            prev_model = row["model"]
        model_col = row["model"] if not is_sub else ""
        line = (
            f"{model_col:<22} {row['dataset']:<14} {row['n']:>6}  "
            f"{_fmt(row['rouge2']):>8}  {_fmt(row['length_error_pct'], 2):>9}  "
            f"{_fmt(row['length_distance'], 1):>8}"
        )
        if show_bart:
            line += f"  {_fmt(row['bart_score'], 3):>10}"
        line += f"  {_fmt(row['avg_gen_length'], 1):>8}"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines) + "\n"


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese summarization models")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--models", type=str, default=None, help="name1=path1,name2=path2")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path(s) to test JSONL. Format: data.jsonl OR Name=path1,Name2=path2")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_bart_score", action="store_true", help="Disable BARTScore")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true", help="Use greedy decoding")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _parse_config(args)

    if not cfg.models:
        parser.error("No models specified. Use --models or provide a config file.")
    if not cfg.test_datasets:
        parser.error("No test data specified. Use --test_data or provide a config file.")

    _setup_logging(cfg.output_dir)
    _seed_everything(cfg.seed)
    logger.info(f"Evaluating {len(cfg.models)} model(s) on {len(cfg.test_datasets)} dataset(s)")

    if cfg.enable_bart_score and not _BARTSCORE_AVAILABLE:
        logger.warning("BARTScore library not found in external/BARTScore — disabling.")
        cfg.enable_bart_score = False

    if not _ROUGE_AVAILABLE:
        logger.warning("HeterSumGraph rouge_all not found — using fallback bigram ROUGE-2.")

    # Load BARTScorer once (shared across all models/datasets)
    bart_scorer = None
    if cfg.enable_bart_score:
        logger.info(f"Loading BARTScorer: {cfg.bart_checkpoint} on {cfg.bart_device}")
        bart_scorer = BARTScorer(
            device=cfg.bart_device,
            max_length=1024,
            checkpoint=cfg.bart_checkpoint,
        )

    all_per_sample: List[Dict] = []

    for dataset_entry in cfg.test_datasets:
        if not os.path.exists(dataset_entry.path):
            logger.warning(f"Dataset not found: {dataset_entry.path} — skipping")
            continue
        samples = _load_test_data(dataset_entry.path, cfg.max_samples)
        logger.info(f"Loaded {len(samples)} samples from '{dataset_entry.name}' ({dataset_entry.path})")

        for model_entry in cfg.models:
            logger.info(f"\n{'='*60}\n{model_entry.name}  ←  {model_entry.path}\n{'='*60}")
            try:
                model, tokenizer = load_eval_model(model_entry.path)
                per_sample = evaluate_model_on_dataset(
                    model_entry.name, model, tokenizer,
                    dataset_entry.name, samples, cfg, bart_scorer,
                )
                all_per_sample.extend(per_sample)

                agg = aggregate(per_sample)
                logger.info(
                    f"[{model_entry.name} / {dataset_entry.name}] "
                    f"ROUGE-2={agg['rouge2']:.4f}  LenErr={agg['length_error_pct']:.1f}%  "
                    f"LenDist={agg['length_distance']:.1f}  "
                    f"BART={_fmt(agg['bart_score'], 3)}  AvgLen={agg['avg_gen_length']:.1f}"
                )

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed: {model_entry.name} on {dataset_entry.name}: {e}", exc_info=True)

    if all_per_sample:
        save_results(all_per_sample, cfg.output_dir, cfg)
    else:
        logger.error("No results collected — check model paths and dataset paths.")


if __name__ == "__main__":
    main()
