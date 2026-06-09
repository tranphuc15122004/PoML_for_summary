#!/usr/bin/env python
"""Evaluate and compare summarization models.

Metrics:
    - ROUGE-1/2/L F1
    - Length Control Error: |words(gen) - words(ref)| (mean over samples)
    - Length Hit Rate: % within ±20% of prompt length requirement
    - BARTScore: conditional log-likelihood mean log P(hyp | article) via causal LM
    - G-Eval: 4-dimension LLM-as-Judge (coherence, consistency, fluency, relevance)
    - Average generated length

Usage:
    # Quick test — no LLM-based metrics
    PYTHONPATH=src python src/SFT_GRPO/evaluate.py \\
        --models "base=/g/data/hn98/dd9648/models/Qwen3.5-4B" \\
        --max_samples 20 --enable_bart_score false --enable_geval false

    # Full eval
    PYTHONPATH=src python src/SFT_GRPO/evaluate.py \\
        --models "base=/g/data/hn98/dd9648/models/Qwen3.5-4B,sft_aug=models/sft_aug_Qwen3.5-4B/final" \\
        --judge_model /g/data/hn98/dd9648/models/Qwen3.5-4B
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SFT_GRPO.config import EvalConfig
from SFT_GRPO.rewards import accuracy_reward, length_reward, rouge_l_f1, rouge_n

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Helpers
# ==============================================================================


def _extract_article(user_msg: str) -> str:
    """Extract article text from formatted user message."""
    parts = user_msg.split("Văn bản:\n", 1)
    return parts[1].strip() if len(parts) > 1 else user_msg.strip()


def compute_rouge(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1."""
    return {
        "rouge_1": rouge_n(generated, reference, n=1),
        "rouge_2": rouge_n(generated, reference, n=2),
        "rouge_l": rouge_l_f1(generated, reference),
    }


def compute_length_error(generated: str, reference: str) -> float:
    """Absolute word count difference between generated and reference."""
    return float(abs(len(generated.split()) - len(reference.split())))


def _parse_target_length(length_req: str) -> float:
    """Extract first numeric target from a length requirement string."""
    m = re.search(r"(\d+)", length_req)
    return float(m.group(1)) if m else 50.0


# ==============================================================================
# Model Loading
# ==============================================================================


def load_eval_model(
    model_path: str,
    base_model_path: Optional[str] = None,
    load_in_4bit: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer. Auto-detects LoRA adapters.

    Args:
        model_path: Local path or HF model ID.
        base_model_path: Base model path for LoRA adapters. Required if model_path
            is a LoRA adapter directory (contains adapter_config.json).
        load_in_4bit: Whether to quantize to 4-bit for memory efficiency.

    Returns:
        (model, tokenizer)
    """
    is_lora = os.path.isdir(model_path) and os.path.exists(
        os.path.join(model_path, "adapter_config.json")
    )

    if is_lora and base_model_path is None:
        # Try to read base model from adapter_config.json
        adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path", model_path)
        logger.info(f"LoRA detected — base model: {base_model_path}")

    tokenizer_path = base_model_path if is_lora else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_path = base_model_path if is_lora else model_path

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
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


def load_judge_model(
    judge_model_path: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the judge model (bf16, no quantization) for BARTScore and G-Eval."""
    logger.info(f"Loading judge model: {judge_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        judge_model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.config.use_cache = False  # disabled for log-likelihood computation
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
    """Generate summaries for a list of formatted prompt strings."""
    summaries: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch,
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
            text = tokenizer.decode(
                output_ids[prompt_len:], skip_special_tokens=True
            ).strip()
            # Strip Qwen3 thinking blocks if present
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
            summaries.append(text)

    return summaries


# ==============================================================================
# BARTScore — conditional log-likelihood via causal LM
# ==============================================================================


@torch.no_grad()
def compute_bart_score(
    hypotheses: List[str],
    articles: List[str],
    judge_model,
    judge_tokenizer,
) -> List[float]:
    """Compute BARTScore as mean log P(hypothesis | article).

    Uses causal LM forward pass: prefix = article context, labels = hypothesis tokens.
    Score is negated loss (higher = better).

    Returns:
        List of float scores, one per sample.
    """
    scores: List[float] = []
    judge_model.eval()

    for hyp, art in tqdm(
        zip(hypotheses, articles), total=len(hypotheses), desc="BARTScore", leave=False
    ):
        if not hyp or not art:
            scores.append(0.0)
            continue
        try:
            prefix = f"Tóm tắt đoạn văn sau:\n{art}\n\nTóm tắt:"
            full_text = prefix + " " + hyp

            prefix_ids = judge_tokenizer(
                prefix, return_tensors="pt", truncation=True, max_length=1024
            ).input_ids.to(judge_model.device)
            full_ids = judge_tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=1280
            ).input_ids.to(judge_model.device)

            prefix_len = prefix_ids.shape[1]
            labels = full_ids.clone()
            labels[0, :prefix_len] = -100  # mask prefix from loss

            loss = judge_model(input_ids=full_ids, labels=labels).loss
            scores.append(float(-loss.item()))  # negate: higher = better
        except Exception as e:
            logger.warning(f"BARTScore failed for sample: {e}")
            scores.append(0.0)

    return scores


# ==============================================================================
# G-Eval — 4-dimension LLM-as-Judge
# ==============================================================================

_GEVAL_PROMPT_TEMPLATE = """Đánh giá bản tóm tắt sau theo 4 tiêu chí, trả về JSON.

Văn bản gốc:
{article}

Bản tóm tắt:
{summary}

Tiêu chí đánh giá (thang điểm 1-5):
- coherence: Nội dung mạch lạc, có tính logic và cấu trúc rõ ràng
- consistency: Thông tin chính xác, trung thực với văn bản gốc, không bịa đặt
- fluency: Ngôn ngữ tự nhiên, đúng ngữ pháp tiếng Việt, dễ đọc
- relevance: Nắm bắt được các ý chính quan trọng nhất của văn bản

Chỉ trả lời bằng JSON theo đúng định dạng này, không kèm giải thích:
{{"coherence": X, "consistency": X, "fluency": X, "relevance": X}}"""

_GEVAL_KEYS = ["coherence", "consistency", "fluency", "relevance"]


def _parse_geval_output(text: str) -> Optional[Dict[str, float]]:
    """Parse JSON from G-Eval output. Returns None on failure."""
    # Try to find JSON block
    m = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group())
        scores = {}
        for k in _GEVAL_KEYS:
            val = float(data.get(k, 3))
            scores[k] = round(max(0.0, min(1.0, (val - 1) / 4.0)), 4)
        return scores
    except (json.JSONDecodeError, ValueError):
        return None


@torch.no_grad()
def compute_geval_batch(
    hypotheses: List[str],
    articles: List[str],
    judge_model,
    judge_tokenizer,
) -> List[Dict[str, float]]:
    """Run G-Eval on each (hypothesis, article) pair.

    Returns:
        List of dicts with keys: coherence, consistency, fluency, relevance.
        Falls back to 0.5 on parse failure.
    """
    results: List[Dict[str, float]] = []
    fallback = {k: 0.5 for k in _GEVAL_KEYS}

    for hyp, art in tqdm(
        zip(hypotheses, articles), total=len(hypotheses), desc="G-Eval", leave=False
    ):
        if not hyp or not art:
            results.append(fallback.copy())
            continue

        # Truncate article to avoid context overflow
        art_words = art.split()
        if len(art_words) > 400:
            art = " ".join(art_words[:400]) + "..."

        prompt = _GEVAL_PROMPT_TEMPLATE.format(article=art, summary=hyp)
        messages = [{"role": "user", "content": prompt}]

        parsed = None
        for attempt in range(2):
            try:
                input_text = judge_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = judge_tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=2048
                ).to(judge_model.device)

                output_ids = judge_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=judge_tokenizer.pad_token_id,
                )
                prompt_len = inputs.input_ids.shape[1]
                output_text = judge_tokenizer.decode(
                    output_ids[0][prompt_len:], skip_special_tokens=True
                ).strip()
                parsed = _parse_geval_output(output_text)
                if parsed:
                    break
            except Exception as e:
                logger.warning(f"G-Eval attempt {attempt+1} failed: {e}")

        results.append(parsed if parsed else fallback.copy())

    return results


# ==============================================================================
# Main evaluation
# ==============================================================================


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    model_name: str = "model",
    max_new_tokens: int = 256,
    batch_size: int = 8,
    judge_model=None,
    judge_tokenizer=None,
    enable_bart_score: bool = True,
    enable_geval: bool = True,
) -> Dict:
    """Run full evaluation on test data.

    Args:
        model: HF model for generation.
        tokenizer: HF tokenizer for generation.
        test_data: List of test samples with "prompt", "reference", "meta".
        model_name: Label for logging.
        max_new_tokens: Max tokens per generation.
        batch_size: Generation batch size.
        judge_model: Model for BARTScore/G-Eval (can be same or different from model).
        judge_tokenizer: Tokenizer for judge model.
        enable_bart_score: Whether to compute BARTScore.
        enable_geval: Whether to compute G-Eval.

    Returns:
        Dict of evaluation metrics.
    """
    prompt_texts: List[str] = []
    refs: List[str] = []
    articles: List[str] = []
    length_reqs: List[str] = []

    for sample in test_data:
        prompt_text = tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_texts.append(prompt_text)
        refs.append(sample.get("reference", ""))

        # Extract article from user message
        user_msg = ""
        for msg in sample.get("prompt", []):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
        articles.append(_extract_article(user_msg))

        meta = sample.get("meta", {})
        length_reqs.append(meta.get("length_requirement", "khoảng 50 từ"))

    logger.info(f"[{model_name}] Generating {len(prompt_texts)} summaries...")
    generated = generate_summaries(
        model, tokenizer, prompt_texts,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        batch_size=batch_size,
    )

    # Per-sample metrics
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougel_scores: List[float] = []
    length_errors: List[float] = []
    length_hits: List[bool] = []

    for gen, ref, lreq in zip(generated, refs, length_reqs):
        r = compute_rouge(gen, ref)
        rouge1_scores.append(r["rouge_1"])
        rouge2_scores.append(r["rouge_2"])
        rougel_scores.append(r["rouge_l"])

        length_errors.append(compute_length_error(gen, ref))

        target = _parse_target_length(lreq)
        if target > 0:
            length_hits.append(abs(len(gen.split()) - target) / target <= 0.20)
        else:
            length_hits.append(True)

    # BARTScore
    bart_scores: List[float] = []
    if enable_bart_score and judge_model is not None:
        logger.info(f"[{model_name}] Computing BARTScore...")
        bart_scores = compute_bart_score(generated, articles, judge_model, judge_tokenizer)
    else:
        bart_scores = [float("nan")] * len(generated)

    # G-Eval
    geval_results: List[Dict[str, float]] = []
    if enable_geval and judge_model is not None:
        logger.info(f"[{model_name}] Computing G-Eval...")
        geval_results = compute_geval_batch(generated, articles, judge_model, judge_tokenizer)
    else:
        geval_results = [{k: float("nan") for k in _GEVAL_KEYS}] * len(generated)

    def _safe_mean(vals):
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(clean)) if clean else float("nan")

    geval_dims = {k: _safe_mean([r[k] for r in geval_results]) for k in _GEVAL_KEYS}
    geval_avg = _safe_mean(list(geval_dims.values()))

    metrics = {
        "model": model_name,
        "samples": len(generated),
        "rouge_1": float(np.mean(rouge1_scores)),
        "rouge_2": float(np.mean(rouge2_scores)),
        "rouge_l": float(np.mean(rougel_scores)),
        "length_error": float(np.mean(length_errors)),
        "length_hit_rate": float(np.mean(length_hits)),
        "bart_score": _safe_mean(bart_scores),
        "geval_coherence": geval_dims["coherence"],
        "geval_consistency": geval_dims["consistency"],
        "geval_fluency": geval_dims["fluency"],
        "geval_relevance": geval_dims["relevance"],
        "geval_avg": geval_avg,
        "avg_gen_length": float(np.mean([len(g.split()) for g in generated])),
    }

    return metrics


# ==============================================================================
# Orchestration
# ==============================================================================


def run_evaluation(cfg: EvalConfig) -> Dict[str, Dict]:
    """Run evaluation for all configured models."""
    test_data: List[Dict] = []
    with open(cfg.test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    logger.info(f"Loaded {len(test_data)} test samples from {cfg.test_data_path}")

    # Load judge model once — shared across all eval models
    judge_model, judge_tokenizer = None, None
    if cfg.enable_bart_score or cfg.enable_geval:
        judge_model, judge_tokenizer = load_judge_model(cfg.judge_model_path)

    results: Dict[str, Dict] = {}

    for model_name, model_path in cfg.model_paths.items():
        logger.info(f"\n{'='*60}\nEvaluating: {model_name} ({model_path})\n{'='*60}")

        if not os.path.isdir(model_path) and "/" not in model_path:
            logger.warning(f"Model path does not exist: {model_path} — skipping")
            continue
        if os.path.isdir(model_path) and not os.listdir(model_path):
            logger.warning(f"Model directory is empty: {model_path} — skipping")
            continue

        try:
            model, tokenizer = load_eval_model(
                model_path,
                base_model_path=cfg.base_model_path,
                load_in_4bit=True,
            )
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                model_name=model_name,
                max_new_tokens=cfg.generation_max_new_tokens,
                batch_size=cfg.batch_size,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                enable_bart_score=cfg.enable_bart_score,
                enable_geval=cfg.enable_geval,
            )
            results[model_name] = metrics

            logger.info(
                f"[{model_name}] R1={metrics['rouge_1']:.4f}  R2={metrics['rouge_2']:.4f}  "
                f"RL={metrics['rouge_l']:.4f}  LenErr={metrics['length_error']:.1f}  "
                f"HitRate={metrics['length_hit_rate']:.2%}  "
                f"BART={metrics['bart_score']:.3f}  GEval={metrics['geval_avg']:.4f}  "
                f"AvgLen={metrics['avg_gen_length']:.1f}"
            )

            # Free GPU memory before loading next model
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}", exc_info=True)
            results[model_name] = {"model": model_name, "error": str(e)}

    _print_summary_table(results, cfg)
    return results


def _print_summary_table(results: Dict[str, Dict], cfg: EvalConfig) -> None:
    """Print a comparison table of all models."""
    print(f"\n{'='*100}")
    print("EVALUATION SUMMARY")
    print(f"{'='*100}")

    header = (
        f"{'Model':<22} {'N':>6} {'R-1':>7} {'R-2':>7} {'R-L':>7} "
        f"{'LenErr':>8} {'HitRate':>9} {'BART':>8} {'GEval':>7} {'AvgLen':>8}"
    )
    print(header)
    print("-" * len(header))

    for model_name in cfg.model_paths:
        if model_name not in results:
            continue
        m = results[model_name]
        if "error" in m:
            print(f"{model_name:<22} ERROR: {m['error']}")
            continue

        def fmt(v):
            if isinstance(v, float) and np.isnan(v):
                return f"{'N/A':>7}"
            return f"{v:>7.4f}"

        print(
            f"{model_name:<22} {m['samples']:>6} {fmt(m['rouge_1'])} {fmt(m['rouge_2'])} "
            f"{fmt(m['rouge_l'])} {m['length_error']:>8.1f} {m['length_hit_rate']:>9.2%} "
            f"{m['bart_score']:>8.3f} {fmt(m['geval_avg'])} {m['avg_gen_length']:>8.1f}"
        )

    print(f"{'='*100}\n")

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
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for quick testing")
    parser.add_argument("--models", type=str, default=None,
                        help="name1=path1,name2=path2")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Override judge model path for BARTScore/G-Eval")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Override base model path for LoRA adapters")
    parser.add_argument("--enable_bart_score", type=str, default="true",
                        choices=["true", "false"],
                        help="Enable BARTScore (default: true)")
    parser.add_argument("--enable_geval", type=str, default="true",
                        choices=["true", "false"],
                        help="Enable G-Eval (default: true)")
    args = parser.parse_args()

    model_paths = None
    if args.models:
        model_paths = {}
        for pair in args.models.split(","):
            if "=" in pair:
                name, path = pair.split("=", 1)
                model_paths[name.strip()] = path.strip()
            else:
                model_paths[pair.strip()] = pair.strip()

    cfg = EvalConfig(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_paths=model_paths or EvalConfig().model_paths,
        enable_bart_score=args.enable_bart_score.lower() == "true",
        enable_geval=args.enable_geval.lower() == "true",
    )
    if args.judge_model:
        cfg.judge_model_path = args.judge_model
    if args.base_model:
        cfg.base_model_path = args.base_model

    # Limit samples for quick testing
    if args.max_samples:
        import json as _json
        test_data: List[Dict] = []
        with open(cfg.test_data_path) as f:
            for line in f:
                if line.strip():
                    test_data.append(_json.loads(line))
                    if len(test_data) >= args.max_samples:
                        break
        logger.info(f"Limited to {len(test_data)} samples (--max_samples)")

        # Patch run_evaluation to use limited data
        original_run = run_evaluation

        def run_evaluation_limited(cfg):
            judge_model, judge_tokenizer = None, None
            if cfg.enable_bart_score or cfg.enable_geval:
                judge_model, judge_tokenizer = load_judge_model(cfg.judge_model_path)

            results: Dict[str, Dict] = {}
            for model_name, model_path in cfg.model_paths.items():
                logger.info(f"\n{'='*60}\nEvaluating: {model_name} ({model_path})\n{'='*60}")
                try:
                    model, tokenizer = load_eval_model(
                        model_path, base_model_path=cfg.base_model_path, load_in_4bit=True
                    )
                    metrics = evaluate_model(
                        model=model, tokenizer=tokenizer, test_data=test_data,
                        model_name=model_name, max_new_tokens=cfg.generation_max_new_tokens,
                        batch_size=cfg.batch_size, judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        enable_bart_score=cfg.enable_bart_score,
                        enable_geval=cfg.enable_geval,
                    )
                    results[model_name] = metrics
                    del model
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name}: {e}", exc_info=True)
                    results[model_name] = {"model": model_name, "error": str(e)}

            _print_summary_table(results, cfg)
            return results

        run_evaluation_limited(cfg)
    else:
        run_evaluation(cfg)
