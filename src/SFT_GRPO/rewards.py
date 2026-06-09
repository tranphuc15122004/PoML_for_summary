"""Reward functions for GRPO training.

Three reward components:
    1. R_acc  — Accuracy: ROUGE-L F1 between generated and reference summary
    2. R_len  — Length adherence: how well word count matches prompt requirement
    3. R_style— Style adherence: LLM-as-Judge evaluation of style matching

Usage:
    from rewards import compute_all_rewards

    reward_dict = compute_all_rewards(
        generated="Bản tóm tắt...",
        reference="Gold summary...",
        length_requirement="khoảng 50 từ",
        style="báo chí",
        judge_pipeline=judge_model,
    )
    # → {"accuracy": 0.82, "length": 1.0, "style": 0.75, "total": 0.845}
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# R_acc — Accuracy Reward (ROUGE-L F1)
# ==============================================================================


def _lcs_length(a: list, b: list) -> int:
    """Compute length of Longest Common Subsequence (word level)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for word_a in a:
        curr = [0] * (len(b) + 1)
        for j, word_b in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if word_a == word_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(generated: str, reference: str) -> float:
    """Compute ROUGE-L F1 score (word-level LCS).

    Args:
        generated: Generated summary.
        reference: Gold reference summary.

    Returns:
        F1 score ∈ [0, 1]. Returns 0 if either is empty.
    """
    if not generated or not reference:
        return 0.0

    gen_tokens = generated.strip().split()
    ref_tokens = reference.strip().split()

    if not gen_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(gen_tokens, ref_tokens)

    precision = lcs_len / len(gen_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def rouge_n(generated: str, reference: str, n: int = 2) -> float:
    """ROUGE-N F1 score using n-gram overlap (no external deps)."""
    from collections import Counter

    def ngrams(words):
        return [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]

    gen_grams = ngrams(generated.lower().split())
    ref_grams = ngrams(reference.lower().split())
    if not gen_grams or not ref_grams:
        return 0.0
    overlap = sum((Counter(gen_grams) & Counter(ref_grams)).values())
    p = overlap / len(gen_grams)
    r = overlap / len(ref_grams)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return round(f1, 4)


def accuracy_reward(generated: str, reference: str) -> float:
    """Accuracy reward: ROUGE-L F1 score.

    Args:
        generated: Model-generated summary.
        reference: Gold reference summary.

    Returns:
        Reward ∈ [0, 1].
    """
    if not reference:
        return 0.0
    if not generated:
        return 0.0
    return rouge_l_f1(generated.strip(), reference.strip())


# ==============================================================================
# R_len — Length Adherence Reward
# ==============================================================================


def _parse_length_requirement(
    length_req: str,
) -> tuple[str, float | tuple[float, float] | None]:
    """Parse a length requirement string.

    Args:
        length_req: E.g. "khoảng 50 từ", "trong khoảng 40-60 từ", "không quá 65 từ"

    Returns:
        Tuple of (type, target):
            - ("exact", target_word_count)     for "khoảng X từ"
            - ("range", (lo, hi))              for "trong khoảng lo-hi từ"
            - ("max", max_word_count)          for "không quá X từ"
    """
    text = length_req.lower().strip()

    # "trong khoảng {lo}-{hi} từ"
    m = re.search(r"trong khoảng\s+(\d+)\s*-\s*(\d+)\s*từ", text)
    if m:
        return ("range", (float(m.group(1)), float(m.group(2))))

    # "không quá {max} từ"
    m = re.search(r"không quá\s+(\d+)\s*từ", text)
    if m:
        return ("max", float(m.group(1)))

    # "khoảng {target} từ"  (default)
    m = re.search(r"khoảng\s+(\d+)\s*từ", text)
    if m:
        return ("exact", float(m.group(1)))

    # Fallback: try to find any number
    m = re.search(r"(\d+)", text)
    if m:
        return ("exact", float(m.group(1)))

    return ("exact", 50.0)  # default guess


def length_reward(generated: str, length_requirement: str) -> float:
    """Compute length adherence reward.

    Evaluates how well the generated summary matches the requested word count.

    Args:
        generated: Generated summary.
        length_requirement: String like "khoảng 50 từ".

    Returns:
        Reward ∈ [0, 1].
    """
    if not generated:
        return 0.0

    actual = len(generated.strip().split())
    req_type, target = _parse_length_requirement(length_requirement)

    if req_type == "exact":
        # "khoảng X từ" → ±20% tolerance
        target_wc = float(target)
        tol = 0.20 * target_wc
        error = abs(actual - target_wc)
        if error <= tol:
            return 1.0
        # Linear decay: reward goes from 1.0 at tol to 0.0 at 3*tol
        score = max(0.0, 1.0 - (error - tol) / (2 * tol))
        return round(score, 4)

    elif req_type == "range":
        lo, hi = float(target[0]), float(target[1])
        if lo <= actual <= hi:
            return 1.0
        if actual < lo:
            return round(max(0.0, actual / lo), 4)
        return round(max(0.0, hi / actual), 4)

    elif req_type == "max":
        max_wc = float(target)
        if actual <= max_wc:
            return 1.0
        score = max(0.0, 1.0 - (actual - max_wc) / max(max_wc, 1))
        return round(score, 4)

    return 0.0


# ==============================================================================
# R_style — Style Adherence Reward (LLM-as-Judge)
# ==============================================================================


def style_reward_llm(
    generated: str,
    style: str,
    judge_pipeline,
    max_retries: int = 2,
) -> float:
    """Evaluate style adherence using an LLM judge.

    The judge is prompted to rate the summary on a scale of 1-5 for the
    requested style. Score is normalized to [0, 1].

    Args:
        generated: Generated summary.
        style: Requested style, e.g. "hài hước", "báo chí".
        judge_pipeline: A callable that takes a prompt string and returns
            generated text. Must implement `judge_pipeline([prompt], max_tokens=5)`.
        max_retries: Number of retries if judge output is unparseable.

    Returns:
        Score ∈ [0, 1].
    """
    if not generated:
        return 0.0

    prompt = (
        f"Trên thang điểm từ 1 đến 5, đánh giá bản tóm tắt sau đây "
        f"tuân thủ phong cách '{style}' như thế nào?\n"
        f"1 = Hoàn toàn không phù hợp\n"
        f"5 = Hoàn toàn phù hợp\n"
        f"Chỉ trả lời bằng MỘT số nguyên từ 1 đến 5, không kèm giải thích.\n\n"
        f"Bản tóm tắt:\n{generated}"
    )

    for attempt in range(max_retries + 1):
        try:
            output = judge_pipeline(prompt, max_tokens=5)[0]["generated_text"]
            # Extract integer from output
            nums = re.findall(r"\d+", output.strip())
            if nums:
                score = int(nums[0])
                score = max(1, min(5, score))  # clamp
                return round((score - 1) / 4.0, 4)
        except Exception as e:
            if attempt < max_retries:
                continue
            logger.warning(f"Style judge failed after {max_retries} retries: {e}")

    return 0.5  # neutral fallback


def style_reward_embedding(
    generated: str,
    style: str,
    style_embeddings: Dict[str, np.ndarray],
    embedding_fn,
) -> float:
    """Fallback: Evaluate style via cosine similarity to style prototypes.

    Args:
        generated: Generated summary.
        style: Requested style.
        style_embeddings: Dict of {style: prototype_embedding_vector}.
        embedding_fn: Callable that takes text → embedding vector.

    Returns:
        Score ∈ [0, 1], normalized by max score across all styles.
    """
    if not generated or style not in style_embeddings:
        return 0.5

    gen_emb = embedding_fn(generated)
    target_vec = style_embeddings[style]

    cos_sim = np.dot(gen_emb, target_vec) / (
        np.linalg.norm(gen_emb) * np.linalg.norm(target_vec) + 1e-8
    )
    # Normalize from [-1, 1] to [0, 1]
    return round(max(0.0, (cos_sim + 1) / 2), 4)


# ==============================================================================
# Composite Reward
# ==============================================================================


def compute_all_rewards(
    generated: str,
    reference: str,
    length_requirement: str,
    style: str,
    judge_pipeline=None,
    w_acc: float = 0.5,
    w_len: float = 0.3,
    w_style: float = 0.2,
    style_embeddings: Optional[Dict] = None,
    embedding_fn=None,
) -> Dict[str, float]:
    """Compute all three reward components and the weighted total.

    Args:
        generated: Model-generated summary.
        reference: Gold reference summary.
        length_requirement: E.g. "khoảng 50 từ".
        style: Requested style.
        judge_pipeline: LLM pipeline for style evaluation.
        w_acc: Weight for accuracy reward.
        w_len: Weight for length reward.
        w_style: Weight for style reward.
        style_embeddings: Optional precomputed style embeddings.
        embedding_fn: Optional embedding function for style fallback.

    Returns:
        Dict with keys: accuracy, length, style, total.
    """
    r_acc = accuracy_reward(generated, reference)

    r_len = length_reward(generated, length_requirement)

    if judge_pipeline is not None:
        r_style = style_reward_llm(generated, style, judge_pipeline)
    elif style_embeddings is not None and embedding_fn is not None:
        r_style = style_reward_embedding(generated, style, style_embeddings, embedding_fn)
    else:
        r_style = 0.5  # default neutral

    r_total = w_acc * r_acc + w_len * r_len + w_style * r_style

    return {
        "accuracy": round(r_acc, 4),
        "length": round(r_len, 4),
        "style": round(r_style, 4),
        "total": round(r_total, 4),
    }


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == "__main__":
    # Test accuracy reward
    gen = "Hôm nay trời đẹp."
    ref = "Hôm nay trời rất đẹp."
    print(f"R_acc: {accuracy_reward(gen, ref):.4f}  (expected ~0.86)")

    # Test length reward
    test_cases = [
        ("khoảng 50 từ", " ".join(["word"] * 50), 1.0),
        ("khoảng 50 từ", " ".join(["word"] * 100), 0.0),
        ("trong khoảng 40-60 từ", " ".join(["word"] * 50), 1.0),
        ("trong khoảng 40-60 từ", " ".join(["word"] * 30), 0.75),
        ("không quá 65 từ", " ".join(["word"] * 65), 1.0),
        ("không quá 65 từ", " ".join(["word"] * 80), 0.7692),
    ]
    print("\nR_len tests:")
    for req, text, expected in test_cases:
        got = length_reward(text, req)
        ok = "✓" if abs(got - expected) < 0.01 else "✗"
        print(f"  {req:40s}  actual={len(text.split()):3d}  got={got:.4f}  exp={expected:.4f}  {ok}")

    # Test composite (no judge)
    result = compute_all_rewards(
        generated="Trời hôm nay đẹp.",
        reference="Hôm nay trời rất đẹp.",
        length_requirement="khoảng 5 từ",
        style="báo chí",
    )
    print(f"\nComposite: {result}")
