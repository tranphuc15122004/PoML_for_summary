"""Reward functions for GRPO training.

Three reward components:
    1. R_acc  — Accuracy: ROUGE-L F1 between generated and reference summary
    2. R_len  — Length adherence: how well word count matches prompt requirement
    3. R_sent — Sentence count adherence: how well sentence count matches requirement

Usage:
    from rewards import compute_all_rewards

    reward_dict = compute_all_rewards(
        generated="Bản tóm tắt...",
        reference="Gold summary...",
        length_requirement="khoảng 50 từ",
        sentence_requirement="khoảng 2 câu",
    )
    # → {"accuracy": 0.82, "length": 1.0, "sentence": 1.0, "total": 0.845}
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

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
# R_sent — Sentence Count Adherence Reward
# ==============================================================================


def _parse_sentence_requirement(
    sent_req: str,
) -> tuple[str, float | tuple[float, float] | None]:
    """Parse a sentence requirement string.

    Args:
        sent_req: E.g. "khoảng 2 câu", "trong khoảng 1-3 câu", "không quá 3 câu"

    Returns:
        Tuple of (type, target):
            - ("exact", target_count)         for "khoảng X câu"
            - ("range", (lo, hi))              for "trong khoảng lo-hi câu"
            - ("max", max_count)               for "không quá X câu"
    """
    text = sent_req.lower().strip()

    # "trong khoảng {lo}-{hi} câu"
    m = re.search(r"trong khoảng\s+(\d+)\s*-\s*(\d+)\s*câu", text)
    if m:
        return ("range", (float(m.group(1)), float(m.group(2))))

    # "không quá {max} câu"
    m = re.search(r"không quá\s+(\d+)\s*câu", text)
    if m:
        return ("max", float(m.group(1)))

    # "khoảng {target} câu"  (default)
    m = re.search(r"khoảng\s+(\d+)\s*câu", text)
    if m:
        return ("exact", float(m.group(1)))

    # Fallback
    m = re.search(r"(\d+)", text)
    if m:
        return ("exact", float(m.group(1)))

    return ("exact", 1.0)  # default guess


def sentence_reward(generated: str, sentence_requirement: str) -> float:
    """Compute sentence count adherence reward.

    Evaluates how well the generated summary matches the requested number of sentences.

    Args:
        generated: Generated summary.
        sentence_requirement: String like "khoảng 2 câu".

    Returns:
        Reward ∈ [0, 1].
    """
    if not generated:
        return 0.0

    # Count sentences by Vietnamese sentence-ending punctuation.
    # Strip decimal-point numbers first to avoid counting "1.5" as a sentence boundary.
    cleaned = re.sub(r'\d+\.\d+', '', generated.strip())
    actual = max(1, len(re.findall(r'[.!?]', cleaned)))

    req_type, target = _parse_sentence_requirement(sentence_requirement)

    if req_type == "exact":
        # "khoảng X câu" → ±1 sentence tolerance
        target_sc = int(target)
        error = abs(actual - target_sc)
        if error <= 1:
            return 1.0
        # Linear decay
        score = max(0.0, 1.0 - (error - 1) / 2.0)
        return round(score, 4)

    elif req_type == "range":
        lo, hi = int(target[0]), int(target[1])
        if lo <= actual <= hi:
            return 1.0
        if actual < lo:
            return round(max(0.0, actual / lo), 4)
        return round(max(0.0, hi / actual), 4)

    elif req_type == "max":
        max_sc = int(target)
        if actual <= max_sc:
            return 1.0
        score = max(0.0, 1.0 - (actual - max_sc) / max(max_sc, 1))
        return round(score, 4)

    return 0.0


# ==============================================================================
# Composite Reward
# ==============================================================================


def compute_all_rewards(
    generated: str,
    reference: str,
    length_requirement: str,
    sentence_requirement: Optional[str] = None,
    w_acc: float = 0.5,
    w_len: float = 0.3,
    w_sent: float = 0.2,
) -> Dict[str, float]:
    """Compute reward components and the weighted total.

    Args:
        generated: Model-generated summary.
        reference: Gold reference summary.
        length_requirement: E.g. "khoảng 50 từ".
        sentence_requirement: Optional. E.g. "khoảng 2 câu". If None, skipped.
        w_acc: Weight for accuracy reward.
        w_len: Weight for length reward.
        w_sent: Weight for sentence reward.

    Returns:
        Dict with keys: accuracy, length, sentence (if applicable), total.
    """
    r_acc = accuracy_reward(generated, reference)

    r_len = length_reward(generated, length_requirement)

    r_sent = sentence_reward(generated, sentence_requirement) if sentence_requirement else 1.0

    # Normalize weights if sentence reward is used
    if sentence_requirement:
        r_total = w_acc * r_acc + w_len * r_len + w_sent * r_sent
    else:
        # Fall back to acc + len only (renormalize)
        total_w = w_acc + w_len
        r_total = (w_acc * r_acc + w_len * r_len) / total_w if total_w > 0 else 0.0

    result: Dict[str, float] = {
        "accuracy": round(r_acc, 4),
        "length": round(r_len, 4),
        "total": round(r_total, 4),
    }
    if sentence_requirement:
        result["sentence"] = round(r_sent, 4)

    return result


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

    # Test sentence reward
    print("\nR_sent tests:")
    sent_cases = [
        ("khoảng 2 câu", "Câu một. Câu hai.", 1.0),
        ("khoảng 2 câu", "Chỉ một câu thôi.", 1.0),  # ±1 tolerance
        ("khoảng 2 câu", "A. B. C. D.", 0.5),  # 4 sentences, error=2 → decay to 0.5
        ("trong khoảng 1-3 câu", "A. B.", 1.0),
        ("trong khoảng 1-3 câu", "A. B. C. D.", 0.75),  # 4 > 3
        ("không quá 2 câu", "A.", 1.0),
        ("không quá 2 câu", "A. B. C.", 0.5),  # 3 > 2
    ]
    for req, text, expected in sent_cases:
        got = sentence_reward(text, req)
        ok = "✓" if abs(got - expected) < 0.01 else "✗"
        actual_sents = max(1, len(re.findall(r'[.!?]', text.strip())))
        print(f"  {req:30s}  actual={actual_sents}  got={got:.4f}  exp={expected:.4f}  {ok}")

    # Test composite with sentence
    result = compute_all_rewards(
        generated="Trời hôm nay đẹp. Tôi đi dạo.",
        reference="Hôm nay trời rất đẹp. Tôi đi dạo.",
        length_requirement="khoảng 5 từ",
        sentence_requirement="khoảng 2 câu",
    )
    print(f"\nComposite (with sentence): {result}")

    # Test composite without sentence (backward compat)
    result2 = compute_all_rewards(
        generated="Trời hôm nay đẹp.",
        reference="Hôm nay trời rất đẹp.",
        length_requirement="khoảng 5 từ",
    )
    print(f"Composite (no sentence): {result2}")
