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
from collections import Counter
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# Degenerate output detection
# ==============================================================================

_MIN_ALPHA_CHARS = 3       # fewer than this many alphabetic chars → degenerate
_MIN_ALPHA_RATIO = 0.12    # < 12% of chars are alphabetic → degenerate (digit/symbol flood)
_MAX_BLOB_LEN = 25         # single token longer than this with no spaces → blob
_MAX_REPETITION_RATIO = 0.60  # one token > 60% of all tokens → repetition loop


def _is_degenerate(text: str) -> bool:
    """Return True if text is clearly degenerate (blob, repetition loop, near-empty).

    Used by compute_all_rewards to zero out constraint rewards (R_len, R_sent) for
    outputs that game "không quá X từ/câu" by producing garbage that satisfies the
    constraint without containing real content.
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # Blob: single long token with no spaces (e.g. "1111100000000...")
    if " " not in stripped and len(stripped) > _MAX_BLOB_LEN:
        return True

    # Too few or too sparse alphabetic characters (e.g. "Is 1111..." or ",,,,,")
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count < _MIN_ALPHA_CHARS:
        return True
    if len(stripped) > 15 and alpha_count / len(stripped) < _MIN_ALPHA_RATIO:
        return True

    # Repetition loop or alternating pattern: detected by two checks on token lists
    words = stripped.split()
    if len(words) >= 5:
        # Single token dominates (e.g. "Nhà , , , , ,")
        top_count = Counter(words).most_common(1)[0][1]
        if top_count / len(words) > _MAX_REPETITION_RATIO:
            return True
        # Low diversity: < 30% unique tokens (catches "Nhà , Nhà , Nhà , ...")
        diversity = len(set(words)) / len(words)
        if diversity < 0.30:
            return True

    return False


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
    """Accuracy reward: 0.5 * ROUGE-1 + 0.5 * ROUGE-L F1.

    ROUGE-1 gives credit for individual word overlap (lenient, good gradient
    signal when generations are paraphrases of short headlines).
    ROUGE-L adds sequence-order quality on top.

    Args:
        generated: Model-generated summary.
        reference: Gold reference summary.

    Returns:
        Reward ∈ [0, 1].
    """
    if not reference or not generated:
        return 0.0
    gen = generated.strip()
    ref = reference.strip()
    r1 = rouge_n(gen, ref, n=1)
    rl = rouge_l_f1(gen, ref)
    return round(0.5 * r1 + 0.5 * rl, 4)


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
        # Minimum floor: output must reach at least 15% of max_wc (or 3 words).
        # Prevents a 1-word blob from earning R_len=1.0 on "không quá X từ" prompts.
        min_wc = max(3, int(max_wc * 0.15))
        if actual < min_wc:
            return 0.0
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

    Total uses multiplicative gating: R_acc acts as a gate so degenerate outputs
    (R_acc ≈ 0) cannot earn reward by satisfying length/sentence constraints alone.

        r_total = R_acc × (1 + w_len×R_len [+ w_sent×R_sent])

    This eliminates the reward-hacking mode where a model generates right-length
    garbage (digit blobs, comma loops) to collect R_len/R_sent while R_acc = 0.

    Args:
        generated: Model-generated summary.
        reference: Gold reference summary.
        length_requirement: E.g. "khoảng 50 từ".
        sentence_requirement: Optional. E.g. "khoảng 2 câu". If None, skipped.
        w_acc: Weight for accuracy reward (unused in gated formula, kept for API compat).
        w_len: Constraint bonus weight for length.
        w_sent: Constraint bonus weight for sentence count.

    Returns:
        Dict with keys: accuracy, length, sentence (if applicable), total.
    """
    r_acc = accuracy_reward(generated, reference)

    # Degenerate outputs (blobs, repetition loops) get zero constraint rewards.
    # This is the first line of defence; the multiplicative gate below is the second.
    if _is_degenerate(generated):
        r_len = 0.0
        r_sent = 0.0
    else:
        r_len = length_reward(generated, length_requirement)
        r_sent = sentence_reward(generated, sentence_requirement) if sentence_requirement else 1.0

    # Multiplicative gate: R_acc=0 → total=0 regardless of constraint scores.
    constraint_bonus = w_len * r_len
    if sentence_requirement:
        constraint_bonus += w_sent * r_sent
    r_total = r_acc * (1.0 + constraint_bonus)

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
    print("=" * 60)
    print("R_acc tests")
    print("=" * 60)
    gen = "Hôm nay trời đẹp."
    ref = "Hôm nay trời rất đẹp."
    print(f"R_acc: {accuracy_reward(gen, ref):.4f}  (expected ~0.86)")

    print("\n" + "=" * 60)
    print("R_len tests")
    print("=" * 60)
    test_cases = [
        ("khoảng 50 từ", " ".join(["word"] * 50), 1.0),
        ("khoảng 50 từ", " ".join(["word"] * 100), 0.0),
        ("trong khoảng 40-60 từ", " ".join(["word"] * 50), 1.0),
        ("trong khoảng 40-60 từ", " ".join(["word"] * 30), 0.75),
        ("không quá 65 từ", " ".join(["word"] * 65), 1.0),
        ("không quá 65 từ", " ".join(["word"] * 80), 0.7692),
        # Floor tests: output too short for "không quá X từ" → must score 0
        ("không quá 21 từ", "Is " + "1" * 200, 0.0),   # blob with 2 tokens
        ("không quá 30 từ", "A", 0.0),                  # 1 word < floor(3)
        ("không quá 30 từ", " ".join(["w"] * 2), 0.0),  # 2 words < floor(3) for max=30
    ]
    print(f"  {'requirement':40s}  {'actual':>6}  {'got':>7}  {'exp':>7}  ok")
    for req, text, expected in test_cases:
        got = length_reward(text, req)
        ok = "✓" if abs(got - expected) < 0.01 else "✗"
        actual_wc = len(text.strip().split())
        print(f"  {req:40s}  {actual_wc:6d}  {got:7.4f}  {expected:7.4f}  {ok}")

    print("\n" + "=" * 60)
    print("R_sent tests")
    print("=" * 60)
    sent_cases = [
        ("khoảng 2 câu", "Câu một. Câu hai.", 1.0),
        ("khoảng 2 câu", "Chỉ một câu thôi.", 1.0),  # ±1 tolerance
        ("khoảng 2 câu", "A. B. C. D.", 0.5),         # 4 sentences, error=2 → 0.5
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

    print("\n" + "=" * 60)
    print("Degenerate detection tests (_is_degenerate)")
    print("=" * 60)
    degen_cases = [
        ("Is " + "1" * 200, True),          # digit blob after 2 words
        ("1" * 50, True),                   # pure digit blob
        ("Nhà , , , , , , , ,", True),      # comma repetition loop
        ("Hôm nay trời rất đẹp và trong lành.", False),  # normal Vietnamese
        ("Tôi đi chợ mua rau.", False),     # normal Vietnamese
        ("word " * 10, True),               # 10 identical tokens = 100% repetition loop
        ("", True),                          # empty
        ("A.", True),                        # only 1 alpha char — too short to be a summary
        ("Vâng.", False),                    # valid Vietnamese (4 alpha chars: V,â,n,g)
        (",,,,,,,,,,,,,,,", True),           # pure punctuation
    ]
    for text, expected in degen_cases:
        got = _is_degenerate(text)
        ok = "✓" if got == expected else "✗"
        preview = repr(text[:40]) + ("..." if len(text) > 40 else "")
        print(f"  {ok}  degen={got!s:5}  {preview}")

    print("\n" + "=" * 60)
    print("Anti-hacking: blob garbage should score total ≈ 0")
    print("=" * 60)
    blob_cases = [
        # Previously scored 0.50 (R_len=1.0, R_sent=1.0, R_acc=0.0)
        ("Is " + "1" * 200, "Gold standard tiếng Việt.", "không quá 21 từ", "không quá 2 câu"),
        # Comma loop
        ("Nhà , , , , , , , ,", "Căn nhà đẹp.", "không quá 10 từ", "không quá 2 câu"),
        # Pure digit blob
        ("1" * 60, "Nội dung quan trọng.", "không quá 30 từ", "khoảng 1 câu"),
    ]
    for gen_text, ref_text, len_req, sent_req in blob_cases:
        result = compute_all_rewards(gen_text, ref_text, len_req, sent_req)
        ok = "✓" if result["total"] < 0.05 else "✗ HACK POSSIBLE"
        print(f"  {ok}  total={result['total']:.4f}  {result}  gen={repr(gen_text[:30])}")

    print("\n" + "=" * 60)
    print("Composite reward — normal text")
    print("=" * 60)
    result = compute_all_rewards(
        generated="Trời hôm nay đẹp. Tôi đi dạo.",
        reference="Hôm nay trời rất đẹp. Tôi đi dạo.",
        length_requirement="khoảng 5 từ",
        sentence_requirement="khoảng 2 câu",
    )
    print(f"Composite (with sentence): {result}")

    result2 = compute_all_rewards(
        generated="Trời hôm nay đẹp.",
        reference="Hôm nay trời rất đẹp.",
        length_requirement="khoảng 5 từ",
    )
    print(f"Composite (no sentence): {result2}")
