#!/usr/bin/env python3
"""Audit the deterministic ACL evaluation snapshot.

The evaluator stores the generated text and the reference metrics, but not the
full prompt metadata in each per-sample record.  This script joins those
records back to the manifest by the canonical source hash and emits one
machine-readable analysis file containing sentence adherence, reward-hacking
rates, and paired bootstrap intervals.  It deliberately consumes saved
generations only; it never loads a model or performs new generation.

Example:
    PYTHONPATH=src:scripts/tools python scripts/tools/analyze_acl_primary_eval.py \
      --run-dir models/eval_results/clean_single_greedy/20260720_065129 \
      --manifest data/test_clean_single_document.jsonl \
      --output /tmp/acl_primary_analysis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "tools"
SRC = ROOT / "src"
for path in (str(TOOLS), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from audit_data_integrity import extract_source, source_hash  # noqa: E402
from SFT_GRPO.rewards import (  # noqa: E402
    _is_degenerate,
    accuracy_reward,
    length_reward,
    sentence_reward,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_no
                rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sentence_count(text: str) -> int:
    import re

    cleaned = re.sub(r"\d+\.\d+", "", (text or "").strip())
    return max(1, len(re.findall(r"[.!?]", cleaned)))


def manifest_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        source = extract_source(row)
        if not source:
            raise ValueError(f"Manifest row has no source: {path}:{row['_line']}")
        key = source_hash(source)
        if key in index:
            raise ValueError(f"Duplicate source hash in manifest: {key}")
        meta = row.get("meta") or {}
        index[key] = {
            "dataset": str(meta.get("dataset", "unknown")).lower(),
            "target_sentences": int(meta.get("target_sentences", 1)),
            "sentence_requirement": str(meta.get("sentence_requirement", "khoảng 1 câu")),
            "length_requirement": str(meta.get("length_requirement", "khoảng 50 từ")),
        }
    return index


def add_row_metrics(row: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    generated = str(row.get("generated", ""))
    reference = str(row.get("reference", ""))
    actual_sentences = sentence_count(generated)
    target_sentences = int(meta["target_sentences"])
    r_acc = accuracy_reward(generated, reference)
    r_len = length_reward(generated, meta["length_requirement"])
    r_sent = sentence_reward(generated, meta["sentence_requirement"])
    degenerate = _is_degenerate(generated)
    # This operational definition follows PROJECT_AUDIT.md.  It is a diagnostic
    # indicator, not evidence that the model optimized these terms causally.
    constraint_only = r_acc <= 0.01 and r_len >= 0.8 and r_sent >= 0.8
    return {
        "model": str(row.get("model", "unknown")),
        "dataset": meta["dataset"],
        "source_hash": row.get("source_hash"),
        "rouge2": float(row.get("rouge2", 0.0)),
        "length_error_pct": float(row.get("length_error_pct", 0.0)),
        "length_distance": float(row.get("length_distance", 0.0)),
        "sentence_count": actual_sentences,
        "target_sentences": target_sentences,
        "sentence_exact": float(actual_sentences == target_sentences),
        "sentence_tolerant": float(abs(actual_sentences - target_sentences) <= 1),
        "sentence_mae": float(abs(actual_sentences - target_sentences)),
        "degenerate": float(degenerate),
        "r_acc": float(r_acc),
        "r_len": float(r_len),
        "r_sent": float(r_sent),
        "near_zero_content": float(r_acc <= 0.01),
        "constraint_only": float(constraint_only),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else math.nan


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "rouge2": mean(rows, "rouge2"),
        "length_error_pct": mean(rows, "length_error_pct"),
        "length_distance": mean(rows, "length_distance"),
        "sentence_exact_pct": 100.0 * mean(rows, "sentence_exact"),
        "sentence_tolerant_pct": 100.0 * mean(rows, "sentence_tolerant"),
        "sentence_mae": mean(rows, "sentence_mae"),
        "degenerate_pct": 100.0 * mean(rows, "degenerate"),
        "zero_content_pct": 100.0 * sum(row["r_acc"] == 0.0 for row in rows) / len(rows),
        "near_zero_content_pct": 100.0 * mean(rows, "near_zero_content"),
        "constraint_only_pct": 100.0 * mean(rows, "constraint_only"),
        "mean_r_acc": mean(rows, "r_acc"),
    }


def bootstrap_delta(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
    *,
    reps: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    by_hash_left = {row["source_hash"]: row for row in left}
    by_hash_right = {row["source_hash"]: row for row in right}
    common = sorted(set(by_hash_left) & set(by_hash_right))
    if not common:
        raise ValueError("No common source hashes for paired bootstrap")
    differences = [
        float(by_hash_right[key_hash][key]) - float(by_hash_left[key_hash][key])
        for key_hash in common
    ]
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(differences)
    for _ in range(reps):
        samples.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    point = sum(differences) / n
    return {
        "n": n,
        "left": "lower/reference",
        "right": "higher/comparison",
        "metric": key,
        "point_delta": point,
        "ci95_low": samples[int(0.025 * (reps - 1))],
        "ci95_high": samples[int(0.975 * (reps - 1))],
        "bootstrap_reps": reps,
        "seed": seed,
    }


def load_generation_rows(run_dir: Path, manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "per_sample").glob("*.jsonl")):
        raw_rows = read_jsonl(path)
        # Older evaluator snapshots did not persist source_hash.  The
        # evaluator iterates every model over the manifest in order, so the
        # dataset-specific manifest order is a safe, auditable fallback.
        dataset_hint = str(raw_rows[0].get("meta_dataset", "")).lower() if raw_rows else ""
        all_keys = list(manifest)
        dataset_keys = [key for key, meta in manifest.items() if meta["dataset"] == dataset_hint]
        ordered_keys = all_keys if len(raw_rows) == len(all_keys) else dataset_keys
        if not ordered_keys and raw_rows:
            raise ValueError(f"Cannot infer manifest dataset for {path}")
        if len(raw_rows) != len(ordered_keys):
            raise ValueError(
                f"Generation count {len(raw_rows)} does not match manifest count "
                f"{len(ordered_keys)} for {path}"
            )
        for index, raw in enumerate(raw_rows):
            key = raw.get("source_hash") or ordered_keys[index]
            if key not in manifest:
                raise ValueError(f"Generation source hash missing from manifest: {key}")
            enriched = dict(raw)
            enriched["source_hash"] = key
            rows.append(add_row_metrics(enriched, manifest[key]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    manifest = manifest_index(args.manifest)
    rows = load_generation_rows(args.run_dir, manifest)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["model"]][row["dataset"]].append(row)

    results: dict[str, Any] = {}
    for model, datasets in sorted(grouped.items()):
        results[model] = {dataset: aggregate(items) for dataset, items in sorted(datasets.items())}
        pooled = [item for items in datasets.values() for item in items]
        results[model]["ALL"] = aggregate(pooled)

    bootstrap: dict[str, Any] = {}
    pairs = [
        ("QWEN3_BASE_sft", "QWEN3_BASE_grpo_sft_v5"),
        ("QWEN3_INSTRUCT_sft", "QWEN3_INSTRUCT_grpo_sft_v5"),
        ("QWEN3_BASE_grpo_fresh_v5", "QWEN3_BASE_grpo_sft_v5"),
        ("QWEN3_INSTRUCT_grpo_fresh_v5", "QWEN3_INSTRUCT_grpo_sft_v5"),
    ]
    for left_name, right_name in pairs:
        if left_name not in grouped or right_name not in grouped:
            continue
        for dataset in sorted(set(grouped[left_name]) & set(grouped[right_name])):
            left = grouped[left_name][dataset]
            right = grouped[right_name][dataset]
            key = f"{left_name}__to__{right_name}__{dataset}"
            bootstrap[key] = {
                metric: bootstrap_delta(left, right, metric, reps=args.bootstrap_reps)
                for metric in ("rouge2", "length_distance", "length_error_pct")
            }

    payload = {
        "run_dir": str(args.run_dir),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_records": len(manifest),
        "generation_records": len(rows),
        "models": sorted(grouped),
        "generation_counts": {
            model: {dataset: len(items) for dataset, items in sorted(datasets.items())}
            for model, datasets in sorted(grouped.items())
        },
        "results": results,
        "bootstrap": bootstrap,
        "definitions": {
            "sentence_exact": "actual punctuation-derived count equals target_sentences",
            "sentence_tolerant": "absolute sentence-count error <= 1",
            "constraint_only": "R_acc <= 0.01 and R_len >= 0.8 and R_sent >= 0.8",
            "bootstrap_delta": "right model minus left model, paired by source_hash",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} generation rows, {len(grouped)} models)")


if __name__ == "__main__":
    main()
