#!/usr/bin/env python3
"""Audit source overlap and build a clean single-document evaluation manifest.

The project stores SFT, GRPO, and evaluation examples in different JSONL
schemas.  This tool extracts the source text from each schema, normalizes it
for hashing, reports cross-split overlap, and optionally writes a clean
VietNews/WikiLingua evaluation file.  It never overwrites an input file.

Example:
    PYTHONPATH=src python scripts/tools/audit_data_integrity.py \
        --output-dir models/data_audit \
        --clean-test data/test_clean_single_document.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_FILES = {
    "sft_train": "data/sft_train.jsonl",
    "sft_val": "data/sft_val.jsonl",
    "grpo_train": "data/grpo_train.jsonl",
    "grpo_val": "data/grpo_val.jsonl",
    "test": "data/test.jsonl",
}


def normalize_source(text: str) -> str:
    """Use the same canonicalization for overlap checks across all schemas."""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def source_hash(text: str) -> str:
    return hashlib.sha256(normalize_source(text).encode("utf-8")).hexdigest()


def _messages(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("messages", "prompt"):
        value = record.get(key)
        if isinstance(value, list):
            yield from (item for item in value if isinstance(item, dict))


def extract_source(record: dict[str, Any]) -> str:
    """Extract source text from raw, SFT, GRPO, or evaluation records."""
    for key in ("source", "article", "document"):
        if isinstance(record.get(key), str):
            return record[key]

    for message in _messages(record):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        for marker in ("Văn bản:\n", "Văn bản:", "Van ban:\n", "Van ban:"):
            if marker in content:
                return content.split(marker, 1)[1].strip()
        return content
    return ""


def dataset_name(record: dict[str, Any], fallback: str) -> str:
    meta = record.get("meta")
    if isinstance(meta, dict) and meta.get("dataset"):
        return str(meta["dataset"]).lower()
    return fallback


def load_jsonl(path: Path, fallback_dataset: str = "unknown") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = extract_source(record)
            if not source:
                raise ValueError(f"No source found in {path}:{line_no}")
            record["_audit_source_hash"] = source_hash(source)
            record["_audit_dataset"] = dataset_name(record, fallback_dataset)
            record["_audit_line"] = line_no
            records.append(record)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["_audit_dataset"] for r in records)
    hashes = [r["_audit_source_hash"] for r in records]
    return {
        "records": len(records),
        "datasets": dict(sorted(counts.items())),
        "unique_sources": len(set(hashes)),
        "duplicate_records_within_file": len(hashes) - len(set(hashes)),
    }


def overlap_rows(
    left: list[dict[str, Any]], right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    right_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in right:
        right_by_hash[record["_audit_source_hash"]].append(record)
    rows = []
    for record in left:
        matches = right_by_hash.get(record["_audit_source_hash"], [])
        if matches:
            rows.append({
                "dataset": record["_audit_dataset"],
                "left_line": record["_audit_line"],
                "right_lines": [m["_audit_line"] for m in matches],
                "source_hash": record["_audit_source_hash"],
            })
    return rows


def strip_audit_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if not k.startswith("_audit_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models/data_audit")
    parser.add_argument(
        "--clean-test",
        default=None,
        help="Optional output JSONL containing clean VietNews/WikiLingua test records",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing the generated JSONL files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    loaded: dict[str, list[dict[str, Any]]] = {}
    missing = []
    for name, relative in SCHEMA_FILES.items():
        path = data_dir / Path(relative).name
        if not path.exists():
            missing.append(str(path))
            continue
        loaded[name] = load_jsonl(path, fallback_dataset=name)

    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_val = loaded["sft_train"] + loaded["sft_val"] + loaded["grpo_train"] + loaded["grpo_val"]
    train_val_hashes = {r["_audit_source_hash"] for r in train_val}
    test_records = loaded["test"]
    leaked = [r for r in test_records if r["_audit_source_hash"] in train_val_hashes]
    clean = [
        r for r in test_records
        if r["_audit_dataset"] in {"vietnews", "wikilingua"}
        and r["_audit_source_hash"] not in train_val_hashes
    ]

    report: dict[str, Any] = {
        "normalization": "Unicode NFKC followed by whitespace collapse",
        "hash": "sha256(normalized source)",
        "files": {name: summarize(records) for name, records in loaded.items()},
        "test_overlap_with_any_train_or_validation": {
            "records": len(leaked),
            "datasets": dict(sorted(Counter(r["_audit_dataset"] for r in leaked).items())),
            "unique_sources": len({r["_audit_source_hash"] for r in leaked}),
        },
        "clean_single_document_test": summarize(clean),
        "overlaps": {},
    }

    for test_name in ("test",):
        for reference_name in ("sft_train", "sft_val", "grpo_train", "grpo_val"):
            rows = overlap_rows(loaded[test_name], loaded[reference_name])
            report["overlaps"][f"{test_name}_vs_{reference_name}"] = {
                "records": len(rows),
                "unique_sources": len({row["source_hash"] for row in rows}),
                "datasets": dict(sorted(Counter(row["dataset"] for row in rows).items())),
            }

    report_path = output_dir / "integrity_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.clean_test:
        clean_path = Path(args.clean_test)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        with clean_path.open("w", encoding="utf-8") as handle:
            for record in clean:
                handle.write(json.dumps(strip_audit_fields(record), ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.clean_test:
        print(f"Wrote clean test: {args.clean_test}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
