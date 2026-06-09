"""Prepare plain (non-augmented) SFT data — no length/style constraints.

Generates one sample per raw article using a simple summarization instruction.
Output: data/sft_train_no_aug.jsonl, data/sft_val_no_aug.jsonl

Usage:
    PYTHONPATH=src python src/dataset/prepare_no_aug.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
# Ensure src/ comes before script directory in sys.path for correct package resolution.
# Python auto-adds the script's directory (src/dataset/) at sys.path[0]; PYTHONPATH adds
# src/ later.  We must move src/ to the front so that 'dataset' resolves to the package
# (src/dataset/__init__.py) rather than to the module (src/dataset/dataset.py).
for p in [_PROJECT_ROOT, _SRC_DIR]:
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# Configure logging before importing project modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from dataset.dataset import DatasetConfig, VietNewsDataset, WikiLinguaDataset

SYSTEM_PROMPT = (
    "Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. "
    "Hãy tóm tắt văn bản được cung cấp một cách ngắn gọn và chính xác."
)


def make_sample(source: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Tóm tắt văn bản sau:\n\n{source}"},
            {"role": "assistant", "content": target},
        ],
        "meta": {
            "source_length": len(source.split()),
            "target_length": len(target.split()),
        },
    }


def save_jsonl(samples: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples):,} samples → {path}")


def main(data_root: str = "VDT_Textsum", out_dir: str = "data"):
    cfg = DatasetConfig(
        data_root=data_root,
        mode="raw",
        max_source_chars=8000,
        max_summary_chars=1500,
    )

    # --- Train split: VietNews train + WikiLingua train ---
    train_samples = []

    logger.info("Loading VietNews/train...")
    vn_train = VietNewsDataset(cfg, split="train")
    for i, s in enumerate(vn_train):
        train_samples.append(make_sample(s["source"], s["target"]))
        if (i + 1) % 20000 == 0:
            logger.info(f"  {i+1}/{len(vn_train)}")

    logger.info("Loading WikiLingua/train...")
    wl_train = WikiLinguaDataset(cfg, split="train")
    for i, s in enumerate(wl_train):
        train_samples.append(make_sample(s["source"], s["target"]))
        if (i + 1) % 5000 == 0:
            logger.info(f"  {i+1}/{len(wl_train)}")

    # --- Val split: first 2K VietNews val + first 500 WikiLingua val ---
    val_samples = []

    vn_val = VietNewsDataset(cfg, split="val")
    for i, s in enumerate(vn_val):
        if i >= 2000:
            break
        val_samples.append(make_sample(s["source"], s["target"]))

    wl_val = WikiLinguaDataset(cfg, split="val")
    for i, s in enumerate(wl_val):
        if i >= 500:
            break
        val_samples.append(make_sample(s["source"], s["target"]))

    if not train_samples:
        raise RuntimeError(
            "Training data is empty — no samples were loaded.\n"
            "Check that VDT_Textsum/vietnews-master/ and VDT_Textsum/wikilingua/ exist."
        )

    save_jsonl(train_samples, os.path.join(out_dir, "sft_train_no_aug.jsonl"))
    save_jsonl(val_samples, os.path.join(out_dir, "sft_val_no_aug.jsonl"))

    logger.info(
        f"\nDone. Train: {len(train_samples):,}  Val: {len(val_samples):,}\n"
        f"(vs augmented: train ×3 = {len(train_samples)*3:,} samples)"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="VDT_Textsum")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()
    main(args.data_root, args.out_dir)
