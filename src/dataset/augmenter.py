"""
Prepare instruction-following data for SFT and prompt-only data for GRPO.

Both SFT and GRPO use the same train splits (overlap is intentional):
- SFT: chat-format messages with assistant response
- GRPO: prompt-only (no assistant), model generates completions during training

Usage:
    from dataset.augmenter import PromptAugmenter, build_all_splits

    augmenter = PromptAugmenter()
    splits = build_all_splits(augmenter, data_root="VDT_Textsum")
    splits.sft_train.save("data/sft_train.jsonl")
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
# Ensure src/ comes before script directory in sys.path for correct package resolution.
for p in [_PROJECT_ROOT, _SRC_DIR]:
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from dataset.dataset import (
    BaseSummarizationDataset,
    DatasetConfig,
    VietNewsDataset,
    WikiLinguaDataset,
    ViMsDataset,
    VLSPDataset,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Length & Sentence Configuration
# ==============================================================================

LENGTH_TEMPLATES: List[str] = [
    # "khoảng X từ" — tolerance ±20% applied in reward
    "khoảng {target} từ",
    # "trong khoảng {lo}-{hi} từ"
    "trong khoảng {lo}-{hi} từ",
    # "không quá {max} từ"
    "không quá {max} từ",
]

SENTENCE_TEMPLATES: List[str] = [
    # "khoảng X câu" — tolerance ±1 sentence applied in reward
    "khoảng {target} câu",
    # "trong khoảng {lo}-{hi} câu"
    "trong khoảng {lo}-{hi} câu",
    # "không quá {max} câu"
    "không quá {max} câu",
]

SYSTEM_PROMPT_SFT = (
    "Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. "
    "Hãy tạo ra bản tóm tắt ngắn gọn, chính xác, "
    "tuân thủ đúng yêu cầu về độ dài và số câu."
)

SYSTEM_PROMPT_GRPO = SYSTEM_PROMPT_SFT

# VietNews targets are newspaper titles. Titles shorter than this threshold
# tend to be click-bait or too brief to serve as quality SFT reference summaries.
# GRPO still uses all VietNews (reward signal is length-aware so short titles are fine).
VIETNEWS_MIN_TARGET_WORDS: int = 10


# ==============================================================================
# PromptAugmenter
# ==============================================================================

@dataclass
class PromptAugmenterConfig:
    """Configuration for PromptAugmenter."""

    length_tolerance: float = 0.2
    """Tolerance for 'khoảng X từ' (±20%)."""

    system_prompt_sft: str = SYSTEM_PROMPT_SFT
    """System prompt for SFT chat samples."""

    system_prompt_grpo: str = SYSTEM_PROMPT_GRPO
    """System prompt for GRPO prompts."""

    seed: int = 42


class PromptAugmenter:
    """Generate instruction-following samples from raw {source, target} pairs.

    Each raw sample produces exactly one SFT variant and one GRPO prompt.
    Length/sentence requirements are derived from the gold summary's statistics.
    """

    def __init__(self, config: Optional[PromptAugmenterConfig] = None):
        self.config = config or PromptAugmenterConfig()
        self._rng = random.Random(self.config.seed)
        # Bộ đếm template để luân phiên cho GRPO (tạo đa dạng)
        self._grpo_template_counter: int = 0

    # ------------------------------------------------------------------
    # Length instruction generation
    # ------------------------------------------------------------------

    def _make_length_req(self, word_count: int, template_idx: int) -> str:
        """Generate a length requirement string from a word count."""
        template = LENGTH_TEMPLATES[template_idx % len(LENGTH_TEMPLATES)]

        if template == "khoảng {target} từ":
            return template.format(target=word_count)

        elif template == "trong khoảng {lo}-{hi} từ":
            lo = max(1, int(word_count * (1 - self.config.length_tolerance)))
            hi = int(word_count * (1 + self.config.length_tolerance))
            return template.format(lo=lo, hi=hi)

        elif template == "không quá {max} từ":
            max_words = int(word_count * (1 + self.config.length_tolerance))
            return template.format(max=max_words)

        return f"khoảng {word_count} từ"

    def _word_count(self, text: str) -> int:
        """Count words in a text (space-delimited for Vietnamese)."""
        return len(text.split())

    def _sentence_count(self, text: str) -> int:
        """Count sentences by Vietnamese sentence-ending punctuation."""
        import re
        count = len(re.findall(r'[.!?]', text.strip()))
        return max(count, 1)

    def _make_sentence_req(self, sent_count: int, template_idx: int) -> str:
        """Generate a sentence requirement string from a sentence count."""
        template = SENTENCE_TEMPLATES[template_idx % len(SENTENCE_TEMPLATES)]

        if template == "khoảng {target} câu":
            return template.format(target=sent_count)

        elif template == "trong khoảng {lo}-{hi} câu":
            lo = max(1, sent_count - 1)
            hi = sent_count + 1
            return template.format(lo=lo, hi=hi)

        elif template == "không quá {max} câu":
            max_sents = max(sent_count, sent_count + 1)
            return template.format(max=max_sents)

        return f"khoảng {sent_count} câu"

    # ------------------------------------------------------------------
    # Per-sample augmentation (SFT)
    # ------------------------------------------------------------------

    def augment_sft(self, source: str, target: str) -> Dict:
        """Generate one SFT sample from a raw {source, target} pair.

        Returns a chat-format dict with system/user/assistant messages.
        """
        wc = self._word_count(target)
        sc = self._sentence_count(target)
        length_req = self._make_length_req(wc, 0)
        sent_req = self._make_sentence_req(sc, 0)

        user_content = (
            f"Yêu cầu:\n"
            f"- Độ dài: {length_req}\n"
            f"- Số câu: {sent_req}\n"
            f"\n"
            f"Văn bản:\n{source}"
        )

        return {
            "messages": [
                {"role": "system", "content": self.config.system_prompt_sft},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target},
            ],
            "meta": {
                "source_length": self._word_count(source),
                "target_length": wc,
                "target_sentences": sc,
                "length_requirement": length_req,
                "sentence_requirement": sent_req,
            },
        }

    # ------------------------------------------------------------------
    # Per-sample augmentation (GRPO prompt)
    # ------------------------------------------------------------------

    def augment_grpo_prompt(
        self, source: str, target: str,
    ) -> Dict:
        """Generate one prompt for GRPO rollout.

        GRPO prompts have NO assistant response — the model generates
        multiple completions during training.

        Returns:
            Dict with "prompt" (list of messages) and "reference" (gold summary)
            and "meta" (length/sentence metadata for reward functions).
        """
        wc = self._word_count(target)
        sc = self._sentence_count(target)
        # Luân phiên template để tạo đa dạng length/sentence requirement
        template_idx = self._grpo_template_counter
        self._grpo_template_counter += 1
        length_req = self._make_length_req(wc, template_idx)
        sent_req = self._make_sentence_req(sc, template_idx)

        user_content = (
            f"Yêu cầu:\n"
            f"- Độ dài: {length_req}\n"
            f"- Số câu: {sent_req}\n"
            f"\n"
            f"Văn bản:\n{source}"
        )

        return {
            "prompt": [
                {"role": "system", "content": self.config.system_prompt_grpo},
                {"role": "user", "content": user_content},
            ],
            "reference": target,
            "meta": {
                "source_length": self._word_count(source),
                "target_length": wc,
                "target_sentences": sc,
                "length_requirement": length_req,
                "sentence_requirement": sent_req,
            },
        }

    # ------------------------------------------------------------------
    # Test prompt (no augment, fixed length)
    # ------------------------------------------------------------------

    def make_test_prompt(
        self, source: str, target: str,
        length_template: int = 0,
    ) -> Dict:
        """Generate a fixed test prompt (no randomness)."""
        wc = self._word_count(target)
        sc = self._sentence_count(target)
        length_req = self._make_length_req(wc, length_template)
        sent_req = self._make_sentence_req(sc, length_template)

        user_content = (
            f"Yêu cầu:\n"
            f"- Độ dài: {length_req}\n"
            f"- Số câu: {sent_req}\n"
            f"\n"
            f"Văn bản:\n{source}"
        )

        return {
            "prompt": [
                {"role": "system", "content": self.config.system_prompt_grpo},
                {"role": "user", "content": user_content},
            ],
            "reference": target,
            "meta": {
                "source_length": self._word_count(source),
                "target_length": wc,
                "target_sentences": sc,
                "length_requirement": length_req,
                "sentence_requirement": sent_req,
            },
        }


# ==============================================================================
# Data Splitting
# ==============================================================================

@dataclass
class DataSplit:
    """A split of processed data (SFT or GRPO format)."""

    name: str
    samples: List[Dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def save(self, path: str) -> None:
        """Save to JSONL (one JSON object per line)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.samples)} samples to {path}")

    def sample(self, n: int, seed: int = 42) -> "DataSplit":
        """Return a random subset."""
        rng = random.Random(seed)
        subset = rng.sample(self.samples, min(n, len(self.samples)))
        return DataSplit(name=f"{self.name}_sample", samples=subset)

    @staticmethod
    def from_jsonl(path: str) -> "DataSplit":
        """Load from JSONL."""
        samples: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return DataSplit(name=os.path.basename(path), samples=samples)


@dataclass
class AllSplits:
    """Container for all data splits.

    Test data được tách riêng theo từng dataset (4 bộ) để
    evaluation có thể tính metric riêng cho từng bộ.
    """

    sft_train: DataSplit
    sft_val: DataSplit
    grpo_train: DataSplit
    grpo_val: DataSplit
    test_vietnews: DataSplit
    test_wikilingua: DataSplit
    test_vlsp: DataSplit
    test_vims: DataSplit
    test: DataSplit  # combined test — used by eval.py default path

    def save_all(self, out_dir: str = "data") -> None:
        """Save all splits to JSONL files.

        Test data được lưu thành 4 file riêng theo từng dataset
        để evaluation có thể thống kê metric theo từng bộ,
        và một file gộp test.jsonl cho eval.py.
        """
        self.sft_train.save(os.path.join(out_dir, "sft_train.jsonl"))
        self.sft_val.save(os.path.join(out_dir, "sft_val.jsonl"))
        self.grpo_train.save(os.path.join(out_dir, "grpo_train.jsonl"))
        self.grpo_val.save(os.path.join(out_dir, "grpo_val.jsonl"))
        # Test files — per dataset
        self.test_vietnews.save(os.path.join(out_dir, "test_vietnews.jsonl"))
        self.test_wikilingua.save(os.path.join(out_dir, "test_wikilingua.jsonl"))
        self.test_vlsp.save(os.path.join(out_dir, "test_vlsp.jsonl"))
        self.test_vims.save(os.path.join(out_dir, "test_vims.jsonl"))
        # Combined test — default path for eval.py
        self.test.save(os.path.join(out_dir, "test.jsonl"))


def _build_dataset(ds_type: str, split: str, config: DatasetConfig):
    """Build a single dataset, handling ViMs's lack of split parameter."""
    if ds_type == "vims":
        return ViMsDataset(config, annotator_idx=0)
    elif ds_type == "vlsp":
        return VLSPDataset(config, split=split)
    elif ds_type == "vietnews":
        return VietNewsDataset(config, split=split)
    elif ds_type == "wikilingua":
        return WikiLinguaDataset(config, split=split)
    else:
        raise ValueError(f"Unknown dataset: {ds_type}")


def build_all_splits(
    augmenter: Optional[PromptAugmenter] = None,
    data_root: str = "VDT_Textsum",
    max_source_chars: int = 8000,
    max_summary_chars: int = 1500,
    test_size_vn: int = 2000,
    test_size_wl: int = 500,
) -> AllSplits:
    """Build all data splits from raw datasets.

    Cả SFT và GRPO đều dùng **train splits** (có thể overlap dữ liệu).
    Val chỉ dùng để validation, test để đánh giá cuối cùng.

    Args:
        augmenter: PromptAugmenter instance. Creates default if None.
        data_root: Path to VDT_Textsum directory.
        max_source_chars: Max source length in chars.
        max_summary_chars: Max summary length in chars.
        test_size_vn: Number of VietNews test samples to include.
        test_size_wl: Number of WikiLingua test samples to include.

    Returns:
        AllSplits containing all splits.
    """
    if augmenter is None:
        augmenter = PromptAugmenter()

    raw_cfg = DatasetConfig(
        data_root=data_root,
        mode="raw",
        max_source_chars=max_source_chars,
        max_summary_chars=max_summary_chars,
    )

    # ------------------------------------------------------------------
    # Train data sources (SFT vs GRPO have different quality requirements)
    #
    # SFT:  VietNews (title >= VIETNEWS_MIN_TARGET_WORDS) + WikiLingua + ViMs 80%
    # GRPO: all VietNews + WikiLingua + VLSP + ViMs 80%
    # ------------------------------------------------------------------
    sft_train_raw: List[Dict] = []   # high-quality sources for SFT
    grpo_train_raw: List[Dict] = []  # all sources for GRPO

    # VietNews: filter short titles for SFT; GRPO uses all
    logger.info("Loading VietNews/train...")
    vn_train = VietNewsDataset(raw_cfg, split="train")
    vn_skipped = 0
    for i, sample in enumerate(vn_train):
        grpo_train_raw.append(sample)
        if len(sample["target"].split()) >= VIETNEWS_MIN_TARGET_WORDS:
            sft_train_raw.append(sample)
        else:
            vn_skipped += 1
        if (i + 1) % 20000 == 0:
            logger.info(f"  Loaded {i+1}/{len(vn_train)} VietNews train samples")
    logger.info(
        f"  VietNews: {len(vn_train)} total, {vn_skipped} skipped for SFT "
        f"(title < {VIETNEWS_MIN_TARGET_WORDS} words)"
    )

    # WikiLingua: high-quality abstractive summaries → both SFT and GRPO
    logger.info("Loading WikiLingua/train...")
    wl_train = WikiLinguaDataset(raw_cfg, split="train")
    for sample in wl_train:
        sft_train_raw.append(sample)
        grpo_train_raw.append(sample)

    # VLSP: human-written summary (abstractive) available via `summary` field.
    # Previously used extractive labels; now uses abstractive `summary` by default.
    # Still GRPO-only because dataset size is small (~300 train).
    logger.info("Loading VLSP/train...")
    try:
        vlsp_train = VLSPDataset(raw_cfg, split="train")
        for sample in vlsp_train:
            if sample.get("target"):
                grpo_train_raw.append(sample)
    except Exception as e:
        logger.warning(f"VLSP/train skipped: {e}")

    # ViMs: human-annotated gold summaries → both SFT and GRPO
    logger.info("Loading ViMs...")
    vims_samples: List[Dict] = []
    split_idx = 0
    try:
        vims = ViMsDataset(raw_cfg, annotator_idx=0)
        vims_samples = list(vims)
        vims_rng = random.Random(42)
        vims_rng.shuffle(vims_samples)
        split_idx = int(len(vims_samples) * 0.8)
        sft_train_raw.extend(vims_samples[:split_idx])
        grpo_train_raw.extend(vims_samples[:split_idx])
    except Exception as e:
        logger.warning(f"ViMs skipped: {e}")

    # ------------------------------------------------------------------
    # 1. SFT TRAIN: high-quality sources only
    # ------------------------------------------------------------------
    sft_train_samples: List[Dict] = []
    for sample in sft_train_raw:
        sft_train_samples.append(
            augmenter.augment_sft(sample["source"], sample["target"])
        )
    logger.info(f"  SFT train: {len(sft_train_samples)} samples")

    # ------------------------------------------------------------------
    # 2. SFT VAL: first 2K VietNews val + first 500 WikiLingua val
    # ------------------------------------------------------------------
    sft_val_samples: List[Dict] = []

    vn_val = VietNewsDataset(raw_cfg, split="val")
    for i, sample in enumerate(vn_val):
        if i >= 2000:
            break
        sft_val_samples.append(augmenter.augment_sft(sample["source"], sample["target"]))

    wl_val = WikiLinguaDataset(raw_cfg, split="val")
    for i, sample in enumerate(wl_val):
        if i >= 500:
            break
        sft_val_samples.append(augmenter.augment_sft(sample["source"], sample["target"]))

    # ------------------------------------------------------------------
    # 3. GRPO TRAIN: all sources (unfiltered VietNews + VLSP + WikiLingua + ViMs)
    # ------------------------------------------------------------------
    grpo_train_samples: List[Dict] = []
    for sample in grpo_train_raw:
        grpo_train_samples.append(
            augmenter.augment_grpo_prompt(sample["source"], sample["target"])
        )
    logger.info(f"  GRPO train: {len(grpo_train_samples)} samples")

    # ------------------------------------------------------------------
    # 4. GRPO VAL: remaining VietNews val + remaining WikiLingua val + ViMs 20%
    # ------------------------------------------------------------------
    grpo_val_samples: List[Dict] = []

    for i, sample in enumerate(vn_val):
        if i < 2000:
            continue
        grpo_val_samples.append(augmenter.augment_grpo_prompt(sample["source"], sample["target"]))

    for i, sample in enumerate(wl_val):
        if i < 500:
            continue
        grpo_val_samples.append(augmenter.augment_grpo_prompt(sample["source"], sample["target"]))

    # VLSP val
    try:
        vlsp_val = VLSPDataset(raw_cfg, split="val")
        for sample in vlsp_val:
            if sample.get("target"):
                grpo_val_samples.append(
                    augmenter.augment_grpo_prompt(sample["source"], sample["target"])
                )
    except Exception as e:
        logger.warning(f"VLSP/val skipped: {e}")

    # ViMs 20%
    for sample in vims_samples[split_idx:]:
        grpo_val_samples.append(
            augmenter.augment_grpo_prompt(sample["source"], sample["target"])
        )

    # ------------------------------------------------------------------
    # 5. TEST: per-dataset files — VietNews, WikiLingua, VLSP, ViMs
    # ------------------------------------------------------------------
    test_vietnews: List[Dict] = []
    test_wikilingua: List[Dict] = []
    test_vlsp: List[Dict] = []
    test_vims: List[Dict] = []

    logger.info("Loading VietNews/test...")
    vn_test = VietNewsDataset(raw_cfg, split="test")
    count = 0
    for sample in vn_test:
        if count >= test_size_vn:
            break
        s = augmenter.make_test_prompt(sample["source"], sample["target"])
        s["meta"]["dataset"] = "vietnews"
        test_vietnews.append(s)
        count += 1

    logger.info("Loading WikiLingua/test...")
    wl_test = WikiLinguaDataset(raw_cfg, split="test")
    count = 0
    for sample in wl_test:
        if count >= test_size_wl:
            break
        s = augmenter.make_test_prompt(sample["source"], sample["target"])
        s["meta"]["dataset"] = "wikilingua"
        test_wikilingua.append(s)
        count += 1

    # VLSP — only use abmusu split (test.label.jsonl has placeholder [0] labels → empty targets)
    for split_name in ["abmusu"]:
        try:
            vlsp_test = VLSPDataset(raw_cfg, split=split_name)
            for sample in vlsp_test:
                if not sample.get("target"):
                    continue
                s = augmenter.make_test_prompt(sample["source"], sample["target"])
                s["meta"]["dataset"] = "vlsp"
                test_vlsp.append(s)
        except Exception as e:
            logger.warning(f"VLSP/{split_name} skipped: {e}")

    # ViMs — all clusters used for test
    logger.info("Loading ViMs for TEST...")
    try:
        vims_all = ViMsDataset(raw_cfg, annotator_idx=0)
        for sample in vims_all:
            s = augmenter.make_test_prompt(sample["source"], sample["target"])
            s["meta"]["dataset"] = "vims"
            test_vims.append(s)
    except Exception as e:
        logger.warning(f"ViMs test skipped: {e}")

    # Combined test for eval.py default path
    test_samples = test_vietnews + test_wikilingua + test_vlsp + test_vims

    # ------------------------------------------------------------------
    # Summary & return
    # ------------------------------------------------------------------
    splits = AllSplits(
        sft_train=DataSplit("sft_train", sft_train_samples),
        sft_val=DataSplit("sft_val", sft_val_samples),
        grpo_train=DataSplit("grpo_train", grpo_train_samples),
        grpo_val=DataSplit("grpo_val", grpo_val_samples),
        test_vietnews=DataSplit("test_vietnews", test_vietnews),
        test_wikilingua=DataSplit("test_wikilingua", test_wikilingua),
        test_vlsp=DataSplit("test_vlsp", test_vlsp),
        test_vims=DataSplit("test_vims", test_vims),
        test=DataSplit("test", test_samples),
    )

    test_total = len(test_vietnews) + len(test_wikilingua) + len(test_vlsp) + len(test_vims)

    logger.info(
        f"\n{'='*60}\n"
        f"Data splits summary:\n"
        f"  SFT train:      {len(splits.sft_train):>8,}  (VietNews≥{VIETNEWS_MIN_TARGET_WORDS}w + WikiLingua + ViMs)\n"
        f"  SFT val:        {len(splits.sft_val):>8,}\n"
        f"  GRPO train:     {len(splits.grpo_train):>8,}  (all VietNews + WikiLingua + VLSP + ViMs)\n"
        f"  GRPO val:       {len(splits.grpo_val):>8,}\n"
        f"  Test — VietNews: {len(test_vietnews):>8,}\n"
        f"  Test — WikiLingua: {len(test_wikilingua):>8,}\n"
        f"  Test — VLSP:     {len(test_vlsp):>8,}\n"
        f"  Test — ViMs:     {len(test_vims):>8,}\n"
        f"  Test — Total:    {test_total:>8,}  → data/test.jsonl\n"
        f"{'='*60}"
    )

    return splits


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    augmenter = PromptAugmenter()
    splits = build_all_splits(
        augmenter,
        data_root="VDT_Textsum",
        max_source_chars=8000,
        max_summary_chars=1500,
    )

    if not splits.sft_train.samples:
        raise RuntimeError(
            "SFT train split is empty — no data was loaded.\n"
            "Check that VDT_Textsum/vietnews-master/ and VDT_Textsum/wikilingua/ exist."
        )

    splits.save_all("data")

    # Print a few examples
    print("\n=== SFT Example ===")
    ex = splits.sft_train[0]
    print(f"Messages: {len(ex['messages'])} roles")
    print(f"System:   {ex['messages'][0]['content'][:80]}...")
    print(f"User:     {ex['messages'][1]['content'][:120]}...")
    print(f"Assist:   {ex['messages'][2]['content'][:80]}...")
    print(f"Meta:     length_req={ex['meta']['length_requirement']}")

    print("\n=== GRPO Example ===")
    ex = splits.grpo_train[0]
    print(f"Prompt: {len(ex['prompt'])} messages")
    print(f"  User: {ex['prompt'][1]['content'][:120]}...")
    print(f"  Reference: {ex['reference'][:60]}...")
    print(f"Meta: length_req={ex['meta']['length_requirement']}")
