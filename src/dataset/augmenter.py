"""
Data augmentation: inject length & style instructions into raw summarization data.

Generates instruction-following chat samples for SFT and prompt-only samples for GRPO,
with clear train/val/test splits.

Usage:
    from dataset.augmenter import PromptAugmenter, build_all_splits

    augmenter = PromptAugmenter()
    splits = build_all_splits(augmenter, data_root="VDT_Textsum")
    splits["sft_train"].save("data/sft_train.jsonl")
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
# Style & Length Configuration
# ==============================================================================

SFT_STYLES: List[str] = [
    "báo chí",
    "trang trọng",
    "học thuật",
    "ngắn gọn súc tích",
    "dạng gạch đầu dòng",
]

GRPO_STYLES: List[str] = SFT_STYLES + [
    "hài hước",
    "thân mật",
    "dành cho trẻ em",
    "mang tính phản biện",
]

LENGTH_TEMPLATES: List[str] = [
    # "khoảng X từ" — tolerance ±20% applied in reward
    "khoảng {target} từ",
    # "trong khoảng {lo}-{hi} từ"
    "trong khoảng {lo}-{hi} từ",
    # "không quá {max} từ"
    "không quá {max} từ",
]

SYSTEM_PROMPT_SFT = (
    "Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. "
    "Hãy tạo ra bản tóm tắt ngắn gọn, chính xác, "
    "tuân thủ đúng yêu cầu về độ dài và phong cách."
)

SYSTEM_PROMPT_GRPO = (
    "Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. "
    "Hãy tạo ra bản tóm tắt chính xác, tuân thủ yêu cầu."
)


# ==============================================================================
# PromptAugmenter
# ==============================================================================

@dataclass
class PromptAugmenterConfig:
    """Configuration for PromptAugmenter."""

    num_variants: int = 3
    """Number of instruction variants to generate per raw sample for SFT."""

    length_tolerance: float = 0.2
    """Tolerance for 'khoảng X từ' (±20%)."""

    sft_styles: List[str] = field(default_factory=lambda: SFT_STYLES.copy())
    """Style pool for SFT data."""

    grpo_styles: List[str] = field(default_factory=lambda: GRPO_STYLES.copy())
    """Style pool for GRPO data."""

    system_prompt_sft: str = SYSTEM_PROMPT_SFT
    """System prompt for SFT chat samples."""

    system_prompt_grpo: str = SYSTEM_PROMPT_GRPO
    """System prompt for GRPO prompts."""

    seed: int = 42


class PromptAugmenter:
    """Generate instruction-following variants from raw {source, target} pairs.

    For each raw sample, creates N variants with different length requirements
    and randomly assigned styles. Length requirements are derived from the
    gold summary's word count (passive strategy).
    """

    def __init__(self, config: Optional[PromptAugmenterConfig] = None):
        self.config = config or PromptAugmenterConfig()
        self._rng = random.Random(self.config.seed)

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

    # ------------------------------------------------------------------
    # Per-sample augmentation (SFT)
    # ------------------------------------------------------------------

    def augment_sft(self, source: str, target: str) -> List[Dict]:
        """Generate N SFT variants from one raw sample.

        Each variant is a chat-format dict with system/user/assistant messages.
        """
        wc = self._word_count(target)
        variants: List[Dict] = []

        for i in range(self.config.num_variants):
            length_req = self._make_length_req(wc, i)
            style = self._rng.choice(self.config.sft_styles)

            user_content = (
                f"Yêu cầu:\n"
                f"- Độ dài: {length_req}\n"
                f"- Phong cách: {style}\n"
                f"\n"
                f"Văn bản:\n{source}"
            )

            variants.append({
                "messages": [
                    {"role": "system", "content": self.config.system_prompt_sft},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": target},
                ],
                # Metadata fields for reward computation / debugging
                "meta": {
                    "source_length": self._word_count(source),
                    "target_length": wc,
                    "length_requirement": length_req,
                    "style": style,
                },
            })

        return variants

    # ------------------------------------------------------------------
    # Per-sample augmentation (GRPO prompt)
    # ------------------------------------------------------------------

    def augment_grpo_prompt(
        self, source: str, target: str,
        style_override: Optional[str] = None,
    ) -> Dict:
        """Generate ONE prompt for GRPO rollout.

        GRPO prompts have NO assistant response — the model generates
        multiple completions during training.

        Returns:
            Dict with "prompt" (list of messages) and "reference" (gold summary)
            and "meta" (length/style metadata for reward functions).
        """
        wc = self._word_count(target)
        # Use a random template
        length_req = self._make_length_req(wc, self._rng.randint(0, 100))
        style = style_override or self._rng.choice(self.config.grpo_styles)

        user_content = (
            f"Yêu cầu:\n"
            f"- Độ dài: {length_req}\n"
            f"- Phong cách: {style}\n"
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
                "length_requirement": length_req,
                "style": style,
            },
        }

    # ------------------------------------------------------------------
    # Test prompt (no augment, fixed style + length)
    # ------------------------------------------------------------------

    def make_test_prompt(
        self, source: str, target: str,
        style: str = "báo chí",
        length_template: int = 0,
    ) -> Dict:
        """Generate a fixed test prompt (no randomness)."""
        wc = self._word_count(target)
        length_req = self._make_length_req(wc, length_template)

        user_content = (
            f"Yêu cầu:\n"
            f"- Độ dài: {length_req}\n"
            f"- Phong cách: {style}\n"
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
                "length_requirement": length_req,
                "style": style,
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
    """Container for all data splits."""

    sft_train: DataSplit
    sft_val: DataSplit
    grpo_train: DataSplit
    grpo_val: DataSplit
    test: DataSplit

    def save_all(self, out_dir: str = "data") -> None:
        """Save all splits to JSONL files."""
        self.sft_train.save(os.path.join(out_dir, "sft_train.jsonl"))
        self.sft_val.save(os.path.join(out_dir, "sft_val.jsonl"))
        self.grpo_train.save(os.path.join(out_dir, "grpo_train.jsonl"))
        self.grpo_val.save(os.path.join(out_dir, "grpo_val.jsonl"))
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
    # 1. SFT TRAIN: VietNews train + WikiLingua train
    # ------------------------------------------------------------------
    sft_train_samples: List[Dict] = []

    logger.info("Loading VietNews/train for SFT...")
    vn_train = VietNewsDataset(raw_cfg, split="train")
    for i, sample in enumerate(vn_train):
        sft_train_samples.extend(augmenter.augment_sft(sample["source"], sample["target"]))
        if (i + 1) % 20000 == 0:
            logger.info(f"  Augmented {i+1}/{len(vn_train)} VietNews train samples")

    logger.info("Loading WikiLingua/train for SFT...")
    wl_train = WikiLinguaDataset(raw_cfg, split="train")
    for i, sample in enumerate(wl_train):
        sft_train_samples.extend(augmenter.augment_sft(sample["source"], sample["target"]))
        if (i + 1) % 5000 == 0:
            logger.info(f"  Augmented {i+1}/{len(wl_train)} WikiLingua train samples")

    # ------------------------------------------------------------------
    # 2. SFT VAL: first 2K VietNews val + first 500 WikiLingua val
    # ------------------------------------------------------------------
    sft_val_samples: List[Dict] = []

    vn_val = VietNewsDataset(raw_cfg, split="val")
    for i, sample in enumerate(vn_val):
        if i >= 2000:
            break
        sft_val_samples.extend(augmenter.augment_sft(sample["source"], sample["target"]))

    wl_val = WikiLinguaDataset(raw_cfg, split="val")
    for i, sample in enumerate(wl_val):
        if i >= 500:
            break
        sft_val_samples.extend(augmenter.augment_sft(sample["source"], sample["target"]))

    # ------------------------------------------------------------------
    # 3. GRPO TRAIN: remaining VietNews val + remaining WikiLingua val
    #    + VLSP train + ViMs 80%
    # ------------------------------------------------------------------
    grpo_train_samples: List[Dict] = []

    # Remaining VietNews val (after first 2000 taken for SFT val)
    for i, sample in enumerate(vn_val):
        if i < 2000:
            continue
        grpo_train_samples.append(augmenter.augment_grpo_prompt(sample["source"], sample["target"]))

    # Remaining WikiLingua val (after first 500 taken for SFT val)
    for i, sample in enumerate(wl_val):
        if i < 500:
            continue
        grpo_train_samples.append(augmenter.augment_grpo_prompt(sample["source"], sample["target"]))

    # VLSP train (285 samples)
    logger.info("Loading VLSP/train for GRPO...")
    try:
        vlsp_train = VLSPDataset(raw_cfg, split="train")
        for sample in vlsp_train:
            if sample["target"]:  # skip empty targets
                grpo_train_samples.append(
                    augmenter.augment_grpo_prompt(sample["source"], sample["target"])
                )
    except Exception as e:
        logger.warning(f"VLSP/train skipped: {e}")

    # ViMs 80% (240/300 clusters)
    logger.info("Loading ViMs for GRPO...")
    try:
        vims = ViMsDataset(raw_cfg, annotator_idx=0)
        vims_samples = list(vims)
        vims_rng = random.Random(42)
        vims_rng.shuffle(vims_samples)
        split_idx = int(len(vims_samples) * 0.8)
        for sample in vims_samples[:split_idx]:
            grpo_train_samples.append(
                augmenter.augment_grpo_prompt(sample["source"], sample["target"])
            )
    except Exception as e:
        logger.warning(f"ViMs skipped: {e}")

    # ------------------------------------------------------------------
    # 4. GRPO VAL: VLSP val + ViMs 20%
    # ------------------------------------------------------------------
    grpo_val_samples: List[Dict] = []

    # VLSP val (15 samples)
    try:
        vlsp_val = VLSPDataset(raw_cfg, split="val")
        for sample in vlsp_val:
            if sample["target"]:
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
    # 5. TEST: VietNews test + WikiLingua test + VLSP test+abmusu
    # ------------------------------------------------------------------
    test_samples: List[Dict] = []

    logger.info("Loading VietNews/test for TEST...")
    vn_test = VietNewsDataset(raw_cfg, split="test")
    count = 0
    for sample in vn_test:
        if count >= test_size_vn:
            break
        test_samples.append(augmenter.make_test_prompt(sample["source"], sample["target"]))
        count += 1

    logger.info("Loading WikiLingua/test for TEST...")
    wl_test = WikiLinguaDataset(raw_cfg, split="test")
    count = 0
    for sample in wl_test:
        if count >= test_size_wl:
            break
        test_samples.append(augmenter.make_test_prompt(sample["source"], sample["target"]))
        count += 1

    # VLSP test + abmusu
    for split_name in ["test", "abmusu"]:
        try:
            vlsp_test = VLSPDataset(raw_cfg, split=split_name)
            for sample in vlsp_test:
                test_samples.append(
                    augmenter.make_test_prompt(sample["source"], sample["target"])
                )
        except Exception as e:
            logger.warning(f"VLSP/{split_name} skipped: {e}")

    # ------------------------------------------------------------------
    # Summary & return
    # ------------------------------------------------------------------
    splits = AllSplits(
        sft_train=DataSplit("sft_train", sft_train_samples),
        sft_val=DataSplit("sft_val", sft_val_samples),
        grpo_train=DataSplit("grpo_train", grpo_train_samples),
        grpo_val=DataSplit("grpo_val", grpo_val_samples),
        test=DataSplit("test", test_samples),
    )

    logger.info(
        f"\n{'='*60}\n"
        f"Data splits summary:\n"
        f"  SFT train:  {len(splits.sft_train):>8,}  (after {augmenter.config.num_variants}x augment)\n"
        f"  SFT val:    {len(splits.sft_val):>8,}\n"
        f"  GRPO train: {len(splits.grpo_train):>8,}\n"
        f"  GRPO val:   {len(splits.grpo_val):>8,}\n"
        f"  Test:       {len(splits.test):>8,}\n"
        f"{'='*60}"
    )

    return splits


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    augmenter = PromptAugmenter()
    splits = build_all_splits(
        augmenter,
        data_root="VDT_Textsum",
        max_source_chars=8000,
        max_summary_chars=1500,
    )

    splits.save_all("data")

    # Print a few examples
    print("\n=== SFT Example ===")
    ex = splits.sft_train[0]
    print(f"Messages: {len(ex['messages'])} roles")
    print(f"System:   {ex['messages'][0]['content'][:80]}...")
    print(f"User:     {ex['messages'][1]['content'][:120]}...")
    print(f"Assist:   {ex['messages'][2]['content'][:80]}...")
    print(f"Meta:     length_req={ex['meta']['length_requirement']}, style={ex['meta']['style']}")

    print("\n=== GRPO Example ===")
    ex = splits.grpo_train[0]
    print(f"Prompt: {len(ex['prompt'])} messages")
    print(f"  User: {ex['prompt'][1]['content'][:120]}...")
    print(f"  Reference: {ex['reference'][:60]}...")
    print(f"Meta: length_req={ex['meta']['length_requirement']}, style={ex['meta']['style']}")
