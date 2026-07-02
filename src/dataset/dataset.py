"""
Dataset classes for post-training an LLM on Vietnamese text summarization.

Supports 4 datasets from VDT_Textsum:
    1. VietNews    – Single-doc abstractive summarization (~150K samples)
    2. ViMs        – Multi-doc abstractive summarization (300 clusters)
    3. VLSP        – Multi-doc extractive summarization (~600 samples)
    4. WikiLingua  – Single-doc abstractive summarization (~19.5K samples)

Each dataset can generate data in two modes:
    - SFT (chat format): {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    - Preference (GRPO/DPO): {"chosen": [...], "rejected": [...]}
"""

from __future__ import annotations

import json
import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class DatasetConfig:
    """Base configuration for all datasets."""

    # Path to the root VDT_Textsum directory
    data_root: str = "VDT_Textsum"

    # Output mode: "sft" for supervised fine-tuning, "preference" for GRPO/DPO,
    # "raw" for raw {source, target} dicts (used by length_profiler etc.)
    mode: str = "sft"

    # System prompt for summarization (used in SFT mode)
    system_prompt: str = (
        "Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. "
        "Hãy tạo ra bản tóm tắt ngắn gọn, chính xác, giữ được ý chính của văn bản gốc."
    )

    # Instruction template for single-document summarization
    single_doc_instruction: str = (
        "Hãy tóm tắt văn bản sau đây một cách ngắn gọn:\n\n{source}"
    )

    # Instruction template for multi-document summarization
    multi_doc_instruction: str = (
        "Hãy tóm tắt các văn bản sau đây thành một bản tóm tắt duy nhất, "
        "bao quát được các ý chính từ tất cả các văn bản:\n\n{source}"
    )

    # Max source length in characters (truncate if longer), 0 = no limit
    max_source_chars: int = 8000

    # Max summary length in characters, 0 = no limit
    max_summary_chars: int = 1500

    # Replace underscore word-segmentation markers with spaces
    replace_underscores: bool = True

    # Seed for reproducibility
    seed: int = 42


# ==============================================================================
# Base Dataset (ABC)
# ==============================================================================

class BaseSummarizationDataset(ABC):
    """Abstract base class for all summarization datasets.

    Subclasses must implement:
        - _raw_samples() -> Iterator[Dict]: yield raw {source, target} dicts
        - name -> str: dataset name
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._samples: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""
        ...

    @abstractmethod
    def _raw_samples(self) -> Iterator[Dict[str, str]]:
        """Yield raw dicts with keys 'source' and 'target'."""
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._load_samples())

    def __getitem__(self, idx: int) -> Dict:
        return self._convert_to_output(self._load_samples()[idx])

    def __iter__(self):
        for sample in self._load_samples():
            yield self._convert_to_output(sample)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self) -> List[Dict]:
        """Lazy-load and cache raw samples."""
        if self._samples is None:
            self._samples = list(self._raw_samples())
            logger.info(f"[{self.name}] Loaded {len(self._samples)} raw samples")
        return self._samples

    def _convert_to_output(self, raw: Dict[str, str]) -> Dict:
        """Convert a raw {source, target} dict to the configured output format."""
        source = self._truncate(raw["source"], self.config.max_source_chars)
        target = self._truncate(raw["target"], self.config.max_summary_chars)

        if self.config.replace_underscores:
            source = source.replace("_", " ")
            target = target.replace("_", " ")

        if self.config.mode == "sft":
            return self._make_sft(source, target)
        elif self.config.mode == "preference":
            return self._make_preference(source, target)
        elif self.config.mode == "raw":
            return {"source": source, "target": target}
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def _make_sft(self, source: str, target: str) -> Dict:
        """Build SFT chat-format sample."""
        instruction = self._get_instruction(source)
        return {
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": target},
            ]
        }

    def _make_preference(self, source: str, target: str) -> Dict:
        """Build preference-format sample (single chosen, no rejected = placeholder).

        In practice, you'd want to pair with a weaker model's output as 'rejected'.
        """
        instruction = self._get_instruction(source)
        return {
            "prompt": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": instruction},
            ],
            "chosen": [{"role": "assistant", "content": target}],
            "rejected": [{"role": "assistant", "content": ""}],
        }

    def _get_instruction(self, source: str) -> str:
        """Subclasses can override to provide different instruction templates."""
        return self.config.single_doc_instruction.format(source=source)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars]
        return text

    @staticmethod
    def _resolve_path(*parts: str) -> str:
        return os.path.join(*parts)


# ==============================================================================
# 1. VietNews Dataset
# ==============================================================================

class VietNewsDataset(BaseSummarizationDataset):
    """Single-document abstractive summarization from Vietnamese news.

    Data layout:
        vietnews-master/data/
        ├── train_tokenized/   (*.txt.seg)
        ├── val_tokenized/
        └── test_tokenized/

    Each .txt.seg file:
        Line 1: Title        → target / summary
        Line 2: (blank)
        Line 3: Sapo/Lead    → source (part 1)
        Line 4: (blank)
        Line 5+: Body        → source (part 2)
        Last line: Caption   → ignore
    """

    SPLIT_MAP = {
        "train": "train_tokenized",
        "val": "val_tokenized",
        "test": "test_tokenized",
    }

    def __init__(self, config: DatasetConfig, split: str = "train"):
        super().__init__(config)
        self.split = split
        if split not in self.SPLIT_MAP:
            raise ValueError(f"split must be one of {list(self.SPLIT_MAP)}, got '{split}'")

    @property
    def name(self) -> str:
        return f"VietNews/{self.split}"

    @property
    def data_dir(self) -> str:
        return self._resolve_path(
            self.config.data_root,
            "vietnews-master/data",
            self.SPLIT_MAP[self.split],
        )

    def _raw_samples(self) -> Iterator[Dict[str, str]]:
        if not os.path.isdir(self.data_dir):
            logger.warning(f"[{self.name}] Directory not found: {self.data_dir}")
            return

        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".seg"):
                continue
            file_path = os.path.join(self.data_dir, fname)
            sample = self._parse_file(file_path)
            if sample is not None:
                yield sample

    def _parse_file(self, file_path: str) -> Optional[Dict[str, str]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"[{self.name}] Error reading {file_path}: {e}")
            return None

        if len(lines) < 1:
            return None

        # Line 1: title → summary
        title = lines[0].strip()

        # Line 3: sapo (if exists), line 5+: body (skip caption = last line)
        sapo = lines[2].strip() if len(lines) >= 3 else ""
        body_lines: List[str] = []
        if len(lines) >= 5:
            # Skip the last line (caption)
            body_lines = [l.strip() for l in lines[4:-1] if l.strip()]

        # Build source: sapo + body
        source_parts = [p for p in [sapo] + body_lines if p]
        source = "\n".join(source_parts)

        if not source or not title:
            return None

        return {"source": source, "target": title}


# ==============================================================================
# 2. WikiLingua Dataset
# ==============================================================================

class WikiLinguaDataset(BaseSummarizationDataset):
    """Single-document abstractive summarization from WikiLingua (Vietnamese).

    Data layout:
        wikilingua/
        ├── train.json
        ├── val.json
        └── test.json

    Each JSON line: {"src": ["sentence1", ...], "tgt": ["summary_sentence1", ...]}
    """

    SPLIT_MAP = {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    }

    def __init__(self, config: DatasetConfig, split: str = "train"):
        super().__init__(config)
        self.split = split
        if split not in self.SPLIT_MAP:
            raise ValueError(f"split must be one of {list(self.SPLIT_MAP)}, got '{split}'")

    @property
    def name(self) -> str:
        return f"WikiLingua/{self.split}"

    @property
    def file_path(self) -> str:
        return self._resolve_path(
            self.config.data_root,
            "wikilingua",
            self.SPLIT_MAP[self.split],
        )

    def _raw_samples(self) -> Iterator[Dict[str, str]]:
        if not os.path.isfile(self.file_path):
            logger.warning(f"[{self.name}] File not found: {self.file_path}")
            return

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                src = sample.get("src", [])
                tgt = sample.get("tgt", [])

                source = " ".join(src) if isinstance(src, list) else str(src)
                target = " ".join(tgt) if isinstance(tgt, list) else str(tgt)

                if source and target:
                    yield {"source": source, "target": target}


# ==============================================================================
# 3. ViMs Dataset (Multi-document)
# ==============================================================================

class ViMsDataset(BaseSummarizationDataset):
    """Multi-document abstractive summarization (Vietnamese).

    Data layout:
        ViMs-Dataset-master/ViMs/
        ├── original/     (300 clusters, each: Cluster_XXX/original/*.txt)
        ├── summary/      (gold summaries: 0.gold.txt, 1.gold.txt per cluster)

    Each original .txt file has metadata headers + Content.
    """

    def __init__(self, config: DatasetConfig, annotator_idx: int = 0):
        """
        Args:
            annotator_idx: Which gold summary to use (0 or 1).
        """
        super().__init__(config)
        self.annotator_idx = annotator_idx

    @property
    def name(self) -> str:
        return f"ViMs/annotator_{self.annotator_idx}"

    @property
    def _base_dir(self) -> str:
        return self._resolve_path(
            self.config.data_root,
            "ViMs-Dataset-master/ViMs",
        )

    def _raw_samples(self) -> Iterator[Dict[str, str]]:
        original_dir = os.path.join(self._base_dir, "original")
        summary_dir = os.path.join(self._base_dir, "summary")

        if not os.path.isdir(original_dir):
            logger.warning(f"[{self.name}] Original dir not found: {original_dir}")
            return

        for cluster_name in sorted(os.listdir(original_dir)):
            cluster_orig = os.path.join(original_dir, cluster_name, "original")
            if not os.path.isdir(cluster_orig):
                continue

            # Read all documents in this cluster
            documents = []
            for fname in sorted(
                os.listdir(cluster_orig),
                key=lambda x: int(re.match(r"\d+", x).group()) if re.match(r"\d+", x) else 0,
            ):
                fpath = os.path.join(cluster_orig, fname)
                if os.path.isfile(fpath):
                    doc = self._parse_doc(fpath)
                    if doc and doc.get("content"):
                        documents.append(doc)

            if not documents:
                continue

            # Build source from all documents in the cluster
            source_parts = []
            for i, doc in enumerate(documents):
                header = f"[Tài liệu {i+1}] {doc.get('title', '')}"
                source_parts.append(f"{header}\n{doc['content']}")
            source = "\n\n---\n\n".join(source_parts)

            # Read gold summary (files are directly in summary/Cluster_XXX/)
            gold_file = os.path.join(summary_dir, cluster_name,
                                     f"{self.annotator_idx}.gold.txt")
            target = ""
            if os.path.isfile(gold_file):
                with open(gold_file, "r", encoding="utf-8") as f:
                    target = f.read().strip()
            else:
                # Fallback to any available gold file in the same directory
                summary_cluster_dir = os.path.join(summary_dir, cluster_name)
                if os.path.isdir(summary_cluster_dir):
                    for gf in sorted(os.listdir(summary_cluster_dir)):
                        if gf.endswith(".gold.txt"):
                            with open(os.path.join(summary_cluster_dir, gf), "r", encoding="utf-8") as f:
                                target = f.read().strip()
                            break

            if source and target:
                yield {"source": source, "target": target}

    def _parse_doc(self, file_path: str) -> Dict[str, str]:
        """Parse a single ViMs document file with metadata headers."""
        sample: Dict[str, str] = {
            "title": "", "source": "", "link": "",
            "published_date": "", "author": "", "tags": "",
            "summary": "", "content": "",
        }
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return sample

        # Split header and content at "Content:"
        content_match = re.search(r"^Content:\s*\n", text, re.MULTILINE)
        if content_match:
            header = text[: content_match.start()]
            content = text[content_match.end() :]
        else:
            header = text
            content = ""

        for line in header.splitlines():
            m = re.match(r"^([A-Za-z][A-Za-z _]*?):\s*(.*)$", line)
            if not m:
                continue
            key = m.group(1).strip().lower().replace(" ", "_")
            if key in sample:
                sample[key] = m.group(2).strip()

        sample["content"] = content.strip()
        return sample

    def _get_instruction(self, source: str) -> str:
        return self.config.multi_doc_instruction.format(source=source)


# ==============================================================================
# 4. VLSP Dataset (Multi-document, Extractive)
# ==============================================================================

class VLSPDataset(BaseSummarizationDataset):
    """Multi-document summarization from VLSP 2022 AbMuSu shared task.

    Data layout:
        vlsp/vlsp/
        ├── train.label.jsonl
        ├── val.label.jsonl
        ├── test.label.jsonl
        └── vlsp_2022_abmusu.label.jsonl

    JSON schema:
        {
            "id": int,
            "text": [[title, sent1, sent2, ...], [title, sent1, ...], ...],
            "summary": [sentence1, sentence2, ...],   # abstractive, human-written
            "label": [flat_index_of_summary_sentences] # extractive indices
        }

    Note: test.label.jsonl has placeholder labels = [0] and no summary field.

    Target priority:
        1. Use `summary` field if present (abstractive, human-written).
        2. Fall back to `label`-based extraction if `summary` is missing/empty.
        3. test.label.jsonl has no gold summary → yields empty target.
    """

    SPLIT_MAP = {
        "train": "train.label.jsonl",
        "val": "val.label.jsonl",
        "test": "test.label.jsonl",
        "abmusu": "vlsp_2022_abmusu.label.jsonl",
    }

    def __init__(self, config: DatasetConfig, split: str = "train"):
        super().__init__(config)
        self.split = split
        if split not in self.SPLIT_MAP:
            raise ValueError(f"split must be one of {list(self.SPLIT_MAP)}, got '{split}'")

    @property
    def name(self) -> str:
        return f"VLSP/{self.split}"

    @property
    def file_path(self) -> str:
        return self._resolve_path(
            self.config.data_root, "vlsp", self.SPLIT_MAP[self.split]
        )

    def _raw_samples(self) -> Iterator[Dict[str, str]]:
        if not os.path.isfile(self.file_path):
            logger.warning(f"[{self.name}] File not found: {self.file_path}")
            return

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                docs = sample.get("text", [])
                summary = sample.get("summary", [])
                labels = sample.get("label", [])

                if not docs:
                    continue

                # Build source: each document as a block
                source_parts = []
                flat_sentences: List[str] = []

                for doc_idx, doc in enumerate(docs):
                    if not doc:
                        continue
                    title = doc[0] if doc else ""
                    sentences = doc[1:] if len(doc) > 1 else []

                    block = f"[Tài liệu {doc_idx + 1}] {title}\n" + " ".join(sentences)
                    source_parts.append(block)

                    # Build flat sentence index for label resolution (fallback)
                    if title:
                        flat_sentences.append(title)
                    flat_sentences.extend(sentences)

                source = "\n\n---\n\n".join(source_parts)

                # Build target:
                #   1. Prefer human-written `summary` (abstractive) if available.
                #   2. Fall back to `label`-based extraction (extractive).
                #   3. test.label.jsonl has placeholder labels — skip.
                if summary:
                    target = " ".join(summary)
                elif labels and not (len(labels) == 1 and labels[0] == 0 and self.split == "test"):
                    summary_sentences = [
                        flat_sentences[i] for i in labels if 0 <= i < len(flat_sentences)
                    ]
                    target = " ".join(summary_sentences)
                else:
                    target = ""

                if source:
                    yield {"source": source, "target": target}

    def _get_instruction(self, source: str) -> str:
        return self.config.multi_doc_instruction.format(source=source)


# ==============================================================================
# 5. Merged Dataset
# ==============================================================================

class MergedDataset:
    """Concatenate multiple BaseSummarizationDataset instances into one.

    Usage:
        merged = MergedDataset([vietnews, wikilingua])
        for sample in merged:
            ...
    """

    def __init__(self, datasets: List[BaseSummarizationDataset]):
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx: int) -> Dict:
        for d in self.datasets:
            if idx < len(d):
                return d[idx]
            idx -= len(d)
        raise IndexError("Index out of range")

    def __iter__(self):
        for d in self.datasets:
            yield from d

    @property
    def name(self) -> str:
        return "+".join(d.name for d in self.datasets)


# ==============================================================================
# 6. Dataset Builder (convenience factory)
# ==============================================================================

def build_dataset(
    dataset_type: str,
    config: Optional[DatasetConfig] = None,
    split: str = "train",
    **kwargs,
) -> BaseSummarizationDataset:
    """Factory function to build a dataset by name.

    Args:
        dataset_type: One of 'vietnews', 'wikilingua', 'vims', 'vlsp', 'merged'.
        config: DatasetConfig instance. Created with defaults if None.
        split: Data split: 'train', 'val', or 'test'.
        **kwargs: Passed to dataset constructor (e.g., annotator_idx for ViMs).

    Returns:
        A BaseSummarizationDataset subclass instance.
    """
    if config is None:
        config = DatasetConfig()

    registry = {
        "vietnews": VietNewsDataset,
        "wikilingua": WikiLinguaDataset,
        "vims": ViMsDataset,
        "vlsp": VLSPDataset,
    }

    if dataset_type == "merged":
        # Build all single-doc datasets merged
        single_doc = [
            VietNewsDataset(config, split=split),
            WikiLinguaDataset(config, split=split),
        ]
        return MergedDataset(single_doc)  # type: ignore[return-value]

    if dataset_type not in registry:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. Choose from {list(registry)} + 'merged'."
        )

    # ViMsDataset does not accept 'split'; only pass split for split-aware datasets
    cls = registry[dataset_type]
    if dataset_type == "vims":
        return cls(config, **kwargs)
    else:
        return cls(config, split=split, **kwargs)


def build_all_datasets(
    config: Optional[DatasetConfig] = None,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, BaseSummarizationDataset]:
    """Build all available datasets for each split.

    Returns:
        Dict mapping "dataset_name/split" → dataset instance.
    """
    if config is None:
        config = DatasetConfig()

    results: Dict[str, BaseSummarizationDataset] = {}

    for split in splits:
        try:
            results[f"vietnews/{split}"] = VietNewsDataset(config, split=split)
        except Exception as e:
            logger.warning(f"Could not build vietnews/{split}: {e}")

        try:
            results[f"wikilingua/{split}"] = WikiLinguaDataset(config, split=split)
        except Exception as e:
            logger.warning(f"Could not build wikilingua/{split}: {e}")

        try:
            results[f"vlsp/{split}"] = VLSPDataset(config, split=split)
        except Exception as e:
            logger.warning(f"Could not build vlsp/{split}: {e}")

    # ViMs has no train/val/test split
    try:
        results["vims/annotator_0"] = ViMsDataset(config, annotator_idx=0)
    except Exception as e:
        logger.warning(f"Could not build vims/annotator_0: {e}")

    try:
        results["vims/annotator_1"] = ViMsDataset(config, annotator_idx=1)
    except Exception as e:
        logger.warning(f"Could not build vims/annotator_0: {e}")

    return results


# ==============================================================================
# CLI demo
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = DatasetConfig(
        data_root="VDT_Textsum",
        mode="sft",
        max_source_chars=3000,
        max_summary_chars=500,
    )

    # --- Quick test for each dataset ---
    test_specs = [
        ("vietnews", {"split": "train"}),
        ("wikilingua", {"split": "train"}),
        ("vims", {"annotator_idx": 0}),
        ("vlsp", {"split": "train"}),
    ]
    for ds_type, kwargs in test_specs:
        try:
            ds = build_dataset(ds_type, config, **kwargs)
            print(f"\n{'='*60}")
            print(f"Dataset: {ds.name}  |  Samples: {len(ds)}")
            sample = ds[0]
            if config.mode == "sft":
                print(f"  System  : {sample['messages'][0]['content'][:80]}...")
                print(f"  User    : {sample['messages'][1]['content'][:100]}...")
                print(f"  Assistant: {sample['messages'][2]['content'][:100]}...")
        except Exception as e:
            print(f"\n[SKIP] {ds_type}: {e}")
