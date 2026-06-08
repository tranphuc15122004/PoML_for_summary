"""
Analyze summary length distribution across all VDT_Textsum datasets.

Computes word count statistics per dataset/split and produces:
    - Histogram data (JSON)
    - Percentiles (p10, p25, p50, p75, p90, p95, p99)
    - Per-dataset summary table

Usage:
    python src/dataset/length_profiler.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Add project root and src dir to sys.path so imports work regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)          # src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)       # project root
# Ensure src/ comes before script directory in sys.path for correct package resolution.
for p in [_PROJECT_ROOT, _SRC_DIR]:
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from dataset.dataset import (
    DatasetConfig,
    VietNewsDataset,
    WikiLinguaDataset,
    ViMsDataset,
    VLSPDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Statistics Computation
# ==============================================================================

def compute_word_counts(dataset, max_samples: int = None) -> List[int]:
    """Compute word count for every target summary in a dataset.

    Args:
        dataset: A BaseSummarizationDataset instance.
        max_samples: Limit to this many samples for quick profiling.

    Returns:
        Sorted list of word counts.
    """
    counts: List[int] = []
    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        # word count on raw target (before chat template conversion)
        target = sample["target"]
        wc = len(target.split())
        counts.append(wc)
    return sorted(counts)


def compute_stats(word_counts: List[int], name: str = "") -> Dict:
    """Compute statistics from a list of word counts.

    Args:
        word_counts: Sorted list of word counts.
        name: Dataset name for logging.

    Returns:
        Dict with keys: name, count, min, max, mean, std, median,
                        p10, p25, p75, p90, p95, p99, histogram.
    """
    arr = np.array(word_counts)
    stats: Dict = {
        "name": name,
        "count": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(round(arr.mean(), 2)),
        "std": float(round(arr.std(), 2)),
        "median": int(np.median(arr)),
        "p10": int(np.percentile(arr, 10)),
        "p25": int(np.percentile(arr, 25)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }

    # Build histogram (buckets of 10 words up to 200, then 50-word buckets)
    histogram: List[Dict] = []
    boundaries = list(range(0, 201, 10)) + list(range(250, 1001, 50))
    lo = 0
    for hi in boundaries[1:]:
        cnt = int(np.sum((arr >= lo) & (arr < hi)))
        histogram.append({"range": f"{lo}-{hi-1}", "count": cnt})
        lo = hi
    cnt = int(np.sum(arr >= lo))
    histogram.append({"range": f"{lo}+", "count": cnt})
    stats["histogram"] = histogram

    logger.info(
        f"[{name or '?'}] n={stats['count']:,}  "
        f"median={stats['median']}  "
        f"mean={stats['mean']}  "
        f"p10={stats['p10']}  p90={stats['p90']}  "
        f"min={stats['min']}  max={stats['max']}"
    )
    return stats


# ==============================================================================
# Main
# ==============================================================================

def profile_all(config: DatasetConfig, max_samples: int = None) -> List[Dict]:
    """Profile all dataset splits and return a list of statistics dicts."""
    all_stats: List[Dict] = []

    # 1. VietNews
    for split in ["train", "val", "test"]:
        try:
            cfg = DatasetConfig(**{**config.__dict__, "mode": "raw"})
            ds = VietNewsDataset(cfg, split=split)
            counts = compute_word_counts(ds, max_samples)
            all_stats.append(compute_stats(counts, f"VietNews/{split}"))
        except Exception as e:
            logger.warning(f"VietNews/{split} skipped: {e}")

    # 2. WikiLingua
    for split in ["train", "val", "test"]:
        try:
            cfg = DatasetConfig(**{**config.__dict__, "mode": "raw"})
            ds = WikiLinguaDataset(cfg, split=split)
            counts = compute_word_counts(ds, max_samples)
            all_stats.append(compute_stats(counts, f"WikiLingua/{split}"))
        except Exception as e:
            logger.warning(f"WikiLingua/{split} skipped: {e}")

    # 3. ViMs (both annotators)
    for ann in [0, 1]:
        try:
            cfg = DatasetConfig(**{**config.__dict__, "mode": "raw"})
            ds = ViMsDataset(cfg, annotator_idx=ann)
            counts = compute_word_counts(ds, max_samples)
            all_stats.append(compute_stats(counts, f"ViMs/annotator_{ann}"))
        except Exception as e:
            logger.warning(f"ViMs/annotator_{ann} skipped: {e}")

    # 4. VLSP
    for split in ["train", "val", "abmusu"]:
        try:
            cfg = DatasetConfig(**{**config.__dict__, "mode": "raw"})
            ds = VLSPDataset(cfg, split=split)
            counts = compute_word_counts(ds, max_samples)
            all_stats.append(compute_stats(counts, f"VLSP/{split}"))
        except Exception as e:
            logger.warning(f"VLSP/{split} skipped: {e}")

    return all_stats


def print_summary_table(all_stats: List[Dict]) -> None:
    """Print a formatted summary table."""
    header = f"{'Dataset':<24} {'n':>8} {'min':>5} {'p10':>5} {'p50':>5} {'p90':>5} {'max':>6} {'mean':>7} {'std':>7}"
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("SUMMARY LENGTH STATISTICS (word count per summary)")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for s in all_stats:
        print(
            f"{s['name']:<24} {s['count']:>8,} {s['min']:>5} {s['p10']:>5} "
            f"{s['median']:>5} {s['p90']:>5} {s['max']:>6} {s['mean']:>7} {s['std']:>7}"
        )
    print(sep)

    # Combined histogram for all datasets
    total_counts = sum((s["histogram"][i]["count"] for s in all_stats for i in range(len(s["histogram"]))))
    _print_combined_histogram(all_stats)


def _print_combined_histogram(all_stats: List[Dict]) -> None:
    """Print a combined histogram using the first dataset's buckets."""
    if not all_stats:
        return
    buckets = all_stats[0]["histogram"]
    print(f"\n{'=' * 60}")
    print("COMBINED HISTOGRAM (all datasets)")
    print(f"{'=' * 60}")
    for i, bucket in enumerate(buckets):
        total = sum(s["histogram"][i]["count"] for s in all_stats if i < len(s["histogram"]))
        if total == 0:
            continue
        bar_len = min(total // max(sum(s["count"] for s in all_stats) // 50, 1), 50)
        bar = "█" * bar_len
        print(f"{bucket['range']:>12}: {bar} {total:,}")


def save_stats(all_stats: List[Dict], save_path: str = "data/length_stats.json") -> None:
    """Save all statistics to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved statistics to {save_path}")


if __name__ == "__main__":
    config = DatasetConfig(
        data_root="VDT_Textsum",
        max_source_chars=0,
        max_summary_chars=0,
    )

    os.makedirs("data", exist_ok=True)
    all_stats = profile_all(config, max_samples=None)
    print_summary_table(all_stats)
    save_stats(all_stats, "data/length_stats.json")
