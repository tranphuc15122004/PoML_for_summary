#!/usr/bin/env python3
"""Generate the section 5.2 baseline-improvement heatmap."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


BASELINES = [
    ("raw_base", "Qwen3-4B-Base (raw)"),
    ("raw_inst", "Qwen3-4B-Instruct (raw)"),
    ("vietai", "VietAI"),
    ("gpt4o", "GPT-4o"),
    ("gpt35", "GPT-3.5-Turbo"),
    ("q14", "Qwen3-14B"),
    ("l70", "Llama3.3-70B"),
    ("p14", "Phi4-14B"),
    ("s20", "Sailor-20B"),
    ("v7", "VinBigdata-7B"),
]

DATASETS = ["VietNews", "WikiLingua", "ViMs", "VLSP"]
METRIC_LABELS = ["$\Delta$R2", "LenRed"]

ROUGE = {
    "base_final": {"VietNews": 28.97, "WikiLingua": 26.35, "ViMs": 18.16, "VLSP": 17.46},
    "inst_final": {"VietNews": 29.62, "WikiLingua": 25.59, "ViMs": 21.27, "VLSP": 24.39},
    "raw_base": {"VietNews": 12.13, "WikiLingua": 3.18, "ViMs": 13.46, "VLSP": 9.49},
    "raw_inst": {"VietNews": 16.48, "WikiLingua": 7.44, "ViMs": 17.83, "VLSP": 15.51},
    "vietai": {"VietNews": 34.24, "WikiLingua": 33.12, "ViMs": None, "VLSP": None},
    "gpt4o": {"VietNews": 21.61, "WikiLingua": 20.65, "ViMs": 44.26, "VLSP": 43.37},
    "gpt35": {"VietNews": 28.13, "WikiLingua": 21.09, "ViMs": 19.28, "VLSP": 35.79},
    "q14": {"VietNews": 18.51, "WikiLingua": 20.24, "ViMs": 44.38, "VLSP": 44.27},
    "l70": {"VietNews": 22.17, "WikiLingua": 19.40, "ViMs": 37.54, "VLSP": 39.02},
    "p14": {"VietNews": 13.17, "WikiLingua": 14.28, "ViMs": 41.85, "VLSP": 42.31},
    "s20": {"VietNews": 17.70, "WikiLingua": 18.74, "ViMs": 39.47, "VLSP": 40.96},
    "v7": {"VietNews": 20.59, "WikiLingua": 21.02, "ViMs": 37.98, "VLSP": 40.23},
}

LENDIST = {
    "base_final": {"VietNews": 1.16, "WikiLingua": 8.35, "ViMs": 89.23, "VLSP": 100.50},
    "inst_final": {"VietNews": 1.52, "WikiLingua": 10.25, "ViMs": 84.45, "VLSP": 82.71},
    "raw_base": {"VietNews": 33.03, "WikiLingua": 61.59, "ViMs": 92.01, "VLSP": 120.33},
    "raw_inst": {"VietNews": 4.01, "WikiLingua": 23.42, "ViMs": 59.80, "VLSP": 70.17},
    "vietai": {"VietNews": None, "WikiLingua": None, "ViMs": None, "VLSP": None},
    "gpt4o": {"VietNews": 8.0, "WikiLingua": 11.0, "ViMs": 187.0, "VLSP": 58.0},
    "gpt35": {"VietNews": 18.0, "WikiLingua": 17.0, "ViMs": 47.0, "VLSP": 30.0},
    "q14": {"VietNews": 50.0, "WikiLingua": 45.0, "ViMs": 131.0, "VLSP": 34.0},
    "l70": {"VietNews": 14.0, "WikiLingua": 13.0, "ViMs": 186.0, "VLSP": 73.0},
    "p14": {"VietNews": 205.0, "WikiLingua": 214.0, "ViMs": 166.0, "VLSP": 134.0},
    "s20": {"VietNews": 38.0, "WikiLingua": 43.0, "ViMs": 268.0, "VLSP": 41.0},
    "v7": {"VietNews": 115.0, "WikiLingua": 89.0, "ViMs": 98.0, "VLSP": 82.0},
}

PANELS = {
    "Base-v5": "base_final",
    "Inst-v5": "inst_final",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the section 5.2 baseline-improvement heatmap."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FormalReport_VDT/latex/figures/section52_tradeoff.pdf"),
        help="Path to the output PDF.",
    )
    return parser.parse_args()


def format_value(value):
    if value is None:
        return "--"
    if abs(value) >= 100:
        return "{:+.0f}".format(value)
    return "{:+.1f}".format(value)


def compute_improvement(final_key, baseline_key, dataset, metric):
    if metric == "rouge":
        final_value = ROUGE[final_key][dataset]
        baseline_value = ROUGE[baseline_key][dataset]
        if final_value is None or baseline_value in (None, 0):
            return None
        return ((final_value - baseline_value) / baseline_value) * 100.0

    final_value = LENDIST[final_key][dataset]
    baseline_value = LENDIST[baseline_key][dataset]
    if final_value is None or baseline_value in (None, 0):
        return None
    return ((baseline_value - final_value) / baseline_value) * 100.0


def build_panel_values(final_key):
    rows = []
    for baseline_key, _ in BASELINES:
        row = []
        for dataset in DATASETS:
            row.append(compute_improvement(final_key, baseline_key, dataset, "rouge"))
            row.append(compute_improvement(final_key, baseline_key, dataset, "length"))
        rows.append(row)
    return rows


def plot_panel(ax, title, values, show_y, norm):
    data = np.array([[np.nan if cell is None else cell for cell in row] for row in values], dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="#e6e6e6")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xticks(range(len(DATASETS) * len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS * len(DATASETS), fontsize=8.3)
    ax.set_yticks(range(len(BASELINES)))
    if show_y:
        ax.set_yticklabels([label for _, label in BASELINES], fontsize=8.8)
        for idx, tick in enumerate(ax.get_yticklabels()):
            if idx < 2:
                tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.tick_params(top=False, bottom=True, left=show_y, right=False, length=0)

    ax.set_xticks(np.arange(-0.5, len(DATASETS) * len(METRIC_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BASELINES), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for xpos in (1.5, 3.5, 5.5):
        ax.axvline(xpos, color="#444444", linewidth=1.4)
    ax.axhline(1.5, color="#444444", linewidth=1.4)

    for dataset_idx, dataset in enumerate(DATASETS):
        center = dataset_idx * 2 + 0.5
        ax.text(
            center,
            1.03,
            dataset,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            text = format_value(value)
            if value is None:
                text_color = "#666666"
            else:
                text_color = "white" if abs(value) >= 80 else "black"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=8.1,
                fontweight="bold" if row_idx < 2 else "normal",
                color=text_color,
            )

    return image



def build_figure(output):
    panel_values = {title: build_panel_values(final_key) for title, final_key in PANELS.items()}
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=300)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), sharey=True, constrained_layout=True)
    image = None
    for idx, (title, values) in enumerate(panel_values.items()):
        image = plot_panel(axes[idx], title, values, show_y=(idx == 0), norm=norm)

    axes[0].set_ylabel("Comparison baselines", fontsize=10)

    cbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02)
    cbar.ax.set_ylabel("Improvement over baseline (%)", rotation=90, fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), bbox_inches="tight")
    plt.close(fig)



def main():
    args = parse_args()
    build_figure(args.output)


if __name__ == "__main__":
    main()
