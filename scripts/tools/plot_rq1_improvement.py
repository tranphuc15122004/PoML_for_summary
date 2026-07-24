#!/usr/bin/env python3
"""Plot baseline-relative improvements for the RQ1 benchmark."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["VietNews", "WikiLingua"]
METRICS = [r"$\Delta$R2$\uparrow$", r"LenRed$\uparrow$"]
BASELINES = [
    "Qwen3-4B-Base (raw)",
    "Qwen3-4B-Instruct (raw)",
    "GPT-4o",
    "GPT-3.5-Turbo",
    "Qwen3-14B",
    "Llama3.3-70B-Instruct",
    "Phi4-14B",
    "Sailor-20B-chat",
    "VinBigdata (7B)",
]

# Values transcribed from the former appendix Table 12. Columns are
# VietNews (Delta R2, LenRed), then WikiLingua (Delta R2, LenRed).
BASE_FINAL = np.array(
    [
        [138.8, 96.5, 728.6, 86.4],
        [75.8, 71.1, 254.2, 64.3],
        [34.1, 85.5, 27.6, 24.1],
        [3.0, 93.6, 24.9, 50.9],
        [56.5, 97.7, 30.2, 81.4],
        [30.7, 91.7, 35.8, 35.8],
        [119.9, 99.4, 84.5, 96.1],
        [63.7, 96.9, 40.6, 80.6],
        [40.7, 99.0, 25.4, 90.6],
    ]
)
INSTRUCT_FINAL = np.array(
    [
        [144.2, 95.4, 704.7, 83.4],
        [79.7, 62.1, 243.9, 56.2],
        [37.1, 81.0, 23.9, 6.8],
        [5.3, 91.6, 21.3, 39.7],
        [60.0, 97.0, 26.4, 77.2],
        [33.6, 89.1, 31.9, 21.2],
        [124.9, 99.3, 79.2, 95.2],
        [67.3, 96.0, 36.6, 76.2],
        [43.9, 98.7, 21.7, 88.5],
    ]
)


def plot(output: Path) -> None:
    data = [BASE_FINAL, INSTRUCT_FINAL]
    titles = [
        "Qwen3-4B-Base + SFT + GRPO v5",
        "Qwen3-4B-Instruct + SFT + GRPO v5",
    ]
    column_labels = [
        f"{dataset}\n{metric}"
        for dataset in DATASETS
        for metric in METRICS
    ]

    # Do not use sharey=True here. With shared y axes, assigning [] as the
    # labels of the second axis can also clear the labels of the first axis.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 7.2),
        gridspec_kw={"wspace": 0.08},
    )

    cmap = plt.cm.YlOrRd
    norm = colors.Normalize(vmin=0, vmax=200)
    image = None

    for index, (axis, values, title) in enumerate(zip(axes, data, titles)):
        image = axis.imshow(
            np.clip(values, 0, 200),
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        axis.set_title(title, fontsize=11, fontweight="bold", pad=10)
        axis.set_xticks(np.arange(4))
        axis.set_xticklabels(column_labels, fontsize=8.5)
        axis.tick_params(axis="x", length=0, pad=5)

        axis.set_yticks(np.arange(len(BASELINES)))
        if index == 0:
            axis.set_yticklabels(BASELINES, fontsize=8.8)
            axis.tick_params(axis="y", length=0, pad=5)
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", length=0)

        # Cell borders.
        axis.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
        axis.grid(
            which="minor",
            color="#333333",
            linestyle="-",
            linewidth=1.0,
        )
        axis.tick_params(which="minor", bottom=False, left=False)

        # Stronger separators between dataset groups and model groups.
        axis.axvline(1.5, color="#222222", linewidth=2.0)
        axis.axhline(1.5, color="#222222", linewidth=2.0)

        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(len(BASELINES) - 0.5, -0.5)

        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                text_color = "white" if value >= 110 else "black"
                axis.text(
                    column,
                    row,
                    f"+{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    fontweight="bold",
                    color=text_color,
                )

    # Reserve a dedicated axis so the colorbar does not overlap the right heatmap.
    fig.subplots_adjust(left=0.23, right=0.89, top=0.89, bottom=0.12)
    colorbar_axis = fig.add_axes([0.915, 0.16, 0.018, 0.68])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("% Improvement", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Figure saved to {output}")


if __name__ == "__main__":
    plot(Path("FormalReport_VDT/latex/figures/rq1_improvement.pdf"))