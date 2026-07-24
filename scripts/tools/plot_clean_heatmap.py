#!/usr/bin/env python3
"""Plot a clean single-document improvement heatmap for the main paper."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

DATASETS = ["VietNews", "WikiLingua"]
PANELS = {
    "Base final: SFT+GRPO v5": {
        "final_r2": [28.69, 26.31], "final_len": [2.3, 10.0],
        "baselines": [
            ("Raw Base", [12.88, 3.02], [30.3, 62.7]),
            ("Base SFT", [26.52, 24.95], [31.0, 33.5]),
            ("Base fresh v5", [20.85, 9.48], [3.9, 22.4]),
        ],
    },
    "Instruct final: SFT+GRPO v5": {
        "final_r2": [30.25, 25.97], "final_len": [1.5, 10.4],
        "baselines": [
            ("Raw Instruct", [16.52, 7.36], [4.0, 22.6]),
            ("Instruct SFT", [26.93, 24.09], [1.1, 10.0]),
            ("Instruct fresh v5", [18.64, 8.03], [3.0, 15.6]),
        ],
    },
}


def improvements(panel):
    values, labels = [], []
    for name, r2, length in panel["baselines"]:
        row = []
        for idx in range(2):
            row.extend([
                (panel["final_r2"][idx] - r2[idx]) / r2[idx] * 100.0,
                (length[idx] - panel["final_len"][idx]) / length[idx] * 100.0,
            ])
        values.append(row)
        labels.append(name)
    return np.asarray(values), labels


def fmt(value):
    return f"{value:+.0f}" if abs(value) >= 100 else f"{value:+.1f}"


def main():
    output = Path("FormalReport_VDT/latex/figures/section52_tradeoff.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=800)
    image = None
    for idx, (title, panel) in enumerate(PANELS.items()):
        values, labels = improvements(panel)
        ax = axes[idx]
        image = ax.imshow(values, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if idx == 0 else [], fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["VN $\\Delta$R2", "VN LenRed", "WL $\\Delta$R2", "WL LenRed"], fontsize=8.5)
        ax.tick_params(length=0)
        ax.set_ylabel("Matched baseline" if idx == 0 else "")
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axvline(1.5, color="#444444", linewidth=1.2)
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                value = values[r, c]
                ax.text(c, r, fmt(value), ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if abs(value) > 100 else "black")
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Relative improvement (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.5, -0.02, "$\\Delta$R2: higher ROUGE-2; LenRed: lower absolute length distance; positive is better.",
             ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Clean heatmap written to {output}")


if __name__ == "__main__":
    main()
