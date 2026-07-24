#!/usr/bin/env python3
"""Generate the section 5.3 stage-wise ablation trajectory figure."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the section 5.3 stage-wise ablation trajectory figure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FormalReport_VDT/latex/figures/section53_ablation.pdf"),
        help="Path to the output figure.",
    )
    return parser.parse_args()


def add_labels(ax, xs, ys, labels, offsets, colors, weight="bold"):
    if isinstance(colors, str):
        colors = [colors] * len(labels)
    for x, y, label, (dx, dy, ha), color in zip(xs, ys, labels, offsets, colors):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            color=color,
            fontweight=weight,
            ha=ha,
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.88,
            ),
        )


def draw_arrows(ax, xs, ys, color):
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.5,
                ls="-",
                mutation_scale=14,
            ),
            zorder=3,
        )


def style_axis(ax, title, xticks, xlim, ylim, show_ylabel=False):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_xaxis()
    ax.set_xlabel("Absolute Length Distance (words; lower is better)", fontsize=9.2)
    if show_ylabel:
        ax.set_ylabel("ROUGE-2 (%)", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=8)


def plot_panels(output_path):
    # Data from the original single-document evaluation in VDT_report.tex.
    # ROUGE-2 is y, LenDist is x; the x-axis is inverted so the desirable region is top-right.

    # VietNews: [Pretrained, SFT, SFT+GRPO v5]
    base_vn_x = [33.03, 5.73, 1.16]
    base_vn_y = [12.13, 26.61, 28.97]
    inst_vn_x = [4.01, 1.09, 1.52]
    inst_vn_y = [16.48, 26.70, 29.62]
    base_fresh_vn_x = 3.80
    base_fresh_vn_y = 20.73
    inst_fresh_vn_x = 2.95
    inst_fresh_vn_y = 18.67

    # WikiLingua: [Pretrained, SFT, SFT+GRPO v5]
    base_wl_x = [61.59, 14.18, 8.35]
    base_wl_y = [3.18, 24.43, 26.35]
    inst_wl_x = [23.42, 9.87, 10.25]
    inst_wl_y = [7.44, 23.93, 25.59]
    base_fresh_wl_x = 22.03
    base_fresh_wl_y = 9.71
    inst_fresh_wl_x = 16.12
    inst_fresh_wl_y = 8.09

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    color_base = "#1f77b4"
    color_inst = "#2ca02c"
    color_fresh_base = "#d62728"
    color_fresh_inst = "#ff7f0e"

    # Panel 1: VietNews
    ax1.plot(
        base_vn_x,
        base_vn_y,
        marker="o",
        markersize=7,
        linewidth=2,
        color=color_base,
        label="Base warm-start trajectory",
    )
    ax1.plot(
        inst_vn_x,
        inst_vn_y,
        marker="s",
        markersize=7,
        linewidth=2,
        color=color_inst,
        label="Instruct warm-start trajectory",
    )
    draw_arrows(ax1, base_vn_x, base_vn_y, color_base)
    draw_arrows(ax1, inst_vn_x, inst_vn_y, color_inst)
    ax1.scatter(
        base_fresh_vn_x,
        base_fresh_vn_y,
        marker="^",
        s=110,
        color=color_fresh_base,
        edgecolor="black",
        zorder=4,
        label="Base fresh GRPO v5 (no SFT)",
    )
    ax1.scatter(
        inst_fresh_vn_x,
        inst_fresh_vn_y,
        marker="v",
        s=110,
        color=color_fresh_inst,
        edgecolor="black",
        zorder=4,
        label="Instruct fresh GRPO v5 (no SFT)",
    )
    add_labels(
        ax1,
        base_vn_x,
        base_vn_y,
        ["Base: Raw", "Base: SFT", "Base: Final"],
        [(-4, 10, "right"), (10, -12, "left"), (-44, 6, "right")],
        color_base,
    )
    add_labels(
        ax1,
        inst_vn_x,
        inst_vn_y,
        ["Inst: Raw", "Inst: SFT", "Inst: Final"],
        [(12, -12, "left"), (-10, -16, "right"), (12, 9, "left")],
        color_inst,
    )
    add_labels(
        ax1,
        [base_fresh_vn_x, inst_fresh_vn_x],
        [base_fresh_vn_y, inst_fresh_vn_y],
        ["Base: Fresh", "Inst: Fresh"],
        [(8, 10, "left"), (10, -12, "left")],
        [color_fresh_base, color_fresh_inst],
    )
    style_axis(ax1, "VietNews", [1, 2, 5, 10, 20, 50], (40, 0.9), (10.5, 31.0), show_ylabel=True)

    # Panel 2: WikiLingua
    ax2.plot(
        base_wl_x,
        base_wl_y,
        marker="o",
        markersize=7,
        linewidth=2,
        color=color_base,
        label="Base warm-start trajectory",
    )
    ax2.plot(
        inst_wl_x,
        inst_wl_y,
        marker="s",
        markersize=7,
        linewidth=2,
        color=color_inst,
        label="Instruct warm-start trajectory",
    )
    draw_arrows(ax2, base_wl_x, base_wl_y, color_base)
    draw_arrows(ax2, inst_wl_x, inst_wl_y, color_inst)
    ax2.scatter(
        base_fresh_wl_x,
        base_fresh_wl_y,
        marker="^",
        s=110,
        color=color_fresh_base,
        edgecolor="black",
        zorder=4,
        label="Base fresh GRPO v5 (no SFT)",
    )
    ax2.scatter(
        inst_fresh_wl_x,
        inst_fresh_wl_y,
        marker="v",
        s=110,
        color=color_fresh_inst,
        edgecolor="black",
        zorder=4,
        label="Instruct fresh GRPO v5 (no SFT)",
    )
    add_labels(
        ax2,
        base_wl_x,
        base_wl_y,
        ["Base: Raw", "Base: SFT", "Base: Final"],
        [(4, 10, "left"), (-14, -14, "right"), (-52, 6, "right")],
        color_base,
    )
    add_labels(
        ax2,
        inst_wl_x,
        inst_wl_y,
        ["Inst: Raw", "Inst: SFT", "Inst: Final"],
        [(8, -12, "left"), (10, 10, "left"), (10, 10, "left")],
        color_inst,
    )
    add_labels(
        ax2,
        [base_fresh_wl_x, inst_fresh_wl_x],
        [base_fresh_wl_y, inst_fresh_wl_y],
        ["Base: Fresh", "Inst: Fresh"],
        [(8, 10, "left"), (10, -12, "left")],
        [color_fresh_base, color_fresh_inst],
    )
    style_axis(ax2, "WikiLingua", [5, 10, 20, 50, 100], (75, 7.2), (2.0, 27.5))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8.4,
        frameon=True,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22, wspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation trajectory plot successfully generated at: {output_path}")


def main():
    args = parse_args()
    plot_panels(args.output)


if __name__ == "__main__":
    main()
