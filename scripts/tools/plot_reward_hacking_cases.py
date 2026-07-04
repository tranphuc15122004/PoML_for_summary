#!/usr/bin/env python3
"""Generate a clearer qualitative reward-hacking case-study figure for the paper."""

from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT = Path("FormalReport_VDT/latex/figures/reward_hacking_cases.pdf")

CASES = [
    {
        "row": "Base fresh\nidx 14",
        "reference": "Điều tra vụ anh rể dùng dao đâm chết em vợ vào rạng sáng",
        "v3_output": "assistant assistant assistant ...",
        "v3_metrics": ["128 tokens", "repetition loop", "ROUGE-2 = 0.00", "LenErr = 814.29%"],
        "v5_output": "Anh C. bị anh rể đâm chết tại nhà.",
        "v5_metrics": ["short recovery", "content-bearing", "headlined summary"],
    },
    {
        "row": "Base SFT-init\nidx 7",
        'reference': '5 năm chưa xử xong vụ án "lạ" có hơn 30 luật sư bào chữa miễn phí',
        "v3_output": "Bị cáo Tuyết ... huỷ án . . . . . . .",
        "v3_metrics": ["245 words", "punctuation-heavy blow-up", "length budget broken", "LenErr = 1189.47%"],
        'v5_output': 'Vụ án "lạ": Huỷ án 2 lần, bị cáo vẫn bị tạm giam 5 năm!',
        "v5_metrics": ["compact rewrite", "budget restored", "content preserved"],
    },
]

COL_X = [0.03, 0.16, 0.39, 0.70]
COL_W = [0.10, 0.21, 0.28, 0.27]
ROW_Y = [0.52, 0.14]
CARD_H = 0.27

COLORS = {
    "neutral_bg": "#fbfbfb",
    "neutral_edge": "#cfcfcf",
    "v3_bg": "#fff3f2",
    "v3_edge": "#ef4444",
    "v5_bg": "#f2fbf4",
    "v5_edge": "#22a25a",
    "chip_bg": "#ffffff",
    "chip_edge": "#d6d6d6",
    "text": "#1c1c1c",
    "muted": "#5f6368",
}


def rounded(ax, x, y, w, h, fc, ec, lw=1.25, r=0.018):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def draw_label_card(ax, x, y, w, h, title):
    rounded(ax, x, y, w, h, "#f7f7f7", COLORS["neutral_edge"])
    ax.text(x + 0.03, y + h - 0.06, title, ha="left", va="top", fontsize=11.5, fontweight="bold", family="DejaVu Sans", color=COLORS["text"])


def draw_reference_card(ax, x, y, w, h, text):
    rounded(ax, x, y, w, h, COLORS["neutral_bg"], COLORS["neutral_edge"])
    ax.text(x + 0.03, y + h - 0.06, fill(text, 30), ha="left", va="top", fontsize=11, family="DejaVu Sans", color=COLORS["text"], linespacing=1.35)


def draw_chips(ax, x, y, w, chips, edge, text_color):
    chip_y = y
    chip_h = 0.045
    cursor_x = x
    max_x = x + w
    for chip in chips:
        chip_w = min(0.0065 * len(chip) + 0.045, w)
        if cursor_x + chip_w > max_x:
            cursor_x = x
            chip_y -= 0.055
        rounded(ax, cursor_x, chip_y, chip_w, chip_h, COLORS["chip_bg"], edge, lw=1.0, r=0.016)
        ax.text(cursor_x + 0.012, chip_y + chip_h / 2, chip, ha="left", va="center", fontsize=9.2, family="DejaVu Sans", color=text_color)
        cursor_x += chip_w + 0.012


def draw_output_card(ax, x, y, w, h, title, body, chips, bg, edge, accent):
    rounded(ax, x, y, w, h, bg, edge)
    ax.text(x + 0.025, y + h - 0.045, title, ha="left", va="top", fontsize=10.2, fontweight="bold", family="DejaVu Sans", color=accent)
    ax.plot([x + 0.025, x + w - 0.025], [y + h - 0.065, y + h - 0.065], color=edge, lw=1.0, alpha=0.8)
    ax.text(x + 0.025, y + h - 0.095, fill(body, 34), ha="left", va="top", fontsize=10.6, family="DejaVu Sans", color=COLORS["text"], linespacing=1.35)
    meta_y = y + 0.085
    ax.text(x + 0.025, meta_y + 0.065, "Failure signature" if "failure" in title.lower() else "Recovered properties", ha="left", va="bottom", fontsize=9.4, fontweight="bold", family="DejaVu Sans", color=COLORS["muted"])
    draw_chips(ax, x + 0.025, meta_y, w - 0.05, chips, edge, accent)


fig = plt.figure(figsize=(13.0, 5.1))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

headers = [
    ("Case", "#1f1f1f"),
    ("Reference", "#1f1f1f"),
    ("v3 output (failure)", "#b42318"),
    ("v5 output (recovered)", "#18794e"),
]
for (label, color), x, w in zip(headers, COL_X, COL_W):
    ax.text(x, 0.94, label, fontsize=12.4, fontweight="bold", color=color, family="DejaVu Sans")
    ax.plot([x, x + w], [0.915, 0.915], color="#252525", lw=1.1)

for case, y in zip(CASES, ROW_Y):
    draw_label_card(ax, COL_X[0], y, COL_W[0], CARD_H, case["row"])
    draw_reference_card(ax, COL_X[1], y, COL_W[1], CARD_H, case["reference"])
    draw_output_card(ax, COL_X[2], y, COL_W[2], CARD_H, "Output snippet (v3 failure)", case["v3_output"], case["v3_metrics"], COLORS["v3_bg"], COLORS["v3_edge"], "#b42318")
    draw_output_card(ax, COL_X[3], y, COL_W[3], CARD_H, "Recovered summary (v5)", case["v5_output"], case["v5_metrics"], COLORS["v5_bg"], COLORS["v5_edge"], "#18794e")

ax.text(0.03, 0.045, "Two representative cases from VietNews. The layout separates the generated snippet from its failure metrics so the reward-hacking pattern is visible at a glance.", fontsize=9.8, color="#444444", family="DejaVu Sans")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
