#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a compact qualitative comparison figure for reward-hacking cases.

Outputs:
    figures/reward_hacking_cases_compact.png
    figures/reward_hacking_cases_compact.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =============================================================================
# Style
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.unicode_minus": False,
})

WHITE = "#FFFFFF"
BLACK = "#111111"
MUTED = "#444444"

FAIL_BG = "#FFF7F7"
FAIL_BORDER = "#E53935"
FAIL_HEADER = "#D83A3A"

OK_BG = "#F5FCF7"
OK_BORDER = "#148F4E"
OK_HEADER = "#168A4B"

NEUTRAL_BG = "#FFFFFF"
NEUTRAL_BORDER = "#CFCFCF"
DIVIDER = "#D3D3D3"


# =============================================================================
# Canvas
# =============================================================================

FIG_W, FIG_H = 16.0, 9.0

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


# =============================================================================
# Geometry
# =============================================================================

LEFT = 0.14
RIGHT = 0.14
TABLE_W = FIG_W - LEFT - RIGHT

# Column proportions:
# Case | Reference | v3 failure | v5 recovered
COL_W = [0.125, 0.235, 0.330, 0.310]

COL_X = [LEFT]
for w in COL_W:
    COL_X.append(COL_X[-1] + TABLE_W * w)

TOP_RULE_Y = 8.82
HEADER_Y = 8.50
HEADER_RULE_Y = 8.24

ROW1_TOP = 8.10
ROW1_BOTTOM = 4.86

ROW_DIVIDER_Y = 4.70

ROW2_TOP = 4.56
ROW2_BOTTOM = 1.18

BOTTOM_RULE_Y = 1.00

CELL_PAD_X = 0.12
CELL_PAD_Y = 0.10
INNER_PAD_X = 0.20
INNER_PAD_Y = 0.16

FS_HEADER = 15.5
FS_CASE = 13.8
FS_REFERENCE = 12.4
FS_SECTION = 13.2
FS_OUTPUT = 12.0
FS_META = 11.5

LINE_H_OUTPUT = 0.32
LINE_H_META = 0.31


# =============================================================================
# Helpers
# =============================================================================

def draw_line(y, x0=None, x1=None, color=BLACK, lw=1.0):
    if x0 is None:
        x0 = COL_X[0]
    if x1 is None:
        x1 = COL_X[-1]

    ax.plot(
        [x0, x1],
        [y, y],
        color=color,
        lw=lw,
        solid_capstyle="butt",
        zorder=5,
    )


def draw_text(
    x,
    y,
    text,
    *,
    size=11,
    color=BLACK,
    weight="normal",
    style="normal",
    ha="left",
    va="top",
    zorder=10,
):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        zorder=zorder,
    )


def cell_bounds(col, y_bottom, y_top):
    x0 = COL_X[col] + CELL_PAD_X
    x1 = COL_X[col + 1] - CELL_PAD_X
    y0 = y_bottom + CELL_PAD_Y
    y1 = y_top - CELL_PAD_Y
    return x0, x1, y0, y1


def draw_box(
    col,
    y_bottom,
    y_top,
    *,
    facecolor=NEUTRAL_BG,
    edgecolor=NEUTRAL_BORDER,
    linewidth=1.0,
):
    x0, x1, y0, y1 = cell_bounds(col, y_bottom, y_top)

    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2,
        )
    )

    return (
        x0 + INNER_PAD_X,
        x1 - INNER_PAD_X,
        y0 + INNER_PAD_Y,
        y1 - INNER_PAD_Y,
    )


def draw_centered_multiline(
    col,
    y_bottom,
    y_top,
    lines,
    *,
    size,
    weight="normal",
    color=BLACK,
    line_height=0.40,
):
    xi, xa, yi, ya = draw_box(col, y_bottom, y_top)

    center_y = (yi + ya) / 2
    start_y = center_y + (len(lines) - 1) * line_height / 2

    for i, line in enumerate(lines):
        draw_text(
            xi,
            start_y - i * line_height,
            line,
            size=size,
            color=color,
            weight=weight,
            va="center",
        )


def draw_output_card(
    col,
    y_bottom,
    y_top,
    *,
    facecolor,
    border_color,
    output_lines,
    section_title,
    meta_items,
):
    xi, xa, yi, ya = draw_box(
        col,
        y_bottom,
        y_top,
        facecolor=facecolor,
        edgecolor=border_color,
        linewidth=1.45,
    )

    y = ya

    # Output heading
    draw_text(
        xi,
        y,
        "Output",
        size=FS_SECTION,
        weight="bold",
    )
    y -= 0.43

    # Generated output
    for line in output_lines:
        draw_text(
            xi,
            y,
            line,
            size=FS_OUTPUT,
            style="italic",
        )
        y -= LINE_H_OUTPUT

    # Divider
    y -= 0.05
    draw_line(
        y,
        x0=xi,
        x1=xa,
        color=border_color,
        lw=1.15,
    )
    y -= 0.22

    # Section heading
    draw_text(
        xi,
        y,
        section_title,
        size=FS_SECTION,
        weight="bold",
    )
    y -= 0.39

    # Bullets
    for item in meta_items:
        draw_text(
            xi + 0.02,
            y,
            f"•  {item}",
            size=FS_META,
            color=BLACK,
        )
        y -= LINE_H_META


# =============================================================================
# Header
# =============================================================================

draw_line(TOP_RULE_Y, lw=1.8)
draw_line(HEADER_RULE_Y, color="#666666", lw=0.8)

centers = [
    (COL_X[i] + COL_X[i + 1]) / 2
    for i in range(4)
]

draw_text(
    centers[0],
    HEADER_Y,
    "Case",
    size=FS_HEADER,
    weight="bold",
    ha="center",
    va="center",
)

draw_text(
    centers[1],
    HEADER_Y,
    "Reference",
    size=FS_HEADER,
    weight="bold",
    ha="center",
    va="center",
)

draw_text(
    centers[2],
    HEADER_Y,
    "v3 output (failure)",
    size=FS_HEADER,
    color=FAIL_HEADER,
    weight="bold",
    ha="center",
    va="center",
)

draw_text(
    centers[3],
    HEADER_Y,
    "v5 output (recovered)",
    size=FS_HEADER,
    color=OK_HEADER,
    weight="bold",
    ha="center",
    va="center",
)


# =============================================================================
# Row 1
# =============================================================================

draw_centered_multiline(
    0,
    ROW1_BOTTOM,
    ROW1_TOP,
    ["Base fresh", "idx 14"],
    size=FS_CASE,
    weight="bold",
    line_height=0.42,
)

draw_centered_multiline(
    1,
    ROW1_BOTTOM,
    ROW1_TOP,
    [
        "Điều tra vụ anh rể dùng dao",
        "đâm chết em vợ vào rạng sáng",
    ],
    size=FS_REFERENCE,
    line_height=0.39,
)

draw_output_card(
    2,
    ROW1_BOTTOM,
    ROW1_TOP,
    facecolor=FAIL_BG,
    border_color=FAIL_BORDER,
    output_lines=[
        "assistant  assistant  assistant  assistant",
        "assistant  assistant  …  (×128 tokens)",
    ],
    section_title="Failure signature",
    meta_items=[
        "Pure repetition loop — no content",
        "ROUGE-2 = 0.00",
        "LenErr = 814.29%",
        "R_acc = 0.00",
    ],
)

draw_output_card(
    3,
    ROW1_BOTTOM,
    ROW1_TOP,
    facecolor=OK_BG,
    border_color=OK_BORDER,
    output_lines=[
        "Anh C. bị anh rể đâm chết tại nhà.",
    ],
    section_title="Recovered properties",
    meta_items=[
        "Short, content-bearing headline",
        "Named entity retained",
        "Budget restored",
        "ROUGE-2 = 0.19",
    ],
)


# =============================================================================
# Divider
# =============================================================================

draw_line(
    ROW_DIVIDER_Y,
    color=DIVIDER,
    lw=0.75,
)


# =============================================================================
# Row 2
# =============================================================================

draw_centered_multiline(
    0,
    ROW2_BOTTOM,
    ROW2_TOP,
    ["Base SFT-init", "idx 7"],
    size=FS_CASE,
    weight="bold",
    line_height=0.42,
)

draw_centered_multiline(
    1,
    ROW2_BOTTOM,
    ROW2_TOP,
    [
        "5 năm chưa xử xong vụ án “lạ”",
        "có hơn 30 luật sư bào chữa miễn phí",
    ],
    size=FS_REFERENCE,
    line_height=0.39,
)

draw_output_card(
    2,
    ROW2_BOTTOM,
    ROW2_TOP,
    facecolor=FAIL_BG,
    border_color=FAIL_BORDER,
    output_lines=[
        "Bị cáo Tuyết bị cấp sơ thẩm 2 lần tuyên phạt",
        "12 năm tù . Bị cáo Tuyết kháng cáo , nên 2 lần",
        "vụ án được đưa ra xét xử phúc thẩm và đều có",
        "chung 1 kết quả , huỷ án . . . . . . . . . . .",
    ],
    section_title="Failure signature",
    meta_items=[
        "245 words — punctuation-heavy blow-up",
        "Meaningful start → punctuation flood",
        "LenErr = 1189.47%",
        "ROUGE-2 = 0.034",
    ],
)

draw_output_card(
    3,
    ROW2_BOTTOM,
    ROW2_TOP,
    facecolor=OK_BG,
    border_color=OK_BORDER,
    output_lines=[
        "Vụ án “lạ” : Huỷ án 2 lần ,",
        "bị cáo vẫn bị tạm giam 5 năm !",
    ],
    section_title="Recovered properties",
    meta_items=[
        "Compact, accurate rewrite",
        "Budget restored",
        "Content preserved",
        "ROUGE-2 = 0.22",
    ],
)


# =============================================================================
# Bottom rule
# =============================================================================

draw_line(BOTTOM_RULE_Y, lw=1.8)


# =============================================================================
# Save
# =============================================================================

script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "reward_hacking_cases_compact.png"
pdf_path = output_dir / "reward_hacking_cases_compact.pdf"

fig.savefig(
    png_path,
    dpi=200,
    facecolor=WHITE,
    bbox_inches=None,
    pad_inches=0,
)

fig.savefig(
    pdf_path,
    facecolor=WHITE,
    bbox_inches=None,
    pad_inches=0,
)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

plt.close(fig)
