"""Candidate Figure 1: fixed probability model plus online ICME decisions.

The figure is a method schematic. Miniature traces, matrices, and event bars are
deterministic explanatory glyphs rather than measured data or performance
results. All drawing, previewing, and export are performed in Python/Matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image
from itertools import combinations


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "Figure1_online_workflow_candidate_v5"

# PASP/AASTeX double-column artwork: exact physical dimensions.
FIG_W_MM = 183.0
FIG_H_MM = 132.0
FIG_W = FIG_W_MM / 25.4
FIG_H = FIG_H_MM / 25.4

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.25,
        "text.color": "#1E252B",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
    }
)

INK = "#1E252B"
MUTED = "#5F6B75"
HAIR = "#C7D0D8"
WHITE = "#FFFFFF"
PALE = "#F7F9FA"
BLUE = "#216EA3"
BLUE_PALE = "#EAF3F8"
PURPLE = "#7053A2"
PURPLE_PALE = "#F1EDF7"
AMBER = "#D48806"
AMBER_PALE = "#FCF3E2"
MAGENTA = "#BB4E85"
MAGENTA_PALE = "#F8EAF1"
TEAL = "#16816F"
TEAL_PALE = "#E7F3F0"
GREY = "#78838D"
GREY_PALE = "#EFF2F4"


def rounded_box(ax, x, y, w, h, *, face=WHITE, edge=HAIR, lw=0.9, radius=0.55, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.06,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=INK, lw=1.0, mutation=7.0, connection="arc3,rad=0", dashed=False, z=7):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 2)) if dashed else "-",
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel_heading(ax, letter, title, y):
    ax.text(1.25, y, letter, fontsize=9.0, fontweight="bold", ha="left", va="top", color=INK, zorder=30)
    ax.text(5.1, y - 0.05, title, fontsize=8.4, fontweight="bold", ha="left", va="top", color=INK, zorder=30)


def lock_icon(ax, x, y, *, scale=1.0, color=INK, face=WHITE):
    bw, bh = 1.55 * scale, 1.15 * scale
    ax.add_patch(Rectangle((x, y), bw, bh, facecolor=face, edgecolor=color, linewidth=0.9, zorder=15))
    ax.add_patch(
        Arc(
            (x + bw / 2, y + bh),
            1.10 * scale,
            1.25 * scale,
            theta1=0,
            theta2=180,
            color=color,
            linewidth=0.9,
            zorder=15,
        )
    )
    ax.add_patch(Circle((x + bw / 2, y + 0.62 * scale), 0.10 * scale, facecolor=color, edgecolor="none", zorder=16))


def signal_icon(ax, x, y, w, h):
    t = np.linspace(0, 1, 140)
    signals = (
        0.76 + 0.08 * np.sin(2 * np.pi * 2.2 * t) + 0.025 * np.sin(2 * np.pi * 9.1 * t),
        0.49 + 0.10 * np.cos(2 * np.pi * 1.4 * t + 0.35),
        0.23 + 0.06 * np.sin(2 * np.pi * 2.8 * t - 0.55),
    )
    for yy, color in zip(signals, (BLUE, PURPLE, TEAL)):
        ax.plot(x + t * w, y + yy * h, color=color, linewidth=1.05, zorder=12)
    for frac in (0.25, 0.5, 0.75):
        ax.plot([x + frac * w] * 2, [y, y + h], color=HAIR, linewidth=0.5, zorder=5)


def feature_matrix(ax, x, y, w, h, *, rows=5, cols=8):
    colors = (BLUE, PURPLE, AMBER, TEAL, GREY)
    gx, gy = w * 0.022, h * 0.07
    cw = (w - gx * (cols - 1)) / cols
    ch = (h - gy * (rows - 1)) / rows
    for r in range(rows):
        for c in range(cols):
            alpha = 0.20 + 0.07 * ((r + 2 * c) % 5)
            ax.add_patch(
                Rectangle(
                    (x + c * (cw + gx), y + (rows - 1 - r) * (ch + gy)),
                    cw,
                    ch,
                    facecolor=mpl.colors.to_rgba(colors[r], alpha),
                    edgecolor="none",
                    zorder=10,
                )
            )


def probability_trace(ax, x, y, w, h, *, threshold_kind=None, threshold_color=MAGENTA):
    t = np.linspace(0, 1, 180)
    p = 0.19 + 0.045 * np.sin(2 * np.pi * 4.2 * t) + 0.025 * np.sin(2 * np.pi * 12.0 * t)
    p += 0.52 * np.exp(-((t - 0.61) / 0.17) ** 2)
    p += 0.14 * np.exp(-((t - 0.23) / 0.075) ** 2)
    p = np.clip(p, 0.04, 0.94)
    ax.plot(x + t * w, y + p * h, color=BLUE, linewidth=1.15, zorder=12)
    if threshold_kind == "dynamic":
        levels = np.array([0.58, 0.67, 0.55, 0.64, 0.60])
        edges = np.linspace(0, 1, len(levels) + 1)
        for i, level in enumerate(levels):
            ax.plot(
                [x + edges[i] * w, x + edges[i + 1] * w],
                [y + level * h] * 2,
                color=threshold_color,
                linewidth=1.25,
                linestyle=(0, (3.5, 2.2)),
                zorder=13,
            )
            if i < len(levels) - 1:
                ax.plot(
                    [x + edges[i + 1] * w] * 2,
                    [y + levels[i] * h, y + levels[i + 1] * h],
                    color=threshold_color,
                    linewidth=1.25,
                    linestyle=(0, (3.5, 2.2)),
                    zorder=13,
                )
    ax.plot([x, x + w], [y, y], color=HAIR, linewidth=0.7, zorder=6)
    ax.plot([x, x], [y, y + h], color=HAIR, linewidth=0.7, zorder=6)


def split_strip(ax, x, y, w):
    entries = ((BLUE, "Val."), (PURPLE, "Train"), (TEAL, "Test"))
    gap = 0.12
    ww = (w - 2 * gap) / 3
    for idx, (color, label) in enumerate(entries):
        left = x + idx * (ww + gap)
        ax.add_patch(Rectangle((left, y), ww, 1.45, facecolor=color, edgecolor=WHITE, linewidth=0.6, zorder=10))
        ax.text(left + ww / 2, y + 0.72, label, fontsize=5.9, fontweight="bold", ha="center", va="center", color=WHITE, zorder=11)


def model_icon(ax, x, y):
    # Sequence window.
    for offset in (0.75, 0.38, 0.0):
        ax.add_patch(Rectangle((x + offset, y + offset), 4.2, 5.0, facecolor=BLUE_PALE, edgecolor=BLUE, linewidth=0.8, zorder=9))
    ax.text(x + 2.45, y - 0.55, "64-observation\nwindow", fontsize=5.25, linespacing=0.92, ha="center", va="top", color=INK, zorder=12)
    arrow(ax, (x + 5.6, y + 2.9), (x + 7.2, y + 2.9), color=MUTED, lw=0.9, mutation=6.2)

    # Convolutional feature filters, intentionally shown without a CNN-LSTM title.
    heights = (2.8, 4.0, 5.0, 4.0, 2.8)
    for i, hh in enumerate(heights):
        ax.add_patch(Rectangle((x + 7.4 + i * 1.05, y + 2.9 - hh / 2), 0.72, hh, facecolor=mpl.colors.to_rgba(BLUE, 0.48 + 0.09 * (i % 3)), edgecolor="none", zorder=10))
    ax.text(x + 9.85, y - 0.55, "local\npatterns", fontsize=5.8, linespacing=0.92, ha="center", va="top", color=INK, zorder=12)
    arrow(ax, (x + 12.8, y + 2.9), (x + 14.2, y + 2.9), color=MUTED, lw=0.9, mutation=6.2)

    rounded_box(ax, x + 14.4, y + 0.2, 5.0, 5.4, face=PURPLE_PALE, edge=PURPLE, lw=0.9, radius=0.45, z=8)
    ax.text(x + 16.9, y + 3.15, "time", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=12)
    ax.text(x + 16.9, y + 1.75, "context", fontsize=6.1, ha="center", va="center", zorder=12)
    arrow(ax, (x + 19.8, y + 2.9), (x + 21.2, y + 2.9), color=MUTED, lw=0.9, mutation=6.2)

    rounded_box(ax, x + 21.4, y + 0.2, 4.2, 5.4, face=AMBER_PALE, edge=AMBER, lw=0.9, radius=0.45, z=8)
    ax.text(x + 23.5, y + 3.25, "p(t)", fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=12)
    ax.text(x + 23.5, y + 1.65, "0 to 1", fontsize=6.0, ha="center", va="center", zorder=12)


def expert_icon(ax, x, y, w, h):
    starts = (0.00, 0.14, 0.31, 0.49)
    spans = (0.94, 0.70, 0.50, 0.31)
    for i, (start, span) in enumerate(zip(starts, spans)):
        yy = y + i * (h - 0.85) / 3
        ax.plot([x, x + w], [yy + 0.38] * 2, color=HAIR, linewidth=0.55, zorder=6)
        ax.add_patch(
            FancyBboxPatch(
                (x + start * w, yy),
                span * w,
                0.76,
                boxstyle="round,pad=0.01,rounding_size=0.26",
                facecolor=mpl.colors.to_rgba(PURPLE, 0.35 + 0.12 * i),
                edgecolor="none",
                zorder=10,
            )
        )
        ax.add_patch(Circle((x + start * w, yy + 0.38), 0.27, facecolor=WHITE, edgecolor=PURPLE, linewidth=0.8, zorder=11))


def binary_strip(ax, x, y, w, h, *, color=TEAL):
    seq = ((0.00, 0.09, 0), (0.09, 0.21, 1), (0.21, 0.34, 0), (0.34, 0.59, 1), (0.59, 0.70, 0), (0.70, 0.94, 1), (0.94, 1.00, 0))
    for left, right, value in seq:
        ax.add_patch(Rectangle((x + left * w, y), (right - left) * w, h, facecolor=color if value else "#E2E7EA", edgecolor=WHITE, linewidth=0.5, zorder=11))


def event_strip(ax, x, y, w, h, *, color=TEAL):
    ax.plot([x, x + w], [y + h / 2] * 2, color=HAIR, linewidth=0.8, zorder=6)
    for left, right in ((0.08, 0.22), (0.35, 0.60), (0.70, 0.94)):
        ax.add_patch(Rectangle((x + left * w, y), (right - left) * w, h, facecolor=color, edgecolor="none", zorder=11))


def contribution_badge(ax, x, y, w, text):
    rounded_box(ax, x, y, w, 4.0, face=MAGENTA_PALE, edge=MAGENTA, lw=1.0, radius=0.5, z=8)
    diamond = Polygon(
        [[x + 1.8, y + 2.0], [x + 2.55, y + 2.75], [x + 3.3, y + 2.0], [x + 2.55, y + 1.25]],
        closed=True,
        facecolor=MAGENTA,
        edgecolor="none",
        zorder=11,
    )
    ax.add_patch(diamond)
    ax.text(x + 4.2, y + 2.0, text, fontsize=6.15, fontweight="bold", ha="left", va="center", color=INK, zorder=12)


def observation_block(ax, x, y, w, h):
    """A card stack containing 64 ordered observations, not a probability trace."""
    for offset, alpha in ((0.75, 0.22), (0.38, 0.34)):
        ax.add_patch(
            FancyBboxPatch(
                (x + offset, y + offset),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.38",
                facecolor=mpl.colors.to_rgba(BLUE, alpha),
                edgecolor=BLUE,
                linewidth=0.65,
                zorder=7,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.38",
            facecolor=WHITE,
            edgecolor=BLUE,
            linewidth=0.9,
            zorder=9,
        )
    )
    cols = rows = 8
    pad_x, pad_y = 0.75, 0.62
    gap_x, gap_y = 0.12, 0.10
    cell_w = (w - 2 * pad_x - (cols - 1) * gap_x) / cols
    cell_h = (h - 2 * pad_y - (rows - 1) * gap_y) / rows
    for row in range(rows):
        for col in range(cols):
            alpha = 0.18 + 0.06 * ((row + 2 * col) % 5)
            ax.add_patch(
                Rectangle(
                    (x + pad_x + col * (cell_w + gap_x), y + pad_y + (rows - 1 - row) * (cell_h + gap_y)),
                    cell_w,
                    cell_h,
                    facecolor=mpl.colors.to_rgba(BLUE, alpha),
                    edgecolor="none",
                    zorder=11,
                )
            )
    rounded_box(ax, x + w - 2.5, y + h - 2.25, 2.05, 1.75, face=BLUE, edge=BLUE, lw=0.8, radius=0.45, z=12)
    ax.text(x + w - 1.47, y + h - 1.37, "64", fontsize=6.1, fontweight="bold", ha="center", va="center", color=WHITE, zorder=13)


def decision_tiles(ax, x, y, w, h):
    """Discrete Online decisions without repeating a probability trace."""
    values = (0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0)
    gap = 0.12
    tile_w = (w - gap * (len(values) - 1)) / len(values)
    for i, value in enumerate(values):
        left = x + i * (tile_w + gap)
        color = TEAL if value else "#DDE4E8"
        ax.add_patch(Rectangle((left, y), tile_w, h, facecolor=color, edgecolor=WHITE, linewidth=0.55, zorder=11))
        ax.text(left + tile_w / 2, y + h / 2, str(value), fontsize=5.55, fontweight="bold", ha="center", va="center", color=WHITE if value else MUTED, zorder=12)
    # A small block boundary reinforces that decisions are made blockwise.
    boundary = x + 6 * (tile_w + gap) - gap / 2
    ax.plot([boundary, boundary], [y - 0.8, y + h + 0.8], color=MAGENTA, linewidth=0.9, linestyle=(0, (2.5, 2)), zorder=8)


def cleanup_sequence(ax, x, y, w, h, values):
    """Draw a short binary sequence with a visible, preserved data gap."""
    slots = len(values)
    gap = 0.10
    tile_w = (w - gap * (slots - 1)) / slots
    for i, value in enumerate(values):
        left = x + i * (tile_w + gap)
        if value is None:
            ax.plot([left + 0.16, left + tile_w * 0.46], [y + 0.2, y + h - 0.2], color=GREY, linewidth=0.9, zorder=11)
            ax.plot([left + tile_w * 0.54, left + tile_w - 0.16], [y + 0.2, y + h - 0.2], color=GREY, linewidth=0.9, zorder=11)
            continue
        ax.add_patch(Rectangle((left, y), tile_w, h, facecolor=TEAL if value else "#DDE4E8", edgecolor=WHITE, linewidth=0.45, zorder=11))


def interval_capsules(ax, x, y, w):
    capsule_h = 3.35
    ax.plot([x, x + w], [y + capsule_h / 2] * 2, color=HAIR, linewidth=1.15, zorder=6)
    intervals = ((0.05, 0.27, "E1"), (0.39, 0.67, "E2"), (0.76, 0.96, "E3"))
    for left, right, label in intervals:
        xx = x + left * w
        ww = (right - left) * w
        rounded_box(ax, xx, y, ww, capsule_h, face=TEAL, edge=TEAL, lw=0.9, radius=0.95, z=10)
        ax.text(xx + ww / 2, y + capsule_h / 2, label, fontsize=6.15, fontweight="bold", ha="center", va="center", color=WHITE, zorder=12)


def one_to_one_diagram(ax, x, y, w):
    """Maximum-cardinality one-to-one matching with unmatched examples."""
    xs_pred = (x + 0.42 * w, x + 0.66 * w, x + 0.89 * w)
    xs_cat = (x + 0.40 * w, x + 0.68 * w, x + 0.92 * w)
    y_pred, y_cat = y + 5.4, y + 1.6

    ax.text(x - 0.15, y_pred, "Prediction", fontsize=6.2, fontweight="bold", ha="left", va="center", color=TEAL, zorder=12)
    ax.text(x - 0.15, y_cat, "Catalog", fontsize=6.2, fontweight="bold", ha="left", va="center", color=PURPLE, zorder=12)

    # Draw connections first so nodes stay clean and readable.
    ax.plot([xs_pred[0], xs_cat[0]], [y_pred, y_cat], color=INK, linewidth=1.55, zorder=7)
    ax.plot([xs_pred[1], xs_cat[1]], [y_pred, y_cat], color=INK, linewidth=1.55, zorder=7)
    for i, xx in enumerate(xs_pred):
        face = TEAL if i < 2 else WHITE
        edge = TEAL if i < 2 else MAGENTA
        ax.add_patch(Circle((xx, y_pred), 1.05, facecolor=face, edgecolor=edge, linewidth=1.25, zorder=11))
        ax.text(xx, y_pred, f"P{i + 1}", fontsize=5.8, fontweight="bold", ha="center", va="center", color=WHITE if i < 2 else MAGENTA, zorder=12)
    for i, xx in enumerate(xs_cat):
        face = PURPLE if i < 2 else WHITE
        edge = PURPLE if i < 2 else GREY
        ax.add_patch(Circle((xx, y_cat), 1.05, facecolor=face, edgecolor=edge, linewidth=1.25, zorder=11))
        ax.text(xx, y_cat, f"C{i + 1}", fontsize=5.8, fontweight="bold", ha="center", va="center", color=WHITE if i < 2 else GREY, zorder=12)


def build_figure():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ------------------------------------------------------------------
    # a | Simple, fixed probability pipeline
    # ------------------------------------------------------------------
    panel_heading(ax, "a", "Generate ICME probabilities", 98.2)

    rounded_box(ax, 5.0, 78.0, 18.0, 15.7, face=BLUE_PALE, edge=BLUE, lw=1.0)
    ax.text(14.0, 91.3, "Solar-wind inputs", fontsize=7.0, fontweight="bold", ha="center", va="center")
    signal_icon(ax, 6.3, 84.4, 8.2, 4.4)
    feature_matrix(ax, 15.5, 84.2, 5.9, 4.8)
    ax.text(14.0, 81.5, "30 measured + 4 derived", fontsize=6.3, ha="center", va="center")
    ax.text(14.0, 79.7, "34 input channels", fontsize=6.1, ha="center", va="center", color=MUTED)

    arrow(ax, (23.5, 85.8), (26.1, 85.8), color=MUTED, lw=1.05)

    rounded_box(ax, 26.4, 78.0, 17.0, 15.7, face=WHITE, edge=HAIR, lw=0.95)
    ax.text(34.9, 91.3, "Prepare the sequence", fontsize=7.0, fontweight="bold", ha="center", va="center")
    steps = ("Fill missing values", "Keep chronology", "Scale with training data")
    for i, label in enumerate(steps):
        yy = 88.4 - i * 2.15
        ax.add_patch(Circle((28.2, yy), 0.28, facecolor=INK, edgecolor="none", zorder=12))
        ax.text(29.3, yy, label, fontsize=6.1, ha="left", va="center", zorder=12)
    split_strip(ax, 28.0, 79.55, 13.8)

    arrow(ax, (43.9, 85.8), (46.1, 85.8), color=MUTED, lw=1.05)

    # The prior CNN-LSTM title is intentionally removed; the retained icons
    # communicate windowing, local feature extraction, temporal memory, and score.
    rounded_box(ax, 46.4, 78.0, 34.1, 15.7, face=PALE, edge=HAIR, lw=0.95)
    # Reserve a separate header strip for the lock and label.  The model glyph
    # begins well below it, so neither the icon nor the wording crowds the p(t)
    # output box or any internal arrow.
    ax.text(63.45, 91.20, "fixed model weights", fontsize=6.4, fontweight="bold", ha="center", va="center")
    model_icon(ax, 49.2, 82.45)

    arrow(ax, (81.0, 85.8), (83.2, 85.8), color=BLUE, lw=1.15)

    rounded_box(ax, 83.5, 78.0, 13.6, 15.7, face=WHITE, edge=BLUE, lw=1.0)
    ax.text(90.3, 91.3, "Probability over time", fontsize=6.85, fontweight="bold", ha="center", va="center")
    probability_trace(ax, 85.1, 84.0, 10.4, 4.9)
    ax.text(90.3, 81.5, "same score stream", fontsize=6.2, ha="center", va="center")
    ax.text(90.3, 79.7, "weights stay fixed online", fontsize=5.75, ha="center", va="center", color=MUTED)

    ax.plot([2.2, 97.8], [75.1, 75.1], color=HAIR, linewidth=0.85, zorder=1)

    # ------------------------------------------------------------------
    # b | Hero: plain-language online update loop
    # ------------------------------------------------------------------
    panel_heading(ax, "b", "Adapt the threshold online", 72.9)
    ax.text(97.0, 72.8, "Online implementation: SAOCP", fontsize=6.2, fontweight="bold", ha="right", va="top", color=PURPLE)
    contribution_badge(
        ax,
        24.7,
        64.8,
        50.6,
        "Study contribution: online ICME decisions that adapt to distribution drift",
    )

    stage_y, stage_h = 39.9, 22.1
    stage_x = (5.0, 28.1, 51.2, 74.3)
    stage_w = 20.7
    faces = (BLUE_PALE, PURPLE_PALE, AMBER_PALE, MAGENTA_PALE)
    edges = (BLUE, PURPLE, AMBER, MAGENTA)
    titles = ("1  Score the next block", "2  Set a threshold", "3  Make decisions", "4  Learn from the block")
    subtitles = (
        "64 observations at a time",
        "use errors from past blocks only",
        "before current labels are known",
        "after all current labels arrive",
    )
    for x, face, edge, title, subtitle in zip(stage_x, faces, edges, titles, subtitles):
        rounded_box(ax, x, stage_y, stage_w, stage_h, face=face, edge=edge, lw=1.0)
        ax.text(x + 1.35, stage_y + stage_h - 3.0, title, fontsize=7.0, fontweight="bold", ha="left", va="center")
        ax.text(x + 1.35, stage_y + stage_h - 5.7, subtitle, fontsize=6.0, ha="left", va="center", color=MUTED)

    observation_block(ax, 9.0, 46.6, 11.6, 6.9)
    ax.text(14.8, 45.0, "ordered block b", fontsize=6.1, fontweight="bold", ha="center", va="center", color=BLUE)
    # The visual center of this lock (including its shackle) is y = 41.85,
    # exactly matching the vertically centred labels-hidden text.
    lock_icon(ax, 7.0, 41.10, scale=0.84, color=INK, face=BLUE_PALE)
    ax.text(9.25, 41.85, "labels hidden", fontsize=6.2, fontweight="bold", ha="left", va="center")

    expert_icon(ax, 30.7, 47.7, 15.6, 5.2)
    ax.text(38.45, 45.2, "several time-scale experts", fontsize=6.2, ha="center", va="center")
    ax.text(38.45, 42.8, "combine their advice", fontsize=6.2, fontweight="bold", ha="center", va="center", color=PURPLE)

    probability_trace(ax, 53.2, 46.5, 16.7, 7.2, threshold_kind="dynamic", threshold_color=MAGENTA)
    ax.text(61.55, 44.2, "blue: probability", fontsize=5.85, ha="center", va="center", color=BLUE)
    ax.text(61.55, 41.95, "pink dashed: current threshold", fontsize=5.85, ha="center", va="center", color=MAGENTA)

    ax.text(76.0, 54.4, "Compare predictions", fontsize=6.25, ha="left", va="center")
    ax.text(76.0, 52.0, "with the revealed labels", fontsize=6.25, ha="left", va="center")
    ax.plot([76.0, 92.6], [49.5, 49.5], color=HAIR, linewidth=0.7, zorder=6)
    ax.text(76.0, 47.0, "Update expert weights", fontsize=6.45, fontweight="bold", ha="left", va="center", color=MAGENTA)
    ax.text(76.0, 44.4, "Use the new threshold", fontsize=6.25, ha="left", va="center")
    ax.text(76.0, 42.0, "for the next block", fontsize=6.25, ha="left", va="center")
    arrow(ax, (92.5, 42.6), (92.5, 54.0), color=MAGENTA, lw=1.15, mutation=7.0, connection="arc3,rad=0.42")

    # Forward arrows. The causal before/after order is stated inside stages 3
    # and 4, so no extra divider or lock is needed between them.
    arrow(ax, (26.1, 51.0), (27.6, 51.0), color=BLUE, lw=1.15)
    arrow(ax, (49.2, 51.0), (50.7, 51.0), color=PURPLE, lw=1.15)
    arrow(ax, (72.0, 51.0), (73.8, 51.0), color=MAGENTA, lw=1.15)

    # Delayed feedback updates the next block's threshold, not the frozen model.
    arrow(
        ax,
        (87.5, 63.0),
        (38.5, 63.1),
        color=MAGENTA,
        lw=1.15,
        mutation=7.0,
        connection="arc3,rad=0.025",
        dashed=True,
        z=5,
    )
    ax.text(
        47.5,
        63.55,
        "completed block informs the next threshold",
        fontsize=5.9,
        ha="center",
        va="center",
        color=INK,
        bbox=dict(facecolor=WHITE, edgecolor="none", pad=0.8),
        zorder=12,
    )

    ax.plot([2.2, 97.8], [35.7, 35.7], color=HAIR, linewidth=0.85, zorder=1)

    # ------------------------------------------------------------------
    # c | Visually distinct event construction and one-to-one evaluation.
    # ------------------------------------------------------------------
    panel_heading(ax, "c", "Construct ICME events", 33.5)

    box_y, box_h = 7.6, 19.5

    rounded_box(ax, 5.0, box_y, 19.2, box_h, face=BLUE_PALE, edge=BLUE, lw=1.0)
    ax.text(14.6, 24.45, "Online decisions", fontsize=7.45, fontweight="bold", ha="center", va="center")
    decision_tiles(ax, 6.8, 17.6, 15.6, 3.2)
    ax.text(14.6, 11.2, "0 = background  ·  1 = ICME", fontsize=6.45, fontweight="bold", ha="center", va="center", color=BLUE)

    arrow(ax, (24.7, 17.35), (26.6, 17.35), color=TEAL, lw=1.15)

    rounded_box(ax, 27.0, box_y, 23.5, box_h, face=TEAL_PALE, edge=TEAL, lw=1.0)
    ax.text(38.75, 24.45, "Gap-aware correction", fontsize=7.45, fontweight="bold", ha="center", va="center")
    ax.text(29.0, 19.8, "before", fontsize=6.15, fontweight="bold", ha="left", va="center", color=MUTED)
    cleanup_sequence(ax, 33.2, 18.6, 15.0, 2.0, (0, 1, 0, 0, 1, 1, 0, 1, None, 1, 1, 0, 1, 1))
    ax.text(29.0, 15.3, "after", fontsize=6.15, fontweight="bold", ha="left", va="center", color=TEAL)
    cleanup_sequence(ax, 33.2, 14.1, 15.0, 2.0, (0, 0, 0, 0, 1, 1, 1, 1, None, 1, 1, 1, 1, 1))
    ax.text(38.75, 10.25, "fix short glitches  ·  preserve gaps", fontsize=6.55, fontweight="bold", ha="center", va="center", color=TEAL)

    arrow(ax, (51.0, 17.35), (52.9, 17.35), color=TEAL, lw=1.15)

    rounded_box(ax, 53.3, box_y, 18.3, box_h, face=WHITE, edge=HAIR, lw=1.0)
    ax.text(62.45, 24.45, "Event intervals", fontsize=7.45, fontweight="bold", ha="center", va="center")
    interval_capsules(ax, 54.8, 16.8, 15.3)
    ax.text(62.45, 11.3, "merge nearby runs", fontsize=6.75, fontweight="bold", ha="center", va="center", color=TEAL)

    arrow(ax, (72.1, 17.35), (74.0, 17.35), color=TEAL, lw=1.15)

    rounded_box(ax, 74.4, box_y, 22.7, box_h, face=PURPLE_PALE, edge=PURPLE, lw=1.0)
    ax.text(85.75, 24.45, "One-to-one scoring", fontsize=7.35, fontweight="bold", ha="center", va="center")
    one_to_one_diagram(ax, 76.0, 16.0, 19.0)
    ax.text(85.75, 13.25, "linked = TP  ·  unlinked = FP/FN", fontsize=6.15, ha="center", va="center")
    ax.text(85.75, 10.40, "Precision  ·  Recall  ·  F1", fontsize=7.0, fontweight="bold", ha="center", va="center", color=PURPLE)

    rounded_box(ax, 5.0, 3.25, 45.8, 2.7, face=GREY_PALE, edge=HAIR, lw=0.75, radius=0.38, z=4)
    ax.text(
        27.9,
        4.60,
        "Reference only: the Static model keeps one validation-selected threshold.",
        fontsize=5.85,
        ha="center",
        va="center",
        color=MUTED,
        zorder=12,
    )
    ax.text(
        97.0,
        4.60,
        "Schematic elements are explanatory, not measured results.",
        fontsize=5.55,
        ha="right",
        va="center",
        color=MUTED,
        zorder=12,
    )

    return fig


def export_and_verify(fig):
    pdf_metadata = {
        "Title": "Online ICME decision workflow",
        "Subject": "Python-generated editable method schematic",
        "Creator": "Matplotlib",
    }
    svg_metadata = {
        "Title": "Online ICME decision workflow",
        "Description": "Python-generated editable method schematic",
        "Creator": "Matplotlib",
    }

    # Draw once and verify that every text bounding box stays inside the canvas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    offenders = []
    for artist in fig.findobj(match=mpl.text.Text):
        if not artist.get_text() or not artist.get_visible():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.x0 < canvas.x0 - 0.5 or bbox.y0 < canvas.y0 - 0.5 or bbox.x1 > canvas.x1 + 0.5 or bbox.y1 > canvas.y1 + 0.5:
            offenders.append(artist.get_text())
    if offenders:
        raise RuntimeError(f"Text outside canvas: {offenders}")

    # Reject substantial text-to-text collisions. Small glyph overhangs are
    # ignored, but any shared area above 10% of the smaller label is a failure.
    visible_text = [
        (artist.get_text(), artist.get_window_extent(renderer=renderer))
        for artist in fig.findobj(match=mpl.text.Text)
        if artist.get_visible() and artist.get_text()
    ]
    collisions = []
    for (text_a, box_a), (text_b, box_b) in combinations(visible_text, 2):
        x0, y0 = max(box_a.x0, box_b.x0), max(box_a.y0, box_b.y0)
        x1, y1 = min(box_a.x1, box_b.x1), min(box_a.y1, box_b.y1)
        if x1 <= x0 or y1 <= y0:
            continue
        intersection = (x1 - x0) * (y1 - y0)
        smaller = min(box_a.width * box_a.height, box_b.width * box_b.height)
        if smaller > 0 and intersection / smaller > 0.10:
            collisions.append((text_a, text_b))
    if collisions:
        raise RuntimeError(f"Text overlap candidates: {collisions}")

    fig.savefig(STEM.with_suffix(".svg"), metadata=svg_metadata)
    fig.savefig(STEM.with_suffix(".pdf"), metadata=pdf_metadata)
    fig.savefig(STEM.with_suffix(".png"), dpi=450, metadata={"Software": "Matplotlib"})
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Convert raster exports to true RGB while preserving their target DPI.
    with Image.open(STEM.with_suffix(".png")) as image:
        image.load()
        png = image.convert("RGB")
    png.save(STEM.with_suffix(".png"), dpi=(450, 450), optimize=True)
    with Image.open(STEM.with_suffix(".tiff")) as image:
        image.load()
        tiff = image.convert("RGB")
    tiff.save(STEM.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")

    expected_png = (round(FIG_W * 450), round(FIG_H * 450))
    expected_tiff = (round(FIG_W * 600), round(FIG_H * 600))
    with Image.open(STEM.with_suffix(".png")) as image:
        assert image.mode == "RGB"
        assert max(abs(a - b) for a, b in zip(image.size, expected_png)) <= 1, (image.size, expected_png)
    with Image.open(STEM.with_suffix(".tiff")) as image:
        assert image.mode == "RGB"
        assert max(abs(a - b) for a, b in zip(image.size, expected_tiff)) <= 1, (image.size, expected_tiff)

    svg = STEM.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<image" not in svg, "SVG contains an embedded raster image"
    assert svg.count("<text") > 0, "SVG text was not preserved as editable text"
    assert svg.count("SAOCP") == 1, "SAOCP should be named exactly once in the figure"
    pdf = STEM.with_suffix(".pdf").read_bytes()
    assert b"/Subtype /Type3" not in pdf, "PDF contains non-editable Type 3 fonts"


def main():
    fig = build_figure()
    export_and_verify(fig)
    print(STEM)


if __name__ == "__main__":
    main()
