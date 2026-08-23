#!/usr/bin/env python3
"""Fig. 1 -- system overview schematic: history -> Selector ->
Speculative decoding (draft / verify / accept-reject / rollback) ->
coordinate output. Diagrams the pipeline described in
src/netllm_litevlm/vp/llama_old_selectable_pipeline.py and
src/netllm_litevlm/speculative/block_verify.py -- no data, pure
schematic. Single- and double-column versions use genuinely different
(horizontal vs. vertical) layouts, not a rescaled copy of the same one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    COLOR_BLACK, COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_VERMILLION,
    DOUBLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN, save_figure, setup,
)

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def box(axis, xy, w, h, text, facecolor="white", fontsize=6.3, linewidth=0.8, zorder=2):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=facecolor, edgecolor=COLOR_BLACK, linewidth=linewidth, zorder=zorder,
    )
    axis.add_patch(rect)
    axis.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
               fontsize=fontsize, zorder=zorder + 1)
    return rect


def arrow(axis, start, end, lw=0.9, connectionstyle="arc3,rad=0", zorder=3):
    patch = mpatches.FancyArrowPatch(
        start, end, arrowstyle="-|>", color=COLOR_BLACK, linewidth=lw,
        mutation_scale=7, connectionstyle=connectionstyle, zorder=zorder,
        shrinkA=1, shrinkB=1,
    )
    axis.add_patch(patch)


def draw_horizontal(figure_width_in, height_in, filename):
    """Double-column layout: everything left to right in one row, the
    speculative block's internal 2x2 loop drawn inside its own box, the
    final output box placed clearly outside (to the right of) that box
    so nothing overlaps it."""
    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    axis.set_xlim(0, 12.0)
    axis.set_ylim(0, 4.3)
    axis.axis("off")

    box(axis, (0.2, 2.75), 1.7, 1.0, "History\n(10-step\nroll/pitch/yaw)")
    box(axis, (2.3, 2.75), 1.7, 1.0, "Selector\n(RecentK /\nAdaptive-K)", facecolor="#EAF2FB")
    arrow(axis, (1.90, 3.25), (2.30, 3.25))

    outer = mpatches.FancyBboxPatch(
        (4.4, 0.35), 5.05, 3.75, boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#FBF3EA", edgecolor=COLOR_ORANGE, linewidth=1.0, zorder=1,
    )
    axis.add_patch(outer)
    axis.text(6.925, 3.90, "Speculative block verification", ha="center", va="center",
               fontsize=6.3, fontweight="bold", zorder=2)
    arrow(axis, (4.00, 3.25), (4.55, 2.9), connectionstyle="arc3,rad=-0.2")

    box(axis, (4.75, 2.35), 1.75, 0.85, "Draft\n(velocity\nextrapolation)", facecolor=COLOR_GREEN)
    box(axis, (7.10, 2.35), 1.75, 0.85, "Verify\n(1 target forward)", facecolor=COLOR_BLUE)
    box(axis, (7.10, 1.05), 1.75, 0.85, "Accept / reject\n(L2 threshold)", facecolor=COLOR_VERMILLION)
    box(axis, (4.75, 1.05), 1.75, 0.85, "KV-cache rollback\n(on reject)")

    arrow(axis, (6.50, 2.775), (7.10, 2.775))
    arrow(axis, (7.975, 2.35), (7.975, 1.90))
    arrow(axis, (7.10, 1.475), (6.50, 1.475))
    axis.text(5.625, 0.85, "reject: re-draft\nfrom carry token", ha="center", fontsize=5.3, zorder=4)
    arrow(axis, (5.625, 1.90), (5.625, 2.35), connectionstyle="arc3,rad=0")
    arrow(axis, (4.75, 1.475), (4.35, 1.475), connectionstyle="arc3,rad=0.3")
    axis.text(3.85, 1.65, "accept:\nadvance\ncarry token,\nloop", ha="center", fontsize=5.3, zorder=4)
    arrow(axis, (4.35, 1.85), (4.75, 2.55), connectionstyle="arc3,rad=-0.3")

    box(axis, (9.95, 2.75), 1.85, 1.0, "Coordinates\n(20-step rollout)")
    arrow(axis, (8.85, 1.475), (10.875, 2.75), connectionstyle="arc3,rad=0.35")
    axis.text(9.9, 1.85, "after 20 steps\ncovered", ha="center", fontsize=5.3, zorder=4)

    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def draw_vertical(figure_width_in, filename):
    """Single-column layout: top-to-bottom flow, speculative loop drawn
    as a compact 2x2 grid below the entry boxes, output at the bottom."""
    height_in = 6.6
    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    axis.set_xlim(0, 6.6)
    axis.set_ylim(0, 10.6)
    axis.axis("off")

    box(axis, (0.4, 9.3), 5.8, 0.9, "History (10-step roll/pitch/yaw)")
    box(axis, (0.4, 7.9), 5.8, 0.9, "Selector (RecentK / Adaptive-K)", facecolor="#EAF2FB")
    arrow(axis, (3.3, 9.3), (3.3, 8.8))

    outer = mpatches.FancyBboxPatch(
        (0.35, 1.55), 5.9, 5.9, boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#FBF3EA", edgecolor=COLOR_ORANGE, linewidth=1.0, zorder=1,
    )
    axis.add_patch(outer)
    axis.text(3.3, 7.15, "Speculative block verification", ha="center", va="center",
               fontsize=6.3, fontweight="bold", zorder=2)
    arrow(axis, (3.3, 7.9), (3.3, 6.75), connectionstyle="arc3,rad=0")

    box(axis, (0.75, 5.75), 2.4, 0.85, "Draft\n(velocity extrapolation)", facecolor=COLOR_GREEN)
    box(axis, (3.45, 5.75), 2.4, 0.85, "Verify\n(1 target forward)", facecolor=COLOR_BLUE)
    box(axis, (3.45, 4.25), 2.4, 0.85, "Accept / reject\n(L2 threshold)", facecolor=COLOR_VERMILLION)
    box(axis, (0.75, 4.25), 2.4, 0.85, "KV-cache rollback\n(on reject)")

    arrow(axis, (3.15, 6.175), (3.45, 6.175))
    arrow(axis, (4.65, 5.75), (4.65, 5.10))
    arrow(axis, (3.45, 4.675), (3.15, 4.675))
    arrow(axis, (1.95, 4.25), (1.95, 5.75), connectionstyle="arc3,rad=0")
    axis.text(1.95, 3.85, "reject: re-draft from carry token", ha="center", fontsize=5.5, zorder=4)

    # Accept path loops back to Draft for the next iteration, routed
    # around the outside (right margin, then along the top) so it never
    # crosses the Verify/Accept-reject boxes or the reject-path arrow.
    line_kwargs = dict(color=COLOR_BLACK, linewidth=0.9, zorder=3, solid_capstyle="round")
    axis.plot([5.85, 6.05], [4.675, 4.675], **line_kwargs)
    axis.plot([6.05, 6.05], [4.675, 6.75], **line_kwargs)
    axis.plot([6.05, 2.10], [6.75, 6.75], **line_kwargs)
    arrow(axis, (2.10, 6.75), (2.10, 6.62))
    axis.text(4.05, 6.83, "accept: loop to next iteration",
               ha="center", va="bottom", fontsize=5.3, zorder=4)

    box(axis, (0.4, 0.25), 5.8, 0.9, "Coordinates (20-step rollout)")
    arrow(axis, (3.3, 1.55), (3.3, 1.15))

    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    draw_horizontal(DOUBLE_COL_WIDTH_IN, 3.3, "fig1_system_overview_2col")
    draw_vertical(SINGLE_COL_WIDTH_IN, "fig1_system_overview_1col")
    print("Wrote fig1_system_overview_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
