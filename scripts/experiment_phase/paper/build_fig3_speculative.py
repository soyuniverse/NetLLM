#!/usr/bin/env python3
"""Fig. 3 -- two panels: (a) LLM forward count per prediction, baseline
vs. speculative; (b) MAE across the acceptance-threshold sweep, y-axis
zoomed to show insensitivity. Full 1,698-sample, same checkpoint.
Source: results/speculative/consolidated/{final_table,consolidated_results}.csv.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    COLOR_BLACK, COLOR_GREEN, COLOR_PURPLE,
    DOUBLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN, save_figure, setup,
)

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_TABLE = PROJECT_ROOT / "results/speculative/consolidated/final_table.csv"
CONSOLIDATED = PROJECT_ROOT / "results/speculative/consolidated/consolidated_results.csv"
THRESHOLDS = [0.35, 0.7, 1.5, 2.5]


def load_rows(path):
    with path.open() as stream:
        return {row["config"]: row for row in csv.DictReader(stream)}


def draw_forward_panel(axis, rows):
    baseline_fwd = float(rows["baseline"]["target_forward_avg"])
    speculative_fwd = float(rows["threshold=0.35_gamma=8"]["target_forward_avg"])
    bars = axis.bar(
        ["baseline", "speculative"], [baseline_fwd, speculative_fwd],
        color=[COLOR_BLACK, COLOR_GREEN], edgecolor="black", linewidth=0.5,
        hatch=[None, "///"],
    )
    for bar, v in zip(bars, [baseline_fwd, speculative_fwd]):
        axis.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center", fontsize=7)
    axis.set_ylabel("LLM forwards / prediction")
    axis.set_ylim(0, 23)
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.set_title("(a)", loc="left", fontsize=8)


def draw_threshold_panel(axis, rows):
    mae = [float(rows[f"threshold={t}_gamma=8"]["mae"]) for t in THRESHOLDS]
    axis.plot(
        [str(t) for t in THRESHOLDS], mae, color=COLOR_PURPLE, marker="D",
        markeredgecolor="black", markeredgewidth=0.4,
    )
    axis.set_xlabel("Acceptance threshold")
    axis.set_ylabel("MAE (degrees)")
    axis.set_ylim(12.7, 13.0)
    axis.grid(True, alpha=0.3)
    axis.set_axisbelow(True)
    axis.set_title("(b)", loc="left", fontsize=8)


def draw(figure_width_in, height_in, filename, wspace):
    final_rows = load_rows(FINAL_TABLE)
    consolidated_rows = load_rows(CONSOLIDATED)
    figure, (left, right) = plt.subplots(1, 2, figsize=(figure_width_in, height_in))
    draw_forward_panel(left, final_rows)
    draw_threshold_panel(right, consolidated_rows)
    figure.subplots_adjust(wspace=wspace)
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def draw_stacked(figure_width_in, filename):
    final_rows = load_rows(FINAL_TABLE)
    consolidated_rows = load_rows(CONSOLIDATED)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(figure_width_in, 4.6))
    draw_forward_panel(top, final_rows)
    draw_threshold_panel(bottom, consolidated_rows)
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    # Single-column: stacked (side-by-side panels would be too narrow to
    # read at 3.3in).
    draw_stacked(SINGLE_COL_WIDTH_IN, "fig3_speculative_1col")
    # Double-column: side by side.
    draw(DOUBLE_COL_WIDTH_IN, 2.6, "fig3_speculative_2col", wspace=0.3)
    print("Wrote fig3_speculative_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
