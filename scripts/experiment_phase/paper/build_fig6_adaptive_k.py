#!/usr/bin/env python3
"""Fig. 6 -- Adaptive-K negative result: overall vs. degraded-group MAE
before/after, direction reversed. Source:
results/speculative/consolidated/adaptive_k_results_stats.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    COLOR_BLUE, COLOR_VERMILLION,
    DOUBLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN, save_figure, setup,
)

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = PROJECT_ROOT / "results/speculative/consolidated/adaptive_k_results_stats.json"


def draw(stats, figure_width_in, height_in, filename):
    groups = ["Overall\n(n=1,698)", "Top-5% degraded\n(n=84)"]
    before_vals = [stats["overall_mae_before_plain_recent_k2_plus_speculative"],
                   stats["degraded_group_mean_before"]]
    after_vals = [stats["overall_mae_after_adaptive_k_plus_speculative"],
                  stats["degraded_group_mean_after"]]

    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    x = range(len(groups))
    width = 0.32
    axis.bar([i - width / 2 for i in x], before_vals, width, label="Before (D)",
              color=COLOR_BLUE, edgecolor="black", linewidth=0.5)
    axis.bar([i + width / 2 for i in x], after_vals, width, label="After (Adaptive-K)",
              color=COLOR_VERMILLION, edgecolor="black", linewidth=0.5, hatch="//")
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        pct = (a - b) / b * 100.0
        sign = "+" if pct > 0 else ""
        axis.annotate(
            f"{sign}{pct:.1f}%", xy=(i, max(b, a) + 0.7), ha="center", fontsize=7, fontweight="bold",
        )
    axis.set_xticks(list(x))
    axis.set_xticklabels(groups)
    axis.set_ylabel("MAE (degrees)")
    axis.set_ylim(0, max(before_vals + after_vals) * 1.18)
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    with STATS_PATH.open() as stream:
        stats = json.load(stream)
    draw(stats, SINGLE_COL_WIDTH_IN, 2.6, "fig6_adaptive_k_reversal_1col")
    draw(stats, DOUBLE_COL_WIDTH_IN, 3.0, "fig6_adaptive_k_reversal_2col")
    print("Wrote fig6_adaptive_k_reversal_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
