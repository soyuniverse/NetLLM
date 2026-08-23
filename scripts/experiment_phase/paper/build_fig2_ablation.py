#!/usr/bin/env python3
"""Fig. 2 -- MAE-latency tradeoff across the four headline configurations
(A baseline, B RecentK-2, C Speculative, D combined), full 1,698-sample,
same checkpoint. Source: results/speculative/consolidated/final_table.csv.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    COLOR_BLACK, COLOR_BLUE, COLOR_GREEN, COLOR_VERMILLION,
    DOUBLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN, save_figure, setup,
)

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_TABLE = PROJECT_ROOT / "results/speculative/consolidated/final_table.csv"

CONFIGS = [
    ("A", "baseline", "baseline", COLOR_BLACK, "o"),
    ("B", "RecentK-2", "B_recent_k2_only", COLOR_BLUE, "s"),
    ("C", "Speculative", "threshold=0.35_gamma=8", COLOR_GREEN, "D"),
    ("D", "RecentK-2+Spec.", "D_recentk2_plus_speculative", COLOR_VERMILLION, "^"),
]


def load_rows():
    with FINAL_TABLE.open() as stream:
        return {row["config"]: row for row in csv.DictReader(stream)}


def draw(figure_width_in, height_in, filename, legend_ncol):
    rows = load_rows()
    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    for letter, label, key, color, marker in CONFIGS:
        row = rows[key]
        mae = float(row["mae"])
        latency = float(row["latency_median_ms"])
        axis.scatter(latency, mae, color=color, marker=marker, s=45, zorder=3, edgecolor="black", linewidth=0.4)
        axis.annotate(f"{letter}", (latency, mae), textcoords="offset points", xytext=(6, 5), fontsize=7)
    axis.margins(0.18)
    axis.set_xlabel("Latency, median (ms)")
    axis.set_ylabel("MAE (degrees)")
    axis.grid(True, alpha=0.3)
    axis.set_axisbelow(True)
    legend_handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor=c, markeredgecolor="black",
                   markersize=6, label=f"{letter}: {label}")
        for letter, label, _, c, m in CONFIGS
    ]
    axis.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
        fontsize=6.5, framealpha=0.9, ncol=legend_ncol,
    )
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    draw(SINGLE_COL_WIDTH_IN, 3.0, "fig2_ablation_tradeoff_1col", legend_ncol=2)
    draw(DOUBLE_COL_WIDTH_IN, 2.8, "fig2_ablation_tradeoff_2col", legend_ncol=4)
    print("Wrote fig2_ablation_tradeoff_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
