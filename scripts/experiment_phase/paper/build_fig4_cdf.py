#!/usr/bin/env python3
"""Fig. 4 -- per-sample MAE CDF across baseline / +Speculative /
+RecentK-2 / +both, full 1,698 samples, same checkpoint. Demonstrates
additive composition: the baseline/+Speculative pair and the
+RecentK-2/+both pair each nearly overlap.
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

RUNS = [
    ("baseline", PROJECT_ROOT / "results/speculative/20260802T075640Z/per_sample_baseline.csv", COLOR_BLACK, "-"),
    ("+Speculative", PROJECT_ROOT / "results/speculative/20260802T082009Z/per_sample_threshold=0.35_gamma=8.csv", COLOR_GREEN, "--"),
    ("+RecentK-2", PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_baseline_selector=recent_k:2.csv", COLOR_BLUE, "-"),
    ("+both (D)", PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv", COLOR_VERMILLION, "--"),
]


def load_mae(path: Path):
    with path.open() as stream:
        return [float(row["mae"]) for row in csv.DictReader(stream)]


def draw(figure_width_in, height_in, filename):
    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    for label, path, color, linestyle in RUNS:
        values = sorted(load_mae(path))
        n = len(values)
        cumulative = [(i + 1) / n for i in range(n)]
        axis.plot(values, cumulative, label=label, color=color, linestyle=linestyle)
    axis.set_xlabel("Per-sample MAE (degrees)")
    axis.set_ylabel("Cumulative fraction")
    axis.grid(True, alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    draw(SINGLE_COL_WIDTH_IN, 2.6, "fig4_cdf_1col")
    draw(DOUBLE_COL_WIDTH_IN, 3.0, "fig4_cdf_2col")
    print("Wrote fig4_cdf_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
