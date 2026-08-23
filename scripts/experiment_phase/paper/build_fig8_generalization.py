#!/usr/bin/env python3
"""Fig. 8 -- generalization to unseen data (Wu2017): in-distribution
(Jin2022, full 1,698) vs. unseen (Wu2017, 300 evenly-sampled) MAE across
the four headline configs. Source: docs/experiment_phase/analysis/
GENERALIZATION_WU2017.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    COLOR_BLUE, COLOR_ORANGE,
    DOUBLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN, save_figure, setup,
)

import matplotlib.pyplot as plt  # noqa: E402

INDIST = {"A": 12.798559396025476, "B": 10.84686681689948, "C": 12.831301641086654, "D": 10.895102344584037}
WU2017 = {"A": 15.895621841192918, "B": 13.606910356814714, "C": 15.927836322117027, "D": 13.64648620796602}
CONFIGS = ["A", "B", "C", "D"]


def draw(figure_width_in, height_in, filename):
    indist_vals = [INDIST[c] for c in CONFIGS]
    wu_vals = [WU2017[c] for c in CONFIGS]

    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    x = range(len(CONFIGS))
    width = 0.32
    axis.bar([i - width / 2 for i in x], indist_vals, width, label="Jin2022 (in-dist., n=1,698)",
              color=COLOR_BLUE, edgecolor="black", linewidth=0.5)
    axis.bar([i + width / 2 for i in x], wu_vals, width, label="Wu2017 (unseen, n=300)",
              color=COLOR_ORANGE, edgecolor="black", linewidth=0.5, hatch="//")
    axis.set_xticks(list(x))
    axis.set_xticklabels([f"{c}" for c in CONFIGS])
    axis.set_xlabel("Configuration (A/B/C/D)")
    axis.set_ylabel("MAE (degrees)")
    axis.set_ylim(0, max(indist_vals + wu_vals) * 1.15)
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    draw(SINGLE_COL_WIDTH_IN, 2.6, "fig8_generalization_1col")
    draw(DOUBLE_COL_WIDTH_IN, 3.0, "fig8_generalization_2col")
    print("Wrote fig8_generalization_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
