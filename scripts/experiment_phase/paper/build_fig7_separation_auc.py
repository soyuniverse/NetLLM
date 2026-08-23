#!/usr/bin/env python3
"""Fig. 7 -- absence of a separating signal: AUC (probability a random
false-positive sample exceeds a random true-positive sample) for 5
history-derived features, none distinguishable from 0.5 (chance).
Source: results/speculative/consolidated/adaptive_k_fp_diagnosis_stats.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    COLOR_BLACK, COLOR_BLUE,
    DOUBLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN, save_figure, setup,
)

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = PROJECT_ROOT / "results/speculative/consolidated/adaptive_k_fp_diagnosis_stats.json"

LABELS = {
    "direction_reversals": "Direction\nreversals",
    "velocity_std": "Velocity\nstd.",
    "velocity_cv": "Velocity\nCV",
    "avg_acceleration": "Avg.\naccel.",
    "avg_velocity": "Avg.\nvelocity",
}


def draw(stats, figure_width_in, height_in, filename):
    ranked = stats["ranked_by_separation"]
    metrics = stats["metrics"]
    aucs = [metrics[name]["auc_fp_exceeds_tp"] for name in ranked]
    ps = [metrics[name]["mannwhitney_p"] for name in ranked]
    labels = [LABELS.get(name, name) for name in ranked]

    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    y_pos = range(len(ranked))
    bars = axis.barh(list(y_pos), aucs, color=COLOR_BLUE, edgecolor="black", linewidth=0.5, height=0.6)
    axis.axvline(0.5, color=COLOR_BLACK, linewidth=1.0, linestyle="--")
    axis.text(0.5, -0.9, "chance\n(AUC=0.5)", ha="center", va="bottom", fontsize=6.5)
    for i, (auc, p) in enumerate(zip(aucs, ps)):
        axis.text(auc + 0.012, i, f"{auc:.3f} (p={p:.2f})", va="center", fontsize=6.5)
    axis.set_yticks(list(y_pos))
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("AUC (false-positive exceeds true-positive)")
    axis.set_xlim(0.40, 0.68)
    axis.set_ylim(len(ranked) - 0.3, -1.15)
    axis.grid(True, axis="x", alpha=0.3)
    axis.set_axisbelow(True)
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    with STATS_PATH.open() as stream:
        stats = json.load(stream)
    draw(stats, SINGLE_COL_WIDTH_IN, 2.6, "fig7_separation_auc_1col")
    draw(stats, DOUBLE_COL_WIDTH_IN, 2.8, "fig7_separation_auc_2col")
    print("Wrote fig7_separation_auc_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
