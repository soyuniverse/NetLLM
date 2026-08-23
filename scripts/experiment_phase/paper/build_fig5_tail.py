#!/usr/bin/env python3
"""Fig. 5 -- history motion speed vs. per-sample MAE degradation (D-A),
all 1,698 samples, top-5%-degraded group highlighted. Re-derives motion
speed from the dataset exactly as tail_analysis.py's motion_stats() does
(CPU only, no model/GPU) and reuses the already-git-tracked per-sample
CSVs for MAE -- no new experiment, same methodology as TAIL_ANALYSIS.md.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (  # noqa: E402
    COLOR_BLACK, COLOR_SKY_BLUE, COLOR_VERMILLION,
    DOUBLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN, save_figure, setup,
)

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from netllm_litevlm.vp.checkpoint_era_runtime import UPSTREAM_VP_ROOT  # noqa: E402

A_CSV = PROJECT_ROOT / "results/speculative/20260802T075640Z/per_sample_baseline.csv"
D_CSV = PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv"
TAIL_STATS = PROJECT_ROOT / "results/speculative/consolidated/tail_analysis_stats.json"


def wrapped_abs_diff_deg(a, b):
    raw = np.abs(a - b) % 360.0
    return np.minimum(raw, 360.0 - raw)


def motion_speed(history):
    step_diffs = wrapped_abs_diff_deg(history[1:], history[:-1])
    return float(step_diffs.mean(axis=1).mean())


def load_histories():
    sys.path.insert(0, str(UPSTREAM_VP_ROOT))
    from config import cfg
    from dataset.load_dataset import create_dataset

    cfg.dataset["Jin2022"] = str(
        PROJECT_ROOT.parent / "NetLLM-source/viewport_prediction/data/viewports/Jin2022"
    )
    test_dataset = create_dataset(
        "Jin2022", his_window=10, fut_window=20, trim_head=30, trim_tail=60,
        frequency=5, step=15, include=["test"],
    )[0]
    return [np.asarray(test_dataset[i][0]) for i in range(len(test_dataset))]


def load_mae(path):
    with path.open() as stream:
        return np.array([float(r["mae"]) for r in csv.DictReader(stream)])


def draw(velocities, diff_d_a, top5_idx, rho, p, figure_width_in, height_in, filename):
    mask = np.zeros(len(velocities), dtype=bool)
    mask[top5_idx] = True

    figure, axis = plt.subplots(figsize=(figure_width_in, height_in))
    axis.scatter(
        velocities[~mask], diff_d_a[~mask], s=6, alpha=0.35, color=COLOR_SKY_BLUE,
        marker="o", linewidths=0, label="rest (n={})".format((~mask).sum()),
    )
    axis.scatter(
        velocities[mask], diff_d_a[mask], s=14, alpha=0.9, color=COLOR_VERMILLION,
        marker="^", edgecolor="black", linewidth=0.3,
        label="top-5% degraded (n={})".format(mask.sum()),
    )
    axis.axhline(0, color=COLOR_BLACK, linewidth=0.8)
    axis.set_xlabel("History avg. motion speed (deg/step)")
    axis.set_ylabel("Per-sample MAE diff, D - A (degrees)")
    axis.grid(True, alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", fontsize=6.5, framealpha=0.9, title=f"Spearman ρ={rho:.3f}, p={p:.1e}", title_fontsize=6.5)
    figure.tight_layout()
    save_figure(figure, filename)
    plt.close(figure)


def main():
    setup()
    with TAIL_STATS.open() as stream:
        tail_stats = json.load(stream)
    rho = tail_stats["correlations_spearman"]["velocity_vs_diff"]["rho"]
    p = tail_stats["correlations_spearman"]["velocity_vs_diff"]["p"]
    degraded_ids = set(s["sample_id"] for s in tail_stats["top5pct_samples"])

    mae_a = load_mae(A_CSV)
    mae_d = load_mae(D_CSV)
    diff_d_a = mae_d - mae_a
    n = len(diff_d_a)
    assert n == 1698

    print("Loading dataset histories (CPU only, no model)...")
    histories = load_histories()
    assert len(histories) == n
    velocities = np.array([motion_speed(h) for h in histories])

    top5_idx = np.array(sorted(degraded_ids))
    assert len(top5_idx) == len(tail_stats["top5pct_samples"])

    draw(velocities, diff_d_a, top5_idx, rho, p, SINGLE_COL_WIDTH_IN, 2.6, "fig5_tail_velocity_1col")
    draw(velocities, diff_d_a, top5_idx, rho, p, DOUBLE_COL_WIDTH_IN, 3.0, "fig5_tail_velocity_2col")
    print("Wrote fig5_tail_velocity_{1col,2col}.{pdf,png}")


if __name__ == "__main__":
    main()
