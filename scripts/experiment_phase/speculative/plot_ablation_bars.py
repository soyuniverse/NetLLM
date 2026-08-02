#!/usr/bin/env python3
"""Ablation bar chart (MAE + latency side by side) for configs A-D', styled
consistently with the other figures in results/speculative/consolidated/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "results/speculative/consolidated"

# (label, MAE, latency_ms) -- from results/speculative/{20260802T075640Z,
# 20260802T082009Z,20260802T101802Z}/results.csv, see
# docs/experiment_phase/speculative/PHASE_B_REAL_RESULTS.md section 2.
CONFIGS = [
    ("A. baseline", 12.798559, 571.656),
    ("B. RecentK-2", 10.846867, 622.952),
    ("C. Speculative\n(th=0.35,g=8)", 12.831302, 124.419),
    ("D. RecentK-2 +\nSpeculative", 10.895102, 122.228),
    ("D'. RecentK-2 +\nSpeculative(th=0.7)", 10.902756, 121.899),
]


def main():
    labels = [c[0] for c in CONFIGS]
    mae = [c[1] for c in CONFIGS]
    latency = [c[2] for c in CONFIGS]
    colors = ["black", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    figure, (mae_axis, latency_axis) = plt.subplots(1, 2, figsize=(11, 4.5))

    mae_axis.bar(labels, mae, color=colors)
    mae_axis.axhline(mae[0], color="black", linestyle="--", alpha=0.4)
    mae_axis.set_ylabel("MAE (degrees, lower is better)")
    mae_axis.set_title("Ablation: MAE")
    mae_axis.grid(True, axis="y", alpha=0.3)
    mae_axis.tick_params(axis="x", labelrotation=20)

    latency_axis.bar(labels, latency, color=colors)
    latency_axis.axhline(latency[0], color="black", linestyle="--", alpha=0.4)
    latency_axis.set_ylabel("Latency Median (ms, lower is better)")
    latency_axis.set_title("Ablation: Latency")
    latency_axis.grid(True, axis="y", alpha=0.3)
    latency_axis.tick_params(axis="x", labelrotation=20)

    figure.suptitle("Configuration Ablation (full 1,698-sample, same checkpoint)")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "ablation_bars.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
