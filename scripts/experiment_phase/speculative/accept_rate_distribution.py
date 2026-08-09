#!/usr/bin/env python3
"""Acceptance-rate distribution for config D (RecentK-2 + speculative,
threshold=0.35, gamma=8), full 1,698-sample test split.

Reads the existing per-sample CSV from the 2026-08-02 real-checkpoint run
(no GPU/checkpoint/dataset access needed -- the per-sample accept_sum and
target_forward_count columns are already on disk). Companion to
tail_analysis.py's scatter plots: this shows the full-population
DISTRIBUTION of accept rate rather than its correlation with degradation.

See docs/experiment_phase/analysis/TAIL_ANALYSIS.md's "Acceptance
mechanism" section for what a per-iteration-position breakdown
(early/mid/late step) would additionally require and why it is not
produced here.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
D_CSV = PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/speculative/consolidated"


def load_csv(path: Path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    rows = load_csv(D_CSV)
    n = len(rows)
    forward_count = np.array([int(r["target_forward_count"]) for r in rows])
    accepted_sum = np.array([int(r["accepted_sum"]) for r in rows])
    # forward_count - 1: exclude the initial warmup forward (seeds the KV
    # cache, verifies nothing), matching tail_analysis.py's own definition.
    accept_rate = accepted_sum / (forward_count - 1)

    stats_summary = {
        "n": n,
        "config": "threshold=0.35_gamma=8_selector=recent_k:2",
        "accept_rate_avg_accepted_per_iteration": {
            "mean": float(accept_rate.mean()),
            "median": float(np.median(accept_rate)),
            "std": float(accept_rate.std()),
            "min": float(accept_rate.min()),
            "max": float(accept_rate.max()),
            "p10": float(np.percentile(accept_rate, 10)),
            "p90": float(np.percentile(accept_rate, 90)),
        },
        "gamma": 8,
        "note": "accept_rate = accepted_sum / (target_forward_count - 1); "
        "max possible per-iteration value is gamma=8 (all drafted "
        "coordinates accepted every iteration).",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "accept_rate_distribution_stats.json"
    with stats_path.open("w") as stream:
        json.dump(stats_summary, stream, indent=2)
        stream.write("\n")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(accept_rate, bins=np.arange(0, 8.5, 0.5), color="tab:green", edgecolor="black", alpha=0.75)
    axis.axvline(float(accept_rate.mean()), color="black", linestyle="--", linewidth=1,
                 label=f"mean={accept_rate.mean():.2f}")
    axis.set_xlabel("Accept Rate (avg accepted coordinates / iteration, max=gamma=8)")
    axis.set_ylabel("Sample Count")
    axis.set_title(f"Acceptance Rate Distribution, Config D (n={n})")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "accept_rate_histogram.png", dpi=160)
    plt.close(figure)

    print(json.dumps(stats_summary, indent=2))
    print(f"wrote {stats_path}")
    print(f"wrote {OUTPUT_DIR / 'accept_rate_histogram.png'}")


if __name__ == "__main__":
    main()
