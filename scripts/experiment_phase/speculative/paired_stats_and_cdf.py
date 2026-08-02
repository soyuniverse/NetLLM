#!/usr/bin/env python3
"""Paired per-sample statistics (headline config D vs. baseline A) and a
NetLLM-Figure-10(b)-style per-sample MAE CDF across baseline / RecentK-2 /
speculative-only / combined.

All four runs share the same 1,698-sample dataset order (verified: same
video/user/timestep per row index across files), so pairing by row index
is valid without needing to re-match on (video, user, timestep).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RUNS = {
    "baseline": PROJECT_ROOT / "results/speculative/20260802T075640Z/per_sample_baseline.csv",
    "recent_k2": PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_baseline_selector=recent_k:2.csv",
    "speculative": PROJECT_ROOT / "results/speculative/20260802T082009Z/per_sample_threshold=0.35_gamma=8.csv",
    "combined": PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "results/speculative/consolidated"


def load_mae(path: Path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return [float(row["mae"]) for row in rows], [
        (row["video"], row["user"], row["timestep"]) for row in rows
    ]


def _percentile(sorted_values, p):
    if not sorted_values:
        raise ValueError("empty")
    n = len(sorted_values)
    rank = (n - 1) * p
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_stats(baseline_mae, headline_mae):
    diffs = sorted(h - b for h, b in zip(headline_mae, baseline_mae))
    degraded = sum(1 for d in diffs if d > 0)
    return {
        "n": len(diffs),
        "median_diff": _percentile(diffs, 0.5),
        "p50_diff": _percentile(diffs, 0.5),
        "p90_diff": _percentile(diffs, 0.9),
        "p99_diff": _percentile(diffs, 0.99),
        "degraded_sample_count": degraded,
        "degraded_sample_fraction": degraded / len(diffs),
        "mean_diff": sum(diffs) / len(diffs),
    }


def plot_cdf(mae_by_run, output_path):
    # baseline/speculative nearly overlap (speculative decoding barely
    # shifts the accuracy distribution), as do recent_k2/combined (the
    # selector is what shifts it) -- dashed vs solid makes the two
    # near-identical pairs distinguishable rather than fully hidden.
    styles = {
        "baseline": ("tab:blue", "-"),
        "speculative": ("tab:green", "--"),
        "recent_k2": ("tab:orange", "-"),
        "combined": ("tab:red", "--"),
    }
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, values in mae_by_run.items():
        ordered = sorted(values)
        n = len(ordered)
        cumulative = [(i + 1) / n for i in range(n)]
        color, linestyle = styles.get(label, (None, "-"))
        axis.plot(ordered, cumulative, label=label, color=color, linestyle=linestyle)
    axis.set_xlabel("Per-sample MAE (degrees)")
    axis.set_ylabel("Cumulative Fraction of Samples")
    axis.set_title("Per-Sample MAE CDF (1,698 samples, same checkpoint)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    mae_by_run = {}
    keys_by_run = {}
    for name, path in RUNS.items():
        mae, keys = load_mae(path)
        mae_by_run[name] = mae
        keys_by_run[name] = keys

    reference_keys = keys_by_run["baseline"]
    for name, keys in keys_by_run.items():
        if keys != reference_keys:
            raise RuntimeError(f"sample order mismatch: {name} vs baseline")

    stats = {
        "combined_vs_baseline": paired_stats(mae_by_run["baseline"], mae_by_run["combined"]),
        # Isolates which component (selector vs speculative decoding)
        # drives the combined config's per-sample behavior: if
        # combined_vs_recent_k2 is near-zero while recent_k2_vs_baseline
        # matches combined_vs_baseline closely, the accuracy shift is
        # attributable to the selector, and speculative decoding adds
        # negligible per-sample noise on top of it.
        "combined_vs_recent_k2": paired_stats(mae_by_run["recent_k2"], mae_by_run["combined"]),
        "recent_k2_vs_baseline": paired_stats(mae_by_run["baseline"], mae_by_run["recent_k2"]),
        "speculative_vs_baseline": paired_stats(mae_by_run["baseline"], mae_by_run["speculative"]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "paired_stats_combined_vs_baseline.json"
    with stats_path.open("w") as stream:
        json.dump(stats, stream, indent=2)
        stream.write("\n")

    cdf_path = OUTPUT_DIR / "mae_cdf.png"
    plot_cdf(mae_by_run, cdf_path)

    print(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")
    print(f"wrote {cdf_path}")


if __name__ == "__main__":
    main()
