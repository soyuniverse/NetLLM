#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = [
    "model",
    "checkpoint",
    "selector",
    "selected_tokens",
    "keep_ratio",
    "mae",
    "rmse",
    "mean_angular_error",
    "test_loss",
    "latency_median_ms",
    "latency_p95_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "metric_valid",
]


def load_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"CSV schema mismatch: expected {REQUIRED_COLUMNS}, got {list(frame.columns)}"
        )
    if frame["selector"].duplicated().any():
        raise ValueError("selector names must be unique")
    frame["metric_valid"] = (
        frame["metric_valid"].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    )
    if frame["metric_valid"].isna().any():
        raise ValueError("metric_valid must contain only true/false")
    return frame.sort_values("keep_ratio")


def _save_line(
    frame: pd.DataFrame,
    x: str,
    ys,
    labels,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for column, label in zip(ys, labels):
        valid = frame[[x, column]].dropna()
        if not valid.empty:
            axis.plot(valid[x], valid[column], marker="o", label=label)
            plotted = True
    if plotted and len(ys) > 1:
        axis.legend()
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No valid benchmark data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def create_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_metrics = frame[frame["metric_valid"]].copy()
    _save_line(
        valid_metrics,
        "keep_ratio",
        ["mae"],
        ["MAE"],
        "MAE",
        "Keep Ratio vs MAE",
        output_dir / "keep_ratio_vs_mae.png",
    )
    _save_line(
        valid_metrics,
        "keep_ratio",
        ["rmse"],
        ["Rotation-aware RMSE"],
        "RMSE",
        "Keep Ratio vs RMSE",
        output_dir / "keep_ratio_vs_rmse.png",
    )
    _save_line(
        frame,
        "keep_ratio",
        ["latency_median_ms", "latency_p95_ms"],
        ["Median", "P95"],
        "Latency (ms)",
        "Keep Ratio vs Inference Latency",
        output_dir / "keep_ratio_vs_latency.png",
    )
    _save_line(
        frame,
        "keep_ratio",
        ["peak_allocated_mib", "peak_reserved_mib"],
        ["Peak allocated", "Peak reserved"],
        "GPU Memory (MiB)",
        "Keep Ratio vs GPU Memory",
        output_dir / "keep_ratio_vs_gpu_memory.png",
    )
    _save_line(
        valid_metrics,
        "latency_median_ms",
        ["mae"],
        ["MAE"],
        "MAE (lower is better)",
        "MAE–Latency Tradeoff",
        output_dir / "accuracy_latency_tradeoff.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    frame = load_results(args.input_csv)
    if args.validate_only:
        print(
            f"CSV schema valid: rows={len(frame)}, "
            f"metric_valid={int(frame['metric_valid'].sum())}"
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    create_plots(frame, args.output_dir)
    print(f"Created 5 figures in {args.output_dir}")


if __name__ == "__main__":
    main()
