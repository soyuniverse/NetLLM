#!/usr/bin/env python3
"""Plot the completed recovered-artifact Llama selector benchmark."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path("/root/NetLLM")
DEFAULT_INPUT = (
    PROJECT / "experiments/vp/llama_benchmark/full/benchmark_summary.csv"
)
DEFAULT_OUTPUT = PROJECT / "experiments/vp/llama_benchmark/figures"


def read_completed_summary(path):
    status = path.parent / "run_status.txt"
    if not status.exists() or "success=true" not in status.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("completed full benchmark status was not found")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6 or any(int(row["sample_count"]) != 1698 for row in rows):
        raise RuntimeError("full benchmark summary is incomplete")
    return rows


def labels(rows):
    return [row["selector"] for row in rows]


def values(rows, key):
    return [float(row[key]) for row in rows]


def decorate(ax, rows, title, xlabel, ylabel):
    ax.set_title(
        "{}\ncheckpoint-era recovered artifact comparison".format(title)
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.01, 0.01,
        "paper reproduction invalid | n={} per configuration".format(
            rows[0]["sample_count"]
        ),
        transform=ax.transAxes, fontsize=8, va="bottom",
    )


def line_figure(rows, x_key, y_key, title, xlabel, ylabel, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = values(rows, x_key)
    y = values(rows, y_key)
    ax.plot(x, y, marker="o", linewidth=1.8)
    for x_value, y_value, label in zip(x, y, labels(rows)):
        marker = "*" if label == "original" else "o"
        size = 150 if label == "original" else 50
        ax.scatter([x_value], [y_value], marker=marker, s=size, zorder=3)
        offset = {
            "original": (-67, -18),
            "identity_keep10": (5, -18),
        }.get(label, (4, 5))
        ax.annotate(label, (x_value, y_value), xytext=offset,
                    textcoords="offset points", fontsize=8)
    decorate(ax, rows, title, xlabel, ylabel)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def gpu_figure(rows, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = values(rows, "keep_ratio")
    allocated = values(rows, "peak_allocated_mib")
    reserved = values(rows, "peak_reserved_mib")
    ax.plot(x, allocated, marker="o", label="peak allocated")
    ax.plot(x, reserved, marker="s", label="peak reserved")
    for x_value, y_value, label in zip(x, allocated, labels(rows)):
        ax.annotate(label, (x_value, y_value), xytext=(4, 5),
                    textcoords="offset points", fontsize=8)
    ax.legend()
    decorate(
        ax, rows, "Keep ratio vs GPU memory", "Initial keep ratio",
        "GPU memory (MiB)",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = read_completed_summary(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("keep_ratio", "mae", "Keep ratio vs MAE",
         "Initial keep ratio", "MAE (degrees)", "keep_ratio_vs_mae.png"),
        ("keep_ratio", "corrected_rmse",
         "Keep ratio vs corrected rotation-aware RMSE",
         "Initial keep ratio", "Corrected RMSE (degrees)",
         "keep_ratio_vs_corrected_rmse.png"),
        ("keep_ratio", "evaluation_loss", "Keep ratio vs evaluation loss",
         "Initial keep ratio", "Normalized MSE loss",
         "keep_ratio_vs_loss.png"),
        ("keep_ratio", "latency_median_ms", "Keep ratio vs latency",
         "Initial keep ratio", "Median latency (ms)",
         "keep_ratio_vs_latency.png"),
        ("mae", "latency_median_ms", "MAE-latency tradeoff",
         "MAE (degrees)", "Median latency (ms)",
         "mae_latency_tradeoff.png"),
        ("processed_sequence_length_sum", "latency_median_ms",
         "Processed sequence length vs latency",
         "Processed sequence-length sum", "Median latency (ms)",
         "processed_tokens_vs_latency.png"),
    ]
    for x_key, y_key, title, xlabel, ylabel, filename in specifications:
        line_figure(
            rows, x_key, y_key, title, xlabel, ylabel,
            args.output_dir / filename,
        )
    gpu_figure(rows, args.output_dir / "keep_ratio_vs_gpu_memory.png")

    expected = sorted(spec[-1] for spec in specifications) + [
        "keep_ratio_vs_gpu_memory.png"
    ]
    print("generated={}".format(",".join(expected)))


if __name__ == "__main__":
    main()
