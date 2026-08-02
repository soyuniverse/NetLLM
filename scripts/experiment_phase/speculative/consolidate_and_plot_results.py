#!/usr/bin/env python3
"""Consolidates this session's speculative + AttentionTopK results into one
table and 3 figures, styled consistently with
scripts/experiment_phase/benchmark/plot_vp_benchmark.py (same figsize,
marker style, grid, dpi).

Two different sample scales are combined deliberately, and kept labeled
rather than blurred together: the 3 figures use the 50-sample smoke grid
(the only run with full threshold coverage across gamma), while the
consolidated table's speculative rows are the trustworthy full
1,698-sample numbers from Task 3 step 3. Mixing them into one number
would misrepresent which results are full-scale.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SMOKE_GRID_DIRS = [
    PROJECT_ROOT / "results/speculative/20260802T081351Z",  # 5 thresholds x 3 gammas
    PROJECT_ROOT / "results/speculative/20260802T081802Z",  # boundary extras, gamma=8
]
FULL_RUN_DIR = PROJECT_ROOT / "results/speculative/20260802T075640Z"  # full baseline
FULL_SELECTED_DIR = PROJECT_ROOT / "results/speculative/20260802T082009Z"  # full 4 configs
ATTENTION_TOPK_SMOKE = PROJECT_ROOT / "experiments/vp/attention_topk_7b_smoke/smoke_result.json"

OUTPUT_DIR = PROJECT_ROOT / "results/speculative/consolidated"


def _read_csv_rows(path: Path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load_smoke_grid():
    rows = []
    seen = set()
    for directory in SMOKE_GRID_DIRS:
        for row in _read_csv_rows(directory / "results.csv"):
            key = (row["config"], row["threshold"], row["gamma"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def load_full_results():
    baseline = _read_csv_rows(FULL_RUN_DIR / "results.csv")[0]
    # speedup_claim_valid/accuracy_preserved live only in summary.json,
    # not results.csv.
    with (FULL_SELECTED_DIR / "summary.json").open() as stream:
        summary = json.load(stream)
    return baseline, summary["configs"]


def build_consolidated_table(baseline, full_configs, attention_topk):
    columns = [
        "config", "mode", "num_samples", "threshold", "gamma", "keep_k",
        "mae", "corrected_rmse", "mean_angular_error", "latency_median_ms",
        "target_forward_avg", "selector_forward_count",
        "speedup_claim_valid", "accuracy_preserved",
    ]
    rows = []
    rows.append({
        "config": "baseline", "mode": "baseline", "num_samples": 1698,
        "threshold": "", "gamma": "", "keep_k": "",
        "mae": baseline["mae"], "corrected_rmse": baseline["corrected_rmse"],
        "mean_angular_error": baseline["mae"],  # mean_angular_error == mae
                                                  # under this project's
                                                  # rotation-aware definition
        "latency_median_ms": baseline["latency_median_ms"],
        "target_forward_avg": baseline["target_forward_avg"],
        "selector_forward_count": "", "speedup_claim_valid": False,
        "accuracy_preserved": True,
    })
    for row in full_configs:
        rows.append({
            "config": row["config"], "mode": "speculative_block_verify",
            "num_samples": 1698, "threshold": row["threshold"],
            "gamma": row["gamma"], "keep_k": "",
            "mae": row["mae"], "corrected_rmse": row["corrected_rmse"],
            "mean_angular_error": row["mae"],
            "latency_median_ms": row["latency_median_ms"],
            "target_forward_avg": row["target_forward_avg"],
            "selector_forward_count": "",
            "speedup_claim_valid": row["speedup_claim_valid"],
            "accuracy_preserved": row["accuracy_preserved"],
        })
    for k, comparison in attention_topk["k_comparison"].items():
        for selector_name, key in (
            ("attention_top_k", "attention_top_k"), ("recent_k", "recent_k")
        ):
            entry = comparison[key]
            rows.append({
                "config": f"{selector_name}_k={k}",
                "mode": f"selector_{selector_name}",
                "num_samples": attention_topk["num_samples"],
                "threshold": "", "gamma": "", "keep_k": k,
                "mae": entry["mae"], "corrected_rmse": entry["corrected_rmse"],
                "mean_angular_error": entry["mae"],
                "latency_median_ms": entry["latency_median_ms"],
                "target_forward_avg": 20,
                "selector_forward_count": entry["selector_forward_count"],
                "speedup_claim_valid": "", "accuracy_preserved": "",
            })
    rows.append({
        "config": "selector_baseline_50sample", "mode": "baseline_50sample",
        "num_samples": attention_topk["num_samples"], "threshold": "",
        "gamma": "", "keep_k": "", "mae": attention_topk["baseline"]["mae"],
        "corrected_rmse": attention_topk["baseline"]["corrected_rmse"],
        "mean_angular_error": attention_topk["baseline"]["mae"],
        "latency_median_ms": attention_topk["baseline"]["latency_median_ms"],
        "target_forward_avg": 20, "selector_forward_count": "",
        "speedup_claim_valid": "", "accuracy_preserved": "",
    })
    return columns, rows


def write_table(columns, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "consolidated_results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUTPUT_DIR / "consolidated_results.md"
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def _save_line(series, xlabel, ylabel, title, output, hline=None, hline_label=None):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, (xs, ys) in series.items():
        axis.plot(xs, ys, marker="o", label=label)
    if hline is not None:
        axis.axhline(hline, color="black", linestyle="--", alpha=0.6, label=hline_label)
    axis.legend()
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def make_plots(smoke_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_gamma = {"2": [], "4": [], "8": []}
    for row in smoke_rows:
        if row["config"] == "baseline" or not row["gamma"]:
            continue
        by_gamma[row["gamma"]].append(row)
    for gamma_rows in by_gamma.values():
        gamma_rows.sort(key=lambda r: float(r["threshold"]))

    # 1) threshold vs forward count, one line per gamma
    forward_series = {
        f"gamma={g}": (
            [float(r["threshold"]) for r in rows],
            [float(r["target_forward_avg"]) for r in rows],
        )
        for g, rows in by_gamma.items() if rows
    }
    _save_line(
        forward_series, "Acceptance Threshold (normalized L2)",
        "Avg Target Forward Count", "Threshold vs Target Forward Count (50-sample smoke)",
        OUTPUT_DIR / "threshold_vs_forward_count.png",
    )

    # 2) threshold vs MAE, baseline horizontal line
    baseline_mae = float(
        next(r for r in smoke_rows if r["config"] == "baseline")["mae"]
    )
    mae_series = {
        f"gamma={g}": (
            [float(r["threshold"]) for r in rows],
            [float(r["mae"]) for r in rows],
        )
        for g, rows in by_gamma.items() if rows
    }
    _save_line(
        mae_series, "Acceptance Threshold (normalized L2)", "MAE (degrees)",
        "Threshold vs MAE (50-sample smoke)",
        OUTPUT_DIR / "threshold_vs_mae.png",
        hline=baseline_mae, hline_label="baseline MAE",
    )

    # 3) MAE-latency tradeoff, all smoke configs + baseline + RecentK(k=2)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    colors = {"2": "tab:blue", "4": "tab:orange", "8": "tab:green"}
    for g, rows in by_gamma.items():
        if not rows:
            continue
        axis.scatter(
            [float(r["latency_median_ms"]) for r in rows],
            [float(r["mae"]) for r in rows],
            label=f"speculative gamma={g}", color=colors[g], marker="o",
        )
    baseline_row = next(r for r in smoke_rows if r["config"] == "baseline")
    axis.scatter(
        [float(baseline_row["latency_median_ms"])], [float(baseline_row["mae"])],
        label="baseline", color="black", marker="*", s=150,
    )
    with ATTENTION_TOPK_SMOKE.open() as stream:
        attention_topk = json.load(stream)
    recent_k2 = attention_topk["k_comparison"]["2"]["recent_k"]
    axis.scatter(
        [recent_k2["latency_median_ms"]], [recent_k2["mae"]],
        label="Recent-K (k=2)", color="red", marker="^", s=100,
    )
    axis.set_xlabel("Latency Median (ms)")
    axis.set_ylabel("MAE (degrees, lower is better)")
    axis.set_title("MAE-Latency Tradeoff (50-sample smoke)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mae_latency_tradeoff.png", dpi=160)
    plt.close(figure)


def main():
    baseline, full_configs = load_full_results()
    with ATTENTION_TOPK_SMOKE.open() as stream:
        attention_topk = json.load(stream)
    columns, rows = build_consolidated_table(baseline, full_configs, attention_topk)
    csv_path, md_path = write_table(columns, rows)

    smoke_rows = load_smoke_grid()
    make_plots(smoke_rows)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote 3 figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
