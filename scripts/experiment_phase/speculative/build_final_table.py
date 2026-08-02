#!/usr/bin/env python3
"""Builds the final integrated results table: everything in
results/speculative/consolidated/consolidated_results.{csv,md} plus the
Selector x Speculative ablation rows (B, D, D') from Task 2's combined
run, all in one place with speedup_claim_valid/accuracy_preserved columns."""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONSOLIDATED_CSV = PROJECT_ROOT / "results/speculative/consolidated/consolidated_results.csv"
ABLATION_SUMMARY = PROJECT_ROOT / "results/speculative/20260802T101802Z/summary.json"

COLUMNS = [
    "config", "mode", "num_samples", "threshold", "gamma", "keep_k",
    "mae", "corrected_rmse", "mean_angular_error", "latency_median_ms",
    "target_forward_avg", "selector_forward_count",
    "speedup_claim_valid", "accuracy_preserved",
]


def load_existing_rows():
    with CONSOLIDATED_CSV.open() as stream:
        return list(csv.DictReader(stream))


def build_ablation_rows():
    with ABLATION_SUMMARY.open() as stream:
        summary = json.load(stream)
    rows = []
    baseline = summary["baseline"]
    rows.append({
        "config": "B_recent_k2_only", "mode": "selector_only", "num_samples": 1698,
        "threshold": "", "gamma": "", "keep_k": 2,
        "mae": baseline["mae"], "corrected_rmse": baseline["corrected_rmse"],
        "mean_angular_error": baseline["mae"],
        "latency_median_ms": baseline["latency_median_ms"],
        "target_forward_avg": baseline["target_forward_avg"],
        "selector_forward_count": "",
        "speedup_claim_valid": False, "accuracy_preserved": True,
    })
    for config in summary["configs"]:
        label = "D_recentk2_plus_speculative" if config["threshold"] == 0.35 else "D_prime_recentk2_plus_speculative_th0.7"
        rows.append({
            "config": label, "mode": "selector_plus_speculative",
            "num_samples": 1698, "threshold": config["threshold"],
            "gamma": config["gamma"], "keep_k": 2,
            "mae": config["mae"], "corrected_rmse": config["corrected_rmse"],
            "mean_angular_error": config["mae"],
            "latency_median_ms": config["latency_median_ms"],
            "target_forward_avg": config["target_forward_avg"],
            "selector_forward_count": "",
            "speedup_claim_valid": config["speedup_claim_valid"],
            "accuracy_preserved": config["accuracy_preserved"],
        })
    return rows


def main():
    output_dir = Path(__file__).resolve()
    rows = load_existing_rows() + build_ablation_rows()

    out_dir = PROJECT_ROOT / "results/speculative/consolidated"
    csv_path = out_dir / "final_table.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})

    md_path = out_dir / "final_table.md"
    header = "| " + " | ".join(COLUMNS) + " |"
    separator = "|" + "|".join(["---"] * len(COLUMNS)) + "|"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
