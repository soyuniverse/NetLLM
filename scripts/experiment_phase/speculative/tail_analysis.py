#!/usr/bin/env python3
"""Tail analysis: which samples degrade under the headline config D
(RecentK-2 + speculative) relative to baseline A, and why.

Re-derives history motion speed/acceleration from the dataset (the
per-sample result CSVs don't store the raw history array) using the same
create_dataset(...) call as every other script this session -- dataset
construction is deterministic, so sample_id lines up with row order in
every per_sample_*.csv already produced.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from netllm_litevlm.vp.checkpoint_era_runtime import UPSTREAM_VP_ROOT

A_CSV = PROJECT_ROOT / "results/speculative/20260802T075640Z/per_sample_baseline.csv"
B_CSV = PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_baseline_selector=recent_k:2.csv"
D_CSV = PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv"

OUTPUT_DIR = PROJECT_ROOT / "results/speculative/consolidated"
DOC_DIR = PROJECT_ROOT / "docs/experiment_phase/analysis"


def load_csv(path: Path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def wrapped_abs_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a-b| accounting for +-180/360 wraparound (roll/yaw); harmless for
    bounded pitch since its diffs never approach 180."""
    raw = np.abs(a - b) % 360.0
    return np.minimum(raw, 360.0 - raw)


def motion_stats(history: np.ndarray):
    # history: [10, 3] raw-degree (roll, pitch, yaw)
    step_diffs = wrapped_abs_diff_deg(history[1:], history[:-1])  # [9,3]
    velocity_per_step = step_diffs.mean(axis=1)  # [9] degrees/step, avg over channels
    avg_velocity = float(velocity_per_step.mean())
    accel_per_step = np.abs(np.diff(velocity_per_step))  # [8]
    avg_acceleration = float(accel_per_step.mean()) if len(accel_per_step) else 0.0
    return avg_velocity, avg_acceleration


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
    histories = []
    for i in range(len(test_dataset)):
        history_np, _future_np, _info = test_dataset[i]
        histories.append(history_np)
    return histories


def main():
    rows_a = load_csv(A_CSV)
    rows_b = load_csv(B_CSV)
    rows_d = load_csv(D_CSV)
    n = len(rows_a)
    assert len(rows_b) == n and len(rows_d) == n

    for ra, rb, rd in zip(rows_a, rows_b, rows_d):
        key = (ra["video"], ra["user"], ra["timestep"])
        assert (rb["video"], rb["user"], rb["timestep"]) == key
        assert (rd["video"], rd["user"], rd["timestep"]) == key

    mae_a = np.array([float(r["mae"]) for r in rows_a])
    mae_b = np.array([float(r["mae"]) for r in rows_b])
    mae_d = np.array([float(r["mae"]) for r in rows_d])
    diff_d_a = mae_d - mae_a
    diff_b_a = mae_b - mae_a
    diff_d_b = mae_d - mae_b
    # exact identity, sanity check the CSVs were paired correctly
    assert np.allclose(diff_d_a, diff_b_a + diff_d_b, atol=1e-6)

    forward_count_d = np.array([int(r["target_forward_count"]) for r in rows_d])
    accepted_sum_d = np.array([int(r["accepted_sum"]) for r in rows_d])
    accept_rate_d = accepted_sum_d / (forward_count_d - 1)

    print("loading histories from dataset...")
    histories = load_histories()
    assert len(histories) == n
    velocities = np.zeros(n)
    accelerations = np.zeros(n)
    for i, history in enumerate(histories):
        velocities[i], accelerations[i] = motion_stats(history)

    # --- top 5% degraded samples ---
    threshold_idx = int(np.ceil(n * 0.95))
    order = np.argsort(diff_d_a)
    top5_idx = order[threshold_idx:]
    top5_idx = top5_idx[np.argsort(-diff_d_a[top5_idx])]  # worst first

    top5_records = []
    for i in top5_idx:
        top5_records.append({
            "sample_id": int(i),
            "video": rows_a[i]["video"], "user": rows_a[i]["user"],
            "timestep": rows_a[i]["timestep"],
            "diff_d_a": float(diff_d_a[i]),
            "diff_b_a": float(diff_b_a[i]), "diff_d_b": float(diff_d_b[i]),
            "avg_velocity_deg_per_step": float(velocities[i]),
            "avg_acceleration_deg_per_step2": float(accelerations[i]),
            "target_forward_count": int(forward_count_d[i]),
            "accept_rate": float(accept_rate_d[i]),
        })

    # --- correlation analysis (all 1,698 samples) ---
    rho_velocity, p_velocity = stats.spearmanr(velocities, diff_d_a)
    rho_accel, p_accel = stats.spearmanr(accelerations, diff_d_a)
    rho_accept, p_accept = stats.spearmanr(accept_rate_d, diff_d_a)
    rho_forward, p_forward = stats.spearmanr(forward_count_d, diff_d_a)

    # --- hypothesis check: does the top-5% degraded group have
    # meaningfully higher motion speed than the rest? ---
    rest_idx = order[:threshold_idx]
    top5_velocity_mean = float(velocities[top5_idx].mean())
    rest_velocity_mean = float(velocities[rest_idx].mean())
    mannwhitney_velocity = stats.mannwhitneyu(
        velocities[top5_idx], velocities[rest_idx], alternative="greater"
    )

    # --- attribution across the full top-5% group ---
    selector_attributed = int(np.sum(np.abs(diff_b_a[top5_idx]) >= np.abs(diff_d_b[top5_idx])))
    speculative_attributed = len(top5_idx) - selector_attributed

    stats_summary = {
        "n": n,
        "top5pct_count": len(top5_idx),
        "correlations_spearman": {
            "velocity_vs_diff": {"rho": rho_velocity, "p": p_velocity},
            "acceleration_vs_diff": {"rho": rho_accel, "p": p_accel},
            "accept_rate_vs_diff": {"rho": rho_accept, "p": p_accept},
            "forward_count_vs_diff": {"rho": rho_forward, "p": p_forward},
        },
        "hypothesis_check_high_motion_concentration": {
            "top5pct_mean_velocity_deg_per_step": top5_velocity_mean,
            "rest_mean_velocity_deg_per_step": rest_velocity_mean,
            "ratio": top5_velocity_mean / rest_velocity_mean if rest_velocity_mean else None,
            "mannwhitneyu_greater_pvalue": float(mannwhitney_velocity.pvalue),
        },
        "attribution_top5pct": {
            "selector_attributed_count": selector_attributed,
            "speculative_attributed_count": speculative_attributed,
            "selector_attributed_fraction": selector_attributed / len(top5_idx),
        },
        "top5pct_samples": top5_records,
    }

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "tail_analysis_stats.json"
    with stats_path.open("w") as stream:
        json.dump(stats_summary, stream, indent=2)
        stream.write("\n")

    # --- plots ---
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(velocities, diff_d_a, s=8, alpha=0.4, color="tab:blue")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("History Avg. Motion Speed (deg/step)")
    axis.set_ylabel("Per-Sample MAE Diff, D - A (degrees)")
    axis.set_title(f"Motion Speed vs. MAE Degradation (Spearman rho={rho_velocity:.3f}, p={p_velocity:.1e})")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "tail_velocity_vs_diff.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(accept_rate_d, diff_d_a, s=8, alpha=0.4, color="tab:orange")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Speculative Accept Rate (avg accepted/iteration, D)")
    axis.set_ylabel("Per-Sample MAE Diff, D - A (degrees)")
    axis.set_title(f"Accept Rate vs. MAE Degradation (Spearman rho={rho_accept:.3f}, p={p_accept:.1e})")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "tail_acceptrate_vs_diff.png", dpi=160)
    plt.close(figure)

    print(json.dumps(stats_summary["correlations_spearman"], indent=2))
    print(json.dumps(stats_summary["hypothesis_check_high_motion_concentration"], indent=2))
    print(json.dumps(stats_summary["attribution_top5pct"], indent=2))
    print(f"wrote {stats_path}")
    print(f"wrote {OUTPUT_DIR / 'tail_velocity_vs_diff.png'}")
    print(f"wrote {OUTPUT_DIR / 'tail_acceptrate_vs_diff.png'}")


if __name__ == "__main__":
    main()
