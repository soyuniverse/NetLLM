#!/usr/bin/env python3
"""Task 3 (adaptive-K) full-scale judgment: per-sample paired before/after
on the pre-existing top-5% degraded group (from TAIL_ANALYSIS.md's
tail_analysis_stats.json, computed against the D=RecentK-2+speculative
per-sample CSVs), plus overall MAE and K-distribution for the new
AdaptiveKSelector full 1,698-sample run.

"Before" reuses the already-verified, git-tracked 2026-08-02 full-1,698
D per-sample CSV (accuracy reproduces bit-for-bit across instances, see
GATE_VERIFICATION_20260823.md) rather than re-running D on this
instance -- consistent with this session's rule that only latency needs
a fresh same-instance baseline, not accuracy.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

D_BEFORE_CSV = PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv"
TAIL_STATS = PROJECT_ROOT / "results/speculative/consolidated/tail_analysis_stats.json"
OUTPUT_DIR = PROJECT_ROOT / "results/speculative/consolidated"
DOC_DIR = PROJECT_ROOT / "docs/experiment_phase/analysis"

NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def load_csv(path: Path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(after_csv_path: str, adaptive_k_run_dir: str):
    fm.fontManager.addfont(NANUM_PATH)
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

    after_csv = Path(after_csv_path).resolve()
    run_dir = Path(adaptive_k_run_dir).resolve()

    before_rows = load_csv(D_BEFORE_CSV)
    after_rows = load_csv(after_csv)
    assert len(before_rows) == len(after_rows) == 1698, (
        f"expected 1698 rows, got before={len(before_rows)} after={len(after_rows)}"
    )
    before_mae = [float(r["mae"]) for r in before_rows]
    after_mae = [float(r["mae"]) for r in after_rows]
    after_k = [int(r["selected_k"]) for r in after_rows]

    with TAIL_STATS.open() as stream:
        tail_stats = json.load(stream)
    degraded_ids = sorted(s["sample_id"] for s in tail_stats["top5pct_samples"])

    # --- Overall MAE (all 1,698) ---
    overall_before = float(np.mean(before_mae))
    overall_after = float(np.mean(after_mae))

    # --- Degraded-group paired before/after ---
    deg_before = np.array([before_mae[i] for i in degraded_ids])
    deg_after = np.array([after_mae[i] for i in degraded_ids])
    deg_diff = deg_after - deg_before  # negative = improved
    n_improved = int((deg_diff < 0).sum())
    n_worsened = int((deg_diff > 0).sum())
    n_unchanged = int((deg_diff == 0).sum())

    # --- K distribution ---
    k_counts = {k: after_k.count(k) for k in sorted(set(after_k))}

    # --- Widened-sample breakdown: degraded-group vs rest-of-population ---
    # (diagnoses whether the overall-MAE regression is a false-positive-
    # widening effect: samples the velocity threshold flagged as risky
    # that were NOT actually in the degraded group, where TAIL_ANALYSIS's
    # own population-wide negative correlation means widening more often
    # costs accuracy than it protects it.)
    degraded_set = set(degraded_ids)
    widened_idx = [i for i, k in enumerate(after_k) if k != 2]
    widened_degraded = [i for i in widened_idx if i in degraded_set]
    widened_rest = [i for i in widened_idx if i not in degraded_set]
    rest_before = np.array([before_mae[i] for i in range(1698) if i not in degraded_set])
    rest_after = np.array([after_mae[i] for i in range(1698) if i not in degraded_set])
    widened_rest_before = np.array([before_mae[i] for i in widened_rest])
    widened_rest_after = np.array([after_mae[i] for i in widened_rest])

    result = {
        "n_total": len(after_mae),
        "overall_mae_before_plain_recent_k2_plus_speculative": overall_before,
        "overall_mae_after_adaptive_k_plus_speculative": overall_after,
        "overall_mae_delta": overall_after - overall_before,
        "overall_mae_delta_pct": (overall_after - overall_before) / overall_before * 100.0,
        "rest_population_n": 1698 - len(degraded_ids),
        "rest_population_mean_before": float(rest_before.mean()),
        "rest_population_mean_after": float(rest_after.mean()),
        "rest_population_mean_diff": float((rest_after - rest_before).mean()),
        "degraded_group_n": len(degraded_ids),
        "degraded_group_mean_before": float(deg_before.mean()),
        "degraded_group_mean_after": float(deg_after.mean()),
        "degraded_group_mean_diff": float(deg_diff.mean()),
        "degraded_group_median_diff": float(np.median(deg_diff)),
        "degraded_group_n_improved": n_improved,
        "degraded_group_n_worsened": n_worsened,
        "degraded_group_n_unchanged": n_unchanged,
        "k_distribution": k_counts,
        "k_distribution_pct": {k: round(v / len(after_k) * 100.0, 2) for k, v in k_counts.items()},
        "widened_total_n": len(widened_idx),
        "widened_from_degraded_group_n": len(widened_degraded),
        "widened_from_rest_population_n": len(widened_rest),
        "widened_rest_mean_before": float(widened_rest_before.mean()) if widened_rest else None,
        "widened_rest_mean_after": float(widened_rest_after.mean()) if widened_rest else None,
        "widened_rest_mean_diff": float((widened_rest_after - widened_rest_before).mean()) if widened_rest else None,
        "widened_rest_n_worsened": int(((widened_rest_after - widened_rest_before) > 0).sum()) if widened_rest else 0,
        "widened_rest_n_improved": int(((widened_rest_after - widened_rest_before) < 0).sum()) if widened_rest else 0,
    }
    print(json.dumps(result, indent=2))

    stats_path = OUTPUT_DIR / "adaptive_k_results_stats.json"
    with stats_path.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")

    # --- Figure 1: degraded-group before/after scatter ---
    figure, axis = plt.subplots(figsize=(7, 4.5))
    lims = [min(deg_before.min(), deg_after.min()) - 1, max(deg_before.max(), deg_after.max()) + 1]
    axis.plot(lims, lims, color="gray", linestyle="--", alpha=0.6, label="y=x (no change)")
    axis.scatter(deg_before, deg_after, color="tab:red", alpha=0.7, s=25)
    axis.set_xlim(lims)
    axis.set_ylim(lims)
    axis.set_xlabel("Before: MAE under plain RecentK-2 + Speculative (D)")
    axis.set_ylabel("After: MAE under Adaptive-K + Speculative")
    axis.set_title(
        f"기존 열화 상위 5% 샘플군 Before/After (n={len(degraded_ids)})\n"
        f"점이 대각선 아래 = 개선({n_improved}), 위 = 악화({n_worsened})"
    )
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.text(
        0.01, 0.01,
        f"출처: before={D_BEFORE_CSV.relative_to(PROJECT_ROOT)} (2026-08-02, full 1,698);\n"
        f"after={after_csv.relative_to(PROJECT_ROOT)} (2026-08-23, full 1,698).",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "adaptive_k_degraded_before_after.png", dpi=160)
    plt.close(figure)

    # --- Figure 2: K distribution histogram ---
    figure, axis = plt.subplots(figsize=(7, 4.5))
    ks = sorted(k_counts.keys())
    counts = [k_counts[k] for k in ks]
    colors = {2: "tab:orange", 4: "tab:blue", 10: "tab:green"}
    bar_colors = [colors.get(k, "gray") for k in ks]
    bars = axis.bar([str(k) for k in ks], counts, color=bar_colors)
    for bar, k in zip(bars, ks):
        pct = k_counts[k] / len(after_k) * 100.0
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                   f"{k_counts[k]}\n({pct:.1f}%)", ha="center", fontsize=9)
    axis.set_xlabel("Selected K (adaptive)")
    axis.set_ylabel("Sample count")
    axis.set_title(f"Adaptive-K 선택 분포 (full 1,698-sample, v_low=2.41, v_high=4.44)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.text(
        0.01, 0.01,
        f"출처: {after_csv.relative_to(PROJECT_ROOT)} (2026-08-23, full 1,698, selected_k column).",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "adaptive_k_distribution_histogram.png", dpi=160)
    plt.close(figure)

    print(f"\nWrote {stats_path}")
    print(f"Wrote {OUTPUT_DIR / 'adaptive_k_degraded_before_after.png'}")
    print(f"Wrote {OUTPUT_DIR / 'adaptive_k_distribution_histogram.png'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: adaptive_k_results.py <after_per_sample_csv> <adaptive_k_run_dir>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
