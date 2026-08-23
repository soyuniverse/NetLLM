#!/usr/bin/env python3
"""Task 2 follow-up diagnosis (2026-08-23): among the 445 samples
Adaptive-K widened away from K=2, why did 63 (true positives, in the
pre-existing top-5% degraded group) improve while 382 (false positives)
worsened on average? No model/GPU involved -- history features only,
re-derived from the dataset exactly as tail_analysis.py does (same
create_dataset(...) call, deterministic construction, sample_id lines up
with row order in every per_sample_*.csv already produced this project).
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from netllm_litevlm.vp.checkpoint_era_runtime import UPSTREAM_VP_ROOT  # noqa: E402

ADAPTIVE_K_CSV = PROJECT_ROOT / "results/speculative/20260823T070043Z/per_sample_threshold=0.35_gamma=8_selector=adaptive_k:2.41:4.44.csv"
TAIL_STATS = PROJECT_ROOT / "results/speculative/consolidated/tail_analysis_stats.json"
OUTPUT_DIR = PROJECT_ROOT / "results/speculative/consolidated"
NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def wrapped_abs_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raw = np.abs(a - b) % 360.0
    return np.minimum(raw, 360.0 - raw)


def signed_wrapped_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference a-b, in [-180, 180)."""
    return ((a - b + 180.0) % 360.0) - 180.0


def history_features(history: np.ndarray) -> dict:
    # history: [10, 3] raw-degree (roll, pitch, yaw)
    step_diffs = wrapped_abs_diff_deg(history[1:], history[:-1])  # [9,3]
    velocity_per_step = step_diffs.mean(axis=1)  # [9]
    avg_velocity = float(velocity_per_step.mean())
    velocity_std = float(velocity_per_step.std())
    velocity_cv = velocity_std / avg_velocity if avg_velocity > 1e-9 else 0.0

    accel_per_step = np.abs(np.diff(velocity_per_step))  # [8]
    avg_acceleration = float(accel_per_step.mean()) if len(accel_per_step) else 0.0

    signed_diffs = signed_wrapped_diff_deg(history[1:], history[:-1])  # [9,3]
    signs = np.sign(signed_diffs)  # [9,3], 0 counts as no direction
    reversal_counts = []
    for ch in range(3):
        s = signs[:, ch]
        nonzero = s[s != 0]
        if len(nonzero) < 2:
            reversal_counts.append(0)
            continue
        reversals = int((nonzero[1:] != nonzero[:-1]).sum())
        reversal_counts.append(reversals)
    avg_reversals = float(np.mean(reversal_counts))

    return {
        "avg_velocity": avg_velocity,
        "velocity_std": velocity_std,
        "velocity_cv": velocity_cv,
        "avg_acceleration": avg_acceleration,
        "direction_reversals": avg_reversals,
    }


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
        histories.append(np.asarray(history_np))
    return histories


def auc_from_mannwhitney(group_a, group_b):
    """Probability that a random group_b value exceeds a random group_a
    value (rank-biserial via the Mann-Whitney U statistic). 0.5 = no
    separation; closer to 0 or 1 = better separation."""
    u_stat, _ = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return float(u_stat / (len(group_a) * len(group_b)))


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def main():
    fm.fontManager.addfont(NANUM_PATH)
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

    with ADAPTIVE_K_CSV.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1698
    selected_k = {int(r["sample_id"]): int(r["selected_k"]) for r in rows}

    with TAIL_STATS.open() as stream:
        tail_stats = json.load(stream)
    degraded_set = set(s["sample_id"] for s in tail_stats["top5pct_samples"])

    widened_ids = sorted(sid for sid, k in selected_k.items() if k != 2)
    true_positive_ids = [sid for sid in widened_ids if sid in degraded_set]
    false_positive_ids = [sid for sid in widened_ids if sid not in degraded_set]
    print(f"widened total: {len(widened_ids)}, true positive: {len(true_positive_ids)}, "
          f"false positive: {len(false_positive_ids)}")
    assert len(true_positive_ids) == 63, f"expected 63 true positives, got {len(true_positive_ids)}"
    assert len(false_positive_ids) == 382, f"expected 382 false positives, got {len(false_positive_ids)}"

    print("Loading dataset histories (CPU only, no model)...")
    histories = load_histories()

    tp_features = [history_features(histories[i]) for i in true_positive_ids]
    fp_features = [history_features(histories[i]) for i in false_positive_ids]

    metric_names = ["avg_velocity", "velocity_std", "velocity_cv", "avg_acceleration", "direction_reversals"]
    results = {}
    for name in metric_names:
        tp_vals = np.array([f[name] for f in tp_features])
        fp_vals = np.array([f[name] for f in fp_features])
        _, p_value = stats.mannwhitneyu(tp_vals, fp_vals, alternative="two-sided")
        auc = auc_from_mannwhitney(tp_vals, fp_vals)
        d = cohens_d(tp_vals, fp_vals)
        results[name] = {
            "tp_mean": float(tp_vals.mean()), "tp_std": float(tp_vals.std()),
            "fp_mean": float(fp_vals.mean()), "fp_std": float(fp_vals.std()),
            "mannwhitney_p": float(p_value),
            "auc_fp_exceeds_tp": auc,
            "separation": abs(auc - 0.5),
            "cohens_d": d,
        }

    ranked = sorted(metric_names, key=lambda n: results[n]["separation"], reverse=True)
    print("\nRanked by separation |AUC - 0.5| (descending):")
    for name in ranked:
        r = results[name]
        print(f"  {name}: AUC={r['auc_fp_exceeds_tp']:.3f}, d={r['cohens_d']:.3f}, "
              f"p={r['mannwhitney_p']:.4f}, TP mean={r['tp_mean']:.3f}, FP mean={r['fp_mean']:.3f}")

    output = {
        "n_true_positive": len(true_positive_ids),
        "n_false_positive": len(false_positive_ids),
        "metrics": results,
        "ranked_by_separation": ranked,
    }
    stats_path = OUTPUT_DIR / "adaptive_k_fp_diagnosis_stats.json"
    with stats_path.open("w") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(f"\nWrote {stats_path}")

    # --- Boxplot of the top-ranked metric ---
    top_metric = ranked[0]
    tp_top = [f[top_metric] for f in tp_features]
    fp_top = [f[top_metric] for f in fp_features]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    box = axis.boxplot(
        [tp_top, fp_top], labels=[f"True Positive\n(n={len(tp_top)})", f"False Positive\n(n={len(fp_top)})"],
        patch_artist=True, widths=0.5,
    )
    for patch, color in zip(box["boxes"], ["tab:green", "tab:red"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axis.set_ylabel(top_metric.replace("_", " "))
    r = results[top_metric]
    axis.set_title(
        f"위드닝된 445개 중 True/False Positive 분리도: {top_metric}\n"
        f"AUC={r['auc_fp_exceeds_tp']:.3f}, Cohen's d={r['cohens_d']:.2f}, p={r['mannwhitney_p']:.4f}"
    )
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.text(
        0.01, 0.01,
        "출처: results/speculative/20260823T070043Z (K 배정), consolidated/tail_analysis_stats.json (열화군 정의),\n"
        "Jin2022 test split에서 history feature 직접 재계산 (모델/GPU 불필요).",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "adaptive_k_fp_diagnosis_boxplot.png", dpi=160)
    plt.close(figure)
    print(f"Wrote {OUTPUT_DIR / 'adaptive_k_fp_diagnosis_boxplot.png'}")


if __name__ == "__main__":
    main()
