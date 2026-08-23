#!/usr/bin/env python3
"""Presentation figure for the adaptive-K negative result: left panel
shows the overall-vs-degraded-group MAE before/after contrast (opposite
directions at a glance), right panel shows the improved/worsened
composition of the 445 widened samples split by true-positive (in the
pre-existing degraded group) vs false-positive. Styled consistently with
build_presentation_figures.py (figsize (11,4.5), dpi=160, NanumGothic).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = PROJECT_ROOT / "results/speculative/consolidated/adaptive_k_results_stats.json"
OUTPUT_DIR = PROJECT_ROOT / "results/presentation_20260816"
NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def main():
    fm.fontManager.addfont(NANUM_PATH)
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

    with STATS_PATH.open() as stream:
        stats = json.load(stream)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: overall vs degraded-group MAE before/after ---
    groups = ["전체 (n=1,698)", "기존 열화 상위 5%\n(n=84)"]
    before_vals = [stats["overall_mae_before_plain_recent_k2_plus_speculative"],
                   stats["degraded_group_mean_before"]]
    after_vals = [stats["overall_mae_after_adaptive_k_plus_speculative"],
                  stats["degraded_group_mean_after"]]
    x = range(len(groups))
    width = 0.35
    left.bar([i - width / 2 for i in x], before_vals, width, label="Before (D)", color="tab:blue")
    left.bar([i + width / 2 for i in x], after_vals, width, label="After (Adaptive-K)", color="tab:red")
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        pct = (a - b) / b * 100.0
        sign = "+" if pct > 0 else ""
        left.annotate(
            f"{sign}{pct:.1f}%", xy=(i, max(b, a) + 0.6), ha="center", fontsize=10,
            color="tab:red" if pct > 0 else "tab:green", fontweight="bold",
        )
    left.set_xticks(list(x))
    left.set_xticklabels(groups)
    left.set_ylabel("MAE (degrees)")
    left.set_title("방향이 반대다: 전체 악화, 열화군 개선")
    left.legend(fontsize=9)
    left.grid(True, axis="y", alpha=0.3)

    # --- Right: widened-sample composition, TP vs FP ---
    tp_n = stats["widened_from_degraded_group_n"]
    fp_n = stats["widened_from_rest_population_n"]
    tp_improved, tp_worsened = 50, 13  # from the diagnosis re-check
    fp_improved, fp_worsened = stats["widened_rest_n_improved"], stats["widened_rest_n_worsened"]

    categories = [f"True Positive\n(n={tp_n})", f"False Positive\n(n={fp_n})"]
    improved_counts = [tp_improved, fp_improved]
    worsened_counts = [tp_worsened, fp_worsened]
    x2 = range(len(categories))
    right.bar(x2, improved_counts, color="tab:green")
    right.bar(x2, worsened_counts, bottom=improved_counts, color="tab:red")
    right.set_ylim(0, 460)
    for i, (imp, wor) in enumerate(zip(improved_counts, worsened_counts)):
        total = imp + wor
        # Small bars (e.g. n=63) can't fit text inside their own segments
        # legibly next to a n=382 bar on the same axis -- always label
        # just outside each segment's top edge instead, with the two
        # labels staggered far enough apart to never collide even when
        # the bar itself is short.
        right.text(i, imp + 30, f"개선 {imp} ({imp/total*100:.0f}%)", ha="center", va="bottom", fontsize=9, color="tab:green", fontweight="bold")
        right.text(i, total + 30, f"악화 {wor} ({wor/total*100:.0f}%)", ha="center", va="bottom", fontsize=9, color="tab:red", fontweight="bold")
    right.set_xticks(list(x2))
    right.set_xticklabels(categories)
    right.set_ylabel("Sample count")
    right.set_title("위드닝된 445개 중 True/False Positive 결과")
    right.grid(True, axis="y", alpha=0.3)

    figure.suptitle("Adaptive-K: 설계는 유효하나 트리거가 부정확하다 (negative result)")
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    figure.text(
        0.01, 0.01,
        "출처: results/speculative/consolidated/adaptive_k_results_stats.json "
        "(20260823T070043Z, full 1,698-sample, 이 인스턴스).",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "adaptive_k_negative_result.png", dpi=160)
    plt.close(figure)
    print(f"Wrote {OUTPUT_DIR / 'adaptive_k_negative_result.png'}")


if __name__ == "__main__":
    main()
