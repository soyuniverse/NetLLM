#!/usr/bin/env python3
"""Figures for GENERALIZATION_WU2017.md: (1) in-distribution (Jin2022,
full 1,698) vs. unseen (Wu2017, 300 evenly-sampled) MAE across the four
headline configs, (2) per-sample speculative accept-rate distribution
comparison, in-dist D vs. unseen D vs. unseen C.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results/speculative/consolidated"

INDIST = {"A": 12.798559396025476, "B": 10.84686681689948, "C": 12.831301641086654, "D": 10.895102344584037}
WU2017 = {"A": 15.895621841192918, "B": 13.606910356814714, "C": 15.927836322117027, "D": 13.64648620796602}

INDIST_D_CSV = PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv"
WU2017_D_CSV = PROJECT_ROOT / "results/speculative/20260823T075033Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv"
WU2017_C_CSV = PROJECT_ROOT / "results/speculative/20260823T074714Z/per_sample_threshold=0.35_gamma=8.csv"


def accept_rates(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return [int(r["accepted_sum"]) / (int(r["target_forward_count"]) - 1) for r in rows]


def fig_mae_comparison():
    configs = ["A", "B", "C", "D"]
    labels = ["A: baseline", "B: RecentK-2", "C: Speculative", "D: RecentK-2+Spec."]
    indist_vals = [INDIST[c] for c in configs]
    wu_vals = [WU2017[c] for c in configs]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    x = range(len(configs))
    width = 0.35
    axis.bar([i - width / 2 for i in x], indist_vals, width, label="Jin2022 (in-dist., n=1,698)", color="tab:blue")
    axis.bar([i + width / 2 for i in x], wu_vals, width, label="Wu2017 (unseen, n=300)", color="tab:orange")
    for i, (iv, wv) in enumerate(zip(indist_vals, wu_vals)):
        axis.text(i - width / 2, iv + 0.2, f"{iv:.2f}", ha="center", fontsize=8)
        axis.text(i + width / 2, wv + 0.2, f"{wv:.2f}", ha="center", fontsize=8)
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels)
    axis.set_ylabel("MAE (degrees)")
    axis.set_ylim(0, max(indist_vals + wu_vals) * 1.22)
    axis.set_title("In-Distribution vs. Unseen-Dataset MAE, 4 Headline Configs", pad=45)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=9)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.text(
        0.01, 0.01,
        "출처: Jin2022 results/speculative/consolidated/final_table.csv; "
        "Wu2017 results/speculative/20260823T074714Z + 20260823T075033Z (이 인스턴스, 2026-08-23).",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "generalization_mae_comparison.png", dpi=160)
    plt.close(figure)


def fig_accept_rate_comparison():
    indist_d = accept_rates(INDIST_D_CSV)
    wu_d = accept_rates(WU2017_D_CSV)
    wu_c = accept_rates(WU2017_C_CSV)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    bins = [x * 0.25 for x in range(16, 27)]  # 4.0 to 6.5 in 0.25 steps
    axis.hist(indist_d, bins=bins, alpha=0.5, label=f"Jin2022 D (n=1698, mean={sum(indist_d)/len(indist_d):.2f})", color="tab:blue", density=True)
    axis.hist(wu_d, bins=bins, alpha=0.5, label=f"Wu2017 D (n=300, mean={sum(wu_d)/len(wu_d):.2f})", color="tab:orange", density=True)
    axis.hist(wu_c, bins=bins, alpha=0.5, label=f"Wu2017 C, no selector (n=300, mean={sum(wu_c)/len(wu_c):.2f})", color="tab:green", density=True)
    axis.set_xlabel("Accept rate (avg accepted/iteration, max 8)")
    axis.set_ylabel("Density")
    axis.set_title("Speculative Accept-Rate Distribution: In-Dist vs. Unseen")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.text(
        0.01, 0.01,
        "출처: per-sample CSVs, accept_rate = accepted_sum/(target_forward_count-1).",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "generalization_accept_rate_comparison.png", dpi=160)
    plt.close(figure)


def main():
    fig_mae_comparison()
    fig_accept_rate_comparison()
    print("Wrote generalization_mae_comparison.png, generalization_accept_rate_comparison.png")


if __name__ == "__main__":
    main()
