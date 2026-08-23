#!/usr/bin/env python3
"""Builds the module-by-module presentation figure package in
results/presentation_20260816/, reorganizing this project's already-
verified 2026-08-02 full-1,698-sample results (and, where noted in the
legend, the 50-sample AttentionTopK-vs-RecentK comparison, the only
scale that selector was ever measured at) into one slide-ready figure
per module. No new experiments -- every number here traces back to a
git-tracked CSV/JSON already referenced by FINAL_RESULTS_SUMMARY.md /
TAIL_ANALYSIS.md / consolidate_and_plot_results.py.

Styled consistently with the existing consolidated figures: figsize
(7, 4.5) single-panel / (11, 4.5) two-panel, dpi=160, tab:blue/orange/
green/red/purple + black, grid alpha=0.3 (see
scripts/experiment_phase/speculative/consolidate_and_plot_results.py
and paired_stats_and_cdf.py).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "presentation_20260816"

NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fm.fontManager.addfont(NANUM_PATH)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

FINAL_TABLE = PROJECT_ROOT / "results/speculative/consolidated/final_table.csv"
ATTN_SMOKE_DIR = "results/speculative/20260802T081351Z + 20260802T081802Z (50-sample smoke grid)"
FULL_BASELINE_DIR = "results/speculative/20260802T075640Z (full 1,698-sample baseline)"
FULL_SELECTED_DIR = "results/speculative/20260802T082009Z (full 1,698-sample, 4 speculative configs)"
COMBO_DIR = "results/speculative/20260802T101802Z (full 1,698-sample, Selector x Speculative ablation)"

CDF_RUNS = {
    "baseline": (PROJECT_ROOT / "results/speculative/20260802T075640Z/per_sample_baseline.csv", "tab:blue", "-"),
    "speculative": (PROJECT_ROOT / "results/speculative/20260802T082009Z/per_sample_threshold=0.35_gamma=8.csv", "tab:green", "--"),
    "recent_k2": (PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_baseline_selector=recent_k:2.csv", "tab:orange", "-"),
    "combined": (PROJECT_ROOT / "results/speculative/20260802T101802Z/per_sample_threshold=0.35_gamma=8_selector=recent_k:2.csv", "tab:red", "--"),
}
CDF_LABELS = {
    "baseline": "baseline",
    "speculative": "+Speculative",
    "recent_k2": "+RecentK-2",
    "combined": "+둘 다 (combined)",
}


def load_final_table_rows():
    with FINAL_TABLE.open() as stream:
        return {row["config"]: row for row in csv.DictReader(stream)}


def footnote(axis_or_fig, text):
    axis_or_fig.text(0.01, 0.01, text, fontsize=7, color="gray", transform=axis_or_fig.transFigure if hasattr(axis_or_fig, "add_axes") else None)


def module1_token_selection(rows):
    k_axis = [10, 8, 6, 4, 2]
    baseline_50 = float(rows["selector_baseline_50sample"]["mae"])
    recent_k_50 = {
        10: baseline_50,  # RecentK(10) == full history == no selection == baseline
        8: float(rows["recent_k_k=8"]["mae"]),
        6: float(rows["recent_k_k=6"]["mae"]),
        4: float(rows["recent_k_k=4"]["mae"]),
        2: float(rows["recent_k_k=2"]["mae"]),
    }
    attn_topk_50 = {
        8: float(rows["attention_top_k_k=8"]["mae"]),
        6: float(rows["attention_top_k_k=6"]["mae"]),
        4: float(rows["attention_top_k_k=4"]["mae"]),
        2: float(rows["attention_top_k_k=2"]["mae"]),
    }
    recent_k2_full1698 = float(rows["B_recent_k2_only"]["mae"])

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        k_axis, [recent_k_50[k] for k in k_axis], color="tab:orange", marker="o",
        label="RecentK (50-sample basis)",
    )
    axis.plot(
        [8, 6, 4, 2], [attn_topk_50[k] for k in [8, 6, 4, 2]], color="tab:green", marker="s",
        label="AttentionTopK (50-sample basis)",
    )
    axis.axhline(baseline_50, color="black", linestyle="--", alpha=0.6, label="baseline (50-sample, K=10 equivalent)")
    axis.scatter(
        [2], [recent_k2_full1698], color="tab:red", marker="*", s=220, zorder=5,
        label="RecentK K=2, full 1,698-sample confirmation",
    )
    axis.annotate(
        f"{recent_k2_full1698:.3f} (full-scale)",
        xy=(2, recent_k2_full1698), xytext=(4.3, recent_k2_full1698 - 0.55),
        fontsize=8, color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1),
    )
    axis.invert_xaxis()
    axis.set_xticks(k_axis)
    axis.set_xlim(10.6, 1.4)
    axis.set_xlabel("K (history length kept)")
    axis.set_ylabel("MAE (degrees, lower is better)")
    axis.set_title("이력 선택: 최근성이 attention 중요도를 능가\n(Recency beats attention importance for history selection)")
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axis.grid(True, alpha=0.3)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.text(
        0.01, 0.01,
        "출처: 50-sample smoke grid (results/speculative/20260802T081351Z+081802Z);\n"
        "K=2 full-scale: results/speculative/20260802T101802Z. 곡선(주황/초록/점선)은 동일 50-sample, 빨간 별만 full 1,698-sample.",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "module1_token_selection.png", dpi=160)
    plt.close(figure)


def module2_speculative(rows):
    figure, (fwd_axis, th_axis) = plt.subplots(1, 2, figsize=(11, 4.5))

    baseline_fwd = float(rows["baseline"]["target_forward_avg"])
    speculative_fwd = float(rows["threshold=0.35_gamma=8"]["target_forward_avg"])
    fwd_axis.bar(["baseline", "speculative\n(th=0.35, γ=8)"], [baseline_fwd, speculative_fwd], color=["black", "tab:green"])
    for i, v in enumerate([baseline_fwd, speculative_fwd]):
        fwd_axis.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)
    fwd_axis.set_ylabel("LLM forward count per prediction")
    fwd_axis.set_title("예측당 LLM forward 수")
    fwd_axis.grid(True, axis="y", alpha=0.3)

    thresholds = [0.35, 0.7, 1.5, 2.5]
    th_mae = [float(rows[f"threshold={t}_gamma=8"]["mae"]) for t in thresholds]
    th_axis.plot([str(t) for t in thresholds], th_mae, color="tab:purple", marker="o")
    th_axis.set_ylim(12.7, 13.0)
    th_axis.set_xlabel("acceptance threshold (normalized L2 space)")
    th_axis.set_ylabel("MAE (degrees)")
    th_axis.set_title("Threshold sweep: MAE 둔감성\n(y-axis zoomed to show flatness)")
    th_axis.grid(True, alpha=0.3)

    figure.suptitle("모듈②: Speculative Decoding — forward 수 절감 + threshold 둔감성")
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.text(
        0.01, 0.01,
        f"출처: {FULL_SELECTED_DIR}.",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "module2_speculative.png", dpi=160)
    plt.close(figure)


def load_mae(path: Path):
    with path.open() as stream:
        return [float(row["mae"]) for row in csv.DictReader(stream)]


def module3_combination():
    figure, axis = plt.subplots(figsize=(7, 4.5))
    mae_by_run = {}
    for name, (path, color, linestyle) in CDF_RUNS.items():
        values = load_mae(path)
        mae_by_run[name] = values
        ordered = sorted(values)
        n = len(ordered)
        cumulative = [(i + 1) / n for i in range(n)]
        axis.plot(ordered, cumulative, label=CDF_LABELS[name], color=color, linestyle=linestyle)

    axis.set_xlabel("Per-sample MAE (degrees)")
    axis.set_ylabel("Cumulative Fraction of Samples")
    axis.set_title("모듈③: 조합 — 두 쌍 겹침 = 가산적 결합\n(Per-Sample MAE CDF, 1,698 samples, same checkpoint)")
    axis.legend(fontsize=9, loc="lower right")
    axis.grid(True, alpha=0.3)

    axis.annotate(
        "baseline ≈ +Speculative\n(speculative alone barely shifts accuracy)",
        xy=(15, 0.55), xytext=(22, 0.18), fontsize=8, color="tab:blue",
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1),
    )
    axis.annotate(
        "+RecentK-2 ≈ +둘 다\n(selector drives the shift;\nspeculative composes additively)",
        xy=(8, 0.62), xytext=(35, 0.62), fontsize=8, color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1),
    )

    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.text(
        0.01, 0.01,
        "출처: results/speculative/20260802T101802Z (RecentK-2 + combined),\n"
        "20260802T075640Z (baseline), 20260802T082009Z (speculative) — 모두 full 1,698-sample.",
        fontsize=6.5, color="gray",
    )
    figure.savefig(OUTPUT_DIR / "module3_combination.png", dpi=160)
    plt.close(figure)


def summary_table(rows):
    configs = [
        ("A", "baseline", "baseline"),
        ("B", "RecentK-2 only", "B_recent_k2_only"),
        ("C", "Speculative only", "threshold=0.35_gamma=8"),
        ("D", "RecentK-2 + Speculative", "D_recentk2_plus_speculative"),
    ]
    baseline_mae = float(rows["baseline"]["mae"])
    baseline_latency = float(rows["baseline"]["latency_median_ms"])

    table_rows = []
    for letter, label, key in configs:
        row = rows[key]
        mae = float(row["mae"])
        latency = float(row["latency_median_ms"])
        forward = float(row["target_forward_avg"])
        delta_pct = (mae - baseline_mae) / baseline_mae * 100.0
        speedup = baseline_latency / latency
        table_rows.append((letter, label, mae, delta_pct, latency, speedup, forward))

    # --- Markdown ---
    md_lines = [
        "# Summary Table — A/B/C/D (full 1,698-sample, same checkpoint)",
        "",
        f"Source: `{FINAL_TABLE.relative_to(PROJECT_ROOT)}` "
        f"({FULL_BASELINE_DIR}; {FULL_SELECTED_DIR}; {COMBO_DIR}).",
        "",
        "| config | MAE (deg) | ΔMAE % vs A | latency median (ms) | speedup vs A | avg forward count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for letter, label, mae, delta_pct, latency, speedup, forward in table_rows:
        md_lines.append(
            f"| {letter}. {label} | {mae:.3f} | {delta_pct:+.2f}% | {latency:.1f} | {speedup:.2f}x | {forward:.2f} |"
        )
    (OUTPUT_DIR / "summary_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # --- PNG render ---
    figure, axis = plt.subplots(figsize=(11, 2.8))
    axis.axis("off")
    col_labels = ["Config", "MAE (deg)", "ΔMAE %", "Latency (ms)", "Speedup", "Avg fwd/pred"]
    cell_text = []
    for letter, label, mae, delta_pct, latency, speedup, forward in table_rows:
        cell_text.append([
            f"{letter}. {label}",
            f"{mae:.3f}",
            f"{delta_pct:+.2f}%",
            f"{latency:.1f}",
            f"{speedup:.2f}x",
            f"{forward:.2f}",
        ])
    table = axis.table(
        cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
        colWidths=[0.30, 0.14, 0.14, 0.16, 0.13, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for col in range(len(col_labels)):
        table[0, col].set_facecolor("#2b2b2b")
        table[0, col].set_text_props(color="white", weight="bold")
    row_colors = {"A": "#f2f2f2", "B": "#fde9d9", "C": "#daf2da", "D": "#fddede"}
    for r, (letter, *_rest) in enumerate(table_rows, start=1):
        for col in range(len(col_labels)):
            table[r, col].set_facecolor(row_colors[letter])
            if col == 0:
                table[r, col].set_text_props(ha="left")
                table[r, col].PAD = 0.03
    axis.set_title("A/B/C/D Summary — full 1,698-sample, same checkpoint", fontsize=12, pad=14)
    figure.tight_layout()
    figure.text(0.01, 0.01, f"출처: {FINAL_TABLE.relative_to(PROJECT_ROOT)}", fontsize=6.5, color="gray")
    figure.savefig(OUTPUT_DIR / "summary_table.png", dpi=160)
    plt.close(figure)

    return table_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_final_table_rows()
    module1_token_selection(rows)
    module2_speculative(rows)
    module3_combination()
    table_rows = summary_table(rows)
    print("Wrote figures to", OUTPUT_DIR)
    for letter, label, mae, delta_pct, latency, speedup, forward in table_rows:
        print(f"{letter}: {label}: MAE={mae:.4f} ({delta_pct:+.2f}%), latency={latency:.1f}ms, speedup={speedup:.2f}x, fwd={forward:.2f}")


if __name__ == "__main__":
    main()
