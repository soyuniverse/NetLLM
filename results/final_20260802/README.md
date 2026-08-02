# Final Presentation Package — 2026-08-02

**These files are copies.** The originals (and per-sample CSVs, JSON
summaries, and additional intermediate runs) live at their recorded
paths under `results/speculative/<timestamp>/` and
`results/speculative/consolidated/` — those are the paths referenced by
`docs/experiment_phase/speculative/PHASE_B_REAL_RESULTS.md` and
`manifests/final_run_manifest.md`, and are the ones to treat as
authoritative if this copy and the original ever diverge. This directory
exists only as a single place to hand off the headline table and figures
without needing to know the individual run timestamps.

See `docs/final/FINAL_RESULTS_SUMMARY.md` for the narrative writeup and
`manifests/final_run_manifest.md` for exact reproduction commands.

- `final_table.csv` / `final_table.md` — baseline, the 4 selected
  speculative configs, the AttentionTopK/RecentK selector comparison, and
  the 3 Selector x Speculative ablation configs, one table.
- `threshold_vs_forward_count.png`, `threshold_vs_mae.png`,
  `mae_latency_tradeoff.png` — 50-sample smoke grid (full threshold x
  gamma coverage).
- `mae_cdf.png` — full 1,698-sample per-sample MAE CDF, baseline /
  RecentK-2 / speculative-only / combined.
- `ablation_bars.png` — full 1,698-sample MAE and latency bars for
  configs A-D'.
