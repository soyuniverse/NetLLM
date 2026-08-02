# Final Run Manifest — 2026-08-02

Reproducibility record for every result referenced in
`docs/final/FINAL_RESULTS_SUMMARY.md` and
`docs/experiment_phase/speculative/PHASE_B_REAL_RESULTS.md`.

## Code

- Repository commit at time of writing: `0ee3ffba2e54a1c63561843879ec7ef19ee9adc4`
  (later commits in this session add this manifest and Task 5's file
  audit; the results themselves were produced at or before this commit).
- Vendored upstream source: `third_party/netllm_upstream/` — commit
  `ee4d8726898610e4ae7df08bdd26728cafb4701f` of
  `https://github.com/duowuyms/NetLLM.git` (see
  `third_party/netllm_upstream/PROVENANCE.md`).

## Environment

| package | version |
|---|---|
| torch | 2.2.0 (cu121) |
| transformers | 4.34.1 |
| peft | 0.6.2 |
| accelerate | 0.24.1 |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce RTX 4090, driver 580.95.05 |
| dtype | float16 throughout |

Installed per `docs/experiment_phase/llama/environment/LLAMA_ENVIRONMENT_PLAN_V2.md`
+ the accelerate/huggingface-hub fix in
`docs/experiment_phase/phase1_5a/PLM_DEPENDENCY_COMPATIBILITY.md`.

## Assets

| asset | path | verification |
|---|---|---|
| base Llama2-7b weights | `/root/llama2-7b-base/` | architecture matches config (32 layers, hidden=4096) |
| VP checkpoint | `/root/NetLLM-assets/checkpoints/try_llama2_7b/` | strict load: adapter missing/unexpected/value-mismatch = 0/0/0; non-PLM missing/unexpected = 0/0 (`scripts/experiment_phase/assets/verify_checkpoint_strict_load.py`, `experiments/vp/asset_recovery/checkpoint_strict_load.json`) |
| Jin2022 dataset | `/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022/` | test split = exactly 1,698 samples; sample 0 = (video=4, user=83, timestep=30), matching prior recorded runs |

Placement and pre-extraction discrepancy handling documented in
`docs/experiment_phase/assets/ASSET_RECOVERY_VERIFICATION.md`.

## Seed

`--seed 0` (the harness's default) for every run below — never overridden.

## Exact commands

All run from the repository root.

```bash
# Full baseline (config A), and the 4 selected speculative configs (C
# is threshold=0.35_gamma=8 in this run's output).
python3 scripts/experiment_phase/speculative/run_speculative_benchmark.py \
  --checkpoint-path /root/NetLLM-assets/checkpoints/try_llama2_7b \
  --dataset-path /root/NetLLM-source/viewport_prediction/data/viewports/Jin2022 \
  --num-samples 1698 --thresholds "" --gammas ""
# -> results/speculative/20260802T075640Z/

python3 scripts/experiment_phase/speculative/run_speculative_benchmark.py \
  --checkpoint-path /root/NetLLM-assets/checkpoints/try_llama2_7b \
  --dataset-path /root/NetLLM-source/viewport_prediction/data/viewports/Jin2022 \
  --num-samples 1698 --thresholds "0.35,0.7,1.5,2.5" --gammas "8"
# -> results/speculative/20260802T082009Z/

# 50-sample smoke grid used to select the above thresholds/gamma (5
# calibrated thresholds x 3 gammas, then 3 boundary-search extras).
python3 scripts/experiment_phase/speculative/run_speculative_benchmark.py \
  --checkpoint-path /root/NetLLM-assets/checkpoints/try_llama2_7b \
  --dataset-path /root/NetLLM-source/viewport_prediction/data/viewports/Jin2022 \
  --num-samples 50
# -> results/speculative/20260802T081351Z/ (defaults: thresholds
#    0.05,0.1,0.2,0.35,0.7; gammas 2,4,8)

python3 scripts/experiment_phase/speculative/run_speculative_benchmark.py \
  --checkpoint-path /root/NetLLM-assets/checkpoints/try_llama2_7b \
  --dataset-path /root/NetLLM-source/viewport_prediction/data/viewports/Jin2022 \
  --num-samples 50 --thresholds "1.0,1.5,2.5" --gammas "8"
# -> results/speculative/20260802T081802Z/

# Selector x Speculative ablation: B (RecentK-2 alone), D and D'
# (RecentK-2 + speculative at threshold 0.35 and 0.7, gamma=8).
python3 scripts/experiment_phase/speculative/run_speculative_benchmark.py \
  --checkpoint-path /root/NetLLM-assets/checkpoints/try_llama2_7b \
  --dataset-path /root/NetLLM-source/viewport_prediction/data/viewports/Jin2022 \
  --num-samples 1698 --selector "recent_k:2" --thresholds "0.35,0.7" --gammas "8"
# -> results/speculative/20260802T101802Z/

# AttentionTopK vs RecentK, real checkpoint, K in {8,6,4,2}, 50 samples.
python3 scripts/experiment_phase/llama/smoke/run_attention_topk_7b_smoke.py
# -> experiments/vp/attention_topk_7b_smoke/smoke_result.json

# Consolidation, paired stats, and figures (read the run directories
# above by their pinned timestamps -- update the constants at the top of
# each script if any run is repeated under a new timestamp).
python3 scripts/experiment_phase/speculative/consolidate_and_plot_results.py
python3 scripts/experiment_phase/speculative/paired_stats_and_cdf.py
python3 scripts/experiment_phase/speculative/plot_ablation_bars.py
python3 scripts/experiment_phase/speculative/build_final_table.py
```

## Gate tests (must pass before any number above is trusted)

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
```

32 passed, 3 skipped (skips are pre-existing, unrelated to this
session's work — a missing Phase 3A retry artifact). Includes
`tests/speculative/test_block_verify.py`'s Selector x
SpeculativeBlockVerifyPipeline compatibility gates
(`test_threshold_zero_matches_baseline_with_recent_k_selector`,
parametrized over k in {4, 6, 10}) added this session.
