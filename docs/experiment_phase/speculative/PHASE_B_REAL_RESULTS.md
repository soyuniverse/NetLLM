# Phase B — Real-Checkpoint Speculative Decoding Results

**Same-checkpoint controlled comparison.** Every number in this document
comes from the same recovered `try_llama2_7b` checkpoint (strict-loaded,
missing/unexpected keys 0/0/0/0/0 — see
`docs/experiment_phase/assets/ASSET_RECOVERY_VERIFICATION.md`) and the
same recovered Jin2022 test split, on one GPU, in fp16. Only the decoding
strategy (baseline / block verification threshold / selector) changes
between rows. As with the original 7.26 selector benchmark, this is a
controlled comparison, not a claim of matching any external/paper
benchmark protocol.

## 0. Asset and baseline reproduction (prerequisite gates)

- Checkpoint strict load: adapter missing/unexpected/value-mismatch = 0/0/0;
  non-PLM missing/unexpected = 0/0.
- Dataset test split: exactly 1,698 samples (matches the 7.26 report).
- Full 1,698-sample baseline MAE: **12.798559** vs. the 7.26 report's
  12.798525 (diff 0.000034, fp16 noise). Corrected RMSE 27.118723 vs.
  27.118540.

Both gates passed before any speculative or selector number below was
treated as trustworthy.

## 1. Speculative block verification — full 1,698-sample results

Threshold is a normalized (Tanh-bounded, ~[-1,1]-per-channel) L2 distance
between draft and target task-head outputs (see
`docs/experiment_phase/speculative/PHASE_A_DESIGN.md` §3 and
`src/netllm_litevlm/speculative/block_verify.py`'s `acceptance_threshold`
docstring) — not degrees. Calibrated empirically against real draft-vs-
target disagreement (10 samples, 40 steps: median 0.174, most mass
0.01-0.7).

Selected via a 50-sample smoke across 5 thresholds × 3 gammas (+3
boundary-search extras at gamma=8) — full grid in
`results/speculative/20260802T081351Z/` and
`results/speculative/20260802T081802Z/`. All 18 smoke configs stayed
within 0.42% of baseline MAE; gamma=8 dominated latency at every
threshold since forward count was already near its floor.

| config | MAE | vs. baseline | corrected RMSE | latency median | target forward avg | speedup_claim_valid | accuracy_preserved |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 12.798559 | — | 27.118723 | 571.7 ms | 20.00 | False | True |
| threshold=0.35, gamma=8 | 12.831302 | +0.26% | 27.141736 | 124.4 ms | 4.21 | **True** | True |
| threshold=0.70, gamma=8 | 12.849404 | +0.40% | 27.154655 | 124.0 ms | 4.08 | **True** | True |
| threshold=1.50, gamma=8 | 12.892513 | +0.74% | 27.236853 | 123.5 ms | 4.03 | **True** | True |
| threshold=2.50, gamma=8 | 12.929292 | +1.02% | 27.327807 | 123.6 ms | 4.02 | **True** | True |

**Verdict criteria (defined in code,
`scripts/experiment_phase/speculative/run_speculative_benchmark.py`):**
- `speedup_claim_valid` = target-forward-count reduction AND latency
  reduction, both measured against baseline.
- `accuracy_preserved` = MAE ≤ baseline MAE × 1.05.

All four selected configs satisfy both. This is the first time in this
project's speculative-decoding work that `speedup_claim_valid=True` has
been recorded against the real checkpoint at full scale — earlier
prototypes either always cost the full 20 forwards
(`continuous_draft_verify.py`, see `PHASE_A_DESIGN.md` §2) or ran only
against a randomly-initialized head (`PHASE_B_7B_SMOKE.md`). No general
"speedup" claim is made beyond these measured numbers: latency was
measured on one GPU, one process, fp16, this checkpoint, this dataset —
not benchmarked against other implementations or hardware.

## 2. AttentionTopK selector — 50-sample preliminary comparison

**Not full-scale.** `experiments/vp/attention_topk_7b_smoke/smoke_result.json`,
50 samples, same real checkpoint. Recorded as measured, not extended to
1,698 samples this session (selector latency doesn't reduce the
baseline's own 20-forward non-cached loop the way block verification
does, so there was no clear efficiency case to justify the extra GPU
time the way there was for the speculative grid).

| selector | K | MAE | corrected RMSE | latency median |
|---|---:|---:|---:|---:|
| baseline (50-sample) | 10 (all) | 11.036768 | 22.132159 | 577.2 ms |
| AttentionTopK | 8 | 11.162241 | 22.245248 | 577.1 ms |
| RecentK | 8 | 11.124346 | 22.332132 | 577.5 ms |
| AttentionTopK | 6 | 11.335791 | 22.511452 | 578.9 ms |
| RecentK | 6 | 10.984009 | 22.271884 | 577.7 ms |
| AttentionTopK | 4 | 11.520563 | 22.998189 | 578.6 ms |
| RecentK | 4 | 10.582027 | 22.058914 | 578.0 ms |
| AttentionTopK | 2 | 11.189642 | 21.210239 | 580.0 ms |
| RecentK | 2 | 10.017190 | 20.199241 | 577.6 ms |

**AttentionTopK's MAE is worse than RecentK's at every K tested, and the
gap widens as K shrinks** (K=8: +0.038, K=6: +0.352, K=4: +0.939, K=2:
+1.172). No claim is made about why beyond the plausible-but-unverified
note in the smoke script: viewport prediction may favor temporal
recency over first-decoder-layer attention salience, since head motion
tends to be locally continuous. This would need more samples and
probably a different importance signal (later layers? multiple layers
averaged?) to investigate further — out of scope for this session.

Selection latency itself did not measurably change baseline latency in
either direction (both selectors ~577-580ms vs. baseline's 577ms) —
expected, since `LlamaOldSelectablePipeline` still performs the full
20-step non-cached autoregressive loop regardless of history length;
selecting fewer history tokens shortens the *initial* sequence slightly
but the dominant cost is the 20 uncached forwards themselves.

## 3. Figures

`results/speculative/consolidated/` (50-sample smoke grid, for full
threshold coverage — the 4-point full-scale table above is the
trustworthy final numbers; these figures are the illustrative shape
across the swept range):

- `threshold_vs_forward_count.png` — gamma=8 forward count drops fastest
  and plateaus near 4.0 by threshold≈1.0; gamma=2/4 plateau higher.
- `threshold_vs_mae.png` — baseline MAE as a horizontal reference; all
  three gamma lines stay within ~0.05 degrees of it across the entire
  swept range up to threshold=2.5.
- `mae_latency_tradeoff.png` — all speculative configurations cluster at
  low latency (~120-330ms) and near-baseline MAE; baseline (star) and
  Recent-K k=2 (triangle) are plotted for reference. Recent-K's MAE
  measurement at 50-sample scale sits below baseline, consistent with the
  K=2 row in §2's table — noted, not explained further (small-sample
  effect is plausible but not verified).

Regenerate with `scripts/experiment_phase/speculative/consolidate_and_plot_results.py`
(reads the specific run directories it's pinned to; update its constants
if a run is re-done under a new timestamp).

## 4. What this does and doesn't establish

**Established**: block verification, on the real fine-tuned checkpoint
and real Jin2022 test data, reduces both measured target-forward-count
and measured wall-clock latency by roughly 4-5x while keeping MAE within
1% of baseline, across a threshold range with no sign of a sharp accuracy
cliff up to at least 2.5x the empirically-typical draft-target
disagreement scale.

**Not established**: general speedup claims independent of this exact
setup (one RTX 4090, fp16, this checkpoint); AttentionTopK as a
worthwhile selector for this task (the opposite trend was measured, at
50-sample scale); accuracy/speed behavior on the full held-out set for
thresholds beyond 2.5 or gammas beyond 8; combined
selector+speculative-decoding behavior (not tested together this
session).
