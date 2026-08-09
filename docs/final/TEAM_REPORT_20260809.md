# Team Report — NetLLM VP: Speculative Decoding + Selector Study

2026-08-09. For the team / 선배. Condenses
`docs/final/FINAL_RESULTS_SUMMARY.md` (full narrative) and
`docs/experiment_phase/analysis/TAIL_ANALYSIS.md` (tail deep-dive) to
~2 pages. All numbers below are from the 2026-08-02 real-checkpoint,
full-1,698-sample run and are unaffected by this session's asset-loss
event (see "다음 확인 사항" below).

## (a) Results summary

Llama2-7B VP pipeline, Jin2022 test split (n=1,698), real fine-tuned
checkpoint:

| config | MAE (deg) | latency median | avg target forwards |
|---|---:|---:|---:|
| A. baseline (no selector, no speculative) | 12.799 | 571.7 ms | 20.00 |
| B. RecentK-2 selector only | 10.847 | 623.0 ms | 20.00 |
| C. Speculative decoding only (th=0.35, γ=8) | 12.831 | 124.4 ms | 4.21 |
| **D. RecentK-2 + Speculative (recommended default)** | **10.895** | **122.2 ms** | **4.01** |

**D is the only configuration that improves accuracy AND reduces
latency simultaneously**, both at full scale against the real
checkpoint — MAE down 14.8% vs. baseline, latency down 78.6%. Paired
per-sample analysis shows this is not a uniform win (47.1% of
individual samples are slightly worse under D, p99 diff +7.77°); see
(c) below for why.

## (b) Implementation approach

- **Block speculative decoding**
  (`src/netllm_litevlm/speculative/block_verify.py`,
  `SpeculativeBlockVerifyPipeline`): replaces the checkpoint-era
  per-step full-sequence recompute (no KV cache) with an incremental
  KV-cache loop. Each iteration embeds one carry token + `gamma`
  drafted coordinates (constant-velocity extrapolation), runs exactly
  one target forward, and verifies all `gamma` drafts against that
  single forward's output.
- **Acceptance criterion**: L2 distance in the task head's own
  Tanh-bounded **normalized** output space (not degrees) — unit space
  matters if you're tuning a threshold for a new checkpoint.
- **AttentionTopKSelector** (`src/netllm_litevlm/selectors/
  attention_topk.py`): top-K history embeddings by first-decoder-layer
  attention. Drop-in `BaseSelector` implementation; empirically loses
  to plain `RecentKSelector` at every K tested (see FINAL_RESULTS_
  SUMMARY.md §"recency dominates narrative" for three independent
  pieces of supporting evidence).
- **Composition**: the same selector instance wraps both the baseline
  and speculative pipelines unchanged — selection only shortens what
  the target LLM's initial prefill sees; the draft model always uses
  the full original history regardless of selector K (traced and
  tested, not just argued).
- **Verification**: threshold=0 equivalence gate passes at atol=1e-5
  (fp32/CPU) / atol=2e-3 (fp16/GPU) — chained KV-cache floating-point
  reassociation noise, not a control-flow bug (isolated and measured in
  `PHASE_A_DESIGN.md`). 32 tests passing, 3 pre-existing unrelated
  skips.

## (c) Tail analysis — key findings

Full doc: `docs/experiment_phase/analysis/TAIL_ANALYSIS.md`.

1. **Hypothesis "degradation concentrates in abrupt-motion segments
   (inertia breaks)" — partially supported, with a correction.** The
   top-5%-worst samples have 2.16x higher motion speed than the rest
   (p=3.0e-24) — worst cases *are* in high-motion segments. But the
   population-wide correlation is **negative** (Spearman rho=−0.400,
   p=2.8e-66): higher motion more often means *improvement*. The real
   shape is a fan spread — high-velocity samples have far greater
   outcome variance both ways; the worst-case tail is drawn from that
   high-variance regime, not from "high speed" as a simple predictor.
2. **Attribution: 100% RecentK-2, not speculative decoding.** For every
   one of the top-5% degraded samples, the selector's own effect
   dominates; speculative decoding's incremental contribution is small
   and often slightly *negative* (partially offsets rather than
   compounds the selector's error).
3. **Acceptance mechanism (new this session)**: accept rate is narrow
   and near-ceiling across the population — mean 6.22/8, median 6.33,
   ~99% of samples in one histogram bin, small low tail down to 4.25
   (`results/speculative/consolidated/accept_rate_histogram.png`). The
   low-acceptance tail overlaps the degraded-sample regime. A finer
   early/mid/late-step breakdown was requested but isn't producible
   from data persisted on this instance (per-iteration accept counts
   were computed in memory but never written to disk) — flagged, not
   guessed.
4. **Suggested next step, narrowly scoped to what the data supports**:
   an adaptive-K selector that widens history specifically under high
   recent motion variance — not a threshold/gamma change, since the
   tail is selector-attributed and the speculative layer already has
   little acceptance-rate headroom left to gain from tuning.

## (d) 하영 전달 패키지 (handoff)

`handoff/` — self-contained integration guide, no repo history needed.
Quick start: `python handoff/smoke_test.py` (CPU, ~5s, no checkpoint
required, verifies interface contracts + threshold=0 equivalence).
Also zipped separately: `handoff_soyun_v1.zip` (see repo root for path
+ sha256, recorded in the file-organization commit). Contents:
`HANDOFF.md` (quick start + module map + recommended config),
`INTERFACE_SPEC.md` (selector contract, pipeline call contract, exact
AdaLoRA integration point), `example_integration.py` (custom selector
skeleton, on/off comparison, alternate-checkpoint loading),
`smoke_test.py` (standalone CPU gate check). Verified runnable from a
fresh repo copy at a different absolute path — no dependency on this
instance's paths.

## (e) 다음 확인 사항

1. **Asset restoration (blocking)**: this instance's `/root/NetLLM-
   assets/`, base model weights, and off-instance backup are all
   absent — the project's third asset loss (see
   `docs/experiment_phase/assets/
   ASSET_RECOVERY_VERIFICATION_20260809.md`). No new benchmark number
   was produced this session; everything above is a citation of the
   2026-08-02 run, which passed its own gate and is git-tracked. Next
   restoration should also persist an off-instance copy that survives
   instance teardown (the previous one did not).
2. Iteration-position accept pattern (early/mid/late step) needs a
   small harness change (persist `accepted_per_iteration` per sample)
   plus a fresh run once assets are back.
3. AdaLoRA integration: 하영/팀원 확인 필요 — run the threshold=0
   equivalence gate against the new adapter before trusting any
   accuracy number from it (see `handoff/INTERFACE_SPEC.md` §3).
4. Wu2017 generalization check was only spot-checked at 200/1,395
   samples; full-scale run would strengthen the generalization claim
   if needed for a paper-style writeup.
