# Team Report — NetLLM VP: Speculative Decoding + Selector Study

2026-08-09. For the team / 선배. Condenses
`docs/final/FINAL_RESULTS_SUMMARY.md` (full narrative) and
`docs/experiment_phase/analysis/TAIL_ANALYSIS.md` (tail deep-dive) to
~2 pages. Full-1,698-sample numbers below are from the 2026-08-02 run;
this instance's asset stack (checkpoint + dataset + base weights) was
independently re-verified 2026-08-09 (see (a) reliability note and
`docs/experiment_phase/assets/GATE_A_VERIFICATION.md`) and reproduces
the 2026-08-02 accuracy numbers to 8 significant figures, so the
2026-08-02 figures are cited directly below rather than rerun at full
scale.

**Reliability (Gate-A + Gate-B, both COMPLETE 2026-08-09)**: staging
zip checksums matched `BACKUP_MANIFEST.md` exactly, checkpoint/dataset
placed and structurally verified, full base+adapter strict load passed
0 missing/unexpected/mismatch, and 50-sample MAE reproduced the
reference to 8 significant figures (11.036768 → 11.036768085417648).
Full record: `docs/experiment_phase/assets/GATE_A_VERIFICATION.md`.

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
**Delivered**: `/root/NetLLM/handoff_soyun_v1.zip`, sha256
`b26cee001e9bef453c1aeea7a35d28efd9df85e6e4c6b9a625500b77aa71d099`
(includes the base-weight `hf download` procedure added after this
session hit the exact problem live — see HANDOFF.md's environment
section). Contents: `HANDOFF.md` (quick start + module map +
recommended config), `INTERFACE_SPEC.md` (selector contract, pipeline
call contract, exact AdaLoRA integration point),
`example_integration.py` (custom selector skeleton, on/off comparison,
alternate-checkpoint loading), `smoke_test.py` (standalone CPU gate
check). Verified runnable from a fresh repo copy at a different
absolute path — no dependency on this instance's paths.

## (e) 다음 확인 사항

1. **Asset restoration — RESOLVED 2026-08-09.** This instance lost its
   checkpoint/dataset/base-weight assets a third time this project
   (see `ASSET_RECOVERY_VERIFICATION_20260809.md`); staging `scp` had
   connectivity problems, so the checkpoint/dataset zips were relayed
   through Google Drive instead, and the base weights were re-pulled
   via `hf download`. Both Gate-A and Gate-B now pass (see reliability
   note above) — this instance is trustworthy for GPU work going
   forward.
2. Iteration-position accept pattern (early/mid/late step) still needs
   a small harness change (persist `accepted_per_iteration` per
   sample, not just its sum) plus a fresh run — assets are available
   now, so this is unblocked whenever someone picks it up.
3. AdaLoRA integration: 하영/팀원 확인 필요 — run the threshold=0
   equivalence gate against the new adapter before trusting any
   accuracy number from it (see `handoff/INTERFACE_SPEC.md` §3).
4. Wu2017 generalization check was only spot-checked at 200/1,395
   samples; full-scale run would strengthen the generalization claim
   if needed for a paper-style writeup.
5. Latency numbers are instance-specific (this instance runs ~16-18%
   slower in absolute terms than the 2026-08-02 instance for the same
   configs, though the relative speedup ratio was close — see
   `NEW_INSTANCE_CALIBRATION.md`). Any future latency claim should be
   measured fresh within one session, not diffed against an older
   instance's numbers.

### Disaster-recovery procedure (established this session — worth reusing)

1. **Off-instance backup manifest first** (`docs/final/
   BACKUP_MANIFEST.md`) — sha256 + size recorded for every
   checkpoint/dataset zip *before* it's needed, so a later re-upload
   has a ground truth to verify against.
2. **Whatever transfer channel actually works** — direct `scp` failed
   on connectivity this time; relaying the same zips through Google
   Drive worked. The channel doesn't matter as long as the checksum
   step below catches any corruption it introduces.
3. **Never trust a re-upload without the two-gate check** — checksum
   match (transfer integrity) is necessary but not sufficient; a
   strict-load + reference-metric reproduction (Gate-B) is what
   actually proves the assets work, not just that the bytes arrived.
