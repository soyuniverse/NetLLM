# Phase B — Real 7B Integration Smoke

**Scope.** This uses the base Llama2-7b weights + `peft_model(base,
"llama", 32)` (PEFT's own default LoRA init — `B=0`, zero adaptation
effect until trained) + a freshly random-initialized
`SimpleLinearTaskHead`, **not** the fine-tuned VP checkpoint (still
unavailable in this instance — see
`docs/experiment_phase/speculative/PHASE_A_DESIGN.md` §4). **Prediction
quality (MAE, RMSE, etc.) is meaningless here and is not reported.** This
smoke only checks (a) whether block verification's control-flow behaves
the same at real 7B scale as it did in the tiny-synthetic-model gate
tests committed in `tests/speculative/test_block_verify.py`, and (b)
resource usage. No speedup claim is made anywhere below — wall-clock
numbers are recorded for reference only.

Script: `scripts/experiment_phase/speculative/run_llama_7b_speculative_smoke.py`.
Raw output: `experiments/vp/llama_7b_speculative_smoke/smoke_result.json`.
Hardware: single RTX 4090 (48GB), fp16, `gamma=4` unless swept.
Input: 5 synthetic viewport trajectories (`his_window=10`) with
realistic-scale roll/pitch/yaw motion — not real Jin2022 data (also
unavailable), just a plausible input scale (roll, yaw in degrees up to
tens; pitch in degrees up to tens), normalized the same way the real
pipeline normalizes Jin2022 (`/180, /90, /180`).

## (a) Threshold=0 equivalence gate

| sample | max abs diff (normalized coords) | target forward count |
|---|---|---|
| 0 | 0.0009766 | 20 |
| 1 | 0.0009766 | 20 |
| 2 | 0.0009766 | 20 |
| 3 | 0.0012207 | 20 |
| 4 | 0.0009766 | 20 |

**Result: PASS.** Forward count matches baseline exactly (20/20) on every
sample; `draft_forward_count == 0` throughout.

**atol finding, updated from the tiny-model diagnostic.** The tiny
synthetic model in `test_block_verify.py` measured chained-KV-cache
floating-point reassociation noise (see
`docs/experiment_phase/speculative/PHASE_A_DESIGN.md` §3) at ~1e-3 in
fp16. The real 7B model's observed max is **0.00122 (~1.2e-3)** —
consistent with that tiny-model estimate, at the same order of magnitude.
This confirms the noise floor is a property of chained fp16 KV-cache
reuse in general, not an artifact of the tiny test model's small hidden
size. **Updated fp16 atol guidance: 2e-3 remains an appropriate gate
threshold** (the committed tiny-model GPU test already used this value;
no change needed there, but this is now confirmed at the scale that
matters).

## (b) Large-threshold forward-count reduction gate

Threshold `3.0` (normalized-coordinate L2; chosen the same way as the
tiny-model test — the task head ends in `Tanh`, bounding normalized
output to `[-1,1]^3`, and this synthetic trajectory's velocity keeps the
draft within a few units of that, so `3.0` was expected to force
acceptance regardless of the untrained random head's specific weights).

| sample | target forward count | accepted per iteration | finite |
|---|---|---|---|
| 0-4 (identical) | 6 (vs. baseline 20) | `[3, 4, 4, 4, 3]` | yes |

**Result: PASS.** Forward count drops from 20 to 6 on every sample; no
non-finite values observed. The per-iteration acceptance pattern
(`3,4,4,4,3` out of `gamma=4` drafts per iteration) shows genuine partial
acceptance, not a degenerate always-0/always-max pattern.

## (c) Gamma sweep — peak GPU memory and OOM

One 20-step inference each, `acceptance_threshold=3.0`, sample 0:

| gamma | target forward count | peak allocated (MiB) | peak reserved (MiB) | OOM |
|---|---|---|---|---|
| 2 | 11 | 13005.08 | 13042.00 | no |
| 4 | 6 | 12999.11 | 13042.00 | no |
| 8 | 4 | 12998.43 | 13042.00 | no |

Baseline (`LlamaOldSelectablePipeline`) peak for the same sample:
12996.12 / 13042.00 MiB (allocated/reserved).

**Result: no OOM at any gamma; memory is flat (~13GB) across gamma.**
Expected — at `his_window=10`/`fut_window=20` the sequence never exceeds
~30 tokens, so activation/KV-cache memory is negligible next to the 7B
weights themselves (~13GB fp16), regardless of how many draft tokens are
batched per verification forward.

## (d) Wall-clock reference (not a speedup claim)

One 20-step inference, sample 0, `gamma=4`, `acceptance_threshold=3.0`:

- baseline: 570.57 ms
- speculative: 179.53 ms (6 target forwards, vs. baseline's 20)

Recorded as a reference number only. This is a single untimed-warmup
single-sample measurement on one GPU with an untrained model and is not a
benchmark; `scripts/experiment_phase/speculative/run_speculative_benchmark.py`
is the proper harness for a real latency/accuracy benchmark once the VP
checkpoint and Jin2022 dataset are restored.

## Summary

| gate | status |
|---|---|
| threshold=0 exactness (atol=2e-3 fp16) | PASS |
| threshold=0 forward count == 20 | PASS |
| large-threshold forward count < 20 | PASS |
| gamma in {2,4,8}: no OOM | PASS |

All checks that can run without the fine-tuned checkpoint and Jin2022
dataset now pass at real 7B scale, matching the tiny-model gate tests
committed in `tests/speculative/test_block_verify.py`. Nothing here
validates prediction accuracy or a real acceptance rate on real head
motion — both require the still-missing checkpoint and dataset.
