# Handoff — NetLLM VP Speculative Decoding + Selector Modules

Audience: 하영 (팀원), integrating these modules into a teammate's
AdaLoRA / patch-selection code. You do not need this repo's history —
everything you need to integrate in ~30 minutes is in this `handoff/`
directory. Start here, then `INTERFACE_SPEC.md` for the exact contracts,
then `example_integration.py` for copy-pasteable code.

## What this package gives you

Two independently-composable pieces bolted onto NetLLM's Llama2-7B
Viewport Prediction (VP) pipeline:

1. **A selector interface** (`src/netllm_litevlm/selectors/`) — pluggable
   history-embedding pruning applied once before the LLM sees the
   sequence. Patch selection is a drop-in implementation of this same
   interface (see `INTERFACE_SPEC.md` §1).
2. **Block speculative decoding** (`src/netllm_litevlm/speculative/
   block_verify.py`) — draft-then-verify decoding for the 20-step
   autoregressive VP rollout, using a real KV cache instead of
   recomputing the whole sequence every step.

Both wrap the existing NetLLM pipeline without modifying it (see
`INTERFACE_SPEC.md` §3 for exactly where AdaLoRA plugs in).

## ⚠️ Environment — read before installing anything

```
transformers==4.34.1
peft==0.6.2
torch==2.2.0
```

**This transformers version predates the `Cache`/`DynamicCache` object.**
`LlamaModel` returns `past_key_values` as a legacy **tuple of
`(key, value)` tuples**, each shaped `[B, num_heads, seq_len,
head_dim]` — not the newer `Cache` class. `block_verify.py`'s KV-cache
slicing (`slice_past_key_values`) is written specifically against this
tuple format. **If your AdaLoRA/patch-selection code was built against a
newer `transformers` with the `Cache` object, do not just `pip install
-U transformers`** — the KV-cache slicing here will silently break
(wrong indexing into a different object shape) rather than raise an
import error. Pin the versions above, or port `slice_past_key_values`
to whatever cache format you upgrade to.

A second precision note: chaining many sequential KV-cache extension
hops (as the 20-step VP rollout does) exposes floating-point
reassociation noise inherent to BLAS matmul kernels — ~1e-5 abs diff on
fp32/CPU, ~2e-3 on fp16/GPU, versus a full non-cached recompute. This is
not a bug; see `INTERFACE_SPEC.md` §2 for the exact equivalence
tolerance used in this repo's own gate tests.

## Environment setup + asset paths

```bash
pip install -r requirements-vp.txt   # includes the pins above
```

| asset | expected path (this repo's convention) |
|---|---|
| base Llama2-7b weights | `/root/llama2-7b-base/` (config.json, tokenizer, `*.safetensors`) |
| fine-tuned VP checkpoint (LoRA adapter) | `<checkpoint_path>/{adapter_config.json, adapter_model.bin, modules_except_plm.bin}` |
| Jin2022 dataset | `<dataset_path>/viewports/Jin2022/video{1..27}/5Hz/simple_5Hz_user*.csv` |

Both checkpoint and dataset paths are passed explicitly as arguments
(`--checkpoint-path`, `--dataset-path` / `checkpoint_path=` in
`load_checkpoint_era_model`) — nothing is hardcoded to a single
absolute path except the base-model default
(`DEFAULT_BASE_MODEL_PATH` in `src/netllm_litevlm/vp/
checkpoint_era_runtime.py:27`), which you can override with
`--base-model-path`.

**As of 2026-08-09, none of these three assets exist on the source
instance this package was built on** — a third asset loss this project
has hit (see `docs/experiment_phase/assets/
ASSET_RECOVERY_VERIFICATION_20260809.md`). The interface contracts,
code, and recommended-config table below are all still valid (verified
against a real checkpoint in the 2026-08-02 session, git-committed and
unaffected by this loss); you just need your own copies of the three
assets above to run anything beyond `smoke_test.py`.

## Quick start

Smoke test — CPU, no checkpoint, no dataset, ~1 minute, confirms the
interface contracts and threshold=0 equivalence gate are intact:

```bash
python handoff/smoke_test.py
```

Full benchmark — GPU, real checkpoint + dataset required, sweeps
selector/threshold/gamma and writes `results/speculative/<timestamp>/`:

```bash
python scripts/experiment_phase/speculative/run_speculative_benchmark.py \
  --checkpoint-path /root/NetLLM-assets/checkpoints/try_llama2_7b \
  --dataset-path /root/NetLLM-source/viewport_prediction/data \
  --selector recent_k:2 --thresholds 0.35 --gammas 8 --num-samples 1698
```

## Module map

```
src/netllm_litevlm/
├── selectors/
│   ├── base.py            BaseSelector (abstract), SelectionOutput (dataclass)
│   ├── identity.py         IdentitySelector — passthrough, no pruning
│   ├── recent_k.py         RecentKSelector(k) — keep last k embeddings
│   └── attention_topk.py   AttentionTopKSelector — top-K by layer-1 attention
│                            (empirically loses to RecentK at every K tested;
│                            kept as a reference alternative-signal implementation)
│
├── speculative/
│   ├── base.py             ContinuousDraftModel (abstract), DraftOutput
│   ├── recent_velocity_draft.py  RecentVelocityDraft — constant-velocity
│   │                              extrapolation draft (deterministic, no LLM call)
│   └── block_verify.py     SpeculativeBlockVerifyPipeline — the real thing.
│                            Owns the KV cache, calls draft_model, verifies
│                            gamma coordinates per target forward.
│
└── vp/
    ├── checkpoint_era_runtime.py   load_checkpoint_era_model() — assembles
    │                                base model + LoRA adapter + task head.
    │                                *** AdaLoRA integration point, see
    │                                INTERFACE_SPEC.md §3 ***
    └── llama_old_selectable_pipeline.py  LlamaOldSelectablePipeline — the
                                            unmodified-behavior baseline
                                            wrapper (selector-only, no
                                            speculative decoding), used as
                                            the equivalence reference.

scripts/experiment_phase/speculative/
└── run_speculative_benchmark.py    CLI harness: constructs both pipelines
                                     with the SAME selector instance, runs
                                     them over real samples, writes CSV/JSON.
```

Dependency direction: `run_speculative_benchmark.py` depends on
`checkpoint_era_runtime.load_checkpoint_era_model()` (model assembly)
and on both `LlamaOldSelectablePipeline` and
`SpeculativeBlockVerifyPipeline` (chosen by CLI flags), each of which
depends on a `BaseSelector` instance (optional, injected) from
`selectors/`. `SpeculativeBlockVerifyPipeline` additionally depends on a
`ContinuousDraftModel` instance from `speculative/` (defaults to
`RecentVelocityDraft`). Nothing in `selectors/` or `speculative/`
depends back on the harness or on `vp/` — they are standalone,
independently testable modules.

```
run_speculative_benchmark.py
        │
        ├── checkpoint_era_runtime.load_checkpoint_era_model()
        │         (assembles base model + LoRA adapter, see §3 of
        │          INTERFACE_SPEC.md for the AdaLoRA swap point)
        │
        ├── LlamaOldSelectablePipeline(model, selector)      ─┐
        └── SpeculativeBlockVerifyPipeline(model, selector,   │  both take
              draft_model, gamma, acceptance_threshold)      ─┘  the same
                    │                        │                   BaseSelector
                    ▼                        ▼                   instance
              selectors.BaseSelector   speculative.ContinuousDraftModel
              (RecentKSelector,        (RecentVelocityDraft, or your own)
               IdentitySelector,
               AttentionTopKSelector,
               or your patch selector)
```

## Recommended default configuration

RecentK-2 selector + speculative block-verify at `threshold=0.35`,
`gamma=8` — the only configuration in this project to date that
improves accuracy AND reduces latency simultaneously versus the
unmodified baseline, measured on the full 1,698-sample Jin2022 test
split against the real fine-tuned checkpoint
(`results/speculative/consolidated/final_table.md`):

| config | MAE (deg) | latency median (ms) | avg target forwards/sample |
|---|---:|---:|---:|
| baseline (no selector, no speculative) | 12.799 | 571.7 | 20.0 |
| RecentK-2 alone | 10.847 | 622.9 | 20.0 |
| **RecentK-2 + speculative, th=0.35, γ=8 (recommended)** | **10.895** | **122.2** | **4.01** |

The accuracy gain is attributable almost entirely to RecentK-2 selection
(confirmed via paired per-sample decomposition,
`docs/experiment_phase/analysis/TAIL_ANALYSIS.md`); speculative decoding
composes additively on top and is what delivers the ~4.7x latency
reduction. If your patch-selection module replaces RecentK-2, re-run
this same benchmark — the latency reduction from speculative decoding
should hold regardless of which selector you plug in (see the
threshold=0 equivalence gate in `INTERFACE_SPEC.md` §2, which is
selector-independent by construction), but the accuracy number is
selector-specific and needs its own measurement.

## If something breaks

1. Run `python handoff/smoke_test.py` first — isolates whether the
   problem is in these modules (interface/equivalence broken) or in
   your integration (wrong shapes, wrong call order, real-model
   numerics).
2. Check `INTERFACE_SPEC.md` §2 for the threshold=0 exactness contract
   — if your integrated pipeline doesn't reproduce the baseline exactly
   at `acceptance_threshold=0`, the bug is almost always a KV-cache
   shape/format mismatch (see the environment warning above), not the
   accept/reject logic itself.
3. `docs/experiment_phase/speculative/PHASE_A_DESIGN.md` has the full
   design rationale and the empirical acceptance-threshold calibration
   (normalized-space L2 disagreement distribution) if you need to
   re-tune thresholds for a different selector or checkpoint.
