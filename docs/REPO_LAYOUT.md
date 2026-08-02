# Repository Layout

Navigation reference for future sessions. The canonical structure below was
established by an earlier reorganization pass
(`docs/final/FILE_ORGANIZATION_RULES.md`, `FILE_ORGANIZATION_AUDIT.md`,
`FILE_REORGANIZATION_RESULT.md`) — read those first for the *rules*
(immutability of referenced paths, move policy, symlink-over-copy). This
file is the current *map*, re-audited and refreshed as of 2026-08-02.

**Audit finding (2026-08-02, re-run twice this day — asset recovery
session then results/AttentionTopK session):** everything added in both
passes already matches the canonical layout those rules define, or an
explicitly user-instructed new path (`results/speculative/`,
`tests/selectors/`, both noted below). No file needed to move either
time. The one correction from the first pass (stray base-model files at
the repo root, moved to `/root/llama2-7b-base/`) still stands; nothing
new like it appeared in the second pass.

```text
/root/NetLLM/                      this repository
/root/llama2-7b-base/              base Llama2-7b weights (config, tokenizer,
                                    *.safetensors, LICENSE) -- NOT in the repo,
                                    matches DEFAULT_BASE_MODEL_PATH in
                                    src/netllm_litevlm/vp/checkpoint_era_runtime.py.
                                    Moved here 2026-08-02 from the repo root,
                                    where they had accidentally landed from a
                                    prior download.

docs/
  final/                           cross-phase conclusions, final manuals
    manuals/                       reinstall/rebuild manuals (root has
                                    compatibility symlinks to these)
  implementation/                  implementation explanations (how the
                                    wrapper code works, not phase evidence)
  experiment_phase/                phase-specific evidence, one dir per phase
    llama/{benchmark,smoke,compatibility,recovery,environment,setup,...}/
    speculative/                   PHASE_A_DESIGN.md (block-verify design),
                                    PHASE_B_7B_SMOKE.md (random-head 7B
                                    structural smoke), PHASE_B_REAL_RESULTS.md
                                    (real-checkpoint controlled comparison --
                                    the headline deliverable)
    assets/                        ASSET_RECOVERY_VERIFICATION.md: RUNBOOK
                                    absence, zip-structure mismatches found
                                    and resolved, strict-load + dataset +
                                    sample-for-sample MAE re-verification
    phase0/ .. phase3a/, recovery/, resume/    earlier phases, stable
  REPO_LAYOUT.md                   this file
  MEETING_NOTES.md, RESEARCH_DIRECTION.md, *.pdf   project-level references

src/netllm_litevlm/                the only package; nothing here wraps or
                                    edits the original NetLLM/upstream source
                                    in place -- new behavior is always a new
                                    file
  selectors/                       IdentitySelector, RecentKSelector, base,
                                    attention_topk.py (AttentionTopKSelector:
                                    top-K by single-decoder-layer attention,
                                    drop-in BaseSelector/SelectionOutput)
  evaluation/                      vp_metrics.py, runtime_benchmark.py --
                                    reused by scripts instead of
                                    reimplementing MAE/RMSE/latency math
  speculative/                     acceptance.py, base.py,
                                    continuous_draft_verify.py (prior
                                    prototype, still target-forward=20 by
                                    design -- see PHASE_A_DESIGN.md),
                                    recent_velocity_draft.py,
                                    block_verify.py (SpeculativeBlockVerify
                                    Pipeline; acceptance_threshold's unit
                                    space -- normalized, not degrees -- is
                                    documented directly on the parameter)
  vp/                               llama_old_selectable_pipeline.py
                                    (baseline wrapper), selectable_pipeline.py,
                                    checkpoint_era_runtime.py (real-model
                                    assembly helper, default base-weight path
                                    and optional real checkpoint_path)

third_party/netllm_upstream/       vendored, unmodified upstream source
                                    (PROVENANCE.md: exact commit, fetch date,
                                    non-modification principle) -- lets this
                                    project import the real LlamaTaskHeadModel2 /
                                    SimpleLinearTaskHead / EmbeddingForViewport
                                    Prediction / peft_model / create_dataset
                                    instead of a stub

tests/
  llama_benchmark/                 baseline pipeline + selector contract tests
  phase3a/                         identity-selector equivalence tests
  speculative/                     acceptance.py, RecentVelocityDraft,
                                    ContinuousDraftVerify,
                                    test_block_verify.py (gate tests)
  selectors/                       test_attention_topk.py -- CPU-only tiny
                                    real LlamaModel (GPU was occupied by a
                                    real benchmark run when written)
  benchmark/                       vp_metrics / selector benchmark tests

scripts/experiment_phase/
  llama/{benchmark,setup}/         Llama-era benchmark/setup scripts
  llama/smoke/                     run_llama_vp_technical_smoke.py,
                                    run_llama_strict_load.py,
                                    run_attention_topk_7b_smoke.py (real
                                    checkpoint, K in {8,6,4,2} vs RecentK)
  speculative/                     run_llama_continuous_speculative_smoke.py
                                    (prior prototype), run_llama_7b_speculative
                                    _smoke.py (random-head structural smoke),
                                    run_speculative_benchmark.py (the real
                                    benchmark harness -- --checkpoint-path/
                                    --dataset-path now resolve for real;
                                    --dry-run still available for machinery
                                    self-tests), consolidate_and_plot_results.py
                                    (merges runs into one table + 3 figures)
  assets/                          verify_checkpoint_strict_load.py
                                    (adapter + non-PLM strict-load re-check,
                                    independent of checkpoint_era_runtime's
                                    own internal assertion)
  benchmark/                       run_vp_benchmark.py, plot_vp_benchmark.py
  phase1/ .. phase3a/               earlier phase scripts, stable

experiments/vp/                    runtime results (never mixed with source)
  asset_recovery/                  checkpoint_strict_load.json + this
                                    session's *.log files (gitignored,
                                    redundant with the committed JSON/CSV)
  attention_topk_7b_smoke/         AttentionTopK vs RecentK, real checkpoint
  llama_7b_speculative_smoke/      random-head 7B structural smoke JSON
  llama_speculative_smoke/         prior draft-and-verify prototype's result
  llama_benchmark/, llama_*        earlier phase runtime outputs, stable

results/speculative/<timestamp>/   run_speculative_benchmark.py output
                                    (results.csv, per_sample_<config>.csv,
                                    summary.json, summary.md) -- distinct
                                    from experiments/ because it's the
                                    harness's own defined output contract.
                                    Now includes real full-1,698-sample runs
                                    (baseline reproduction, 4 selected
                                    speculative configs) alongside the
                                    50-sample smoke grids used to select them.
  consolidated/                    consolidate_and_plot_results.py's merged
                                    table + 3 figures across all of the above

manifests/{llama,final}/           inventories and SHA-256 checksums for the
                                    prior reorganization pass

patches/experiment_phase/          proposed-but-not-applied diffs (e.g. the
                                    accelerate/huggingface-hub dependency fix)

configs/                           benchmark config JSON (vp_benchmark.json)

Root-level compatibility symlinks (see FILE_REORGANIZATION_RESULT.md for
why these exist instead of the regular files living at the root):
  NETLLM_LLAMA_재설치_매뉴얼.md -> docs/final/manuals/...
  netllm_rebuild_manual.md      -> docs/final/manuals/...
  setup_netllm_llama.sh         -> scripts/experiment_phase/llama/setup/...
  setup_netllm_repro_master.sh  -> scripts/experiment_phase/llama/setup/...

Root-level regular files:
  README.md, claude.md           project overview / session context
  이거따라해.md                    plain-language Vast.ai instance recovery
                                  guide (exactly the scenario this session
                                  started from)
  setup.sh                       environment setup entry point
  requirements-vp.txt            pinned VP dependency versions
  constraints-vp-plm.txt         pip constraints file
```

## What changed 2026-08-02 (asset-recovery pass, morning)

- Moved stray base-model files (`config.json`, `generation_config.json`,
  `*.safetensors`, tokenizer files, `LICENSE.txt`) from the repo root to
  `/root/llama2-7b-base/` — they were never git-tracked (`.gitignore`
  already excluded `*.safetensors`/`*.bin`), just sitting in the working
  tree from a prior download that landed in the wrong place.
- Added `third_party/netllm_upstream/`, `results/speculative/`, and the
  files then listed under `speculative/` and `vp/` above.

## What changed 2026-08-02 (real-checkpoint pass, afternoon) — the big one

**The VP fine-tuned checkpoint and Jin2022 dataset are no longer
missing.** They were re-uploaded to `NetLLM-assets/staging/` and placed
at `/root/NetLLM-assets/checkpoints/try_llama2_7b/` and
`/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022/` (see
`docs/experiment_phase/assets/ASSET_RECOVERY_VERIFICATION.md` for the
placement/verification details, including two pre-extraction
discrepancies that were reported and resolved rather than silently
patched over). Every speculative-decoding and selector result up to this
point in the project was either structural-only (random head) or
degenerate (target-forward always 20). This pass produced the first real
numbers: full 1,698-sample baseline reproduces the pre-loss 7.26 report
exactly (within fp16 noise), and four selected speculative configurations
are the first to record `speedup_claim_valid=True` against the real
checkpoint at full scale. `docs/experiment_phase/speculative/
PHASE_B_REAL_RESULTS.md` is the consolidated writeup.

Also added `src/netllm_litevlm/selectors/attention_topk.py` and
`tests/selectors/` (new test subdirectory, mirrors the `src/` layout).
