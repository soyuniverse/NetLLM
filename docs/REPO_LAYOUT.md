# Repository Layout

Navigation reference for future sessions. The canonical structure below was
established by an earlier reorganization pass
(`docs/final/FILE_ORGANIZATION_RULES.md`, `FILE_ORGANIZATION_AUDIT.md`,
`FILE_REORGANIZATION_RESULT.md`) — read those first for the *rules*
(immutability of referenced paths, move policy, symlink-over-copy). This
file is the current *map*, re-audited and refreshed as of 2026-08-02.

**Audit finding (2026-08-02):** everything added this session already
matches the canonical layout those rules define. The only correction
needed was moving stray base-model download files that had landed at the
repo root back outside the repository (rule: "Models, checkpoints,
datasets, ZIP files, and environments stay outside the repository
workflow and are never committed") — done, see below. No other file was
moved; historical Phase 0-4 paths remain exactly where prior manifests
reference them, per that policy's explicit immutability rule.

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
    speculative/                   this session's design + smoke docs:
                                    PHASE_A_DESIGN.md, PHASE_B_7B_SMOKE.md
    phase0/ .. phase3a/, recovery/, resume/    earlier phases, stable
  REPO_LAYOUT.md                   this file
  MEETING_NOTES.md, RESEARCH_DIRECTION.md, *.pdf   project-level references

src/netllm_litevlm/                the only package; nothing here wraps or
                                    edits the original NetLLM/upstream source
                                    in place -- new behavior is always a new
                                    file
  selectors/                       IdentitySelector, RecentKSelector, base
  evaluation/                      vp_metrics.py, runtime_benchmark.py --
                                    reused by scripts instead of
                                    reimplementing MAE/RMSE/latency math
  speculative/                     acceptance.py, base.py,
                                    continuous_draft_verify.py (prior
                                    prototype, still target-forward=20 by
                                    design -- see PHASE_A_DESIGN.md),
                                    recent_velocity_draft.py,
                                    block_verify.py (this session's
                                    SpeculativeBlockVerifyPipeline)
  vp/                               llama_old_selectable_pipeline.py
                                    (baseline wrapper), selectable_pipeline.py,
                                    checkpoint_era_runtime.py (this session's
                                    real-model assembly helper)

third_party/netllm_upstream/       vendored, unmodified upstream source
                                    (PROVENANCE.md: exact commit, fetch date,
                                    non-modification principle) -- added this
                                    session so the real LlamaTaskHeadModel2 /
                                    SimpleLinearTaskHead / EmbeddingForViewport
                                    Prediction / peft_model / create_dataset
                                    can be imported instead of stubbed

tests/
  llama_benchmark/                 baseline pipeline + selector contract tests
  phase3a/                         identity-selector equivalence tests
  speculative/                     acceptance.py, RecentVelocityDraft,
                                    ContinuousDraftVerify, and this session's
                                    test_block_verify.py (gate tests)
  benchmark/                       vp_metrics / selector benchmark tests

scripts/experiment_phase/
  llama/{benchmark,setup,smoke}/   Llama-era benchmark/setup/smoke scripts
  speculative/                     run_llama_continuous_speculative_smoke.py
                                    (prior prototype),
                                    run_llama_7b_speculative_smoke.py (this
                                    session's real-7B structural smoke),
                                    run_speculative_benchmark.py (this
                                    session's benchmark harness, blocked on
                                    the missing checkpoint/dataset -- see
                                    --dry-run for the machinery self-test)
  benchmark/                       run_vp_benchmark.py, plot_vp_benchmark.py
  phase1/ .. phase3a/               earlier phase scripts, stable

experiments/vp/                    runtime results (never mixed with source)
  llama_7b_speculative_smoke/      this session's 7B smoke JSON
  llama_speculative_smoke/         prior draft-and-verify prototype's result
  llama_benchmark/, llama_*        earlier phase runtime outputs, stable

results/speculative/<timestamp>/   this session's new convention for
                                    run_speculative_benchmark.py output
                                    (results.csv, summary.json, summary.md)
                                    -- distinct from experiments/ because it
                                    is the harness's own defined output
                                    contract, not a one-off smoke/dev run

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

## What changed this session (2026-08-02)

- Moved stray base-model files (`config.json`, `generation_config.json`,
  `*.safetensors`, tokenizer files, `LICENSE.txt`) from the repo root to
  `/root/llama2-7b-base/` — they were never git-tracked (`.gitignore`
  already excluded `*.safetensors`/`*.bin`), just sitting in the working
  tree from a prior download that landed in the wrong place.
- Added `third_party/netllm_upstream/`, `results/speculative/`, and the
  files listed under `speculative/` and `vp/` above. Everything else is
  unchanged.
