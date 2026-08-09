# Repository Layout

Navigation reference for future sessions. The canonical structure below was
established by an earlier reorganization pass
(`docs/final/FILE_ORGANIZATION_RULES.md`, `FILE_ORGANIZATION_AUDIT.md`,
`FILE_REORGANIZATION_RESULT.md`) — read those first for the *rules*
(immutability of referenced paths, move policy, symlink-over-copy). This
file is the current *map*, re-audited and refreshed as of 2026-08-02.

**Audit finding (2026-08-02, re-run four times this day — asset
recovery session, results/AttentionTopK session, Selector-x-Speculative-
ablation + final-package session, then this tail-analysis +
generalization + backup session):** everything added in all four passes
already matches the canonical layout those rules define, or an
explicitly user-instructed new path (`results/speculative/`,
`tests/selectors/`, `results/final_<date>/`, `docs/experiment_phase/
analysis/`, all noted below). No file needed to move in any pass. The
one correction from the first pass (stray base-model files at the repo
root, moved to `/root/llama2-7b-base/`) still stands; nothing new like
it appeared in later passes.

```text
/root/NetLLM/                      this repository
  handoff/                         self-contained integration package for
                                    하영 (teammate) -- no repo history needed,
                                    ~30min integration target, see
                                    handoff/HANDOFF.md. HANDOFF.md (quick
                                    start + module map + recommended
                                    config), INTERFACE_SPEC.md (selector
                                    contract, pipeline call contract, exact
                                    AdaLoRA integration point),
                                    example_integration.py, smoke_test.py
                                    (standalone CPU gate check, no
                                    checkpoint needed). Also packaged as
                                    handoff_soyun_v1.zip at the repo root
                                    (gitignored like all zips, see
                                    manifests/final_run_manifest.md-style
                                    provenance note in
                                    docs/final/TEAM_REPORT_20260809.md for
                                    its sha256) -- verified runnable from a
                                    fresh repo copy at a different absolute
                                    path.
/root/llama2-7b-base/              base Llama2-7b weights (config, tokenizer,
                                    *.safetensors, LICENSE) -- NOT in the repo,
                                    matches DEFAULT_BASE_MODEL_PATH in
                                    src/netllm_litevlm/vp/checkpoint_era_runtime.py.
                                    Moved here 2026-08-02 from the repo root,
                                    where they had accidentally landed from a
                                    prior download.
/root/backup_20260802/             off-instance backup copy (results/,
                                    docs/+manifests/, the two staging
                                    zips) -- NOT in the repo, see
                                    docs/final/BACKUP_MANIFEST.md. Not a
                                    duplicate of the git history: this
                                    instance has lost checkpoint/dataset
                                    assets twice already, so this exists
                                    as a second copy pending an
                                    off-instance scp download.

docs/
  final/                           cross-phase conclusions, final manuals
    manuals/                       reinstall/rebuild manuals (root has
                                    compatibility symlinks to these)
    FINAL_RESULTS_SUMMARY.md       the research narrative end-to-end,
                                    condensing PHASE_B_REAL_RESULTS.md for
                                    presentation (goal status ->
                                    implementation -> verification ->
                                    performance -> conclusion/next-work,
                                    same structure as the 7.26 report;
                                    includes a section 6 mapping all 7
                                    figures to a presentation flow;
                                    2026-08-09: tail-analysis section
                                    gained an acceptance-mechanism
                                    addendum, existing content preserved)
    TEAM_REPORT_20260809.md        ~2-page team/선배 summary: results
                                    table, implementation (bullet style),
                                    tail-analysis key findings, handoff/
                                    package pointer, open follow-ups
    BACKUP_MANIFEST.md             off-instance backup record (this
                                    instance lost checkpoint/dataset
                                    assets twice already): file list,
                                    sizes, sha256, scp download commands
  implementation/                  implementation explanations (how the
                                    wrapper code works, not phase evidence)
  experiment_phase/                phase-specific evidence, one dir per phase
    llama/{benchmark,smoke,compatibility,recovery,environment,setup,...}/
    speculative/                   PHASE_A_DESIGN.md (block-verify design),
                                    PHASE_B_7B_SMOKE.md (random-head 7B
                                    structural smoke), PHASE_B_REAL_RESULTS.md
                                    (real-checkpoint controlled comparison,
                                    including the Selector x Speculative
                                    combination ablation and its paired
                                    per-sample statistics -- the full
                                    version of FINAL_RESULTS_SUMMARY.md)
    assets/                        ASSET_RECOVERY_VERIFICATION.md: RUNBOOK
                                    absence, zip-structure mismatches found
                                    and resolved, strict-load + dataset +
                                    sample-for-sample MAE re-verification.
                                    ASSET_RECOVERY_VERIFICATION_20260809.md:
                                    a THIRD asset loss -- unlike the two
                                    prior recoveries, no assets were found
                                    anywhere on this instance (not even the
                                    staging zips). Gate FAILED; documents
                                    what was checked and the decision to
                                    proceed with GPU/checkpoint-independent
                                    work only. See "What changed 2026-08-09"
                                    below.
    analysis/                      TAIL_ANALYSIS.md: which samples degrade
                                    under the headline combined config and
                                    why (high-motion-variance regime,
                                    attributable to RecentK-2 selection,
                                    not speculative decoding -- Spearman
                                    correlations + Mann-Whitney test).
                                    2026-08-09 addendum: full-population
                                    acceptance-rate distribution (narrow,
                                    near-ceiling, mean 6.22/8) + why an
                                    iteration-position accept breakdown
                                    isn't producible from data persisted on
                                    this instance, + a scoped "next work"
                                    paragraph (adaptive-K, not
                                    adaptive-threshold).
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
                                    test_block_verify.py (gate tests,
                                    including Selector x
                                    SpeculativeBlockVerifyPipeline
                                    compatibility: draft-velocity
                                    selector-independence, and
                                    threshold=0 exactness parametrized
                                    over RecentKSelector(k) for k in
                                    {4,6,10})
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
                                    --dataset-path resolve for real;
                                    --selector "none"/"identity"/"recent_k:K"
                                    wraps both the baseline and speculative
                                    pipelines with the same selector
                                    instance, for the combination ablation;
                                    --dry-run still available for machinery
                                    self-tests), consolidate_and_plot_results.py
                                    (threshold/MAE/tradeoff figures),
                                    paired_stats_and_cdf.py (per-sample
                                    paired diffs + CDF across baseline/
                                    RecentK-2/speculative/combined),
                                    plot_ablation_bars.py (MAE+latency bars
                                    for configs A-D'), build_final_table.py
                                    (merges everything into one final table),
                                    tail_analysis.py (per-sample degradation
                                    vs. history motion speed/acceleration/
                                    accept rate, Spearman + Mann-Whitney),
                                    wu2017_generalization_spotcheck.py
                                    (200-sample distribution-shift check),
                                    accept_rate_distribution.py (2026-08-09:
                                    full-population accept-rate histogram,
                                    reads only the existing per-sample CSV --
                                    no GPU/checkpoint/dataset needed)
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
  wu2017_generalization_spotcheck/ 200-sample distribution-shift check,
                                    real checkpoint (fine-tuned on Jin2022)
  llama_7b_speculative_smoke/      random-head 7B structural smoke JSON
  llama_speculative_smoke/         prior draft-and-verify prototype's result
  llama_benchmark/, llama_*        earlier phase runtime outputs, stable

results/speculative/<timestamp>/   run_speculative_benchmark.py output
                                    (results.csv, per_sample_<config>.csv,
                                    summary.json, summary.md) -- distinct
                                    from experiments/ because it's the
                                    harness's own defined output contract.
                                    Includes full-1,698-sample runs
                                    (baseline reproduction, 4 selected
                                    speculative configs, and the Selector x
                                    Speculative ablation: B/D/D') alongside
                                    the 50-sample smoke grids used to select
                                    thresholds/gamma.
  consolidated/                    merged tables + all 7 figures: the 3
                                    threshold/MAE/tradeoff figures, mae_cdf.png,
                                    ablation_bars.png, tail_velocity_vs_diff.png,
                                    tail_acceptrate_vs_diff.png,
                                    final_table.{csv,md},
                                    paired_stats_combined_vs_baseline.json,
                                    tail_analysis_stats.json,
                                    accept_rate_histogram.png +
                                    accept_rate_distribution_stats.json
                                    (2026-08-09 addendum, 8th figure)
results/final_<date>/              copies (not moves) of the final table +
                                    5 figures for handoff -- its own
                                    README.md says explicitly that the
                                    results/speculative/ originals are
                                    authoritative if they ever diverge

manifests/{llama,final}/           inventories and SHA-256 checksums for the
                                    prior reorganization pass
manifests/final_run_manifest.md    commit hash, package versions, GPU
                                    model, asset paths + verification,
                                    seed, and the exact command for every
                                    run behind FINAL_RESULTS_SUMMARY.md

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

## What changed 2026-08-02 (Selector x Speculative ablation + final
package, evening) — the headline result

Extended `run_speculative_benchmark.py` with `--selector` so the same
selector instance can wrap both the baseline and speculative pipelines,
enabling the combination ablation (`results/speculative/20260802T101802Z/`):
RecentK-2 alone, and RecentK-2 + speculative decoding at two thresholds.
**Config D (RecentK-2 + speculative) is the first configuration in this
project to achieve an accuracy improvement AND a latency reduction
simultaneously**, both measured at full 1,698-sample scale against the
real checkpoint. Paired per-sample analysis
(`paired_stats_and_cdf.py`) decomposes this: the accuracy shift is
attributable almost entirely to RecentK-2 selection, not speculative
decoding, which composes additively on top of it (`PHASE_B_REAL_RESULTS.md`
§2 has the full breakdown, including a real, not-uniformly-positive
per-sample story — 47% of individual samples are slightly worse under
the combined config despite the aggregate mean/median improving).

Added `docs/final/FINAL_RESULTS_SUMMARY.md` (research narrative),
`manifests/final_run_manifest.md` (reproducibility record), and
`results/final_20260802/` (copies for handoff) as the session's closing
package.

## What changed 2026-08-02 (tail analysis + generalization + backup,
night) — closing the last scientific gaps and de-risking the instance

Answered the open question from the previous pass ("why do 47% of
samples degrade under config D"): `docs/experiment_phase/analysis/
TAIL_ANALYSIS.md` finds it's a high-motion-variance regime (top-5%-worst
samples have 2.16x the history motion speed of the rest, p=3.0e-24),
not a simple "high motion is bad" effect (population-wide correlation is
actually negative), and attributes it 100% to RecentK-2 selection in the
tail, with no exceptions. Ran a 200-sample generalization spot-check on
Wu2017 (found already present from the earlier data.zip extraction,
unseen during this checkpoint's Jin2022 fine-tuning) confirming both the
accuracy improvement and latency reduction hold under distribution
shift.

Given this instance has lost its checkpoint/dataset assets twice
already, packaged an off-instance backup (`/root/backup_20260802/`,
outside the repo, `docs/final/BACKUP_MANIFEST.md`) of results/,
docs/+manifests/, and the two staging zips, with sha256 checksums for
every file. Added `*.tar.gz`/`*.tar`/`*.zip` to `.gitignore` as a
defense-in-depth safety net (the backup directory was already outside
the repo, so this doesn't change what gets committed, but closes a real
gap for the repo tree generally).

## What changed 2026-08-09 (third asset loss + handoff package + tail
addendum + team report)

**Third asset loss on this instance**: `/root/NetLLM-assets/` (staging,
checkpoints), `/root/NetLLM-source/.../Jin2022`, `/root/llama2-7b-base/`,
and `/root/backup_20260802/` are all entirely absent — not merely
unverified, not present anywhere on the filesystem. The session brief
believed a backup had been restored to staging; it had not. Documented
in `docs/experiment_phase/assets/ASSET_RECOVERY_VERIFICATION_20260809.md`
rather than silently worked around. Given this, the session proceeded
with GPU/checkpoint/dataset-independent work only, citing (not
recomputing) the 2026-08-02 session's git-tracked, already-gated
results wherever a number was needed.

Added `handoff/` (new top-level directory, see the entry above) — a
self-contained integration package for 하영, built and verified without
needing any of the missing assets (`smoke_test.py` uses a tiny CPU
model, no checkpoint). Verified it runs correctly from a fresh copy of
this repo at a different absolute path, both as a plain directory and
extracted from `handoff_soyun_v1.zip`.

Extended `docs/experiment_phase/analysis/TAIL_ANALYSIS.md` with an
acceptance-mechanism addendum (full-population accept-rate histogram,
`scripts/experiment_phase/speculative/accept_rate_distribution.py` +
`results/speculative/consolidated/accept_rate_histogram.png`) — this
one required no GPU access either, since it re-reads an existing
git-tracked per-sample CSV. The requested finer iteration-position
breakdown could not be produced (raw per-iteration accept data was
computed in memory during the original run but never persisted to
disk) and is documented as blocked rather than approximated.

Added `docs/final/TEAM_REPORT_20260809.md` (team/선배-facing summary)
and updated `docs/final/FINAL_RESULTS_SUMMARY.md`'s tail-analysis
section with the acceptance-mechanism addendum, preserving all existing
content.
