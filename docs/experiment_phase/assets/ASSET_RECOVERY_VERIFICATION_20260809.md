# Asset Recovery Verification — 2026-08-09 (GATE: FAILED)

This document records a Task 0 integrity-gate attempt for a new instance
started 2026-08-09. Unlike the 2026-08-02 recovery
(`ASSET_RECOVERY_VERIFICATION.md`), **no assets were found to verify.**
This is the project's third asset loss (see `claude.md` for the first,
`ASSET_RECOVERY_VERIFICATION.md` for the second).

## What the task instructions assumed

The session brief stated the backup had been "restored to
`/root/NetLLM-assets/staging/`" and asked for the standard three-step
re-verification (extraction listing, checkpoint strict-load, 50-sample
baseline MAE against the 11.0368 reference).

## What was actually found

Every non-git-tracked asset path this project depends on was checked
directly and is absent on this instance:

| path | expected content | status |
|---|---|---:|
| `/root/NetLLM-assets/` | staging zips, checkpoints/ | **directory does not exist** |
| `/root/NetLLM-assets/staging/{try_llama2_7b.zip,data.zip}` | backup zips | **absent** |
| `/root/NetLLM-assets/checkpoints/try_llama2_7b/` | LoRA adapter + modules_except_plm.bin | **absent** |
| `/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022/` | dataset | **absent** |
| `/root/llama2-7b-base/` | base Llama2-7b weights | **absent** |
| `/root/backup_20260802/` | off-instance backup copy referenced in `docs/final/BACKUP_MANIFEST.md` | **absent** |

A filesystem-wide search (`find / -iname "try_llama2_7b*"`, `find /
-iname "Jin2022"`) found no trace of any of these outside two unrelated
stub directories under `experiments/vp/phase1*_runtime/results/
regression/Jin2022` (pre-existing git-tracked regression-test fixtures,
not the real dataset). `/root/` contains only this repository
(`/root/NetLLM`) and standard shell/tooling dotfiles — no other project
directory exists on the instance at all.

**Conclusion: the gate cannot be run.** There is no checkpoint to
strict-load, no dataset to split-count, and no baseline to reproduce.
This is not a verification failure (mismatched keys, wrong sample
count) — it is a total absence of the input the gate operates on.

## Decision (user, this session)

Informed of this finding, the user chose to proceed with all
GPU/checkpoint/dataset-independent work this session (the handoff
package, referencing already-git-committed and previously-gated
analysis results, documentation, and file organization) while treating
every number that would require a fresh GPU run as blocked pending
actual asset restoration. No new benchmark, accuracy, or latency figure
is produced or claimed in this session's outputs; every number cited
from `docs/experiment_phase/analysis/TAIL_ANALYSIS.md`,
`results/speculative/`, and `docs/final/FINAL_RESULTS_SUMMARY.md` is a
citation of the 2026-08-02 session's results, which passed their own
Task 0 gate at the time (see `ASSET_RECOVERY_VERIFICATION.md`) and are
git-tracked (unaffected by this instance's loss of untracked assets).

One concrete consequence: Task 2 item 5's "iteration-position accept
pattern" (early/mid/late-step draft acceptance) requires per-sample
per-iteration accept lists that were computed in memory during the
2026-08-02 run (`SpeculativeBlockVerifyPipeline.accepted_per_iteration`,
`src/netllm_litevlm/speculative/block_verify.py:98,201,222`) but never
persisted to disk — `run_speculative_benchmark.py` only writes the
per-sample *sum* (`accepted_sum`) to CSV
(`scripts/experiment_phase/speculative/run_speculative_benchmark.py:206,365`).
This specific sub-analysis cannot be produced without a fresh
instrumented run against the real checkpoint and dataset, and is
recorded as blocked in `docs/experiment_phase/analysis/TAIL_ANALYSIS.md`
rather than fabricated or silently omitted.

## Recommendation for next asset restoration

Whoever re-uploads `try_llama2_7b.zip`/`data.zip` next should also
persist an off-instance copy somewhere that survives instance teardown
(the previous `/root/backup_20260802/` did not — it was itself lost),
and should confirm the upload landed on the *current* instance's
filesystem, not a previous one, before any session trusts it.
