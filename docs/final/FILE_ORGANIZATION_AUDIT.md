# File Organization Audit

## Organization status

This was an organization-only, read-only audit of the existing repository
artifacts. No experiment, inference, benchmark, smoke test, training, or
source validation run was executed.

- Repository root: `/root/NetLLM`
- Files scanned before these required outputs: 246
- Bytes scanned before these required outputs: 3,292,336,345
- JSON files parsed: 49; errors: 0
- CSV files checked: 21; header/row-width errors: 0
- Existing PNG files: 7
- Existing `.pyc` or `__pycache__`: 0
- Matplotlib cache directories: 0
- Files moved: 0
- Existing files modified or overwritten: 0
- Existing runtime, log, CSV, JSON, PNG, or Markdown removed: 0

## What was scanned

The audit covered the project root, `docs`, `src`, `scripts`, `tests`,
`experiments`, and `manifests`, plus the Git index and local exclude file.
It checked:

- duplicate source/script/test/document basenames;
- report and runtime locations against the requested canonical structure;
- JSON parsing and CSV header/row-width consistency;
- PNG references from Markdown;
- `.pyc`, `__pycache__`, temporary files, and Matplotlib caches;
- `/workspace` and legacy `/venv/` path literals in Python and shell files;
- staged or tracked ZIP/model/checkpoint/dataset files;
- large files in the repository and large files under Git control;
- references to apparent move candidates.

The external asset tree and both upstream source trees were not scanned for
organization or modified.

## Duplicate basenames

Three basename groups were found:

1. `__init__.py` in the selector, evaluation, and speculative packages.
2. `base.py` in the selector and speculative packages.
3. `setup_netllm_llama.sh` at the repository root and under `scripts/`.

The first two are intentional Python package modules. The setup scripts have
different SHA-256 values and the root path is referenced by the reinstall
manual, setup audit, runtime audit, artifact manifest, and online-download
wrapper. Neither copy was moved.

## Files moved

None.

No candidate satisfied all of the required conditions of being clearly
misplaced, newly generated, unreferenced, and safe to move without editing
existing content.

## Files intentionally not moved

The following were left unchanged because they are staged, referenced, part of
preserved historical phase output, or explicitly protected:

- `setup_netllm_llama.sh`
- `scripts/setup_netllm_llama.sh`
- `scripts/setup_netllm_llama.sh.empty`
- `NETLLM_LLAMA_재설치_매뉴얼.md`
- the 14 reports directly under `docs/experiment_phase/llama/`
- historical runtime directories:
  - `experiments/vp/llama_compatibility/`
  - `experiments/vp/llama_data_audit/`
  - `experiments/vp/llama_environment_v2/`
  - `experiments/vp/llama_setup/`
  - `experiments/vp/llama_source_recovery/`
  - `experiments/vp/llama_strict_load/`
  - `experiments/vp/llama_vp_technical_smoke/`
- `data.zip` and `try_llama2_7b.zip`

The Llama reports and historical runtimes are referenced by existing
documents, scripts, or `manifests/llama/*`. Moving them would break recorded
paths, and content edits were prohibited.

## Cache and temporary files

No `.pyc`, `__pycache__`, or Matplotlib cache existed, so none was removed.

`scripts/setup_netllm_llama.sh.empty` is a zero-byte, temp-like file, but it is
already staged and therefore treated as user state. It was not deleted or
moved.

## Format and figure-reference findings

All JSON and CSV files passed structural checks.

Two PNG basenames referenced by
`docs/experiment_phase/phase3a/PHASE3A_FINAL_RESULT.md` do not exist anywhere
in the repository:

- `keep_ratio_vs_rmse.png`
- `accuracy_latency_tradeoff.png`

The five other referenced Phase 3A figure basenames have matching files. No
figure was generated because experiment and visualization execution was
prohibited.

## Legacy path literals

There are 64 `/workspace` or legacy `/venv/` matches in 19 Python/shell files.
They are concentrated in preserved Phase 1–3 scripts and one retry test. No
Phase 4 Llama benchmark or speculative source was changed.

These paths were reported only. Correcting them would require content edits
and could change preserved reproduction scripts.

## Large files

Large untracked, locally excluded files at the project root:

| File | Size (bytes) | Git state |
|---|---:|---|
| `data.zip` | 3,199,081,523 | ignored, untracked, unstaged |
| `try_llama2_7b.zip` | 77,861,701 | ignored, untracked, unstaged |

No tracked or staged repository file is 10 MiB or larger. The tracked/staged
files at least 1 MiB are two preserved Phase 1 regression-detail CSV files
(2,307,776 bytes each) and `docs/liteVLM.pdf` (1,064,097 bytes).

## Git staged-risk check

No ZIP, model weight, checkpoint, adapter, or dataset file is staged.
`data.zip` and `try_llama2_7b.zip` already have exact local entries in
`.git/info/exclude`; the exclude file did not need modification.

The following pre-existing non-asset files remain staged and were not changed:

- `NETLLM_LLAMA_재설치_매뉴얼.md`
- `setup_netllm_llama.sh`
- `scripts/setup_netllm_llama.sh`
- `scripts/setup_netllm_llama.sh.empty`

## Final recommended repository layout

New work should use:

```text
docs/final/
docs/implementation/
docs/experiment_phase/llama/{benchmark,smoke,compatibility,recovery}/
docs/experiment_phase/speculative/
src/netllm_litevlm/{selectors,vp,evaluation,speculative}/
scripts/experiment_phase/llama/{benchmark,setup,smoke}/
scripts/experiment_phase/speculative/
tests/{llama_benchmark,phase3a,speculative}/
experiments/vp/{recovery,llama_selector_equivalence,llama_benchmark,llama_speculative_smoke}/
manifests/{llama,final}/
```

Existing referenced historical paths should remain stable until a separate
reference-migration task is explicitly authorized.

## Remaining manual actions

1. Review the four staged setup/manual files before committing, especially the
   zero-byte `scripts/setup_netllm_llama.sh.empty`.
2. Decide whether the two missing Phase 3A figures are planned-only artifacts
   or should be generated in a later experiment-authorized task.
3. If root setup/manual files should move, first authorize a reference-aware
   migration that can update the existing documents and manifests together.
4. Decide whether the 19 legacy-path scripts should remain as historical
   records or receive a separately reviewed path-portability update.
5. Keep the two root ZIP files unstaged; move them externally only through a
   separately authorized asset-management task.

