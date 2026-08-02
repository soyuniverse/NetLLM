# Provenance

## Source

- Repository: `https://github.com/duowuyms/NetLLM.git`
- Commit: `ee4d8726898610e4ae7df08bdd26728cafb4701f` (2024-07-30 23:01:55 +08:00)
- This is the exact checkpoint-era commit identified in
  `docs/experiment_phase/llama/recovery/TASK_HEAD_SOURCE_ARCHAEOLOGY.md`
  (the published `try_llama2_7b` checkpoint's own README names this commit
  and its `run_old.py` entry point).
- Fetched: 2026-08-02, via `raw.githubusercontent.com` at the pinned commit
  SHA (not a branch/tag — content cannot drift under us).

## Files vendored (and why)

Only the files needed to assemble and run the checkpoint-era Llama VP path
directly (`LlamaTaskHeadModel2` + `SimpleLinearTaskHead` +
`EmbeddingForViewportPrediction` + `peft_model`), matching the exact
call pattern already proven in this repo's
`scripts/experiment_phase/speculative/run_llama_continuous_speculative_smoke.py`
and cross-checked against `run_old.py`'s own assembly order (see below):

- `viewport_prediction/config.py` — `cfg`, imported at module load time by
  `models/old/pipeline.py` (only actually read if `using_multimodal=True`,
  which this project never sets).
- `viewport_prediction/models/old/llama.py` — `LlamaTaskHeadModel2`.
- `viewport_prediction/models/old/networking_head.py` — `SimpleLinearTaskHead`.
- `viewport_prediction/models/old/pipeline.py` — `EmbeddingForViewportPrediction`
  / `EmbeddingModelViewportPrediction`.
- `viewport_prediction/models/low_rank.py` — `peft_model` (LoRA wrapping
  helper).
- `viewport_prediction/utils/normalize.py` — `normalize_data`/`denormalize_data`.
- `viewport_prediction/dataset/load_dataset.py` — `create_dataset`, used by
  `scripts/experiment_phase/speculative/run_speculative_benchmark.py` to
  load real Jin2022 test samples once the dataset is restored (its own
  `--dry-run` mode does not use this file).
- `viewport_prediction/run_old.py` — kept for reference only (not imported
  anywhere in this repo); its lines ~254-292 are the canonical assembly
  order this project's scripts follow: `load_plm` (this project uses
  `LlamaTaskHeadModel2.from_pretrained` directly instead — same effect for
  the llama path) → `peft_model(plm, plm_type, rank)` →
  `SimpleLinearTaskHead(input_dim=hidden_size, output_dim=3,
  fut_window=...)` → `plm.set_task_head(task_head)` →
  `EmbeddingForViewportPrediction(plm, ...)`.
- `viewport_prediction/README.md` — kept for reference (names the
  checkpoint and the `run_old.py` invocation).

**Deliberately not vendored:** `models/old/plms_utils.py` (a generic
`load_plm` dispatcher spanning gpt2/mistral/opt/llama — pulling it in
would require vendoring `models/old/gpt2.py`, `mistral.py`, `opt.py` too,
none of which this project's Llama-only path needs) and everything else in
the upstream `viewport_prediction/` tree (other model backends, the
training loop, the bundled Jin2022 CSV data, notebooks, etc.) — out of
scope for what this project's wrapper code calls.

## Non-modification principle

Every file under `viewport_prediction/` in this directory is a verbatim
fetch (verified importable and byte-identical to the fetched blobs — see
git history of this directory for the untouched initial commit). **Do not
edit these files.** They exist only so
`sys.path.insert(0, "third_party/netllm_upstream/viewport_prediction")`
lets this project's own wrapper code (`src/netllm_litevlm/...`, never
modified either) import the real checkpoint-era classes instead of a
synthetic stand-in. Any behavior change belongs in a new wrapper file
under `src/netllm_litevlm/`, per this project's absolute rule that the
original NetLLM source is never modified in place.
