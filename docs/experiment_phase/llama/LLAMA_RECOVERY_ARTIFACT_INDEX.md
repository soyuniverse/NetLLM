# Llama Recovery Artifact Index

## Outcome

The checkpoint-era source was recovered, strict load completed, and a real
Jin2022 technical smoke passed. No external migration loader was created
because the checkpoint is natively strict-compatible with the recovered source.

## Phase index

| Gate | Category | Canonical artifacts | Status |
| --- | --- | --- | --- |
| 0 | recovery | `recovery/CHECKPOINT_RECOVERY_START_STATE.md` | complete |
| 1 | source | `recovery/TASK_HEAD_SOURCE_ARCHAEOLOGY.md`, `llama_source_recovery/source_candidates.json` | exact source found |
| 2 | compatibility | `recovery/CHECKPOINT_COMPATIBILITY_RECLASSIFICATION.md`, `reclassification_matrix.json` | A |
| 3 | migration | no source/test/runtime created | not applicable |
| 4 | provenance | `recovery/MULTIMODAL_MODE_EVIDENCE.md` | proven non-multimodal |
| 5 | environment | `environment/*_V2.md`, `llama_environment_v2/*` | pass |
| 6 | load | `smoke/LLAMA_STRICT_LOAD_RESULT.md`, `llama_strict_load/*` | pass |
| 7 | VP smoke | `smoke/LLAMA_VP_TECHNICAL_SMOKE_RESULT.md`, `llama_vp_technical_smoke/*` | final run pass |
| 8 | readiness | `LLAMA_BENCHMARK_READINESS.md` | latency only |
| 9 | requirements | `REMAINING_TEAM_INPUT_REQUIREMENTS_V2.md` | four provenance inputs |
| 10 | organization | this index, `FILE_ORGANIZATION_RESULT_PHASE3.md`, recovery manifests | complete |

Document paths are relative to `docs/experiment_phase/llama`; runtime paths are
relative to `experiments/vp`.

## Preserved Gate 7 attempts

- Root runtime: initial post-validation trace failure; output itself was finite
  with shape `[1,20,3]`.
- `diagnostics/run1_trace_hook`: confirmed the hook observed no calls because it
  was below the PEFT wrapper.
- `run2_20260726`: corrected hook, full contract pass.

The detailed per-file fields, hashes, sizes, references, and Git states are in
`manifests/llama/recovery_artifact_index.json`.
