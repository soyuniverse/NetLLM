# Llama Benchmark Artifact Index

## Scope

This index covers the selector-equivalence, pilot/full benchmark, visualization, implementation-summary, and continuous VP speculative prototype artifacts created in Phase 4.

## Canonical locations

| Phase | Category | Location | Status |
|---|---|---|---|
| Gates 0–3 | contract/equivalence | `docs/experiment_phase/llama/benchmark/`, `experiments/vp/llama_selector_equivalence/` | final |
| Gates 2–5 | selector source/tests/runners | `src/netllm_litevlm/vp/`, `tests/llama_benchmark/`, `scripts/experiment_phase/llama/benchmark/` | complete |
| Gates 4–5 | pilot/full results | `experiments/vp/llama_benchmark/pilot/`, `experiments/vp/llama_benchmark/full/` | final, partial=false |
| Gate 6 | figures | `experiments/vp/llama_benchmark/figures/` | seven full-result figures |
| Gate 7 | implementation | `docs/implementation/LLAMA_VP_SELECTOR_IMPLEMENTATION_SUMMARY.md` | complete |
| Gate 8 | speculative source/tests/smoke | `src/netllm_litevlm/speculative/`, `tests/speculative/`, `scripts/experiment_phase/speculative/`, `experiments/vp/llama_speculative_smoke/` | technical smoke final |
| Gates 9–10 | readiness/organization | `docs/experiment_phase/llama/`, `manifests/llama/` | complete |

The machine-readable per-file index is `manifests/llama/benchmark_artifact_index.json`. File digests are in `manifests/llama/benchmark_file_checksums.sha256`.

## Runtime status

- Identity equivalence: success, three max absolute differences 0.0
- Pilot: success, 128 samples per configuration
- Full: success, 1,698 samples per configuration, partial=false
- Figures: 7/7 generated from the full summary
- Speculative smoke: success, control flow valid, speedup claim invalid

Generic runtime basenames such as `progress.json`, `summary.json`, and `per_sample_metrics.csv` intentionally repeat inside isolated configuration directories. No duplicate source, script, test, documentation, or figure artifact was found.

## Exclusions

External model, checkpoint, adapter, dataset, ZIP, environment, and both upstream source trees are referenced but not copied into this repository index. Existing Phase 0–3 artifacts were preserved and are not duplicated here.

