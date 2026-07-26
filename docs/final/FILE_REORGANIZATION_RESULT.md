# File Reorganization Result

## Outcome

Six root or top-level setup/manual regular files were moved into canonical
locations. Their contents were not edited. SHA-256 matched before and after
every move.

Existing path references were preserved with relative compatibility symlinks.
No model, checkpoint, dataset, ZIP, environment, runtime result, upstream
source tree, benchmark value, or code logic was changed.

## Canonical moves

| Previous path | Canonical regular-file path | SHA-256 preserved |
|---|---|---|
| `NETLLM_LLAMA_재설치_매뉴얼.md` | `docs/final/manuals/NETLLM_LLAMA_재설치_매뉴얼.md` | yes |
| `netllm_rebuild_manual.md` | `docs/final/manuals/netllm_rebuild_manual.md` | yes |
| `setup_netllm_llama.sh` | `scripts/experiment_phase/llama/setup/setup_netllm_llama.sh` | yes |
| `scripts/setup_netllm_llama.sh` | `scripts/experiment_phase/llama/setup/legacy/setup_netllm_llama_uploaded_copy.sh` | yes |
| `scripts/setup_netllm_llama.sh.empty` | `scripts/experiment_phase/llama/setup/legacy/setup_netllm_llama.sh.empty` | yes |
| `setup_netllm_repro_master.sh` | `scripts/experiment_phase/llama/setup/setup_netllm_repro_master.sh` | yes |

The two non-identical Llama setup scripts were preserved separately. The
zero-byte staged variant was preserved under `legacy/` rather than deleted.

## Compatibility

The six previous paths are now symlinks to their canonical regular files.
This preserves existing manuals, wrappers, manifests, and user commands
without editing their contents.

## Intentionally stable paths

Referenced Phase 0–4 reports, runtime outputs, figures, source files, and
manifests were not moved. External ZIP files and both upstream trees were not
touched.

## Verification records

- Before: `manifests/final/file_move_checksums_before.sha256`
- After: `manifests/final/file_move_checksums_after.sha256`
- Mapping: `manifests/final/file_reorganization_manifest.json`

