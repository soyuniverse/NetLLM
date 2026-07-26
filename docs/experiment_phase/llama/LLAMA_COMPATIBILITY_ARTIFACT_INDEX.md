# Llama Compatibility Artifact Index

## Gate outcome

Gate 0 through Gate 3 completed. Gate 2 classified the checkpoint as
`D. structurally-incompatible`; therefore Gate 4 through Gate 7 were not
started. Gate 8 requirements and Gate 9 organization/integrity records were
then produced.

## Repository artifacts

| Gate | Category | Artifact | Purpose |
| --- | --- | --- | --- |
| 0 | safety | `compatibility/GIT_ASSET_SAFETY_RESULT.md` | ZIP staging/exclude result |
| 1 | report | `compatibility/CHECKPOINT_FORENSIC_AUDIT.md` | adapter and non-PLM forensic summary |
| 1 | runtime | `experiments/vp/llama_compatibility/adapter_key_manifest.json` | adapter tensor metadata |
| 1 | runtime | `experiments/vp/llama_compatibility/modules_except_plm_key_manifest.json` | non-PLM tensor metadata |
| 2 | report | `compatibility/CHECKPOINT_UPSTREAM_COMPATIBILITY.md` | compatibility decision |
| 2 | runtime | `experiments/vp/llama_compatibility/compatibility_matrix.json` | module matrix |
| 3 | report | `data/TEAM_UPSTREAM_DATASET_COMPARISON.md` | cooked CSV relationship |
| 3 | runtime | `experiments/vp/llama_data_audit/dataset_comparison.json` | data comparison |
| 8 | requirements | `REMAINING_TEAM_INPUT_REQUIREMENTS.md` | unresolved team evidence |
| 9 | organization | `FILE_ORGANIZATION_RESULT_PHASE2.md` | organization and integrity result |
| 9 | manifest | `manifests/llama/compatibility_artifact_index.json` | detailed artifact metadata |
| 9 | manifest | `manifests/llama/compatibility_file_checksums.sha256` | repository-file checksums |

Paths without a repository prefix in the table are relative to
`docs/experiment_phase/llama`.

No model, checkpoint, dataset, ZIP, or environment directory was placed in the
Git repository as part of this audit.
