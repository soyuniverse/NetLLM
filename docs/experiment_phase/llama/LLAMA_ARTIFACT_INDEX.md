# Llama asset intake artifact index

- 작업일: 2026-07-26 UTC
- 최종 진행 Gate: D
- hard stop: checkpoint/dataset completeness 실패

## Repository 산출물

### Gate A — setup failure

- `docs/experiment_phase/llama/setup/LLAMA_SETUP_FAILURE_ANALYSIS.md`
- `experiments/vp/llama_setup/offline_environment_audit.txt`
- `scripts/experiment_phase/llama/setup/setup_llama_online_download_offline_run.sh`

### Gate B — Hugging Face access

- `docs/experiment_phase/llama/setup/LLAMA_HF_ACCESS_RESULT.md`

### Gate C/D — uploaded asset audit

- `docs/experiment_phase/llama/assets/UPLOADED_ARCHIVE_AUDIT.md`
- `docs/experiment_phase/llama/assets/LLAMA_CHECKPOINT_CLASSIFICATION.md`
- `docs/experiment_phase/llama/assets/TEAM_DATASET_CLASSIFICATION.md`

### Gate I — organization

- `docs/experiment_phase/llama/LLAMA_ARTIFACT_INDEX.md`
- `docs/experiment_phase/llama/FILE_ORGANIZATION_RESULT.md`
- `manifests/llama/artifact_index.json`
- `manifests/llama/repository_file_checksums.sha256`

## External 산출물과 asset

- access check:
  `/root/NetLLM-assets/llama/access_check/config.json`
- archive manifests:
  `/root/NetLLM-assets/manifests/*_archive_manifest.json`
- checkpoint manifest:
  `/root/NetLLM-assets/manifests/checkpoint_manifest.sha256`
- dataset manifest:
  `/root/NetLLM-assets/manifests/dataset_manifest.sha256`
- canonical checkpoint:
  `/root/NetLLM-assets/checkpoints/try_llama2_7b`
- canonical team data:
  `/root/NetLLM-assets/datasets/team_data`

사용자가 업로드한 원본 ZIP:

- `/root/NetLLM/try_llama2_7b.zip`
- `/root/NetLLM/data.zip`

두 ZIP은 기존 위치에 그대로 있으며 삭제, overwrite 또는 Git staging하지 않았다.

## Pre-existing asset

다음 Llama base는 이번 작업 시작 전에 이미 존재했다.

```text
/root/NetLLM-assets/llama/base
```

Lock revision은 `01c7f73d771dfac7d292323805ebc428287df4f9`이며 local/Git checksum
manifest는 일치한다. Gate D 실패 후 다운로드를 실행하지 않았으므로 이번 작업의 generated
asset으로 분류하지 않는다.

각 artifact의 machine-readable category, checksum, size, dependency 및 status는
`manifests/llama/artifact_index.json`에 기록한다.
