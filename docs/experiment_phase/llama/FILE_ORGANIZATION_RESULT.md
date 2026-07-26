# Llama 작업 파일 정리 결과

- 정리 시각: 2026-07-26 UTC
- 기존 파일 내용 변경: 없음
- 기존 log/runtime/result 삭제 또는 overwrite: 없음
- 사용자 ZIP 삭제/이동: 없음

## Inventory와 분류

신규 repository 파일은 다음 canonical category에만 배치했다.

```text
docs/experiment_phase/llama/setup/
docs/experiment_phase/llama/assets/
docs/experiment_phase/llama/
scripts/experiment_phase/llama/setup/
experiments/vp/llama_setup/
manifests/llama/
```

Model, checkpoint, dataset 및 generated asset manifest는 repository 밖
`/root/NetLLM-assets`에 유지했다.

## Extraction asset 이동

안전 검사를 통과한 ZIP은 staging에 한 번 압축 해제했다. 분류 후 동일 파일의 중복을 남기지
않기 위해 다음 canonical external path로 filesystem move했다.

```text
/root/NetLLM-assets/checkpoints/try_llama2_7b
/root/NetLLM-assets/datasets/team_data
```

이동 전후 relative-path checksum manifest는 byte-for-byte 일치했다.

| asset | manifest SHA-256 |
|---|---|
| checkpoint | `44cbaaa6a174207bd98c21030200ad4244f09b5273a3ec0355ece0830519c1c6` |
| dataset | `4cbc567ebc3783102c996b46fffe965b617815a6a9e487e0a33a59aa4fa17399` |

Staging에는 extracted asset 사본이 남지 않았다. 원본 ZIP은 보존했다.

## Reference scan

- 기존 tracked script/test가 신규 staging path를 참조하지 않음
- checkpoint/data canonical path는 신규 문서와 artifact index만 참조
- 기존 Phase 3A/GPT-2 path는 변경하지 않음
- upstream source path는 변경하지 않음

## Cleanup

이번 작업이 생성한 다음 임시 파일만 제거했다.

- `/tmp/netllm_*_zip_test.txt`
- `/tmp/netllm_*_zip_list.txt`
- `/tmp/netllm_*_manifest_before.sha256`

신규 `__pycache__`, `.pyc`, 임시 PNG 또는 matplotlib cache는 repository/upstream에 남지
않도록 확인했다. 기존 사용자 파일은 삭제하지 않았다.

## 검증

- repository `git diff --check`: 통과
- setup wrapper `bash -n`: 통과
- upstream commit/status/diff: unchanged/clean
- upstream `__pycache__`: `0`
- GPT-2 environment/artifact hash: unchanged
