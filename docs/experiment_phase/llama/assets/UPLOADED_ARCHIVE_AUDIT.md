# 업로드 archive audit

- audit 시각: 2026-07-26 UTC
- Gate C 결과: 성공
- 검사 방식: read-only `sha256sum`, `unzip -t`, `unzip -l`, ZIP central-directory 검사
- 압축 해제: Gate C 중에는 하지 않음

## try_llama2_7b.zip

| 항목 | 값 |
|---|---|
| absolute path | `/root/NetLLM/try_llama2_7b.zip` |
| archive size | `77,861,701 bytes` |
| SHA-256 | `57062c71a3e103ae610ccbc499feee22dc46d25e32b8179cac20d6d2e32dec53` |
| ZIP integrity | pass |
| entries / files / directories | `6 / 4 / 2` |
| uncompressed size | `84,061,409 bytes` |
| top-level | `try_llama2_7b` |
| absolute/`../` path | 없음 |
| symlink entry | 없음 |

대표 파일:

```text
try_llama2_7b/try_llama2_7b/adapter_config.json
try_llama2_7b/try_llama2_7b/adapter_model.bin
try_llama2_7b/try_llama2_7b/modules_except_plm.bin
try_llama2_7b/try_llama2_7b/README.md
```

LoRA checkpoint 후보 구조다. Completeness는 Gate D에서 내용 분류 후 판정한다.

## data.zip

| 항목 | 값 |
|---|---|
| absolute path | `/root/NetLLM/data.zip` |
| archive size | `3,199,081,523 bytes` |
| SHA-256 | `9c3b700524b63082ab8e85fba72a24d34c81c5b9f782f5a93efd204716476e8d` |
| ZIP integrity | pass |
| entries / files / directories | `49,135 / 48,957 / 178` |
| uncompressed size | `3,456,022,732 bytes` |
| top-level | `data` |
| absolute/`../` path | 없음 |
| symlink entry | 없음 |

첫 listing에는 기존 GPT-2 fine-tuned 결과와 viewport CSV가 보인다. Processed
`Jin2022images/saliencyMap` 및 `features` 존재 여부는 Gate D에서 전체 tree를 분류한다.

## 안전 판정

두 archive 모두 CRC/integrity와 path traversal 검사를 통과했다. 원본 ZIP은 사용자 업로드
파일로 그대로 보존하며 수정·삭제·덮어쓰지 않는다.
