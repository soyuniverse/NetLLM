# Git asset safety 결과

- 확인 시각: 2026-07-26 UTC
- Gate 0 결과: 성공

## Staged 상태

작업 전:

```text
A  try_llama2_7b.zip
?? data.zip
```

`try_llama2_7b.zip`만 Git index에 staged되어 있었다. 사용자 지침에 따라 content를
변경하지 않고 `git restore --staged try_llama2_7b.zip`을 실행했다. `data.zip`은 처음부터
untracked였으므로 index 변경이 없었다.

작업 후 두 ZIP 모두 Git status에 나타나지 않도록 local exclude를 적용했다.

## Local-only exclude

`.git/info/exclude`에 중복 없이 다음 두 entry를 추가했다.

```text
try_llama2_7b.zip
data.zip
```

`.gitignore`와 tracked repository file은 수정하지 않았다.

## Repository 내부 asset 검사

Repository 내부에서 발견된 대형 asset은 사용자가 업로드한 다음 원본 ZIP뿐이다.

| 파일 | 크기 | 상태 |
|---|---:|---|
| `/root/NetLLM/try_llama2_7b.zip` | 77,861,701 bytes | preserved, ignored |
| `/root/NetLLM/data.zip` | 3,199,081,523 bytes | preserved, ignored |

Model weight, extracted checkpoint 및 extracted dataset은 repository 외부 canonical path에
있다.

```text
/root/NetLLM-assets/llama/base
/root/NetLLM-assets/checkpoints/try_llama2_7b
/root/NetLLM-assets/datasets/team_data
```

ZIP/model/checkpoint/dataset content는 변경하거나 이동하지 않았다.
