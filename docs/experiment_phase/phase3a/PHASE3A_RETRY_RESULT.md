# Phase 3A Retry 결과: reporting compatibility

- 확인 시각: 2026-07-20 UTC
- Retry 결과: **실행 전 환경 확인 실패 — equivalence runner 미실행**
- retry wrapper exit code: `2`
- Phase 3A 최종 상태: **미완료**

## 1. 기존 실패 원인과 수정

기존 실패는 model output 계산이 아니라 `max_difference()`의 reporting-only 좌표 변환에서 발생했다.

```text
torch==2.1.0+cu118
torch.unravel_index 없음
```

`difference`, `argmax`, maximum absolute difference, 좌우 값 조회, `torch.equal`, 그리고
`torch.allclose(rtol=0, atol=1e-7)`는 변경하지 않았다. Flat index만 tensor shape의 마지막
dimension부터 나머지와 몫을 계산하는 dependency-free 방식으로 변환하도록 수정했다.

Selector, `SelectablePipeline`, tensor 연산 순서, tolerance, random seed, model, dataset,
cache, LayerNorm, feedback 및 batch-size contract는 변경하지 않았다.

## 2. 기존 shell Python 경로 불일치 조사

수정 전 실제 shell wrapper에는 다음 값이 있었다.

```text
PYTHON_BIN=/venv/vp_netllm/bin/python
```

반면 보존된 Phase 3A 결과 문서의 traceback은
`/venv/vp_netllm_repro/lib/python3.8/site-packages/torch/__init__.py`에서 발생했고,
`experiments/vp/phase3a_runtime/run_status.txt`의 freeze hash는 다음 값이다.

```text
4601ad2592a119fc91953a4cc142783db59d8a1cdb548097205a1ac5c057ffbe
```

이 값은 Phase 2B status/result에 `/venv/vp_netllm_repro/bin/python`으로 명시된 freeze hash와
일치한다. 따라서 기존 wrapper의 interpreter line과 기록된 실제 실패 환경 사이에 불일치가
있었으며, 기존 wrapper의 해당 한 줄과 신규 retry wrapper는 모두 다음 경로로 맞췄다.

```text
/venv/vp_netllm_repro/bin/python
```

기존 raw `identity_equivalence.log`는 현재 checkout에 존재하지 않아 직접 재검사할 수 없었다.
조사 근거는 보존된 `run_status.txt`, upstream before/after 기록, Phase 3A traceback 문서 및
Phase 2B 환경 기록이다.

## 3. Retry 실행 상태

신규 retry wrapper를 한 번 호출했으나 다음 필수 경로가 현재 실행 환경에 없다.

```text
/workspace/NetLLM
/workspace/NetLLM-source
/venv/vp_netllm_repro/bin/python
/workspace/NetLLM-artifacts/plms/gpt2/base
```

Preflight가 equivalence 및 test 실행 전에 종료했으며 exit code는 `2`다. Package 설치,
환경 생성, source clone, artifact download 또는 대체 Python 사용은 금지 조건 때문에 수행하지
않았다. 현재 기본 Python은 `/opt/conda/bin/python` (`Python 3.10.13`, `torch 2.2.0`)이므로
요구된 repro 환경을 대신할 수 없다.

## 4. 검증 결과

요구된 `/venv/vp_netllm_repro/bin/python` 실행 결과는 다음과 같다.

| 항목 | 결과 |
|---|---|
| IdentitySelector unit test | 미실행 — repro Python 없음 |
| equivalence runner | 미실행 — preflight 종료 |
| retry JSON | 미생성 |
| full Phase 3A tests | 미실행 — JSON 성공 후에만 실행하도록 보존 |
| Original vs Disabled | 미측정 |
| Original vs Identity | 미측정 |
| Disabled vs Identity | 미측정 |
| maximum absolute difference | 미측정 |
| sequence length / forward count | 미측정 |

보조 검증으로 현재 세션 Python에서 기존 IdentitySelector unit test 3개와 수동 flat-index 변환의
focused check는 통과했다. 이는 요구된 repro 환경의 unit/equivalence 결과로 간주하지 않는다.
두 shell wrapper의 `bash -n`과 `git diff --check`도 통과했다.

## 5. 생성 및 보존 상태

구현된 retry 전용 파일:

- `scripts/experiment_phase/phase3a/run_phase3a_identity_equivalence_retry.sh`
- `tests/phase3a/test_vp_identity_equivalence_retry.py`
- `docs/experiment_phase/phase3a/PHASE3A_RETRY_RESULT.md`

환경이 충족된 성공 실행에서 생성하도록 구성된 output:

- `experiments/vp/phase3a_retry_runtime/identity_equivalence.json`
- `experiments/vp/phase3a_retry_runtime/identity_equivalence.log`
- `experiments/vp/phase3a_retry_runtime/tests.log`
- `experiments/vp/phase3a_retry_runtime/run_status.txt`

이번 preflight 실패에서는 위 runtime directory와 output이 생성되지 않았다. 기존
`experiments/vp/phase3a_runtime/`의 tracked status 및 upstream 기록은 수정하거나 삭제하지
않았다.

## 6. 원본 및 환경 무결성

현재 실행 환경에는 `/workspace/NetLLM-source`가 없어 upstream commit/status/diff를 새로 측정할
수 없다. 보존된 기존 before/after 기록은 commit
`105bcf070f2bec808f7b14f8f5a953de6e4e6e54`, empty status/diff, `__pycache__` 0개를 나타낸다.
이번 작업은 해당 upstream 경로, repro environment 또는 GPT-2 artifact에 접근하거나 변경하지
않았다. Package install/upgrade/downgrade도 수행하지 않았다.

Repro environment와 artifact가 실제로 mount된 환경에서 retry wrapper는 실행 전후 freeze hash,
artifact fingerprint, upstream commit/status/diff 및 `__pycache__` count를
`run_status.txt`에 기록하고 변경이 있으면 실패하도록 구성했다.

## 7. 최종 판정과 다음 단계

Reporting compatibility 수정과 retry 실행 도구는 준비됐지만 수치 equivalence가 실행되지
않았으므로 Phase 3A 완료를 주장할 수 없다. `/workspace` source/artifact와
`/venv/vp_netllm_repro`가 보존된 원래 실행 환경에서 신규 retry wrapper를 실행해 JSON 및 전체
test 성공을 확인해야 한다.

그 전에는 Recent-K, ScoreTopK, training, LoRA, adaptation 또는 다음 phase로 진행하면 안 된다.
