# Phase 3A 결과: IdentitySelector scaffold

- 실행 시각: 2026-07-11 UTC
- 최종 결과: **실패 — equivalence metric 기록 단계의 external wrapper API 오류**
- runner exit code: `1`
- test status: `not_started` — runner 성공 후 실행하도록 구성되어 있어 전체 test suite는 시작되지 않음
- upstream NetLLM: `/workspace/NetLLM-source`
- upstream commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`

## 1. 구현된 extension 구조

원본 `Pipeline`을 수정하거나 monkey-patch하지 않고 composition 방식의 `SelectablePipeline`을 구현했다.

```text
upstream Pipeline instance
├── plm
├── conv1d
├── embed_vp
├── embed_ln
└── networking_head
        ↑ 동일한 module instance를 SelectablePipeline이 재사용
```

selector 삽입 위치:

```text
history temporal embedding concatenate
→ upstream embed_ln
→ optional selector
→ attention mask
→ original-style autoregressive loop
```

feedback embedding에는 selector와 LayerNorm을 다시 적용하지 않는다. Cache 미사용, full-sequence 20회, 마지막 unused feedback 계산과 batch-size 1 전제를 보존했다.

## 2. Selector contract

다음 interface를 구현했다.

- `SelectionOutput`
- `BaseSelector`
- `IdentitySelector`

`IdentitySelector` 구현은 입력 embeddings와 attention mask object를 그대로 반환하며 다음 metadata를 생성한다.

```text
selected_indices=[0,1,...,L-1]
original_length=L
selected_length=L
scores=None
preserves_order=True
```

입력 tensor를 수정하는 연산은 없다.

## 3. 성공한 사전 unit test

equivalence runner 전에 `test_identity_selector.py`의 unit test 3개를 CPU와 사용 가능한 CUDA에서 실행했다.

```text
test_preserves_none_attention_mask: PASS
test_preserves_tensor_contract_without_in_place_mutation: PASS
test_rejects_invalid_shapes: PASS
Ran 3 tests: OK
```

확인 범위:

- shape, dtype, device, gradient contract 보존
- attention mask object 및 값 보존
- embeddings object 및 값 보존
- token 순서와 selected indices 보존
- no in-place mutation
- invalid input shape 거부

## 4. equivalence 실행 상태

실제 tracked Jin2022 test index 0과 동일한 model/Pipeline module instance를 사용하도록 runner를 구성했다.

```text
dataset=Jin2022 test
index=0
video=4
user=83
timestep=30
B=1, H=10, F=20, E=1024
```

실행 log상 original model load와 세 경로 호출 이후 첫 comparison 계산 지점에 도달했다. 그러나 결과 JSON을 쓰기 전에 comparison helper가 실패했으므로 다음 값은 영구 기록 및 검증되지 않았다.

- Original vs disabled maximum absolute difference
- Original vs Identity maximum absolute difference
- 세 output shape/value/hash
- 세 경로 sequence length와 forward count report
- Identity runtime selection metadata

따라서 equivalence 성공을 주장하지 않는다.

## 5. 정확한 실패 원인

분류: **external comparison/reporting wrapper의 Torch API compatibility**

실패 위치:

```text
scripts/experiment_phase/phase3a/run_phase3a_identity_equivalence.py
max_difference()
```

traceback:

```text
Traceback (most recent call last):
  File ".../run_phase3a_identity_equivalence.py", line 420, in <module>
    main()
  File ".../run_phase3a_identity_equivalence.py", line 401, in main
    report = run_equivalence(Path(args.output))
  File ".../run_phase3a_identity_equivalence.py", line 259, in run_equivalence
    "original_vs_disabled": max_difference(
  File ".../run_phase3a_identity_equivalence.py", line 105, in max_difference
    index = list(torch.unravel_index(torch.tensor(flat_index), difference.shape))
  File "/venv/vp_netllm_repro/lib/python3.8/site-packages/torch/__init__.py", line 1833, in __getattr__
    raise AttributeError(...)
AttributeError: module 'torch' has no attribute 'unravel_index'
```

환경 확인:

```text
torch==2.1.0+cu118
hasattr(torch, "unravel_index")=False
numpy==1.24.4
hasattr(numpy, "unravel_index")=True
```

Model, dataset, selector insertion, Pipeline, CUDA, shape 또는 NetworkingHead에서 발생한 오류가 아니다. Maximum difference 값 자체를 계산한 뒤 그 flat maximum index를 좌표로 변환하는 진단 코드에서 발생했다.

## 6. 적용하지 않은 수정

재시도 후보는 `torch.unravel_index`를 현재 설치된 NumPy의 `numpy.unravel_index` 또는 수동 index 계산으로 교체하는 것이다. 이 변경은 tolerance, seed, model, Pipeline 및 output 값에 영향을 주지 않는 reporting-only 수정이다.

그러나 실패 시 runtime을 삭제·덮어쓰거나 임의로 계속 시도하지 말라는 지침에 따라 다음을 수행하지 않았다.

- 실패 runtime 삭제 또는 overwrite
- runner 수정 및 재실행
- tolerance 변경
- random seed 변경
- package 변경
- source 변경
- 다른 model/Pipeline 경로로 우회

## 7. runtime 산출물

생성됨:

- `experiments/vp/phase3a_runtime/identity_equivalence.log`
- `experiments/vp/phase3a_runtime/run_status.txt`
- `experiments/vp/phase3a_runtime/upstream_before.txt`
- `experiments/vp/phase3a_runtime/upstream_after.txt`

생성되지 않음:

- `experiments/vp/phase3a_runtime/identity_equivalence.json`
- `experiments/vp/phase3a_runtime/tests.log`

기존 runtime은 보존했으며 삭제하거나 덮어쓰지 않았다.

## 8. 무결성

| 항목 | 작업 전 | 실패 후 |
|---|---|---|
| upstream commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` | 동일 |
| upstream status/diff | 빈 결과 | 빈 결과 |
| upstream `__pycache__` | 0개 | 0개 |
| repro environment freeze hash | `4601ad2592a119fc91953a4cc142783db59d8a1cdb548097205a1ac5c057ffbe` | 동일 |
| GPT-2 artifact fingerprint | `9eb853117884a343db1500f673fb1b0f79104e40074405d1bffabd8b067a0680` | 동일 |

Training, backward, optimizer, LoRA, adaptation, multimodal, pruning, Recent-K 및 learned scorer는 실행하거나 구현하지 않았다.

## 9. 성공 조건 판정

| 조건 | 판정 |
|---|---|
| 원본 source 변경 0개 | 통과 |
| IdentitySelector unit contract | 통과 |
| 실제 sample runner 진입 | 통과 |
| Original/disabled/Identity 수치 equivalence 기록 | 실패/미완료 |
| max absolute difference `≤1e-7` 증명 | 미완료 |
| `identity_equivalence.json` 생성 | 실패 |
| 전체 Phase 3A test suite | 미실행 |
| 환경/artifact 보존 | 통과 |

Phase 3A는 완료 상태가 아니다.

## 10. 다음 단계 진행 조건

사용자 검토 후 다음 두 사항의 승인이 필요하다.

1. reporting-only index 변환을 Torch 2.1 호환 방식으로 수정
2. 기존 실패 runtime을 보존한 채 별도 retry runtime 경로에서 equivalence를 다시 한 번 실행

승인 전에는 Recent-K, ScoreTopK, training, adaptation 또는 Patch Selection으로 진행하지 않는다.
