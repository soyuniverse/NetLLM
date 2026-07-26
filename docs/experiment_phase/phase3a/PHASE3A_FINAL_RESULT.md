# NetLLM VP Phase 3A 최종 결과

- 실행 시각: 2026-07-26 UTC
- 결과: 성공
- runtime: `/root/NetLLM/experiments/vp/phase3a_final_runtime`
- tolerance: `atol=1e-7`, `rtol=0`

## Identity equivalence

실제 Git-tracked Jin2022 test sample을 사용했다.

```text
dataset index=0
video=4
user=83
timestep=30
B=1
H=10
F=20
```

Original Pipeline, SelectablePipeline의 selector disabled 경로,
SelectablePipeline + IdentitySelector 경로는 하나의 upstream Pipeline과 동일 module/weight
instance를 공유했다.

| 비교 | max absolute difference | 판정 |
|---|---:|---|
| Original vs Disabled | `0.0` | 통과 |
| Original vs Identity | `0.0` | 통과 |
| Disabled vs Identity | `0.0` | 통과 |

세 output은 exact equal이며 SHA-256도 모두
`cfaef687619a44e6d639543e343ef1db1d5f24c810476154df05169dd4c041fe`로
일치했다.

## Tensor 및 sequence contract

- output shape: `[1,20,3]`
- finite: `True`
- 각 경로 sequence lengths: `10,11,...,29`
- 각 경로 GPT-2 forward count: `20`
- `past_key_values` 전달: 없음
- cache reused: `False`
- Identity original length: `10`
- Identity selected length: `10`
- Identity selected indices: `[0,1,2,3,4,5,6,7,8,9]`
- selector 호출: initial history에 1회
- autoregressive feedback에 selector 적용: 없음

Reporting helper는 `torch.unravel_index`를 사용하지 않는다. Flat index를 shape의 마지막
dimension부터 나머지/몫으로 변환하는 dependency-free `unravel_flat_index()`를 사용한다.

## Test

Phase 3A unit/result test 6개가 통과했다. Benchmark 준비 test를 포함한 최종 통합 검증은
13개 모두 통과했다.

## Benchmark framework 준비

다음 configuration을 정의했다.

| selector | selected tokens | keep ratio | indices |
|---|---:|---:|---|
| baseline_original | 10 | 1.0 | `0..9` |
| identity_keep10 | 10 | 1.0 | `0..9` |
| recent_k8 | 8 | 0.8 | `2..9` |
| recent_k6 | 6 | 0.6 | `4..9` |
| recent_k4 | 4 | 0.4 | `6..9` |
| recent_k2 | 2 | 0.2 | `8..9` |

Dry-run은 selector contract, selector latency interface, CSV schema 및 plot input validation만
수행했다. 학습 checkpoint가 없으므로 MAE, RMSE, Mean Angular Error, Test Loss,
end-to-end inference latency 및 GPU peak memory는 기록하지 않았고 `metric_valid=False`로
표시했다. CPU zero embedding으로 측정한 selector-only latency는 metadata에만 분리했으며
model 성능 또는 accuracy로 해석하지 않는다.

Metric interface는 upstream `compute_rmse()` 동작을 재현하는 `upstream_rmse`와 circular
wrap을 적용하는 `corrected_rotation_aware_rmse`를 별도 field로 보존한다. CSV의 `rmse`는
향후 corrected rotation-aware RMSE를 기록한다.

Plot script는 다음 별도 figure를 생성하도록 준비했다.

- `keep_ratio_vs_mae.png`
- `keep_ratio_vs_rmse.png`
- `keep_ratio_vs_latency.png`
- `keep_ratio_vs_gpu_memory.png`
- `accuracy_latency_tradeoff.png`

마지막 파일명은 visualization contract를 유지하지만 y축과 title은 generic accuracy가 아니라
명시적으로 MAE를 사용한다.

## 무결성

| 항목 | 실행 전 | 실행 후 |
|---|---|---|
| upstream commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` | 동일 |
| upstream status/diff | clean | clean |
| upstream `__pycache__` | `0` | `0` |
| raw pip freeze SHA-256 | `731a5031a3fb94909d541db8b66a41299d0977716e09d26eaa682ec3154d0311` | 동일 |
| artifact content manifest | `f3fdcf85dd2a8d38b329048ebb0349bcc94e1c6a04aa08ec20a4c0334ed74f14` | 동일 |

## 범위

Random initialization 결과를 성능 수치로 보고하지 않았다. Llama2-7B, LoRA, training,
full benchmark 및 speculative decoding은 실행하지 않았다.
