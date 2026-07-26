# NetLLM VP resume 상태

- audit 시각: 2026-07-26 UTC
- project: `/root/NetLLM`
- upstream: `/root/NetLLM-source`
- upstream commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- Gate 0 결과: 성공
- R0/R1/R2 재실행: 하지 않음

## 기존 산출물 audit

| 산출물 | 상태 | SHA-256 |
|---|---|---|
| `docs/experiment_phase/recovery/ENVIRONMENT_RECOVERY_REPORT.md` | 존재 | `76c2283ebe3580e7361d331aeb67a16d89308129b68825496523d6853723f9d1` |
| `experiments/vp/recovery/environment_manifest.json` | 존재, JSON valid | `08d564d219918d2d8410fcbadbc5b8ddf6c354cacedc710f0d7a22d698c53222` |
| `experiments/vp/phase3a_final_runtime/identity_equivalence.json` | 존재, JSON valid | `6235adc0f129bbe4238fff89bf316950409823b7b065468dc2561845cb8f130b` |
| `docs/experiment_phase/phase3a/PHASE3A_FINAL_RESULT.md` | 존재 | `4a96632a80fcab94d2db2067782d10b43fd6490c0230931fe34be1bc30291287` |
| `configs/vp_benchmark.json` | 존재, JSON valid | `41fe9aa569e633b0b09bb6ee0464ef84536c9d8709987278c6dd616927decfa4` |
| `src/netllm_litevlm/selectors/recent_k.py` | 존재 | `5720e8fe6a560aa0fac4271421fc3e7cb5d4ec6a8ef809d69988ed2df274c53a` |
| `src/netllm_litevlm/evaluation/vp_metrics.py` | 존재 | `996915c99a542b887d42764fdeb1aa4bc44027aa4a2111927b4cfb9f9620e4f4` |
| `experiments/vp/benchmark_dry_run/vp_benchmark.csv` | 존재 | `381552cf8bf82cd0b8a38073f708b6c18a25e061bea9dd689d7009a010d5bee7` |
| `scripts/experiment_phase/benchmark/plot_vp_benchmark.py` | 존재 | `96d14664732bfece3b80d415e680425f8eccc419ce85d7683456ae069979d980` |
| `experiments/vp/phase3a_final_runtime/tests.log` | 존재, 6/6 pass | `2d0fc20596107831bf5bba74ec560e356f9addc971e68b1bab896156052a50d2` |
| `experiments/vp/benchmark_dry_run/tests.log` | 존재, 13/13 pass | `c02b75e98f5429e3ce5a9dcd47ecbdb57cbc231fa7037ede14812d048e6f965e` |

## Phase 3A 확정

`identity_equivalence.json`을 다시 계산하지 않고 읽어 검증했다.

| 조건 | 기록값 | 판정 |
|---|---:|---|
| `success` | `True` | 통과 |
| Original vs Disabled max diff | `0.0` | 통과 |
| Original vs Identity max diff | `0.0` | 통과 |
| Disabled vs Identity max diff | `0.0` | 통과 |
| output shape | `[1,20,3]` | 통과 |
| sequence lengths | `10..29` | 통과 |
| GPT-2 forward count | `20` | 통과 |
| cache reused | `False` | 통과 |
| Identity indices | `[0,1,2,3,4,5,6,7,8,9]` | 통과 |

Tolerance는 `atol=1e-7`, `rtol=0`이다. Phase 3A는 완료 상태로 확정한다.

## R2 확정

Dry-run CSV에는 다음 6개 selector가 있다.

```text
baseline_original
identity_keep10
recent_k8
recent_k6
recent_k4
recent_k2
```

모든 row의 `metric_valid=False`이며 MAE, RMSE, Mean Angular Error, Test Loss,
end-to-end latency와 GPU memory field는 비어 있다. 이는 trained checkpoint가 없는 상태를
정확히 나타내며 random head 성능을 보고하지 않는다.

현재 CSV schema는 Phase 3A 준비 contract의 14개 column이다. 향후 Gate 5 실제 Llama
benchmark에서는 요청된 신규 schema에 `sample_count`를 추가해야 한다.

## 무결성

- upstream commit: 요구 commit과 일치
- upstream status/diff: clean
- upstream `__pycache__`: `0`
- GPT-2 environment raw `pip freeze` hash:
  `731a5031a3fb94909d541db8b66a41299d0977716e09d26eaa682ec3154d0311`
- GPT-2 artifact content manifest:
  `f3fdcf85dd2a8d38b329048ebb0349bcc94e1c6a04aa08ec20a4c0334ed74f14`

Gate 0에서는 기존 runtime/log를 삭제하거나 덮어쓰지 않았고 test, forward, benchmark를
재실행하지 않았다.
