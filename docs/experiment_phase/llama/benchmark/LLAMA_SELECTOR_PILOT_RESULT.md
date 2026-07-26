# Llama VP Selector Pilot Result

## 판정

- 상태: **성공**
- 결과 명칭: `recovered-artifact controlled comparison`
- test subset: checkpoint-era Jin2022 test split의 고정된 선두 128개 sample
- configuration: `original`, `identity_keep10`, `recent_k8`, `recent_k6`, `recent_k4`, `recent_k2`
- 모든 configuration sample count: 128
- 모든 output finite: true
- OOM: 없음
- strict load missing/unexpected key: 0/0
- 총 실행 시간: 494.566초

## Identity control

- MAE 차이: 0.0
- upstream RMSE 차이: 0.0
- corrected rotation-aware RMSE 차이: 0.0
- mean angular error 차이: 0.0
- evaluation loss 차이: 0.0
- sample별 prediction SHA-256: 모두 동일

## Pilot summary

| selector | keep ratio | MAE | upstream RMSE | corrected RMSE | loss | median latency (ms) | p95 (ms) | peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 1.0 | 11.911071 | 35.982593 | 25.757154 | 0.040911 | 615.369 | 636.868 | 12996.031 |
| identity_keep10 | 1.0 | 11.911071 | 35.982593 | 25.757154 | 0.040911 | 616.794 | 636.608 | 12996.127 |
| recent_k8 | 0.8 | 11.923862 | 35.787629 | 25.748246 | 0.040486 | 616.211 | 627.809 | 12993.938 |
| recent_k6 | 0.6 | 12.002620 | 35.426524 | 26.224424 | 0.039643 | 614.568 | 664.881 | 12991.750 |
| recent_k4 | 0.4 | 11.508642 | 34.595656 | 25.349628 | 0.037807 | 611.689 | 616.225 | 12989.562 |
| recent_k2 | 0.2 | 10.523219 | 31.965498 | 22.203652 | 0.032371 | 609.806 | 645.622 | 12987.373 |

이 pilot 수치는 subset 결과이며 최종 test-set 결과로 해석하지 않는다. 특히 configuration별 latency 차이는 작아 full run에서 다시 측정한다.

## Metric validity

- `technical_smoke_valid=True`
- `comparative_quality_valid=True`
- `paper_reproduction_valid=False`

동일한 base artifact, checkpoint, checkpoint-era source, test order, seed, batch size를 사용하고 selector만 변경했으며 random VP component가 없으므로 configuration 간 비교는 유효하다. 다만 training 당시 immutable base revision, checkpoint epoch/step 및 선택 criterion이 확인되지 않아 논문 수치 재현이나 공식 NetLLM benchmark로 주장하지 않는다.

## Full-run gate

- full test sample count: 1,698
- pilot 선형 환산 예상 시간: 6,560.724초 (약 1.822시간)
- 6시간 이하이므로 full test-set benchmark 실행 조건을 만족한다.

