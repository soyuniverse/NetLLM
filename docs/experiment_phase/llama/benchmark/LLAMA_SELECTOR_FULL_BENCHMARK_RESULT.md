# Llama VP Selector Full Benchmark Result

## Final status

- 상태: **성공**
- 결과 명칭: `recovered-artifact controlled comparison`
- source commit: `ee4d8726898610e4ae7df08bdd26728cafb4701f`
- test split: checkpoint-era Jin2022 test split 전체
- sample count: configuration별 1,698
- configuration count: 6
- total measured inference count: 10,188
- 실행 시간: 6,313.241초 (약 1.754시간)
- partial: false
- OOM/non-finite/missing/unexpected key: 없음

## Controlled comparison

| selector | tokens | keep | MAE | upstream RMSE | corrected RMSE | mean angular error | evaluation loss | median ms | p95 ms | selector median ms | allocated MiB | reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 10 | 1.0 | 12.798525 | 35.706529 | 27.118540 | 12.798525 | 0.041198 | 613.938 | 629.506 | 0.000 | 12996.031 | 13086.000 |
| identity_keep10 | 10 | 1.0 | 12.798525 | 35.706529 | 27.118540 | 12.798525 | 0.041198 | 614.441 | 623.718 | 0.044 | 12996.127 | 13086.000 |
| recent_k8 | 8 | 0.8 | 12.531576 | 35.694498 | 26.509241 | 12.531576 | 0.041147 | 612.180 | 624.310 | 0.056 | 12993.938 | 13086.000 |
| recent_k6 | 6 | 0.6 | 12.101893 | 35.574892 | 25.430680 | 12.101893 | 0.040837 | 649.555 | 659.016 | 0.059 | 12991.750 | 13086.000 |
| recent_k4 | 4 | 0.4 | 11.566495 | 35.217756 | 24.230301 | 11.566495 | 0.039984 | 609.122 | 630.408 | 0.056 | 12989.562 | 13086.000 |
| recent_k2 | 2 | 0.2 | 10.847409 | 34.911707 | 22.487223 | 10.847409 | 0.039207 | 605.789 | 617.219 | 0.055 | 12987.373 | 13086.000 |

`upstream RMSE`는 checkpoint-era upstream 구현의 rotation 미보정 동작을 보존한 값이다. `corrected RMSE`는 360도 회전을 고려한 최소 각도 오차로 별도 계산했다.

## Identity control

- Original vs Identity MAE difference: 0.0
- upstream RMSE difference: 0.0
- corrected RMSE difference: 0.0
- mean angular error difference: 0.0
- evaluation loss difference: 0.0
- 1,698개 prediction SHA-256 sequence: 모두 동일

따라서 wrapper가 selector disabled/Identity 경로에서 원본 checkpoint-era inference를 변경하지 않는다는 control이 유지됐다.

## Runtime interpretation

각 sample은 cache 없이 PLM을 20번 forward한다. 전체 구성별 forward count는 33,960이다. initial token 감소에 따라 processed sequence-length sum은 662,220에서 390,540으로 감소했다. 그러나 autoregressive loop의 20회 full forward가 유지되어 latency 개선은 작고, `recent_k6`은 해당 run에서 오히려 느렸다. 이 결과만으로 일반적인 속도 향상을 주장하지 않는다.

## Validity

- `technical_smoke_valid=True`
- `comparative_quality_valid=True`
- `paper_reproduction_valid=False`

동일 base artifact, checkpoint, checkpoint-era source, split, order, seed, batch size와 random component가 없는 조건에서 selector만 변경했으므로 configuration 간 비교는 유효하다. Immutable training base revision, checkpoint epoch/step 및 validation selection criterion이 불명확해 논문 수치 재현 또는 공식 NetLLM benchmark라는 주장은 하지 않는다.

