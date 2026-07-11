# Phase 1 결과: Original VP Regression Baseline Smoke Test

- 실행 시각: 2026-07-11 10:16:53 UTC
- 실행 결과: **성공**
- baseline exit code: `0`
- runner final exit code: `0`
- elapsed time: `20` seconds

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.8.10 |
| Python executable | `/venv/vp_netllm/bin/python` |
| PyTorch | 2.1.0+cu118 |
| device | CPU |
| NetLLM root | `/workspace/NetLLM-source` |
| NetLLM commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` |
| VP source | `/workspace/NetLLM-source/viewport_prediction` |
| output root | `/workspace/NetLLM/experiments/vp/phase1_runtime` |

package 설치, upgrade/downgrade, model 다운로드, 학습, GPT-2/PLM/TRACK/multimodal 실행은 수행하지 않았다.

## 2. 실행 명령

프로젝트 기준 `scripts/experiment_phase/phase1/run_phase1_vp_regression.sh` / `/workspace/NetLLM/scripts/experiment_phase/phase1/run_phase1_vp_regression.sh`를 다음과 같이 정확히 한 번 실행했다.

```bash
cd /workspace/NetLLM
./scripts/experiment_phase/phase1/run_phase1_vp_regression.sh
```

runner는 `/venv/vp_netllm/bin/python -B`와 `PYTHONDONTWRITEBYTECODE=1`을 사용하고, 원본 `run_baseline` module을 import했다. runtime에 다음 값만 외부 output path로 override했다.

```text
cfg.models_dir=/workspace/NetLLM/experiments/vp/phase1_runtime/models
cfg.results_dir=/workspace/NetLLM/experiments/vp/phase1_runtime/results
```

## 3. 사용 dataset 및 설정

- dataset: tracked cooked `Jin2022`
- source path: upstream 기준 `viewport_prediction/data/viewports/Jin2022` / `/workspace/NetLLM-source/viewport_prediction/data/viewports/Jin2022`
- repository에 존재하는 Jin2022 CSV: 2,268개, 모두 Git tracked
- configured test split: video 6개 × user 21명 = 126 video-user pair
- test split 누락 파일: 0개
- 실제 `ViewportDataset` test sample 수: 1,698개
- 실제 DataLoader batch 수: 27개

| 설정 | 값 |
|---|---:|
| model | `regression` |
| mode | test only |
| device | `cpu` |
| history window | 10 |
| future window | 20 |
| batch size | 64 |
| seed | 1 |
| sample step | 15 |
| dataset frequency | 5 Hz |
| trim head | 30 |
| trim tail | 60 |

첫 실제 batch에서 runtime instrumentation으로 확인한 shape는 다음과 같다.

- prediction: `(64, 20, 3)`
- ground truth: `(64, 20, 3)`
- video metadata: `(64,)`

synthetic 또는 fake data는 사용하지 않았다.

## 4. 결과

원본 `ResultNotebook`이 summary의 `video=-1, user=-1` row에 기록한 전체 metric은 다음과 같다.

| metric | 값 |
|---|---:|
| 전체 MAE | `19.473604202270508` |
| 전체 RMSE | `77.839111328125` |

주의: upstream `viewport_prediction/utils/metrics.py::compute_rmse()`는 `rotation` argument를 받지만 내부에서 `compute_mse(data1, data2)`를 호출할 때 이를 전달하지 않는다. 위 RMSE는 원본 code가 실제로 기록한 값이며 Phase 1에서는 수정하거나 재정의하지 않았다.

## 5. output 파일

모든 실행 output은 `/workspace/NetLLM/experiments/vp/phase1_runtime` 아래에만 생성됐다.

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `logs/20260711T101653Z_vp_regression_cpu_jin2022.log` | 10,605 bytes | `8666bd0b3e3f138917a3db2a8f4980778bdb044fa78788f4d6954ca4058262da` |
| `logs/20260711T101653Z_run_status.txt` | 808 bytes | 실행 상태 및 metric 기록 |
| `logs/20260711T101653Z_upstream_before.txt` | 92 bytes | 실행 전 integrity 기록 |
| `logs/20260711T101653Z_upstream_after.txt` | 92 bytes | 실행 후 integrity 기록 |
| `results/regression/Jin2022/5Hz/result_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv` | 5,512 bytes | `c8c293e6f65dca5ebd3a460f7ba4ea7514ffa1dddad2d3fbb0a2a939755667b3` |
| `results/regression/Jin2022/5Hz/details_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv` | 2,307,776 bytes | `408cfd0ef02b608158cadef2e69761fb2459d06bc523dcb8b322034023292a1e` |

summary는 128줄(header + 126 pair + 전체 row), detail은 sample별 `pred`/`gt` 두 줄씩 총 3,396줄이다. detail 파일은 `.csv` 확장자지만 원본 `ResultNotebook.write()` 동작상 tabular CSV가 아니라 `pred:`/`gt:` free-form text다. regression은 학습하지 않으므로 checkpoint/model file은 생성되지 않았다.

## 6. 주요 log 마지막 부분

```text
|   25  |  49  | 41.20805358886719  | 124.84782409667969 |
|   -1  |  -1  | 19.473604202270508 |  77.839111328125   |
+-------+------+--------------------+--------------------+
Results saved at /workspace/NetLLM/experiments/vp/phase1_runtime/results/regression/Jin2022/5Hz/result_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv
Detail results saved at /workspace/NetLLM/experiments/vp/phase1_runtime/results/regression/Jin2022/5Hz/details_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv
```

traceback은 발생하지 않았다.

## 7. 원본 Git 무결성 비교

| 항목 | 실행 전 | 실행 후 |
|---|---|---|
| commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` | 동일 |
| `git status --porcelain=v2 --untracked-files=all` | 빈 결과 | 빈 결과 |
| `git diff --name-status HEAD` | 빈 결과 | 빈 결과 |
| source modified file | 0개 | 0개 |
| source untracked file | 0개 | 0개 |
| `__pycache__` | 없음 | 없음 |

원본 안에 log, result, checkpoint 또는 untracked file이 생성되지 않았다. reset, clean, checkout, source file 삭제는 수행하지 않았다.

## 8. 성공 조건 판정

| 조건 | 판정 |
|---|---|
| exit code 0 | 통과 |
| tracked Jin2022 사용 | 통과 |
| 전체 MAE/RMSE 기록 | 통과 |
| output이 external runtime root에만 존재 | 통과 |
| 원본 실행 전후 clean | 통과 |
| 원본 source 변경 0개 | 통과 |

## 9. 다음 단계 진행 가능 여부

Original VP LinearRegression baseline smoke test 관점에서는 Phase 1이 완료되어 다음 단계 검토가 가능하다.

다만 PLM 단계에는 Phase 0에서 확인한 `peft`/`accelerate` import 불일치와 pretrained PLM 부재가 여전히 남아 있다. package 또는 source 변경은 이번 단계에서 수행하지 않았으며, 다음 단계는 사용자 검토와 별도 승인 후에만 진행해야 한다.
