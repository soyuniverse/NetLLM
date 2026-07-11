# Phase 1 제안: VP regression baseline smoke test

## 1. 목표

Phase 1은 신규 연구 기능이나 PLM을 실행하지 않고, cooked Jin2022를 이용한 원본 `LinearRegression` VP baseline의 단일 end-to-end test를 재현하는 단계로 제한한다.

- dataset load
- default collate
- normalization
- regression autoregressive prediction
- denormalization
- MAE/RMSE 계산
- 프로젝트 쪽 log/result 생성
- 실행 전후 원본 Git clean 확인

TRACK, PLM, multimodal, training은 Phase 1 범위에서 제외한다.

## 2. 실행 전 prerequisite

1. 사용자가 Phase 0 문서와 아래 외부-output 방식에 승인해야 한다.
2. `/venv/vp_netllm/bin/python`에서 `run_baseline` import가 계속 성공해야 한다.
3. cooked Jin2022 2,268개 CSV와 test split 파일이 존재해야 한다.
4. 프로젝트 기준 `experiments/vp/phase1_runtime/` / `/workspace/NetLLM/experiments/vp/phase1_runtime`을 writable output root로 사용하도록 승인해야 한다.
5. 실행 직전 `/workspace/NetLLM-source` Git status가 clean이어야 한다.
6. 원본을 수정하는 현재 `scripts/run_vp_regression_cpu.sh`는 그대로 실행하지 않는다.

## 3. 제안하는 정확한 명령

아래 명령은 **Phase 0에서 실행하지 않았다**. 사용자 승인 후 Phase 1에서만 실행한다. 원본 module을 import하되 `cfg.models_dir`와 `cfg.results_dir`를 외부 project path로 runtime override한다.

```bash
git -C /workspace/NetLLM-source status --porcelain=v2 --untracked-files=all
mkdir -p /workspace/NetLLM/experiments/vp/phase1_runtime/logs
cd /workspace/NetLLM-source/viewport_prediction
set -o pipefail
PYTHONPATH=/workspace/NetLLM-source/viewport_prediction \
/venv/vp_netllm/bin/python - <<'PY' 2>&1 | tee /workspace/NetLLM/experiments/vp/phase1_runtime/logs/vp_regression_cpu_jin2022.log
from types import SimpleNamespace
import run_baseline as entry

entry.cfg.models_dir = "/workspace/NetLLM/experiments/vp/phase1_runtime/models"
entry.cfg.results_dir = "/workspace/NetLLM/experiments/vp/phase1_runtime/results"

args = SimpleNamespace(
    train=False,
    test=True,
    device="cpu",
    model="regression",
    compile=False,
    resume=False,
    train_dataset="Jin2022",
    test_dataset="Jin2022",
    his_window=10,
    fut_window=20,
    trim_head=30,
    trim_tail=60,
    dataset_frequency=5,
    sample_step=15,
    epochs=40,
    epochs_per_valid=3,
    lr=2e-4,
    weight_decay=1e-5,
    bs=64,
    model_path=None,
    seed=1,
)
entry.run(args)
PY
status=${PIPESTATUS[0]}
git -C /workspace/NetLLM-source status --porcelain=v2 --untracked-files=all
exit "$status"
```

## 4. 예상 생성물

- log: 프로젝트 기준 `experiments/vp/phase1_runtime/logs/vp_regression_cpu_jin2022.log` / `/workspace/NetLLM/experiments/vp/phase1_runtime/logs/vp_regression_cpu_jin2022.log`
- summary CSV: 프로젝트 기준 `experiments/vp/phase1_runtime/results/regression/Jin2022/5Hz/result_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv`
- detail text: 같은 directory의 `details_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv`
- regression은 weight 학습이나 checkpoint 저장을 하지 않는다. `models/regression/Jin2022/5Hz` directory만 생성될 수 있다.

## 5. 성공 조건

1. process exit code가 0이다.
2. fake/synthetic data가 아니라 tracked Jin2022 test split을 사용한다.
3. result summary에 각 video/user pair와 전체 `(-1,-1)` MAE/RMSE가 기록된다.
4. prediction/ground truth shape mismatch error가 없다.
5. 실행 전후 `/workspace/NetLLM-source` Git status가 모두 빈 결과다.
6. output은 모두 `/workspace/NetLLM/experiments/vp/phase1_runtime` 아래에만 생성된다.

## 6. 실패 시 진단 기준

| 실패 유형 | 우선 확인 |
|---|---|
| `FileNotFoundError` | Jin2022 split, cwd, `config.py` 상대경로 해석 |
| import error | `/venv/vp_netllm/bin/python`, NumPy/pandas/scipy/sklearn 상태 |
| shape error | actual `history=[B,10,3]`, `future=[B,20,3]`, DataLoader batch |
| result write error | 외부 runtime directory 권한과 disk space |
| metric 이상 | normalization/denormalization, rotation-aware MAE/RMSE 입력 단위 |
| 원본 status 변화 | 즉시 중단하고 생성 경로를 보고하며 삭제/reset하지 않음 |

실패하더라도 package 설치, source patch, dataset 다운로드로 바로 넘어가지 않는다. 원인을 기록하고 사용자 승인을 기다린다.

## 7. 원본 무결성 확인

실행 전후 아래 명령 결과를 log와 최종 보고에 그대로 남긴다.

```bash
git -C /workspace/NetLLM-source rev-parse HEAD
git -C /workspace/NetLLM-source status --porcelain=v2 --untracked-files=all
git -C /workspace/NetLLM-source diff --name-status HEAD
```

expected commit은 `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`이며 status/diff는 빈 결과여야 한다.

