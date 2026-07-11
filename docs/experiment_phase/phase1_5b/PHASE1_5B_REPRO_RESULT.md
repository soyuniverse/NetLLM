# Phase 1.5B 결과: 독립 VP 환경 재현

- 실행 시각: 2026-07-11 UTC
- 결과: **성공**
- 신규 환경: `/venv/vp_netllm_repro`
- upstream NetLLM: `/workspace/NetLLM-source`
- upstream commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- runtime output: `/workspace/NetLLM/experiments/vp/phase1_5b_runtime`

## 1. 승인된 dependency 변경

`requirements-vp.txt`에서 다음 한 줄만 변경했다.

```diff
-accelerate==0.32.1
+accelerate==0.24.1
```

다른 requirement와 `setup.sh`는 수정하지 않았다. `setup.sh`는 기존부터 `python -m pip install -r "$SCRIPT_DIR/requirements-vp.txt"`를 실행하므로 별도 setup logic 변경은 필요하지 않다.

Phase 1.5A의 제안 조합은 신규 `constraints-vp-plm.txt`에 byte 단위로 동일하게 확정했다.

## 2. 작업 전 상태

| 항목 | 결과 |
|---|---|
| 프로젝트 HEAD | `d821f1a9fed38b2400e13a228e8f634f890954bc` |
| 프로젝트 working tree | clean |
| upstream status/diff | 빈 결과 / 빈 결과 |
| `/venv/vp_netllm_repro` | 존재하지 않음 |
| `experiments/vp/phase1_5b_runtime` | 존재하지 않음 |
| 디스크 여유 | 약 90GB |
| 기존 환경 freeze hash | `c3717faa56495a1b8e38cac864f1151aa40bdfa25e85c38842d52bb469a47631` |
| overlay freeze hash | `2d1f31014089defd0dfd704b64baa0c490d7960154f127aab3f6ae50102686d5` |
| overlay tree hash | `09aaa1b1de8b6a83a581f93a2e572f75c8c53c68149cee07dca657e87fefa27e` |
| 기존 Accelerate pin | `accelerate==0.32.1` |

## 3. 독립 환경 생성

다음 script를 한 번 실행했다.

```bash
cd /workspace/NetLLM
./scripts/create_phase1_5b_repro_env.sh
```

script가 수행한 핵심 명령은 다음과 같다.

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda create --prefix /venv/vp_netllm_repro python=3.8.10 -y
/venv/vp_netllm_repro/bin/python -m pip install --upgrade pip
/venv/vp_netllm_repro/bin/python -m pip install torch==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
/venv/vp_netllm_repro/bin/python -m pip install \
  -r /workspace/NetLLM/requirements-vp.txt
```

환경 생성 결과:

```text
python_executable=/venv/vp_netllm_repro/bin/python
sys_prefix=/venv/vp_netllm_repro
sys_base_prefix=/venv/vp_netllm_repro
site_packages=['/venv/vp_netllm_repro/lib/python3.8/site-packages']
user_site_enabled=False
```

- `--system-site-packages`를 사용하지 않았다.
- `/venv/vp_netllm`과 `/venv/vp_netllm_plmtest`의 site-packages가 `sys.path`에 없음을 assertion으로 확인했다.
- 신규 환경 크기: 약 5.1GB
- 신규 환경 freeze hash: `4601ad2592a119fc91953a4cc142783db59d8a1cdb548097205a1ac5c057ffbe`

전체 Conda/Pip resolver output과 `pip freeze`는 `experiments/vp/phase1_5b_runtime/logs/20260711T145516Z_environment_creation.log`에 기록했다.

## 4. 최종 version과 import 검사

| package | version |
|---|---:|
| Python | `3.8.10` |
| `torch` | `2.1.0+cu118` |
| `transformers` | `4.34.1` |
| `peft` | `0.6.2` |
| `accelerate` | `0.24.1` |
| `huggingface-hub` | `0.17.3` |
| `safetensors` | `0.5.3` |
| `tokenizers` | `0.14.1` |
| `numpy` | `1.24.4` |
| `opencv-python-headless` | `4.8.1.78` |
| `yacs` | `0.1.8` |

`python -m pip check` 결과:

```text
No broken requirements found.
```

다음 import가 모두 성공했다.

```text
torch, transformers, peft, accelerate, huggingface_hub,
cv2, yacs, numpy, pandas, scipy, sklearn, einops,
run_baseline, run_plm
```

CUDA 확인 결과:

```text
torch.version.cuda=11.8
torch.cuda.is_available()=True
torch.cuda.get_device_name(0)=NVIDIA GeForce RTX 4090
```

검사 명령과 log:

```bash
./scripts/check_phase1_5b_repro_env.sh
```

`experiments/vp/phase1_5b_runtime/logs/20260711T145608Z-environment-check.log`

## 5. Jin2022 regression 재현

다음 wrapper를 정확히 한 번 실행했다.

```bash
cd /workspace/NetLLM
./scripts/run_phase1_5b_regression.sh
```

설정:

```text
Python=/venv/vp_netllm_repro/bin/python
model=regression
mode=test only
device=cpu
train_dataset=Jin2022
test_dataset=Jin2022
history=10
future=20
batch_size=64
seed=1
sample_step=15
dataset_frequency=5
trim_head=30
trim_tail=60
```

실제 tracked cooked Jin2022 test split 1,698 sample, 27 batch를 사용했다. 첫 batch shape는 prediction과 ground truth 모두 `(64, 20, 3)`이었다.

| 항목 | Phase 1 | Phase 1.5B | 비교 |
|---|---:|---:|---|
| exit code | `0` | `0` | 동일 |
| MAE | `19.473604202270508` | `19.473604202270508` | 정확히 동일 |
| RMSE | `77.839111328125` | `77.839111328125` | 정확히 동일 |
| elapsed time | 20초 | 20초 | 동일 |

summary와 detail 파일도 Phase 1 결과와 SHA-256이 각각 완전히 동일하다.

```text
summary: c8c293e6f65dca5ebd3a460f7ba4ea7514ffa1dddad2d3fbb0a2a939755667b3
detail:  408cfd0ef02b608158cadef2e69761fb2459d06bc523dcb8b322034023292a1e
```

## 6. 생성 output

모든 실행 output은 `experiments/vp/phase1_5b_runtime/` 아래에 생성했다.

- 환경 생성 log: `logs/20260711T145516Z_environment_creation.log`
- 환경 검사 log: `logs/20260711T145608Z-environment-check.log`
- regression log: `logs/20260711T145628Z_vp_regression_cpu_jin2022.log`
- 실행 상태: `logs/20260711T145628Z_run_status.txt`
- upstream 실행 전후 기록: `logs/20260711T145628Z_upstream_before.txt`, `logs/20260711T145628Z_upstream_after.txt`
- summary: `results/regression/Jin2022/5Hz/result_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv`
- detail: `results/regression/Jin2022/5Hz/details_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv`

## 7. upstream 무결성

| 항목 | 작업 전 | 작업 후 |
|---|---|---|
| commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` | 동일 |
| status porcelain | 빈 결과 | 빈 결과 |
| diff name-status | 빈 결과 | 빈 결과 |
| `__pycache__` | 0개 | 0개 |

원본에 log, result, cache, checkpoint 또는 untracked file을 생성하지 않았다.

## 8. 기존 환경 보존

| 대상 | 작업 전 | 작업 후 | 판정 |
|---|---|---|---|
| `/venv/vp_netllm` freeze hash | `c3717faa...a47631` | `c3717faa...a47631` | 동일 |
| `/venv/vp_netllm_plmtest` freeze hash | `2d1f3101...186d5` | `2d1f3101...186d5` | 동일 |
| `/venv/vp_netllm_plmtest` tree hash | `09aaa1b1...fa27e` | `09aaa1b1...fa27e` | 동일 |
| `setup.sh` SHA-256 | `3190db18...9ff8` | `3190db18...9ff8` | 동일 |

## 9. 제외 범위와 다음 단계

- pretrained GPT-2/Llama, checkpoint, image, feature, dataset을 다운로드하지 않았다.
- GPT-2/PLM model load, forward, adaptation, training을 실행하지 않았다.
- source patch를 적용하지 않았다.
- selector 또는 LiteVLM 기능을 구현하지 않았다.

Phase 1.5B 성공 조건은 모두 충족했다. `accelerate==0.24.1`은 VP의 재현 가능한 공식 dependency pin으로 사용할 수 있다. 실제 PLM model을 사용하는 후속 단계에는 pretrained PLM 부재와 기존 VP source의 runtime 불확실성이 여전히 blocker로 남는다.
