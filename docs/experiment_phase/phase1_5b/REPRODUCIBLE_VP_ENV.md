# 재현 가능한 VP 환경 구성

## 1. 목표와 전제 조건

이 절차는 기존 Python 환경의 site-packages를 참조하지 않는 독립적인 VP 환경을 `/venv/vp_netllm_repro`에 생성한다.

필요 조건:

- Ubuntu/Linux 환경
- NVIDIA driver와 CUDA 11.8 wheel을 지원하는 GPU
- `/opt/miniforge3`에 설치된 Conda/Miniforge
- 프로젝트 wrapper: `/workspace/NetLLM`
- clean upstream NetLLM: `/workspace/NetLLM-source`
- upstream commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- tracked cooked Jin2022 data: `/workspace/NetLLM-source/viewport_prediction/data/viewports/Jin2022`
- 최소 6GB 이상의 환경 설치 여유 공간

환경 생성 script는 upstream code, model 또는 dataset을 내려받지 않는다. 프로젝트와 upstream clone은 먼저 별도로 준비해야 한다.

## 2. 새 Vast.ai instance 설치 순서

### 2.1 repository 상태 확인

```bash
cd /workspace/NetLLM
git status --short
git -C /workspace/NetLLM-source rev-parse HEAD
git -C /workspace/NetLLM-source status --porcelain=v2 --untracked-files=all
git -C /workspace/NetLLM-source diff --name-status HEAD
```

upstream commit은 다음 값이어야 하며 status와 diff는 비어 있어야 한다.

```text
105bcf070f2bec808f7b14f8f5a953de6e4e6e54
```

### 2.2 대상 경로와 디스크 확인

```bash
test ! -e /venv/vp_netllm_repro
df -h /venv /workspace
```

대상 환경이 이미 있으면 삭제하거나 덮어쓰지 말고 원인을 먼저 확인한다.

### 2.3 독립 환경 생성

```bash
cd /workspace/NetLLM
./scripts/create_phase1_5b_repro_env.sh
```

script는 다음 순서로 실행한다.

1. `conda create --prefix /venv/vp_netllm_repro python=3.8.10 -y`
2. 신규 환경 내부 pip upgrade
3. PyTorch `2.1.0+cu118` wheel 설치
4. 프로젝트 `requirements-vp.txt` 설치
5. 환경 경로, `pip check`, `pip freeze` 기록

`--system-site-packages`는 사용하지 않는다. 설치 log는 `experiments/vp/phase1_5b_runtime/logs/`에 timestamped 파일로 저장된다.

## 3. 검증된 exact version

```text
Python==3.8.10
torch==2.1.0+cu118
transformers==4.34.1
peft==0.6.2
accelerate==0.24.1
huggingface-hub==0.17.3
safetensors==0.5.3
tokenizers==0.14.1
numpy==1.24.4
opencv-python-headless==4.8.1.78
yacs==0.1.8
```

PLM 핵심 조합은 `constraints-vp-plm.txt`에 별도로 고정했다.

## 4. 환경과 import 확인

```bash
cd /workspace/NetLLM
./scripts/check_phase1_5b_repro_env.sh
```

검사 항목:

- Python 3.8.10과 절대 executable
- `sys.prefix`, `sys.base_prefix`, `site.getsitepackages()`
- 기존 환경과 overlay site-packages 미참조
- exact package version
- `python -m pip check`
- 핵심 runtime import
- CUDA runtime, availability, GPU 이름
- upstream `run_baseline`, `run_plm` import

정상 독립 환경의 핵심 출력은 다음과 같다.

```text
sys_prefix=/venv/vp_netllm_repro
sys_base_prefix=/venv/vp_netllm_repro
site_packages=['/venv/vp_netllm_repro/lib/python3.8/site-packages']
legacy_site_packages_referenced=no
overlay_site_packages_referenced=no
No broken requirements found.
run_baseline import: OK
run_plm import: OK
```

## 5. regression 재현 확인

결과 경로가 비어 있을 때 다음을 한 번 실행한다.

```bash
cd /workspace/NetLLM
./scripts/run_phase1_5b_regression.sh
```

성공 기준:

```text
exit code=0
MAE=19.473604202270508
RMSE=77.839111328125
metric_match_phase1=yes
upstream_clean_after=yes
```

wrapper는 기존 결과를 덮어쓰지 않고, 원본 밖의 `experiments/vp/phase1_5b_runtime/`에만 log와 result를 쓴다.

## 6. Accelerate 0.24.1 고정 이유

최초 조합의 `accelerate==0.32.1`은 import 시 `huggingface_hub.split_torch_state_dict_into_shards`를 요구하지만, 고정된 `huggingface-hub==0.17.3`에는 이 symbol이 없다.

Hub를 0.21.0으로 올리면 해당 symbol은 제공되지만 `tokenizers==0.14.1`의 `huggingface-hub>=0.16.4,<0.18` 조건을 위반한다. 반면 `accelerate==0.24.1`은 다음 조건을 모두 만족한다.

- `peft==0.6.2`의 `accelerate>=0.21.0`
- `transformers==4.34.1`의 호환 범위
- `huggingface-hub==0.17.3`
- `tokenizers==0.14.1`
- `pip check` 성공
- `peft`, `accelerate`, `run_plm` import 성공

따라서 다른 핵심 package를 바꾸지 않는 최소 수정이다.

## 7. 재현 실패 시 점검

### 대상 환경이 이미 존재함

자동 삭제하지 않는다. 다음으로 environment identity를 확인하고 사용자 판단 후 처리한다.

```bash
/venv/vp_netllm_repro/bin/python --version
/venv/vp_netllm_repro/bin/python -c 'import site,sys; print(sys.prefix); print(site.getsitepackages())'
```

### package version 불일치

```bash
/venv/vp_netllm_repro/bin/python -m pip freeze
/venv/vp_netllm_repro/bin/python -m pip check
rg -n '^accelerate==' /workspace/NetLLM/requirements-vp.txt
```

다른 version을 임의로 시도하지 말고 environment creation log의 resolver output과 `requirements-vp.txt`를 비교한다.

### `run_plm` import 실패

```bash
cd /workspace/NetLLM-source/viewport_prediction
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/NetLLM-source/viewport_prediction \
/venv/vp_netllm_repro/bin/python -B -c 'import run_plm'
```

traceback에서 실제 실패 package와 module을 확인한다. source patch나 package 무작위 변경은 수행하지 않는다.

### CUDA 미인식

```bash
nvidia-smi
/venv/vp_netllm_repro/bin/python -c \
  'import torch; print(torch.version.cuda); print(torch.cuda.is_available())'
```

driver visibility와 PyTorch wheel version을 구분해 확인한다.

### regression metric 불일치

- cooked Jin2022가 upstream commit에 tracked되어 있는지 확인한다.
- history/future/sample step/frequency/trim/batch size/seed 설정을 확인한다.
- summary와 detail 파일을 Phase 1 checksum과 비교한다.
- 원본 source나 metric 구현은 수정하지 않는다.

## 8. 범위 제한

이 절차는 dependency, import, CUDA visibility와 LinearRegression baseline만 검증한다. GPT-2/Llama download 또는 load, PLM forward, adaptation, training, checkpoint, multimodal path와 LiteVLM selector는 포함하지 않는다.
