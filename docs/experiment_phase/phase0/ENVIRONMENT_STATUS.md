# Phase 0 환경 상태

## 1. OS, GPU, CUDA

| 항목 | 확인 결과 |
|---|---|
| 실행 형태 | unprivileged Docker container |
| OS | Ubuntu 24.04.4 LTS (Noble) |
| kernel | Linux 6.8.0-107-generic, x86_64 |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB (`torch` 보고 25,252,724,736 bytes) |
| GPU 수 | 1 |
| NVIDIA driver | 595.58.03 |
| `nvidia-smi` CUDA compatibility | 13.2 |
| `nvcc` | CUDA 13.0, V13.0.88 |
| PyTorch CUDA runtime | 11.8 |
| `torch.cuda.is_available()` | `True` |

`nvidia-smi`의 CUDA 13.2, system toolkit 13.0, PyTorch wheel의 cu118은 서로 다른 항목이다. 이번 조사에서는 GPU availability만 확인했으며 실제 model kernel 실행은 하지 않았다.

## 2. conda 및 Python

- 조사 shell에는 `CONDA_DEFAULT_ENV`, `CONDA_PREFIX`, `VIRTUAL_ENV`가 설정되어 있지 않았다. 즉 자동 활성화된 conda/venv는 없었다.
- conda executable: `/opt/miniforge3/condabin/conda`
- 존재하는 환경: `/opt/miniforge3`, `/venv/main`, `/venv/vp_netllm`
- VP 대상 환경: `/venv/vp_netllm`
- Python: 3.8.10
- Python executable: `/venv/vp_netllm/bin/python`
- pip: 25.0.1
- pip invocation path: `/venv/vp_netllm/lib/python3.8/site-packages/pip`

모든 Python 검사는 환경 활성화 대신 `/venv/vp_netllm/bin/python` 절대경로로 수행했다.

## 3. 요청 dependency import 결과

| module | 설치 version | import 결과 |
|---|---:|---|
| `torch` | 2.1.0+cu118 | OK |
| `cv2` | 4.8.1 | OK |
| `yacs` | 0.1.8 | OK |
| `transformers` | 4.34.1 | OK |
| `peft` | 0.6.2 | 실패 |
| `numpy` | 1.24.4 | OK |
| `pandas` | 2.0.3 | OK |
| `scipy` | 1.10.1 | OK |
| `sklearn` | 1.3.2 | OK |
| `accelerate` | 0.32.1 | 실패 |
| `einops` | 0.8.1 | OK |

`peft`와 `accelerate`는 설치되어 있지만 다음 symbol을 현재 `huggingface-hub==0.17.3`에서 import하지 못한다.

```text
cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub'
```

그 결과 upstream 기준 `viewport_prediction/run_plm.py` / `/workspace/NetLLM-source/viewport_prediction/run_plm.py` import도 실패한다. `pip check`는 `No broken requirements found`를 반환했지만 실제 runtime import는 실패하므로 `pip check`만으로 정상 상태라고 볼 수 없다.

## 4. VP 관련 추가 import

| 항목 | 결과 | 영향 |
|---|---|---|
| `run_baseline` | OK | regression/velocity entry import 가능 |
| `models.pipeline` | OK | current VP pipeline module 자체는 import 가능 |
| `run_plm` | 실패 | PLM adaptation/test 진입 불가 |
| `torchvision` | 미설치 | ViT feature 및 saliency 추출 script import 불가 |
| `dataset.extract_features` | 실패 | `torchvision` 누락 |
| `dataset.extract_saliency` | 실패 | `torchvision` 누락 |
| `PIL` | 10.4.0, OK | 단독 import 가능 |
| `matplotlib` | 3.7.5, OK | 단독 import 가능 |
| `prettytable` | 3.11.0, OK | result utility import 가능 |

## 5. 환경 변경 없이 확인된 범위

- `pip install`, `conda install`, `apt install`, package upgrade/downgrade를 실행하지 않았다.
- CUDA sample, model forward, training, inference를 실행하지 않았다.
- environment issue의 원인 후보는 설치된 distribution metadata와 실제 import error까지만 확인했다.
- 정확한 dependency 조합 변경은 Phase 1 전 사용자 승인이 필요한 환경 변경 사항이다.

