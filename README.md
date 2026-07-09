# NetLLM VP Vast.ai setup

이 저장소는 NetLLM 원본 코드를 직접 수정하지 않고, Vast.ai 인스턴스를 Destroy한 뒤에도 `git clone -> bash setup.sh`로 viewport prediction 실험 환경을 다시 만드는 wrapper repo입니다.

## Vast.ai 권장 환경

- Template: `vastai/pytorch_cuda-12.9.2-auto/jupyter`
- GPU: RTX 4090 24GB 1장
- 이전 확인 환경:
  - Driver `575.57.08`
  - CUDA driver API `12.9`
  - `nvcc` CUDA `12.8`
  - conda: `/opt/miniforge3/condabin/conda`

## 설치

새 인스턴스에서 setup repo를 clone한 뒤 실행합니다. setup repo는 `/workspace/NetLLM`에 둬도 됩니다. `setup.sh`가 원본 NetLLM 코드는 `/workspace/NetLLM-source`에 따로 clone합니다.

```bash
cd /workspace
git clone <YOUR_SETUP_REPO_URL> NetLLM
cd NetLLM
bash setup.sh
```

설치 스크립트는 다음을 수행합니다.

- `/workspace/research_logs` 생성
- Vast.ai/GPU/CUDA/디스크 정보를 `00_vast_env_check.txt`에 기록
- Codex CLI/Jupyter/VS Code Remote 관련 상태를 확인하고 `02_dev_tools_check.txt`에 기록
- conda env `/venv/vp_netllm` 생성 및 Python `3.8.10` 설치
- `torch==2.1.0` cu118 설치
- `requirements-vp.txt` 설치
- `https://github.com/duowuyms/NetLLM.git`를 `/workspace/NetLLM-source`에 clone
- commit `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` checkout
- torch/cuda, `cv2`, `yacs` import 확인

기본값으로 Codex CLI 설치를 시도합니다. 개발 도구 확인/설치를 건너뛰려면 다음처럼 실행합니다.

```bash
INSTALL_DEV_TOOLS=0 bash setup.sh
```

Codex CLI 설치만 건너뛰려면 다음처럼 실행합니다.

```bash
INSTALL_CODEX_CLI=0 bash setup.sh
```

Codex CLI가 설치되어도 로그인/인증은 새 인스턴스에서 다시 해야 할 수 있습니다. VS Code Remote Server는 보통 VS Code로 새 SSH 접속을 할 때 자동으로 다시 설치됩니다.

## 환경 확인

```bash
bash scripts/check_env.sh
```

Python, pip, torch, CUDA 사용 가능 여부, GPU 이름, `cv2`, `yacs`, NetLLM commit, VP 디렉토리 파일 목록을 출력합니다.

## CPU baseline

```bash
bash scripts/run_vp_regression_cpu.sh
```

`/workspace/NetLLM-source/viewport_prediction`에서 Jin2022 regression CPU test를 실행하고 로그를 `logs/` 아래에 저장합니다.

주의: clone 직후 VP data 디렉토리가 비어 있거나 데이터셋이 없으면 baseline 실행은 실패할 수 있습니다. 이건 숨길 문제가 아니라 데이터 준비가 필요한 정상적인 실패입니다.

## GPT-2 adaptation 1 epoch

```bash
bash scripts/run_vp_gpt2_adapt_e1.sh
```

백그라운드에서 GPT-2 base adaptation을 1 epoch 실행하고 PID를 `logs/vp_gpt2_adapt.pid`에 저장합니다. 스크립트는 `tail -f`를 자동 실행하지 않습니다.

모니터링:

```bash
tail -f /workspace/NetLLM-source/viewport_prediction/logs/<LOG_FILE>
watch -n 2 nvidia-smi
```

GPT-2 실행에는 NetLLM이 기대하는 PLM 경로가 필요합니다. 기본 코드 기준으로 `/workspace/downloaded_plms/gpt2/base`에 모델 파일이 있어야 합니다.

## 이전에 확인한 에러

- `ModuleNotFoundError: No module named 'cv2'`
  - `opencv-python-headless==4.8.1.78`로 해결
- `ModuleNotFoundError: No module named 'yacs'`
  - `yacs==0.1.8`로 해결

## Vast.ai 운영 메모

`Destroy`는 `/workspace`, conda env, NetLLM clone, 로그, 데이터, 체크포인트를 삭제합니다. 이 wrapper repo의 `setup.sh`, `requirements-vp.txt`, `scripts/`, `README.md`는 반드시 GitHub에 push해두세요.

`Stop`은 디스크를 보존하므로 환경은 남지만, storage 비용은 계속 붙을 수 있습니다. 며칠 뒤 다시 시작할 계획이면 비용과 재설치 시간을 비교해서 `Stop` 또는 `Destroy + git clone + bash setup.sh` 중 선택하면 됩니다.
