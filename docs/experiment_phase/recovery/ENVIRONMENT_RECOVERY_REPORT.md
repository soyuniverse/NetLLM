# NetLLM VP 환경 복구 보고서

- 확인 시각: 2026-07-26T07:11:28Z
- 결과: 성공
- hostname: `29d90f362497`
- 작업 디렉터리: `/root/NetLLM`

## 실제 사용 경로

| 항목 | 경로 |
|---|---|
| project repository | `/root/NetLLM` |
| upstream NetLLM source | `/root/NetLLM-source` |
| Python environment | `/root/venvs/vp_netllm_repro` |
| Python executable | `/root/venvs/vp_netllm_repro/bin/python` |
| GPT-2 Medium artifact | `/root/NetLLM-artifacts/plms/gpt2/base` |
| Hugging Face cache | `/root/NetLLM-artifacts/hf_cache` |
| Jin2022 cooked dataset | `/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022` |
| Phase 3A extension source | `/root/NetLLM/src/netllm_litevlm` |

`/workspace` 또는 `/venv`를 가정하지 않고 `/root`, `/workspace`, `/venv`, `/opt`를
탐색했다. 기존 compatible environment, upstream clone 및 artifact가 없어서 서로 독립된
신규 `/root` 경로에 복구했다. 기존 경로나 runtime은 삭제 또는 덮어쓰지 않았다.

## Python, PyTorch 및 CUDA

| 항목 | 값 |
|---|---|
| Python | `3.8.10` |
| torch | `2.1.0+cu118` |
| PyTorch CUDA runtime | `11.8` |
| CUDA toolkit (`nvcc`) | `12.1` (`V12.1.105`) |
| CUDA available | `True` |
| GPU | `NVIDIA GeForce RTX 4090` |
| GPU memory | `24564 MiB` |
| NVIDIA driver | `560.35.03` |
| transformers | `4.34.1` |
| peft | `0.6.2` |
| accelerate | `0.24.1` |
| huggingface-hub | `0.17.3` |
| pip check | 성공 (`No broken requirements found.`) |

현재 checkout의 과거 environment 생성 script는 cu121 조합을 담고 있어 실행하지 않았다.
사용자 지정 exact cu118 조합과 `requirements-vp.txt`의 package를 신규 prefix에만 설치했다.
기존 기본 환경인 `/opt/conda`의 Python 3.10 / torch 2.2 cu121은 VP 재현 실행에 사용하지
않았다.

## Upstream source와 dataset

- origin: `https://github.com/duowuyms/NetLLM.git`
- commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- checkout: detached
- 작업 전 status/diff: clean
- 작업 후 status/diff: clean
- 작업 전/후 upstream `__pycache__`: `0`
- Git-tracked Jin2022 cooked CSV: `2268`개
- R1 sample CSV:
  `viewport_prediction/data/viewports/Jin2022/video4/5Hz/simple_5Hz_user83.csv`

Upstream source는 clone 및 고정 commit checkout 이후 read-only로 사용했으며 code, dataset,
configuration을 수정하지 않았다.

## GPT-2 Medium artifact

- repository: `openai-community/gpt2-medium`
- revision: `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`
- artifact file 수: `9`
- total size: `1,522,851,760 bytes`
- relative-name content manifest SHA-256:
  `f3fdcf85dd2a8d38b329048ebb0349bcc94e1c6a04aa08ec20a4c0334ed74f14`

파일별 SHA-256는 기존
`docs/experiment_phase/phase2a/GPT2_ARTIFACT_MANIFEST.md`의 9개 값과 모두 일치한다.
R1/R2 실행 전/후 content manifest checksum도 동일하다.

## 환경 무결성

- raw `pip freeze` SHA-256:
  `731a5031a3fb94909d541db8b66a41299d0977716e09d26eaa682ec3154d0311`
- sorted `pip freeze` SHA-256:
  `1ca86448bc2eb262ce840f4801eea898234f40cbd3384451bf640c56c590947f`
- R1/R2 실행 전후 hash: 동일
- package install/upgrade/downgrade after environment recovery: 없음
- upstream commit/status/diff/`__pycache__`: 전후 동일
- artifact checksum: 전후 동일

## 범위 제한

Llama2, LoRA, training, adaptation, full benchmark 및 speculative decoding은 실행하지 않았다.
GPT-2는 Phase 3A equivalence forward에만 사용했다.
