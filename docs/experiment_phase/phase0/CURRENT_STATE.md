# Phase 0 현재 상태

- 감사 시각: 2026-07-11 UTC
- 프로젝트 repository root: `.` / `/workspace/NetLLM`
- NetLLM 원본 repository root: upstream 기준 `.` / `/workspace/NetLLM-source`
- 조사 원칙: 읽기 전용 상태 확인만 수행했으며 학습, package 설치, dataset/model 다운로드는 수행하지 않았다.

## 1. 현재 완료된 작업

- 필수 참고자료 4개를 확인했다.
  - 프로젝트 기준 `docs/논문리딩4NetLLM.pdf` / `/workspace/NetLLM/docs/논문리딩4NetLLM.pdf`
  - 프로젝트 기준 `docs/liteVLM.pdf` / `/workspace/NetLLM/docs/liteVLM.pdf`
  - 프로젝트 기준 `docs/RESEARCH_DIRECTION.md` / `/workspace/NetLLM/docs/RESEARCH_DIRECTION.md`
  - 프로젝트 기준 `docs/MEETING_NOTES.md` / `/workspace/NetLLM/docs/MEETING_NOTES.md`
- 프로젝트 Git 상태와 별도 NetLLM 원본 Git 상태를 확인했다.
- `setup.sh`, `requirements-vp.txt`, `README.md`, `scripts/`의 자동화 범위를 읽었다.
- 현재 VP entry point, dataset loader, baseline, PLM loader, multimodal path, networking head, loss/metric/result 경로를 소스 수준에서 추적했다.
- cooked Jin2022/Wu2017 dataset의 디렉터리·파일 수와 대표 CSV 구조를 확인했다.
- `/venv/vp_netllm`의 dependency version과 요청된 import를 검사했다.
- 실제 Jin2022 한 trace를 읽어 `ViewportDataset` item 및 기본 `DataLoader` batch shape를 확인했다. 모델 forward, baseline test, PLM forward는 실행하지 않았다.

## 2. 아직 검증되지 않은 작업

- VP regression/velocity/TRACK baseline의 end-to-end 실행
- VP PLM adaptation 또는 inference
- VP `--multimodal` 실행
- pretrained PLM, VP checkpoint 및 TRACK checkpoint load
- precomputed image feature의 실제 tensor shape와 key coverage
- PLM 내부 hidden state와 GPU 상의 실제 runtime tensor shape
- checkpoint 저장·복구와 result CSV의 재현성
- 논문 성능 재현
- LiteVLM 모듈의 연결 또는 성능 측정

## 3. repository 구조

### 프로젝트 wrapper

- 프로젝트 기준 `setup.sh` / `/workspace/NetLLM/setup.sh`
- 프로젝트 기준 `requirements-vp.txt` / `/workspace/NetLLM/requirements-vp.txt`
- 프로젝트 기준 `scripts/` / `/workspace/NetLLM/scripts`
- 프로젝트 기준 `docs/` / `/workspace/NetLLM/docs`
- source code는 포함하지 않고 별도 원본 clone을 준비·호출하는 wrapper repository다.

### NetLLM 원본

- upstream 기준 `viewport_prediction/` / `/workspace/NetLLM-source/viewport_prediction`
- upstream 기준 `adaptive_bitrate_streaming/` / `/workspace/NetLLM-source/adaptive_bitrate_streaming`
- upstream 기준 `cluster_job_scheduling/` / `/workspace/NetLLM-source/cluster_job_scheduling`
- VP의 cooked viewport data는 upstream 기준 `viewport_prediction/data/viewports/` / `/workspace/NetLLM-source/viewport_prediction/data/viewports`에 Git tracked 상태로 포함되어 있다.

## 4. setup 및 자동화 범위

### 이미 자동화된 것

- 환경/GPU/CUDA/disk/memory 기록
- `/venv/vp_netllm` 생성 또는 재사용과 Python 3.8.10 선택
- PyTorch 2.1.0+cu118 및 `requirements-vp.txt` 설치
- 원본 `https://github.com/duowuyms/NetLLM.git` clone과 commit `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` checkout
- `torch`, `cv2`, `yacs`의 제한된 import 확인
- 원본 VP 파일 목록 미리보기

### 부분적으로 자동화된 것

- `scripts/check_env.sh`는 Python/pip, torch/CUDA, `cv2`, `yacs`, 원본 commit만 확인한다. `transformers`, `peft`, `accelerate` 등 실제 PLM import chain은 확인하지 않는다.
- `scripts/run_vp_regression_cpu.sh`는 원본 command와 log 저장을 묶지만 원본 디렉터리 안에 `logs/` 및 `data/results/`를 생성한다.
- `scripts/run_vp_gpt2_adapt_e1.sh`는 GPT-2 adaptation을 background로 시작하지만 pretrained GPT-2 존재 여부와 전체 import chain을 사전 검사하지 않는다.

### 자동화되지 않은 것

- raw image, saliency map, precomputed image feature 준비
- pretrained GPT-2/Llama 등 PLM 준비
- VP/TRACK/NetLLM checkpoint 준비
- data/checkpoint checksum 또는 schema 검증
- 원본 밖의 writable result/checkpoint/log 경로로 출력 redirect
- dependency 간 runtime compatibility 검증

### 실행 여부가 검증되지 않은 것

- 두 run script 모두 이번 Phase 0에서 실행하지 않았다.
- 과거 `setup.sh` 실행 log는 존재하지만, 현재 `peft`/`accelerate` import 실패 때문에 “PLM 환경 전체가 정상”이라고 결론낼 수 없다.

## 5. 데이터 및 checkpoint 상태

| 항목 | 상태 | 근거 |
|---|---|---|
| cooked Jin2022 | Present and structurally verified | 27 video, video당 84 user CSV, 총 2,268개; 대표 파일 4열 |
| cooked Wu2017 | Present and structurally verified | 18 video, video당 48 user CSV, 총 864개; 대표 파일 4열 |
| raw viewport dataset | Path referenced by code but missing | `dataset/preprocess.py`의 `/data2/...` 절대경로가 없음 |
| raw image/saliency map | Missing | `viewport_prediction/data/images` 자체가 없음 |
| precomputed image feature | Path referenced by code but missing | `config.py`의 `data/images/*images/features`가 없음 |
| pretrained PLM | Path referenced by code but missing | 실제 code 기준 `/workspace/NetLLM-source/downloaded_plms`가 없음 |
| VP/NetLLM checkpoint | Missing | `data/ft_plms`가 없음 |
| TRACK checkpoint | Missing | `data/models`가 없음 |
| VP result/log | Missing | `data/results`, VP `logs`가 없음 |

## 6. 현재 연구 구현 진행도

- Phase 0 repository audit만 완료했다.
- 신규 selector, Patch/Frame Selection, Token Selection, speculative decoding 코드는 0%이며 생성하지 않았다.
- 최초 실제 실행 대상은 여전히 VP regression baseline이다.
- PLM 및 multimodal 단계는 현재 dependency, PLM, image feature blocker가 해소되기 전에는 진행할 수 없다.

