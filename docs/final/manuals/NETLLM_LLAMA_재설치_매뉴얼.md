# NetLLM Llama2-7B 단일 스크립트 재설치 매뉴얼

## 1. 목적

`setup_netllm_llama.sh` 하나로 새 Vast.ai 인스턴스의 Llama 재현 환경을 복구한다.

자동 수행 범위는 다음과 같다.

1. 사용자 NetLLM Git 저장소 확인 또는 clone
2. 원본 NetLLM source 확인 또는 고정 commit으로 clone
3. Hugging Face CLI 설치
4. Hugging Face 인증 확인
5. `meta-llama/Llama-2-7b-hf` 다운로드 또는 기존 파일 동기화
6. 모델 필수 파일 및 SHA256 검증
7. NetLLM이 기대하는 모델 경로에 심볼릭 링크 생성
8. GPT-2 환경과 분리된 Llama 전용 Conda 환경 생성
9. PyTorch 2.2.0/CUDA 12.1 및 NetLLM 의존성 설치
10. 선택적으로 팀 checkpoint와 VP dataset 다운로드·압축 해제
11. Llama2-7B local-only GPU smoke test
12. 모델 revision, checksum, 환경 manifest, 한국어 결과 문서 생성

다음 작업은 자동화할 수 없다.

- Hugging Face 웹페이지에서 최초 Llama2 사용 약관 동의 및 접근권한 신청
- 접근권한이 없는 계정을 자동 승인하는 작업
- 비공개 Google Drive 파일의 로그인·권한 승인
- fine-tuned checkpoint와 dataset이 정확한 원본인지 사람 대신 판정하는 작업

Hugging Face 접근권한은 계정에 남는다. 인스턴스를 폐기하더라도 접근권한을 다시 신청할 필요는 없지만, 새 서버에서 같은 계정으로 다시 로그인해야 한다.

---

## 2. 제공 파일

- `setup_netllm_llama.sh`: 통합 설치 스크립트
- `NETLLM_LLAMA_재설치_매뉴얼.md`: 현재 문서

권장 저장 위치:

```text
/root/NetLLM/scripts/setup_netllm_llama.sh
/root/NetLLM/docs/setup/NETLLM_LLAMA_재설치_매뉴얼.md
```

---

## 3. 현재 인스턴스에서 최초 등록

현재 파일을 VS Code Remote-SSH로 Vast.ai 서버에 업로드한 뒤 실행한다.

```bash
cd /root/NetLLM
mkdir -p scripts docs/setup

cp /업로드한/경로/setup_netllm_llama.sh \
  scripts/setup_netllm_llama.sh

cp /업로드한/경로/NETLLM_LLAMA_재설치_매뉴얼.md \
  docs/setup/NETLLM_LLAMA_재설치_매뉴얼.md

chmod +x scripts/setup_netllm_llama.sh
bash -n scripts/setup_netllm_llama.sh
```

`bash -n`이 아무 출력 없이 종료되면 셸 문법 검사를 통과한 것이다.

Git에는 스크립트와 매뉴얼만 먼저 등록한다.

```bash
git add \
  scripts/setup_netllm_llama.sh \
  docs/setup/NETLLM_LLAMA_재설치_매뉴얼.md

git status --short
git commit -m "Add reproducible NetLLM Llama setup script"
git push
```

`git add .`는 사용하지 않는다.

---

## 4. 현재 인스턴스에서 실행

현재는 Llama2-7B base가 이미 `/root/NetLLM-assets/llama/base`에 있으므로 다음 명령만 실행하면 된다.

```bash
cd /root/NetLLM
bash scripts/setup_netllm_llama.sh
```

스크립트는 `hf download --local-dir`를 다시 호출하지만 기존 파일과 다운로드 metadata를 재사용한다. 정상 파일을 무조건 처음부터 다시 받는 방식이 아니다.

실행 중 Hugging Face 로그인이 필요하면 브라우저 인증 URL과 코드가 표시된다. 내 PC 브라우저에서 같은 Hugging Face 계정으로 승인한다.

현재 단계에서 팀 checkpoint와 dataset URL이 준비되지 않았다면 다음 경고는 정상이다.

```text
CHECKPOINT_URL이 없어 팀 checkpoint 다운로드를 생략함
DATASET_URL이 없어 팀 dataset 다운로드를 생략함
```

이 경우에도 다음 작업은 완료된다.

- Llama base 검증
- 전용 Conda 환경 구성
- NetLLM 경로 연결
- local-only smoke test
- 재현 manifest 생성

---

## 5. checkpoint와 dataset까지 한 번에 설치

팀의 Google Drive 공유 링크가 준비됐다면 실행 전에 환경변수를 설정한다.

```bash
export CHECKPOINT_URL='try_llama2_7b.zip의 실제 Google Drive 공유 링크'
export DATASET_URL='data.zip의 실제 Google Drive 공유 링크'

cd /root/NetLLM
bash scripts/setup_netllm_llama.sh
```

다운로드한 ZIP의 공식 checksum을 알고 있다면 함께 설정한다.

```bash
export CHECKPOINT_SHA256='64자리 SHA256 값'
export DATASET_SHA256='64자리 SHA256 값'
```

스크립트는 다음 경로를 사용한다.

```text
/root/NetLLM-assets/staging/try_llama2_7b.zip
/root/NetLLM-assets/staging/data.zip
/root/NetLLM-assets/checkpoints/try_llama2_7b
/root/NetLLM-assets/datasets/team_data
```

공유 링크에 로그인이나 조직 권한이 필요하면 `gdown`만으로 다운로드되지 않을 수 있다. 그 경우 ZIP을 직접 `staging` 폴더에 업로드한 뒤 스크립트를 다시 실행한다.

---

## 6. 새 Vast.ai 인스턴스에서 복구

### 6.1 권장 인스턴스 조건

- PyTorch 2.2.0 / CUDA 12.1 devel 계열 템플릿
- Conda 사용 가능
- Llama inference용 충분한 GPU memory
- LoRA 학습을 수행한다면 30GB 미만 GPU에서 자동 시작하지 않음
- 모델·checkpoint·dataset을 고려한 충분한 storage

### 6.2 Git clone

```bash
cd /root
git clone https://github.com/soyuniverse/NetLLM.git
cd /root/NetLLM
```

SSH 연결이 준비된 경우 기존 SSH URL을 사용해도 된다.

### 6.3 스크립트 실행

```bash
chmod +x scripts/setup_netllm_llama.sh
bash scripts/setup_netllm_llama.sh
```

스크립트가 처리하는 순서는 다음과 같다.

```text
Git 확인
→ Hugging Face CLI 확인
→ 계정 로그인
→ gated model 접근 확인
→ Llama2-7B 다운로드
→ SHA256 검증
→ Conda 환경 생성
→ dependency 설치
→ 경로 연결
→ GPU smoke test
→ 결과 문서 생성
```

즉, 새 인스턴스에서 과거의 Hugging Face 설치 명령을 하나씩 다시 입력할 필요가 없다. 최초 웹 접근 승인만 계정에 유지되어 있으면 된다.

---

## 7. 비대화식 실행

브라우저 로그인을 사용할 수 없는 자동화 상황에서는 `HF_TOKEN`을 셸 환경변수로 전달한다.

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN
echo
export HF_TOKEN

bash /root/NetLLM/scripts/setup_netllm_llama.sh

unset HF_TOKEN
```

토큰을 다음 위치에 기록하지 않는다.

- Git tracked 파일
- Markdown 문서
- `.env` 파일을 Git에 포함
- Codex 프롬프트
- 셸 명령 인자로 직접 노출

Vast.ai template의 secret 기능을 사용할 수 있다면 `HF_TOKEN`을 secret 환경변수로 주입하는 방식이 더 적합하다.

---

## 8. 주요 설정값

환경변수로 기본값을 변경할 수 있다.

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `NETLLM_REPO` | `/root/NetLLM` | 사용자 프로젝트 |
| `UPSTREAM_REPO` | `/root/NetLLM-source` | 원본 NetLLM source |
| `NETLLM_ASSETS` | `/root/NetLLM-assets` | Git 외부 자산 root |
| `LLAMA_BASE` | `/root/NetLLM-assets/llama/base` | Llama base 모델 |
| `LLAMA_ENV` | `/root/venvs/vp_netllm_llama` | Llama 전용 Conda 환경 |
| `MODEL_ID` | `meta-llama/Llama-2-7b-hf` | Hugging Face model ID |
| `MODEL_REVISION` | lock 파일 또는 원격 SHA | 모델 revision |
| `RUN_SMOKE_TEST` | `1` | base model GPU smoke test |
| `INSTALL_TEAM_ASSETS` | `1` | 팀 자산 처리 활성화 |
| `CHECKPOINT_URL` | 빈 값 | checkpoint Drive 링크 |
| `DATASET_URL` | 빈 값 | dataset Drive 링크 |
| `MIN_FREE_GB` | `25` | 모델 미설치 시 최소 여유 공간 |

예를 들어 smoke test를 나중으로 미루려면 다음처럼 실행한다.

```bash
RUN_SMOKE_TEST=0 bash scripts/setup_netllm_llama.sh
```

다른 asset root를 사용하려면 다음처럼 실행한다.

```bash
NETLLM_ASSETS=/data/NetLLM-assets \
LLAMA_BASE=/data/NetLLM-assets/llama/base \
bash scripts/setup_netllm_llama.sh
```

Vast Volume을 `/data`에 연결한 경우 이 방법을 사용할 수 있다.

---

## 9. 재실행 동작

스크립트는 재실행 가능한 형태로 작성됐다.

기존 항목이 있으면 다음처럼 동작한다.

| 기존 상태 | 재실행 동작 |
|---|---|
| Git 저장소 존재 | clone하지 않고 유지 |
| Llama 파일 존재 | local-dir metadata 기반 동기화·검증 |
| Conda 환경 존재 | 삭제하지 않고 고정 package 재검증 |
| checkpoint ZIP 존재 | 다시 받지 않고 재사용 |
| dataset ZIP 존재 | 다시 받지 않고 재사용 |
| 압축 해제 대상 폴더가 비어 있지 않음 | 자동 덮어쓰지 않고 경고 |
| checksum lock 존재 | 현재 모델과 비교 후 불일치 시 중단 |

checkpoint나 dataset을 강제로 교체해야 한다면 기존 폴더를 임의로 덮어쓰지 말고 별도 백업 후 정리한다.

---

## 10. 성공 판정

실행 마지막에 다음 문구가 나와야 한다.

```text
PASS: Llama2-7B local-only load and inference
설치 완료
```

생성되는 주요 산출물:

```text
/root/NetLLM/docs/experiment_phase/llama/manifests/
├── llama2-7b-hf.lock
└── llama2-7b-hf.sha256

/root/NetLLM/experiments/vp/llama_repro/
├── environment_manifest.json
└── llama_environment_freeze.txt

/root/NetLLM/docs/experiment_phase/llama/
└── LLAMA_ENVIRONMENT_RESULT_KO.md
```

NetLLM 모델 호환 경로:

```text
/root/downloaded_plms/llama/base
→ /root/NetLLM-assets/llama/base
```

확인 명령:

```bash
readlink -f /root/downloaded_plms/llama/base

/root/venvs/vp_netllm_llama/bin/python - <<'PY'
import torch
import transformers
import peft
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(transformers.__version__)
print(peft.__version__)
PY
```

기대값:

```text
PyTorch 2.2.0+cu121
CUDA available=True
Transformers 4.34.1
PEFT 0.6.2
```

---

## 11. Git에 저장할 것과 저장하지 않을 것

### Git에 저장

```text
scripts/setup_netllm_llama.sh
docs/setup/NETLLM_LLAMA_재설치_매뉴얼.md
모델 revision lock
모델 SHA256 manifest
Python freeze 목록
환경 manifest
한국어 결과 문서
실험 설정과 결과
```

### Git에 저장하지 않음

```text
/root/NetLLM-assets/llama/base
/root/NetLLM-assets/checkpoints
/root/NetLLM-assets/datasets
/root/NetLLM-assets/staging
/root/venvs
Hugging Face token
Google Drive 인증정보
```

실행 후 다음 명령으로 점검한다.

```bash
cd /root/NetLLM
git status --short
find /root/NetLLM -type f -size +100M -print
```

100MB가 넘는 모델·dataset 파일이 Git repository 아래에 나타나면 commit하지 않는다.

---

## 12. 인스턴스 폐기 전 확인

Vast.ai 인스턴스를 Destroy하면 일반 container storage의 `/root` 파일은 사라진다. 따라서 다음 세 조건을 충족한 뒤 폐기한다.

1. 코드, script, manifest, 결과 문서를 GitHub에 push했다.
2. 팀 checkpoint와 dataset의 원본이 Google Drive·외장 디스크·object storage 등에 남아 있다.
3. 모델은 Hugging Face에서 같은 revision으로 재다운로드할 수 있다.

확인 명령:

```bash
cd /root/NetLLM

git status --short
git log -1 --oneline

git ls-remote origin HEAD

cat docs/experiment_phase/llama/manifests/llama2-7b-hf.lock
sha256sum \
  docs/experiment_phase/llama/manifests/llama2-7b-hf.sha256
```

작업 중인 변경이 남아 있으면 Destroy하지 않는다.

---

## 13. 현재 프로젝트의 다음 순서

이 스크립트는 환경과 자산 배치까지 담당한다. 설치 성공 후의 실험 Gate는 다음 순서로 유지한다.

1. checkpoint 전체성 확인
   - `adapter_config.json`
   - `adapter_model.safetensors` 또는 `adapter_model.bin`
   - `modules_except_plm.bin`
2. 정확한 원본 VP dataset 구조 확인
   - `Jin2022images`
   - `saliencyMap`
   - `features`
3. 실제 Jin2022 sample 1개 Llama baseline smoke test
4. 원본 Llama baseline metric 검증
5. Llama Identity equivalence
6. Recent-K 8/6/4/2 benchmark
7. MAE, RMSE, Loss, latency, GPU memory 시각화
8. 모든 benchmark 성공 후 continuous draft-and-verify prototype

GPT-2 Phase 3A 환경과 기존 결과는 이 스크립트가 수정하지 않는다.
