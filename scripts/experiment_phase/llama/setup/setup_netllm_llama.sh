#!/usr/bin/env bash
# NetLLM Llama2-7B 재현 환경 통합 설치 스크립트
# 목적: 새 Vast.ai 인스턴스에서도 코드, Llama2 base model, 전용 Conda 환경,
#       경로 연결, checksum, smoke test를 한 번에 복구한다.
# 주의: Hugging Face의 Llama2 접근 승인은 브라우저에서 최초 1회 별도로 받아야 한다.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

########################################
# 0. 사용자 설정값
########################################
NETLLM_REPO="${NETLLM_REPO:-/root/NetLLM}"
NETLLM_GIT_URL="${NETLLM_GIT_URL:-https://github.com/soyuniverse/NetLLM.git}"
CLONE_REPO_IF_MISSING="${CLONE_REPO_IF_MISSING:-1}"

UPSTREAM_REPO="${UPSTREAM_REPO:-/root/NetLLM-source}"
UPSTREAM_GIT_URL="${UPSTREAM_GIT_URL:-https://github.com/duowuyms/NetLLM.git}"
UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-105bcf070f2bec808f7b14f8f5a953de6e4e6e54}"
CLONE_UPSTREAM_IF_MISSING="${CLONE_UPSTREAM_IF_MISSING:-1}"

NETLLM_ASSETS="${NETLLM_ASSETS:-/root/NetLLM-assets}"
LLAMA_BASE="${LLAMA_BASE:-${NETLLM_ASSETS}/llama/base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${NETLLM_ASSETS}/checkpoints/try_llama2_7b}"
DATASET_DIR="${DATASET_DIR:-${NETLLM_ASSETS}/datasets/team_data}"
STAGING_DIR="${STAGING_DIR:-${NETLLM_ASSETS}/staging}"
LOG_DIR="${LOG_DIR:-${NETLLM_ASSETS}/logs}"

LLAMA_ENV="${LLAMA_ENV:-/root/venvs/vp_netllm_llama}"
HF_CLI_PATH="${HF_CLI_PATH:-}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
MODEL_REVISION="${MODEL_REVISION:-}"

# 선택 자산: 값이 비어 있으면 다운로드하지 않고 안내만 한다.
CHECKPOINT_URL="${CHECKPOINT_URL:-}"
DATASET_URL="${DATASET_URL:-}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-}"
DATASET_SHA256="${DATASET_SHA256:-}"

RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
INSTALL_TEAM_ASSETS="${INSTALL_TEAM_ASSETS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT

# 실행 환경 고정 버전
PYTHON_VERSION="${PYTHON_VERSION:-3.8.10}"
TORCH_VERSION="${TORCH_VERSION:-2.2.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.17.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.2.0}"
NUMPY_VERSION="${NUMPY_VERSION:-1.24.4}"
MUNCH_VERSION="${MUNCH_VERSION:-4.0.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.34.1}"
PEFT_VERSION="${PEFT_VERSION:-0.6.2}"
ACCELERATE_VERSION="${ACCELERATE_VERSION:-0.24.1}"
HF_HUB_RUNTIME_VERSION="${HF_HUB_RUNTIME_VERSION:-0.17.3}"
SENTENCEPIECE_VERSION="${SENTENCEPIECE_VERSION:-0.1.99}"
SAFETENSORS_VERSION="${SAFETENSORS_VERSION:-0.4.1}"
GDOWN_VERSION="${GDOWN_VERSION:-5.2.0}"

########################################
# 1. 공통 함수
########################################
now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(now)" "$*"; }
warn() { printf '[%s] 경고: %s\n' "$(now)" "$*" >&2; }
die() { printf '[%s] 오류: %s\n' "$(now)" "$*" >&2; exit 1; }

on_error() {
    local rc=$?
    local line=${1:-unknown}
    printf '\n[%s] 설치 실패: line=%s, exit_code=%s\n' "$(now)" "$line" "$rc" >&2
    printf '로그를 확인하십시오: %s\n' "${SETUP_LOG:-아직 생성되지 않음}" >&2
    exit "$rc"
}
trap 'on_error $LINENO' ERR

command_exists() { command -v "$1" >/dev/null 2>&1; }

assert_safe_path() {
    local path=$1
    local label=$2
    [[ -n "$path" ]] || die "$label 경로가 비어 있음"
    case "$path" in
        /|/root|/home|/usr|/opt|/var|/mnt)
            die "$label 경로가 너무 광범위하여 중단함: $path"
            ;;
    esac
}

run_as_root_apt() {
    if ! command_exists apt-get; then
        return 0
    fi
    local missing=()
    local cmd
    for cmd in git curl unzip sha256sum; do
        command_exists "$cmd" || missing+=("$cmd")
    done
    command_exists ca-certificates || true
    if ((${#missing[@]} > 0)); then
        log "필수 시스템 도구 설치: ${missing[*]}"
        apt-get update -y
        DEBIAN_FRONTEND=noninteractive apt-get install -y \
            git curl unzip ca-certificates coreutils
    fi
}

find_conda() {
    local candidate
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        printf '%s\n' "$CONDA_EXE"
        return 0
    fi
    if command_exists conda; then
        command -v conda
        return 0
    fi
    for candidate in \
        /opt/conda/bin/conda \
        /root/miniconda3/bin/conda \
        /root/anaconda3/bin/conda \
        /usr/local/conda/bin/conda; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

free_gb_at() {
    local path=$1
    df -Pk "$path" | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

validate_sha256_if_given() {
    local file=$1
    local expected=$2
    local label=$3
    [[ -z "$expected" ]] && return 0
    local actual
    actual="$(sha256_file "$file")"
    [[ "$actual" == "$expected" ]] || die "$label SHA256 불일치: expected=$expected actual=$actual"
    log "$label SHA256 일치"
}

########################################
# 2. 디렉터리와 로그 준비
########################################
assert_safe_path "$NETLLM_REPO" "NETLLM_REPO"
assert_safe_path "$UPSTREAM_REPO" "UPSTREAM_REPO"
assert_safe_path "$NETLLM_ASSETS" "NETLLM_ASSETS"
assert_safe_path "$LLAMA_ENV" "LLAMA_ENV"

mkdir -p \
    "$NETLLM_ASSETS" \
    "$LLAMA_BASE" \
    "$CHECKPOINT_DIR" \
    "$DATASET_DIR" \
    "$STAGING_DIR" \
    "$LOG_DIR" \
    "$(dirname "$LLAMA_ENV")"

SETUP_LOG="${LOG_DIR}/setup_llama_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$SETUP_LOG") 2>&1

log "NetLLM Llama 통합 설치 시작"
log "프로젝트 경로: $NETLLM_REPO"
log "자산 경로: $NETLLM_ASSETS"
log "Llama base 경로: $LLAMA_BASE"
log "Llama 환경 경로: $LLAMA_ENV"

run_as_root_apt

########################################
# 3. Git 저장소 복구
########################################
if [[ ! -d "$NETLLM_REPO/.git" ]]; then
    [[ "$CLONE_REPO_IF_MISSING" == "1" ]] || die "NetLLM 저장소가 없음: $NETLLM_REPO"
    [[ ! -e "$NETLLM_REPO" || -z "$(find "$NETLLM_REPO" -mindepth 1 -maxdepth 1 2>/dev/null)" ]] \
        || die "Git 저장소가 아닌 비어 있지 않은 경로가 존재함: $NETLLM_REPO"
    rm -rf "$NETLLM_REPO"
    log "사용자 NetLLM 저장소 clone"
    git clone "$NETLLM_GIT_URL" "$NETLLM_REPO"
else
    log "기존 사용자 NetLLM 저장소 유지"
fi

if [[ ! -d "$UPSTREAM_REPO/.git" ]]; then
    if [[ "$CLONE_UPSTREAM_IF_MISSING" == "1" ]]; then
        [[ ! -e "$UPSTREAM_REPO" || -z "$(find "$UPSTREAM_REPO" -mindepth 1 -maxdepth 1 2>/dev/null)" ]] \
            || die "Upstream Git 저장소가 아닌 비어 있지 않은 경로가 존재함: $UPSTREAM_REPO"
        rm -rf "$UPSTREAM_REPO"
        log "원본 NetLLM upstream clone"
        git clone "$UPSTREAM_GIT_URL" "$UPSTREAM_REPO"
        git -C "$UPSTREAM_REPO" checkout --detach "$UPSTREAM_COMMIT"
    else
        warn "Upstream 저장소가 없지만 clone을 비활성화함: $UPSTREAM_REPO"
    fi
else
    log "기존 upstream 저장소 유지"
fi

if [[ -d "$UPSTREAM_REPO/.git" ]]; then
    CURRENT_UPSTREAM_COMMIT="$(git -C "$UPSTREAM_REPO" rev-parse HEAD)"
    [[ "$CURRENT_UPSTREAM_COMMIT" == "$UPSTREAM_COMMIT" ]] \
        || die "Upstream commit 불일치: expected=$UPSTREAM_COMMIT actual=$CURRENT_UPSTREAM_COMMIT"
    [[ -z "$(git -C "$UPSTREAM_REPO" status --porcelain --untracked-files=no)" ]] \
        || die "Upstream tracked 파일이 수정됨. 원본 저장소를 정리한 뒤 다시 실행하십시오."
    log "Upstream commit 확인: $CURRENT_UPSTREAM_COMMIT"
fi

########################################
# 4. Hugging Face CLI 설치와 인증
########################################
if [[ -n "$HF_CLI_PATH" && -x "$HF_CLI_PATH" ]]; then
    HF_BIN="$HF_CLI_PATH"
elif command_exists hf; then
    HF_BIN="$(command -v hf)"
elif [[ -x "$HOME/.local/bin/hf" ]]; then
    HF_BIN="$HOME/.local/bin/hf"
else
    log "Hugging Face standalone CLI 설치"
    curl -LsSf https://hf.co/cli/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
    command_exists hf || die "hf CLI 설치 후 실행 파일을 찾지 못함"
    HF_BIN="$(command -v hf)"
fi

log "Hugging Face CLI: $HF_BIN"
"$HF_BIN" version || "$HF_BIN" --help >/dev/null

if ! "$HF_BIN" auth whoami >/dev/null 2>&1; then
    if [[ -n "${HF_TOKEN:-}" ]]; then
        log "HF_TOKEN을 사용해 Hugging Face 인증"
        "$HF_BIN" auth login --token "$HF_TOKEN"
    else
        log "Hugging Face 로그인이 필요합니다. 브라우저 인증 절차를 완료하십시오."
        "$HF_BIN" auth login
    fi
fi

HF_ACCOUNT="$($HF_BIN auth whoami | head -n 1 | tr -d '\r')"
log "Hugging Face 계정 확인: $HF_ACCOUNT"

########################################
# 5. 모델 revision 결정
########################################
LOCK_DIR="$NETLLM_REPO/docs/experiment_phase/llama/manifests"
LOCK_FILE="$LOCK_DIR/llama2-7b-hf.lock"
SHA_FILE="$LOCK_DIR/llama2-7b-hf.sha256"
mkdir -p "$LOCK_DIR"

if [[ -z "$MODEL_REVISION" && -f "$LOCK_FILE" ]]; then
    MODEL_REVISION="$(awk -F= '$1=="revision" {print $2}' "$LOCK_FILE" | tail -n 1)"
fi

if [[ -z "$MODEL_REVISION" ]]; then
    MODEL_INFO_JSON="$(mktemp)"
    if "$HF_BIN" models info "$MODEL_ID" --expand sha --format json > "$MODEL_INFO_JSON" 2>/dev/null; then
        MODEL_REVISION="$(python3 - "$MODEL_INFO_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    obj = json.load(f)
print(obj.get("sha", ""))
PY
)"
    fi
    rm -f "$MODEL_INFO_JSON"
fi

if [[ -z "$MODEL_REVISION" ]]; then
    warn "원격 SHA 자동 조회에 실패하여 revision=main을 사용합니다. 첫 성공 후 생성되는 lock 파일을 Git에 commit하십시오."
    MODEL_REVISION="main"
fi

log "모델: $MODEL_ID"
log "요청 revision: $MODEL_REVISION"

########################################
# 6. 디스크와 gated 접근 검사
########################################
FREE_GB="$(free_gb_at "$NETLLM_ASSETS")"
MODEL_READY=0
if [[ -f "$LLAMA_BASE/config.json" ]] \
   && [[ -f "$LLAMA_BASE/tokenizer.model" ]] \
   && find "$LLAMA_BASE" -maxdepth 1 -name '*.safetensors' -type f | grep -q .; then
    MODEL_READY=1
fi

if [[ "$MODEL_READY" == "0" && "$FREE_GB" -lt "$MIN_FREE_GB" ]]; then
    die "자산 filesystem 여유 공간이 ${FREE_GB}GB뿐입니다. 최소 ${MIN_FREE_GB}GB 이상 확보하십시오."
fi
log "자산 filesystem 여유 공간: ${FREE_GB}GB"

# 작은 파일로 인증과 접근권한을 먼저 검사한다.
log "Llama2 gated repository 접근 검사"
"$HF_BIN" download "$MODEL_ID" config.json \
    --revision "$MODEL_REVISION" \
    --local-dir "$LLAMA_BASE"

########################################
# 7. Llama2-7B base 다운로드
########################################
log "Llama2-7B base 파일 동기화 시작"
"$HF_BIN" download "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --include '*.safetensors' \
    --include '*.json' \
    --include '*.model' \
    --include 'README.md' \
    --include 'LICENSE*' \
    --local-dir "$LLAMA_BASE"

[[ -f "$LLAMA_BASE/config.json" ]] || die "config.json 없음"
[[ -f "$LLAMA_BASE/tokenizer.model" ]] || die "tokenizer.model 없음"
find "$LLAMA_BASE" -maxdepth 1 -name '*.safetensors' -type f | grep -q . \
    || die "safetensors weight 없음"

log "Llama base 필수 파일 확인 완료"

########################################
# 8. checksum 생성 또는 기존 manifest 검증
########################################
LOCAL_SHA_FILE="$LLAMA_BASE/SHA256SUMS.local.txt"
TMP_SHA_FILE="$(mktemp)"
(
    cd "$LLAMA_BASE"
    find . \
        -type f \
        ! -path './.cache/*' \
        ! -name 'SHA256SUMS.local.txt' \
        ! -name 'SHA256_VERIFIED_AT.txt' \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "$TMP_SHA_FILE"

if [[ -f "$SHA_FILE" ]]; then
    log "Git에 고정된 모델 checksum 검증"
    (
        cd "$LLAMA_BASE"
        sha256sum -c "$SHA_FILE"
    )
else
    log "첫 실행: 현재 모델 checksum manifest 생성"
    cp "$TMP_SHA_FILE" "$SHA_FILE"
fi

mv "$TMP_SHA_FILE" "$LOCAL_SHA_FILE"
(
    cd "$LLAMA_BASE"
    sha256sum -c "$LOCAL_SHA_FILE"
)
date -Is > "$LLAMA_BASE/SHA256_VERIFIED_AT.txt"

# 정확한 revision을 가능한 경우 다시 확인한다.
if [[ "$MODEL_REVISION" == "main" ]]; then
    MODEL_INFO_JSON="$(mktemp)"
    if "$HF_BIN" models info "$MODEL_ID" --expand sha --format json > "$MODEL_INFO_JSON" 2>/dev/null; then
        RESOLVED_REVISION="$(python3 - "$MODEL_INFO_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    obj = json.load(f)
print(obj.get("sha", "main"))
PY
)"
        [[ -n "$RESOLVED_REVISION" ]] && MODEL_REVISION="$RESOLVED_REVISION"
    fi
    rm -f "$MODEL_INFO_JSON"
fi

cat > "$LOCK_FILE" <<EOF_LOCK
asset_role=llama2_base
model_id=$MODEL_ID
revision=$MODEL_REVISION
weight_format=safetensors
config_file=config.json
tokenizer_file=tokenizer.model
verified_at=$(cat "$LLAMA_BASE/SHA256_VERIFIED_AT.txt")
EOF_LOCK

log "모델 revision/checksum lock 생성: $LOCK_FILE"
log "Git에 commit할 checksum manifest: $SHA_FILE"

########################################
# 9. NetLLM 호환 경로 연결
########################################
NETLLM_EXPECTED_PARENT="${NETLLM_EXPECTED_PARENT:-/root/downloaded_plms/llama}"
NETLLM_EXPECTED_PATH="$NETLLM_EXPECTED_PARENT/base"
mkdir -p "$NETLLM_EXPECTED_PARENT"

if [[ -L "$NETLLM_EXPECTED_PATH" ]]; then
    CURRENT_LINK="$(readlink -f "$NETLLM_EXPECTED_PATH")"
    if [[ "$CURRENT_LINK" != "$(readlink -f "$LLAMA_BASE")" ]]; then
        rm "$NETLLM_EXPECTED_PATH"
        ln -s "$LLAMA_BASE" "$NETLLM_EXPECTED_PATH"
    fi
elif [[ -e "$NETLLM_EXPECTED_PATH" ]]; then
    die "호환 경로에 기존 일반 파일/디렉터리가 있음: $NETLLM_EXPECTED_PATH"
else
    ln -s "$LLAMA_BASE" "$NETLLM_EXPECTED_PATH"
fi

log "NetLLM 모델 경로 연결: $NETLLM_EXPECTED_PATH -> $(readlink -f "$NETLLM_EXPECTED_PATH")"

########################################
# 10. Llama 전용 Conda 환경
########################################
CONDA_BIN="$(find_conda || true)"
[[ -n "$CONDA_BIN" ]] || die "Conda를 찾지 못했습니다. PyTorch 2.2.0/CUDA 12.1 Vast 템플릿 또는 Conda가 있는 인스턴스를 사용하십시오."
log "Conda 실행 파일: $CONDA_BIN"

if [[ ! -f "$LLAMA_ENV/conda-meta/history" ]]; then
    log "Llama 전용 Conda 환경 생성: Python $PYTHON_VERSION"
    "$CONDA_BIN" create -y -p "$LLAMA_ENV" "python=$PYTHON_VERSION" pip
else
    log "기존 Llama Conda 환경 재사용"
fi

ENV_PY="$LLAMA_ENV/bin/python"
ENV_GDOWN="$LLAMA_ENV/bin/gdown"
[[ -x "$ENV_PY" ]] || die "환경 Python을 찾지 못함: $ENV_PY"

log "pip 기본 도구 업데이트"
"$ENV_PY" -m pip install --upgrade \
    pip==24.0 setuptools wheel

log "PyTorch $TORCH_VERSION + CUDA 12.1 wheel 설치/검증"
"$ENV_PY" -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url https://download.pytorch.org/whl/cu121

log "NetLLM Llama 실행 dependency 설치/검증"
"$ENV_PY" -m pip install \
    "numpy==$NUMPY_VERSION" \
    "munch==$MUNCH_VERSION" \
    "transformers==$TRANSFORMERS_VERSION" \
    "peft==$PEFT_VERSION" \
    "accelerate==$ACCELERATE_VERSION" \
    "huggingface-hub==$HF_HUB_RUNTIME_VERSION" \
    "sentencepiece==$SENTENCEPIECE_VERSION" \
    "safetensors==$SAFETENSORS_VERSION" \
    "gdown==$GDOWN_VERSION"

"$ENV_PY" -m pip check

########################################
# 11. 선택: 팀 checkpoint와 dataset 다운로드
########################################
download_drive_asset() {
    local url=$1
    local output=$2
    local label=$3
    if [[ -f "$output" ]]; then
        log "$label ZIP이 이미 존재하여 재사용: $output"
        return 0
    fi
    log "$label Google Drive 다운로드"
    "$ENV_GDOWN" --fuzzy "$url" -O "$output"
}

if [[ "$INSTALL_TEAM_ASSETS" == "1" ]]; then
    if [[ -n "$CHECKPOINT_URL" ]]; then
        CHECKPOINT_ZIP="$STAGING_DIR/try_llama2_7b.zip"
        download_drive_asset "$CHECKPOINT_URL" "$CHECKPOINT_ZIP" "fine-tuned checkpoint"
        unzip -t "$CHECKPOINT_ZIP" >/dev/null
        validate_sha256_if_given "$CHECKPOINT_ZIP" "$CHECKPOINT_SHA256" "checkpoint ZIP"
        if [[ -z "$(find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
            unzip -q "$CHECKPOINT_ZIP" -d "$CHECKPOINT_DIR"
        else
            warn "checkpoint 디렉터리가 비어 있지 않아 자동 압축 해제를 생략함: $CHECKPOINT_DIR"
        fi
    else
        warn "CHECKPOINT_URL이 없어 팀 checkpoint 다운로드를 생략함"
    fi

    if [[ -n "$DATASET_URL" ]]; then
        DATASET_ZIP="$STAGING_DIR/data.zip"
        download_drive_asset "$DATASET_URL" "$DATASET_ZIP" "원본 VP dataset"
        unzip -t "$DATASET_ZIP" >/dev/null
        validate_sha256_if_given "$DATASET_ZIP" "$DATASET_SHA256" "dataset ZIP"
        if [[ -z "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
            unzip -q "$DATASET_ZIP" -d "$DATASET_DIR"
        else
            warn "dataset 디렉터리가 비어 있지 않아 자동 압축 해제를 생략함: $DATASET_DIR"
        fi
    else
        warn "DATASET_URL이 없어 팀 dataset 다운로드를 생략함"
    fi
fi

# checkpoint/data 존재 여부는 보고만 하고, base 환경 설치 실패로 처리하지 않는다.
CHECKPOINT_ROOTS="$(find "$CHECKPOINT_DIR" -type f -name adapter_config.json -printf '%h\n' 2>/dev/null | sort -u || true)"
MODULES_FILES="$(find "$CHECKPOINT_DIR" -type f -name modules_except_plm.bin -print 2>/dev/null | sort || true)"
JIN_ROOTS="$(find "$DATASET_DIR" -type d -name Jin2022images -print 2>/dev/null | sort || true)"
SALIENCY_ROOTS="$(find "$DATASET_DIR" -type d -name saliencyMap -print 2>/dev/null | sort || true)"
FEATURE_ROOTS="$(find "$DATASET_DIR" -type d -name features -print 2>/dev/null | sort || true)"

########################################
# 12. 환경 검증과 local-only smoke test
########################################
ENV_REPORT="$("$ENV_PY" - <<'PY'
import json
import torch
import numpy
import transformers
import peft
import accelerate
import huggingface_hub

report = {
    "python_torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
    "numpy": numpy.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}
print(json.dumps(report, ensure_ascii=False))
PY
)"

log "환경 검증 결과: $ENV_REPORT"

echo "$ENV_REPORT" | "$ENV_PY" -c \
'import json,sys; d=json.load(sys.stdin); assert d["cuda_available"], "CUDA available=False"'

SMOKE_STATUS="생략"
if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
    log "Llama2-7B local-only GPU smoke test 시작"
    LLAMA_BASE="$LLAMA_BASE" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "$ENV_PY" - <<'PY'
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = os.environ["LLAMA_BASE"]
assert torch.cuda.is_available(), "CUDA available=False"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=False)
started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    path,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": 0},
)
load_seconds = time.perf_counter() - started
inputs = tokenizer("Viewport prediction uses historical head movement to", return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
text = tokenizer.decode(output[0], skip_special_tokens=True)
peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
print("생성 결과:", text)
print(f"모델 로드 시간: {load_seconds:.2f}s")
print(f"Peak allocated: {peak_gib:.2f} GiB")
print("PASS: Llama2-7B local-only load and inference")
PY
    SMOKE_STATUS="성공"
fi

########################################
# 13. manifest와 한국어 결과 문서 생성
########################################
MANIFEST_DIR="$NETLLM_REPO/experiments/vp/llama_repro"
DOC_DIR="$NETLLM_REPO/docs/experiment_phase/llama"
mkdir -p "$MANIFEST_DIR" "$DOC_DIR"

PIP_FREEZE_FILE="$MANIFEST_DIR/llama_environment_freeze.txt"
"$ENV_PY" -m pip freeze | sort > "$PIP_FREEZE_FILE"
PIP_FREEZE_SHA="$(sha256_file "$PIP_FREEZE_FILE")"

ENV_JSON_FILE="$MANIFEST_DIR/environment_manifest.json"
ENV_REPORT="$ENV_REPORT" \
MODEL_ID="$MODEL_ID" \
MODEL_REVISION="$MODEL_REVISION" \
LLAMA_BASE="$LLAMA_BASE" \
LLAMA_ENV="$LLAMA_ENV" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
DATASET_DIR="$DATASET_DIR" \
PIP_FREEZE_SHA="$PIP_FREEZE_SHA" \
UPSTREAM_COMMIT="$UPSTREAM_COMMIT" \
python3 - "$ENV_JSON_FILE" <<'PY'
import json, os, sys
runtime = json.loads(os.environ["ENV_REPORT"])
manifest = {
    "model": {
        "id": os.environ["MODEL_ID"],
        "revision": os.environ["MODEL_REVISION"],
        "path": os.environ["LLAMA_BASE"],
    },
    "environment": {
        "path": os.environ["LLAMA_ENV"],
        "pip_freeze_sha256": os.environ["PIP_FREEZE_SHA"],
        **runtime,
    },
    "assets": {
        "checkpoint_path": os.environ["CHECKPOINT_DIR"],
        "dataset_path": os.environ["DATASET_DIR"],
    },
    "upstream_commit": os.environ["UPSTREAM_COMMIT"],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

RESULT_MD="$DOC_DIR/LLAMA_ENVIRONMENT_RESULT_KO.md"
cat > "$RESULT_MD" <<EOF_MD
# NetLLM Llama2-7B 환경 구현 결과

## 1. 실행 결과

- 실행 시각: $(date -Is)
- 실행 로그: \`$SETUP_LOG\`
- Llama base model smoke test: **$SMOKE_STATUS**
- Hugging Face 계정: \`$HF_ACCOUNT\`

## 2. 경로

- 사용자 프로젝트: \`$NETLLM_REPO\`
- 원본 NetLLM source: \`$UPSTREAM_REPO\`
- 외부 자산 root: \`$NETLLM_ASSETS\`
- Llama2-7B base: \`$LLAMA_BASE\`
- Llama 전용 Conda 환경: \`$LLAMA_ENV\`
- checkpoint: \`$CHECKPOINT_DIR\`
- dataset: \`$DATASET_DIR\`
- NetLLM 호환 링크: \`$NETLLM_EXPECTED_PATH\`

## 3. 고정 버전

- 모델 ID: \`$MODEL_ID\`
- 모델 revision: \`$MODEL_REVISION\`
- Python: \`$PYTHON_VERSION\`
- PyTorch: \`$TORCH_VERSION\` / CUDA 12.1 wheel
- NumPy: \`$NUMPY_VERSION\`
- Transformers: \`$TRANSFORMERS_VERSION\`
- PEFT: \`$PEFT_VERSION\`
- Accelerate: \`$ACCELERATE_VERSION\`
- huggingface-hub(runtime): \`$HF_HUB_RUNTIME_VERSION\`
- upstream commit: \`$UPSTREAM_COMMIT\`

## 4. 자산 확인

### Fine-tuned checkpoint

- adapter_config.json 탐색 결과:
\`\`\`text
${CHECKPOINT_ROOTS:-없음}
\`\`\`

- modules_except_plm.bin 탐색 결과:
\`\`\`text
${MODULES_FILES:-없음}
\`\`\`

### 원본 VP dataset

- Jin2022images 탐색 결과:
\`\`\`text
${JIN_ROOTS:-없음}
\`\`\`

- saliencyMap 탐색 결과:
\`\`\`text
${SALIENCY_ROOTS:-없음}
\`\`\`

- features 탐색 결과:
\`\`\`text
${FEATURE_ROOTS:-없음}
\`\`\`

## 5. Git에 저장할 파일

다음 파일은 크기가 작고 재현에 필요하므로 commit 대상이다.

- \`docs/experiment_phase/llama/manifests/llama2-7b-hf.lock\`
- \`docs/experiment_phase/llama/manifests/llama2-7b-hf.sha256\`
- \`experiments/vp/llama_repro/environment_manifest.json\`
- \`experiments/vp/llama_repro/llama_environment_freeze.txt\`
- \`docs/experiment_phase/llama/LLAMA_ENVIRONMENT_RESULT_KO.md\`

다음 항목은 Git에 올리지 않는다.

- Llama weight
- checkpoint 및 adapter weight
- \`modules_except_plm.bin\`
- dataset 및 ZIP
- Hugging Face token
- Conda 환경 디렉터리

## 6. 다음 Gate

1. checkpoint에 \`adapter_config.json\`, adapter weight, \`modules_except_plm.bin\`이 함께 있는지 확정한다.
2. dataset에 정확한 \`Jin2022images/saliencyMap\` 및 \`features\` 구조가 있는지 확정한다.
3. 실제 Jin2022 sample 1개로 원본 Llama NetLLM baseline smoke test를 수행한다.
4. baseline 성공 후에만 Identity equivalence와 Recent-K benchmark를 진행한다.

LoRA 학습은 GPU memory가 충분한지 별도 확인하기 전에는 시작하지 않는다.
EOF_MD

########################################
# 14. 최종 요약
########################################
log "설치 완료"
log "모델 크기: $(du -sh "$LLAMA_BASE" | awk '{print $1}')"
log "환경 manifest: $ENV_JSON_FILE"
log "한국어 결과 문서: $RESULT_MD"
log "pip freeze SHA256: $PIP_FREEZE_SHA"

printf '\n다음 명령으로 Git 반영 대상을 확인하십시오.\n'
printf '  cd %q\n' "$NETLLM_REPO"
printf '  git status --short\n'
printf '\n중요: git add . 를 사용하지 말고 manifest·문서·script만 선택적으로 commit하십시오.\n'
