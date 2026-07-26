#!/usr/bin/env bash
# NetLLM × LiteVLM / Llama VP reproducibility master setup
# Purpose: rebuild the project on a new Vast.ai-style instance after disposing the current one.
# Safe defaults: no training, no benchmark, no overwrite of existing dirs unless explicitly removed by user.

set -euo pipefail

PROJECT_REPO="${PROJECT_REPO:-https://github.com/soyuniverse/NetLLM.git}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/NetLLM}"
CURRENT_SOURCE_ROOT="${CURRENT_SOURCE_ROOT:-/root/NetLLM-source}"
ERA_SOURCE_ROOT="${ERA_SOURCE_ROOT:-/root/NetLLM-source-checkpoint-era}"
ASSET_ROOT="${ASSET_ROOT:-/root/NetLLM-assets}"
LLAMA_ENV="${LLAMA_ENV:-/root/venvs/vp_netllm_llama}"
GPT2_ENV="${GPT2_ENV:-/root/venvs/vp_netllm_repro}"

CURRENT_COMMIT="${CURRENT_COMMIT:-105bcf070f2bec808f7b14f8f5a953de6e4e6e54}"
ERA_COMMIT="${ERA_COMMIT:-ee4d8726898610e4ae7df08bdd26728cafb4701f}"
LLAMA_REPO="${LLAMA_REPO:-meta-llama/Llama-2-7b-hf}"
LLAMA_REVISION="${LLAMA_REVISION:-01c7f73d771dfac7d292323805ebc428287df4f9}"

CHECKPOINT_ZIP="${CHECKPOINT_ZIP:-$PROJECT_ROOT/try_llama2_7b.zip}"
DATA_ZIP="${DATA_ZIP:-$PROJECT_ROOT/data.zip}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ASSET_ROOT/checkpoints/try_llama2_7b}"
TEAM_DATA_DIR="${TEAM_DATA_DIR:-$ASSET_ROOT/datasets/team_data}"
LLAMA_BASE_DIR="${LLAMA_BASE_DIR:-$ASSET_ROOT/llama/base}"
HF_CACHE="${HF_CACHE:-$ASSET_ROOT/hf_cache}"

log(){ printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
fail(){ echo "ERROR: $*" >&2; exit 2; }
need_cmd(){ command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"; }

usage(){
  cat <<USAGE
Usage: bash setup_netllm_repro_master.sh <command>

Commands:
  preflight       Check GPU, disk, network tools, and important paths.
  clone           Clone project repo and both NetLLM upstream commits.
  env-llama       Create /root/venvs/vp_netllm_llama with fixed Llama runtime.
  download-llama  Download Llama2-7B base from Hugging Face to external asset dir.
  unpack-assets   Unpack try_llama2_7b.zip and data.zip into external asset dir.
  verify          Verify key paths and import/load prerequisites without training.
  smoke           Run existing strict-load / VP technical-smoke scripts if present.
  all             Run preflight, clone, env-llama, download-llama, unpack-assets, verify.

Environment overrides:
  PROJECT_REPO, PROJECT_ROOT, CURRENT_SOURCE_ROOT, ERA_SOURCE_ROOT, ASSET_ROOT
  LLAMA_REPO, LLAMA_REVISION, CHECKPOINT_ZIP, DATA_ZIP

Before running download-llama, make sure Hugging Face access is ready:
  hf auth login
  hf auth whoami

This script does NOT run training, full benchmark, fine-tuning, or destructive cleanup.
USAGE
}

preflight(){
  log "Preflight"
  need_cmd git; need_cmd python; need_cmd sha256sum; need_cmd unzip
  nvidia-smi || fail "nvidia-smi failed; GPU is not visible"
  df -h /root || true
  mkdir -p "$ASSET_ROOT" "$HF_CACHE" /root/venvs
  log "Paths"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "CURRENT_SOURCE_ROOT=$CURRENT_SOURCE_ROOT"
  echo "ERA_SOURCE_ROOT=$ERA_SOURCE_ROOT"
  echo "ASSET_ROOT=$ASSET_ROOT"
  echo "LLAMA_ENV=$LLAMA_ENV"
}

clone_one(){
  local url="$1" dir="$2" commit="$3"
  if [[ -e "$dir/.git" ]]; then
    log "Existing git repo found: $dir"
  else
    log "Cloning $url -> $dir"
    git clone "$url" "$dir"
  fi
  git -C "$dir" fetch --all --tags
  git -C "$dir" checkout --detach "$commit"
  local actual; actual="$(git -C "$dir" rev-parse HEAD)"
  [[ "$actual" == "$commit" ]] || fail "$dir commit mismatch: $actual != $commit"
  [[ -z "$(git -C "$dir" status --porcelain=v2 --untracked-files=all)" ]] || fail "$dir is not clean after checkout"
}

clone_all(){
  log "Clone project and upstream sources"
  if [[ -e "$PROJECT_ROOT/.git" ]]; then
    log "Existing project repo found: $PROJECT_ROOT"
  else
    git clone "$PROJECT_REPO" "$PROJECT_ROOT"
  fi
  clone_one "https://github.com/duowuyms/NetLLM.git" "$CURRENT_SOURCE_ROOT" "$CURRENT_COMMIT"
  clone_one "https://github.com/duowuyms/NetLLM.git" "$ERA_SOURCE_ROOT" "$ERA_COMMIT"
}

create_llama_env(){
  log "Create Llama runtime environment"
  if [[ -e "$LLAMA_ENV" ]]; then
    log "Environment already exists; not modifying: $LLAMA_ENV"
    "$LLAMA_ENV/bin/python" --version
    return 0
  fi
  [[ -f /opt/miniforge3/etc/profile.d/conda.sh ]] || fail "Miniforge conda not found at /opt/miniforge3"
  # shellcheck source=/dev/null
  source /opt/miniforge3/etc/profile.d/conda.sh
  conda create --prefix "$LLAMA_ENV" python=3.8.10 -y
  "$LLAMA_ENV/bin/python" -m pip install --upgrade pip
  "$LLAMA_ENV/bin/python" -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
  "$LLAMA_ENV/bin/python" -m pip install \
    numpy==1.24.4 pandas==2.0.3 scipy==1.10.1 scikit-learn==1.3.2 \
    transformers==4.34.1 peft==0.6.2 accelerate==0.24.1 huggingface-hub==0.17.3 \
    safetensors==0.5.3 tokenizers==0.14.1 sentencepiece==0.1.99 \
    opencv-python-headless==4.8.1.78 yacs==0.1.8 einops==0.8.1 \
    prettytable==3.11.0 matplotlib==3.7.5 tqdm munch==4.0.0
  "$LLAMA_ENV/bin/python" -m pip check
}

download_llama(){
  log "Download Llama base"
  [[ -x "$LLAMA_ENV/bin/python" ]] || fail "Llama env missing: $LLAMA_ENV"
  if [[ -d "$LLAMA_BASE_DIR" && -f "$LLAMA_BASE_DIR/config.json" ]]; then
    log "Llama base already exists; not overwriting: $LLAMA_BASE_DIR"
    return 0
  fi
  mkdir -p "$HF_CACHE" "$(dirname "$LLAMA_BASE_DIR")"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  export HF_HOME="$HF_CACHE"
  export TRANSFORMERS_CACHE="$HF_CACHE"
  "$LLAMA_ENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
repo_id = "$LLAMA_REPO"
revision = "$LLAMA_REVISION"
local_dir = "$LLAMA_BASE_DIR"
print(f"Downloading {repo_id}@{revision} -> {local_dir}")
snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir, local_dir_use_symlinks=False)
PY
}

unpack_zip_once(){
  local zip="$1" target="$2" label="$3"
  if [[ -d "$target" ]]; then
    log "$label already exists; not unpacking: $target"
    return 0
  fi
  [[ -f "$zip" ]] || fail "$label zip not found: $zip"
  mkdir -p "$(dirname "$target")" "$ASSET_ROOT/staging"
  local stage="$ASSET_ROOT/staging/${label}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$stage"
  log "Unpacking $zip -> $stage"
  unzip -q "$zip" -d "$stage"
  if [[ -d "$stage/$(basename "$target")" ]]; then
    mv "$stage/$(basename "$target")" "$target"
  else
    mv "$stage" "$target"
  fi
  log "$label unpacked to $target"
}

unpack_assets(){
  log "Unpack checkpoint and dataset assets"
  unpack_zip_once "$CHECKPOINT_ZIP" "$CHECKPOINT_DIR" "try_llama2_7b"
  unpack_zip_once "$DATA_ZIP" "$TEAM_DATA_DIR" "team_data"
}

verify(){
  log "Verify key paths"
  [[ -d "$PROJECT_ROOT" ]] || fail "project missing: $PROJECT_ROOT"
  [[ -d "$CURRENT_SOURCE_ROOT/viewport_prediction" ]] || fail "current upstream missing"
  [[ -d "$ERA_SOURCE_ROOT/viewport_prediction" ]] || fail "checkpoint-era upstream missing"
  [[ -x "$LLAMA_ENV/bin/python" ]] || fail "Llama Python missing"
  [[ -f "$LLAMA_BASE_DIR/config.json" ]] || fail "Llama config missing"
  [[ -d "$CHECKPOINT_DIR" ]] || fail "checkpoint dir missing"
  [[ -d "$TEAM_DATA_DIR" ]] || fail "team data dir missing"
  [[ -z "$(git -C "$CURRENT_SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)" ]] || fail "current upstream not clean"
  [[ -z "$(git -C "$ERA_SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)" ]] || fail "checkpoint-era upstream not clean"
  "$LLAMA_ENV/bin/python" - <<'PY'
import torch, transformers, peft, accelerate
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('accelerate', accelerate.__version__)
PY
  log "Llama base files"
  find "$LLAMA_BASE_DIR" -maxdepth 1 -type f | sort | sed 's#^#  #' | head -40
  log "Checkpoint files"
  find "$CHECKPOINT_DIR" -maxdepth 2 -type f | sort | sed 's#^#  #' | head -40
}

smoke(){
  log "Run available strict-load and technical-smoke runners"
  local strict="$PROJECT_ROOT/scripts/experiment_phase/llama/smoke/run_llama_strict_load.py"
  local vp="$PROJECT_ROOT/scripts/experiment_phase/llama/smoke/run_llama_vp_technical_smoke.py"
  [[ -f "$strict" ]] || fail "strict-load runner not found: $strict"
  [[ -f "$vp" ]] || fail "VP smoke runner not found: $vp"
  export PROJECT_ROOT CURRENT_SOURCE_ROOT ERA_SOURCE_ROOT ASSET_ROOT LLAMA_ENV LLAMA_BASE_DIR CHECKPOINT_DIR TEAM_DATA_DIR
  "$LLAMA_ENV/bin/python" -B "$strict"
  "$LLAMA_ENV/bin/python" -B "$vp"
}

cmd="${1:-help}"
case "$cmd" in
  help|-h|--help) usage ;;
  preflight) preflight ;;
  clone) clone_all ;;
  env-llama) create_llama_env ;;
  download-llama) download_llama ;;
  unpack-assets) unpack_assets ;;
  verify) verify ;;
  smoke) smoke ;;
  all) preflight; clone_all; create_llama_env; download_llama; unpack_assets; verify ;;
  *) usage; fail "unknown command: $cmd" ;;
esac
