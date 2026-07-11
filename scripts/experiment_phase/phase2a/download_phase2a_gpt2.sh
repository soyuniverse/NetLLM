#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/NetLLM"
PYTHON_BIN="/venv/vp_netllm/bin/python"
ARTIFACT_ROOT="/workspace/NetLLM-artifacts"
HF_CACHE="$ARTIFACT_ROOT/hf_cache"
TARGET="$ARTIFACT_ROOT/plms/gpt2/base"
RUNTIME_ROOT="$PROJECT_ROOT/experiments/vp/phase2a_runtime"
LOG_DIR="$RUNTIME_ROOT/logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_DIR/${TIMESTAMP}_gpt2_download.log"
REPO_ID="openai-community/gpt2-medium"
REVISION="6dcaa7a952f72f9298047fd5137cd6e4f05f41da"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

echo "PHASE2A_DOWNLOAD_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PHASE2A_REPO_ID=$REPO_ID"
echo "PHASE2A_REVISION=$REVISION"
echo "PHASE2A_TARGET=$TARGET"
echo "PHASE2A_HF_CACHE=$HF_CACHE"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: reproducible environment Python not found: $PYTHON_BIN" >&2
    exit 2
fi

if [[ -e "$TARGET" ]]; then
    echo "ERROR: artifact target already exists; it was not overwritten: $TARGET" >&2
    find "$TARGET" -maxdepth 2 -printf '%y %p %s bytes\n' || true
    exit 2
fi

mkdir -p "$HF_CACHE" "$(dirname "$TARGET")"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" -B - "$REPO_ID" "$REVISION" "$TARGET" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download

repo_id, revision, target = sys.argv[1:]
info = HfApi().model_info(repo_id=repo_id, revision=revision)
if info.sha != revision:
    raise RuntimeError(f"Resolved revision mismatch: expected {revision}, got {info.sha}")

allow_patterns = [
    "README.md",
    "config.json",
    "generation_config.json",
    "generation_config_for_text_generation.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

result = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=target,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
    resume_download=True,
)
print(f"resolved_revision={info.sha}")
print(f"snapshot_path={result}")
PY

echo "===== Downloaded artifact files ====="
find "$TARGET" -type f -printf '%P|%s bytes\n' | sort
echo "===== SHA-256 ====="
find "$TARGET" -type f -print0 | sort -z | xargs -0 sha256sum
echo "PHASE2A_DOWNLOAD_FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PHASE2A_DOWNLOAD_LOG=$LOG_PATH"
