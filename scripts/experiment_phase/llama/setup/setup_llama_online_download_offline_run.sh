#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 077

PROJECT_ROOT="${NETLLM_REPO:-/root/NetLLM}"
SETUP_SCRIPT="${LLAMA_SETUP_SCRIPT:-$PROJECT_ROOT/setup_netllm_llama.sh}"

if [[ ! -f "$SETUP_SCRIPT" ]]; then
    printf '오류: setup script가 없습니다: %s\n' "$SETUP_SCRIPT" >&2
    exit 2
fi

printf 'Online auth/download를 위해 HF offline 변수를 제거합니다.\n'
printf 'Setup script: %s\n' "$SETUP_SCRIPT"

exec env \
    -u HF_HUB_OFFLINE \
    -u TRANSFORMERS_OFFLINE \
    -u HF_DATASETS_OFFLINE \
    bash "$SETUP_SCRIPT" "$@"
