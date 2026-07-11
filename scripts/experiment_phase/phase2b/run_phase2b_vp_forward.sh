#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/workspace/NetLLM"
SOURCE_ROOT="/workspace/NetLLM-source"
VP_ROOT="$SOURCE_ROOT/viewport_prediction"
PYTHON_BIN="/venv/vp_netllm_repro/bin/python"
ARTIFACT_ROOT="/workspace/NetLLM-artifacts"
ARTIFACT_PATH="$ARTIFACT_ROOT/plms/gpt2/base"
RUNTIME_ROOT="$PROJECT_ROOT/experiments/vp/phase2b_runtime"
SCRIPT_PATH="$PROJECT_ROOT/scripts/experiment_phase/phase2b/run_phase2b_vp_forward.py"
TRACE_PATH="$RUNTIME_ROOT/phase2b_tensor_trace.json"
LOG_PATH="$RUNTIME_ROOT/phase2b_forward.log"
BEFORE_PATH="$RUNTIME_ROOT/upstream_before.txt"
AFTER_PATH="$RUNTIME_ROOT/upstream_after.txt"
STATUS_PATH="$RUNTIME_ROOT/run_status.txt"

artifact_fingerprint() {
    sha256sum \
        "$ARTIFACT_PATH/config.json" \
        "$ARTIFACT_PATH/model.safetensors" \
        "$ARTIFACT_PATH/tokenizer.json" \
        "$ARTIFACT_PATH/merges.txt" \
        "$ARTIFACT_PATH/vocab.json" | sha256sum | cut -d' ' -f1
}

record_upstream() {
    local destination="$1"
    {
        echo "commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
        echo "status_begin"
        git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all
        echo "status_end"
        echo "diff_begin"
        git -C "$SOURCE_ROOT" diff --name-status HEAD
        echo "diff_end"
        echo "pycache_count=$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
    } > "$destination"
}

if [[ -e "$RUNTIME_ROOT" ]]; then
    echo "ERROR: Phase 2B runtime path already exists; nothing was overwritten: $RUNTIME_ROOT" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: Python runner not found: $SCRIPT_PATH" >&2
    exit 2
fi
if [[ ! -d "$ARTIFACT_PATH" ]]; then
    echo "ERROR: GPT-2 artifact not found: $ARTIFACT_PATH" >&2
    exit 2
fi

before_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
before_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
before_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
if [[ -n "$before_status" || -n "$before_diff" || "$before_pycache" -ne 0 ]]; then
    echo "ERROR: upstream is not clean before Phase 2B" >&2
    exit 2
fi

before_freeze_hash="$($PYTHON_BIN -m pip freeze | sha256sum | cut -d' ' -f1)"
before_artifact_hash="$(artifact_fingerprint)"
mkdir -p "$RUNTIME_ROOT"
record_upstream "$BEFORE_PATH"

export HF_HOME="$ARTIFACT_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$ARTIFACT_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VP_ROOT"
export HTTP_PROXY="http://127.0.0.1:9"
export HTTPS_PROXY="http://127.0.0.1:9"
export ALL_PROXY="socks5://127.0.0.1:9"

start_epoch="$(date +%s)"
cd "$VP_ROOT" || exit 2
"$PYTHON_BIN" -B "$SCRIPT_PATH" --output "$TRACE_PATH" 2>&1 | tee "$LOG_PATH"
python_status=${PIPESTATUS[0]}
end_epoch="$(date +%s)"

record_upstream "$AFTER_PATH"
after_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
after_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
after_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
after_freeze_hash="$($PYTHON_BIN -m pip freeze | sha256sum | cut -d' ' -f1)"
after_artifact_hash="$(artifact_fingerprint)"

final_status=$python_status
upstream_clean=yes
environment_unchanged=yes
artifact_unchanged=yes
if [[ -n "$after_status" || -n "$after_diff" || "$after_pycache" -ne 0 ]]; then
    upstream_clean=no
    if [[ $final_status -eq 0 ]]; then final_status=3; fi
fi
if [[ "$before_freeze_hash" != "$after_freeze_hash" ]]; then
    environment_unchanged=no
    if [[ $final_status -eq 0 ]]; then final_status=4; fi
fi
if [[ "$before_artifact_hash" != "$after_artifact_hash" ]]; then
    artifact_unchanged=no
    if [[ $final_status -eq 0 ]]; then final_status=5; fi
fi

{
    echo "python_exit_code=$python_status"
    echo "final_exit_code=$final_status"
    echo "wall_elapsed_seconds=$((end_epoch - start_epoch))"
    echo "upstream_clean_after=$upstream_clean"
    echo "environment_unchanged=$environment_unchanged"
    echo "artifact_unchanged=$artifact_unchanged"
    echo "freeze_hash_before=$before_freeze_hash"
    echo "freeze_hash_after=$after_freeze_hash"
    echo "artifact_fingerprint_before=$before_artifact_hash"
    echo "artifact_fingerprint_after=$after_artifact_hash"
    echo "trace_path=$TRACE_PATH"
    echo "log_path=$LOG_PATH"
} > "$STATUS_PATH"

echo "PHASE2B_PYTHON_EXIT_CODE=$python_status"
echo "PHASE2B_FINAL_EXIT_CODE=$final_status"
echo "PHASE2B_UPSTREAM_CLEAN_AFTER=$upstream_clean"
echo "PHASE2B_ENVIRONMENT_UNCHANGED=$environment_unchanged"
echo "PHASE2B_ARTIFACT_UNCHANGED=$artifact_unchanged"
echo "PHASE2B_TRACE_PATH=$TRACE_PATH"

exit "$final_status"
