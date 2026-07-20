#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/workspace/NetLLM"
SOURCE_ROOT="/workspace/NetLLM-source"
VP_ROOT="$SOURCE_ROOT/viewport_prediction"
PYTHON_BIN="/venv/vp_netllm_repro/bin/python"
ARTIFACT_ROOT="/workspace/NetLLM-artifacts"
ARTIFACT_PATH="$ARTIFACT_ROOT/plms/gpt2/base"
RUNTIME_ROOT="$PROJECT_ROOT/experiments/vp/phase3a_runtime"
RUNNER="$PROJECT_ROOT/scripts/experiment_phase/phase3a/run_phase3a_identity_equivalence.py"
OUTPUT_PATH="$RUNTIME_ROOT/identity_equivalence.json"
LOG_PATH="$RUNTIME_ROOT/identity_equivalence.log"
TEST_LOG_PATH="$RUNTIME_ROOT/tests.log"
STATUS_PATH="$RUNTIME_ROOT/run_status.txt"
BEFORE_PATH="$RUNTIME_ROOT/upstream_before.txt"
AFTER_PATH="$RUNTIME_ROOT/upstream_after.txt"

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
    echo "ERROR: Phase 3A runtime already exists; nothing was overwritten: $RUNTIME_ROOT" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNNER" || ! -d "$ARTIFACT_PATH" ]]; then
    echo "ERROR: required Python, runner, or GPT-2 artifact is missing" >&2
    exit 2
fi

before_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
before_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
before_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
if [[ -n "$before_status" || -n "$before_diff" || "$before_pycache" -ne 0 ]]; then
    echo "ERROR: upstream is not clean before Phase 3A" >&2
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
export PYTHONPATH="$PROJECT_ROOT/src:$VP_ROOT"
export HTTP_PROXY="http://127.0.0.1:9"
export HTTPS_PROXY="http://127.0.0.1:9"
export ALL_PROXY="socks5://127.0.0.1:9"

start_epoch="$(date +%s)"
cd "$VP_ROOT" || exit 2
"$PYTHON_BIN" -B "$RUNNER" --output "$OUTPUT_PATH" 2>&1 | tee "$LOG_PATH"
runner_status=${PIPESTATUS[0]}

test_status=not_started
if [[ $runner_status -eq 0 ]]; then
    cd "$PROJECT_ROOT" || exit 2
    "$PYTHON_BIN" -B -m unittest discover -s tests/phase3a -p 'test_*.py' -v \
        2>&1 | tee "$TEST_LOG_PATH"
    test_status=${PIPESTATUS[0]}
fi
end_epoch="$(date +%s)"

record_upstream "$AFTER_PATH"
after_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
after_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
after_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
after_freeze_hash="$($PYTHON_BIN -m pip freeze | sha256sum | cut -d' ' -f1)"
after_artifact_hash="$(artifact_fingerprint)"

final_status=$runner_status
if [[ "$test_status" != "not_started" && "$test_status" -ne 0 && $final_status -eq 0 ]]; then
    final_status=6
fi
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
    echo "runner_exit_code=$runner_status"
    echo "test_exit_code=$test_status"
    echo "final_exit_code=$final_status"
    echo "wall_elapsed_seconds=$((end_epoch - start_epoch))"
    echo "upstream_clean_after=$upstream_clean"
    echo "environment_unchanged=$environment_unchanged"
    echo "artifact_unchanged=$artifact_unchanged"
    echo "freeze_hash_before=$before_freeze_hash"
    echo "freeze_hash_after=$after_freeze_hash"
    echo "artifact_fingerprint_before=$before_artifact_hash"
    echo "artifact_fingerprint_after=$after_artifact_hash"
    echo "output_path=$OUTPUT_PATH"
} > "$STATUS_PATH"

echo "PHASE3A_RUNNER_EXIT_CODE=$runner_status"
echo "PHASE3A_TEST_EXIT_CODE=$test_status"
echo "PHASE3A_FINAL_EXIT_CODE=$final_status"
echo "PHASE3A_UPSTREAM_CLEAN_AFTER=$upstream_clean"
echo "PHASE3A_ENVIRONMENT_UNCHANGED=$environment_unchanged"
echo "PHASE3A_ARTIFACT_UNCHANGED=$artifact_unchanged"
echo "PHASE3A_OUTPUT_PATH=$OUTPUT_PATH"

exit "$final_status"
