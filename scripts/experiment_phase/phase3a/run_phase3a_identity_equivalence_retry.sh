#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/workspace/NetLLM"
SOURCE_ROOT="/workspace/NetLLM-source"
VP_ROOT="$SOURCE_ROOT/viewport_prediction"
PYTHON_BIN="/venv/vp_netllm_repro/bin/python"
ARTIFACT_ROOT="/workspace/NetLLM-artifacts"
ARTIFACT_PATH="$ARTIFACT_ROOT/plms/gpt2/base"
RUNTIME_ROOT="$PROJECT_ROOT/experiments/vp/phase3a_retry_runtime"
RUNNER="$PROJECT_ROOT/scripts/experiment_phase/phase3a/run_phase3a_identity_equivalence.py"
OUTPUT_PATH="$RUNTIME_ROOT/identity_equivalence.json"
LOG_PATH="$RUNTIME_ROOT/identity_equivalence.log"
TEST_LOG_PATH="$RUNTIME_ROOT/tests.log"
STATUS_PATH="$RUNTIME_ROOT/run_status.txt"

artifact_fingerprint() {
    sha256sum \
        "$ARTIFACT_PATH/config.json" \
        "$ARTIFACT_PATH/model.safetensors" \
        "$ARTIFACT_PATH/tokenizer.json" \
        "$ARTIFACT_PATH/merges.txt" \
        "$ARTIFACT_PATH/vocab.json" | sha256sum | cut -d' ' -f1
}

if [[ -e "$RUNTIME_ROOT" ]]; then
    echo "ERROR: Phase 3A retry runtime already exists; nothing was overwritten: $RUNTIME_ROOT" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNNER" || ! -d "$ARTIFACT_PATH" ]]; then
    echo "ERROR: required repro Python, runner, or GPT-2 artifact is missing" >&2
    exit 2
fi
if [[ ! -d "$SOURCE_ROOT/.git" || ! -d "$VP_ROOT" ]]; then
    echo "ERROR: upstream NetLLM checkout is missing" >&2
    exit 2
fi

python_executable="$($PYTHON_BIN -c 'import sys; print(sys.executable)')"
if [[ "$python_executable" != "$PYTHON_BIN" ]]; then
    echo "ERROR: unexpected Python executable: $python_executable" >&2
    exit 2
fi

before_commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
before_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
before_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
before_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
if [[ -n "$before_status" || -n "$before_diff" || "$before_pycache" -ne 0 ]]; then
    echo "ERROR: upstream is not clean before Phase 3A retry" >&2
    exit 2
fi

python_version="$($PYTHON_BIN --version 2>&1)"
torch_version="$($PYTHON_BIN -c 'import torch; print(torch.__version__)')"
cuda_available="$($PYTHON_BIN -c 'import torch; print(torch.cuda.is_available())')"
before_freeze_hash="$($PYTHON_BIN -m pip freeze | sha256sum | cut -d' ' -f1)"
before_artifact_hash="$(artifact_fingerprint)"

mkdir -p "$RUNTIME_ROOT"

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
cd "$PROJECT_ROOT" || exit 2
{
    echo "===== IdentitySelector unit test ====="
    "$PYTHON_BIN" -B -m unittest tests.phase3a.test_identity_selector -v
} 2>&1 | tee "$TEST_LOG_PATH"
unit_test_status=${PIPESTATUS[0]}

runner_status=not_started
full_test_status=not_started
if [[ $unit_test_status -eq 0 ]]; then
    cd "$VP_ROOT" || exit 2
    "$PYTHON_BIN" -B "$RUNNER" --output "$OUTPUT_PATH" 2>&1 | tee "$LOG_PATH"
    runner_status=${PIPESTATUS[0]}

    if [[ $runner_status -eq 0 && -f "$OUTPUT_PATH" ]]; then
        cd "$PROJECT_ROOT" || exit 2
        {
            echo "===== Full Phase 3A tests ====="
            "$PYTHON_BIN" -B -m unittest discover -s tests/phase3a -p 'test_*.py' -v
        } 2>&1 | tee -a "$TEST_LOG_PATH"
        full_test_status=${PIPESTATUS[0]}
    fi
fi
end_epoch="$(date +%s)"

after_commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
after_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
after_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
after_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
after_freeze_hash="$($PYTHON_BIN -m pip freeze | sha256sum | cut -d' ' -f1)"
after_artifact_hash="$(artifact_fingerprint)"

final_status=0
if [[ $unit_test_status -ne 0 ]]; then
    final_status=7
elif [[ "$runner_status" == "not_started" || $runner_status -ne 0 ]]; then
    final_status=1
elif [[ "$full_test_status" == "not_started" || $full_test_status -ne 0 ]]; then
    final_status=6
fi

upstream_clean=yes
environment_unchanged=yes
artifact_unchanged=yes
if [[ "$before_commit" != "$after_commit" || -n "$after_status" || -n "$after_diff" || "$after_pycache" -ne 0 ]]; then
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
    echo "python_executable=$python_executable"
    echo "python_version=$python_version"
    echo "torch_version=$torch_version"
    echo "cuda_available=$cuda_available"
    echo "upstream_commit_before=$before_commit"
    echo "upstream_commit_after=$after_commit"
    echo "upstream_status_before_begin"
    printf '%s\n' "$before_status"
    echo "upstream_status_before_end"
    echo "upstream_diff_before_begin"
    printf '%s\n' "$before_diff"
    echo "upstream_diff_before_end"
    echo "upstream_status_after_begin"
    printf '%s\n' "$after_status"
    echo "upstream_status_after_end"
    echo "upstream_diff_after_begin"
    printf '%s\n' "$after_diff"
    echo "upstream_diff_after_end"
    echo "upstream_pycache_before=$before_pycache"
    echo "upstream_pycache_after=$after_pycache"
    echo "unit_test_exit_code=$unit_test_status"
    echo "runner_exit_code=$runner_status"
    echo "full_test_exit_code=$full_test_status"
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
    echo "runner_log_path=$LOG_PATH"
    echo "tests_log_path=$TEST_LOG_PATH"
} > "$STATUS_PATH"

echo "PHASE3A_RETRY_UNIT_TEST_EXIT_CODE=$unit_test_status"
echo "PHASE3A_RETRY_RUNNER_EXIT_CODE=$runner_status"
echo "PHASE3A_RETRY_FULL_TEST_EXIT_CODE=$full_test_status"
echo "PHASE3A_RETRY_FINAL_EXIT_CODE=$final_status"
echo "PHASE3A_RETRY_UPSTREAM_CLEAN_AFTER=$upstream_clean"
echo "PHASE3A_RETRY_ENVIRONMENT_UNCHANGED=$environment_unchanged"
echo "PHASE3A_RETRY_ARTIFACT_UNCHANGED=$artifact_unchanged"
echo "PHASE3A_RETRY_OUTPUT_PATH=$OUTPUT_PATH"

exit "$final_status"
