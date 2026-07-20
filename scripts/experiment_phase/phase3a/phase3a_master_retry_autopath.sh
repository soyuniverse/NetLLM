#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# NetLLM Phase 3A Master Retry
# - Run this ONLY inside the original Vast.ai instance.
# - Does not modify /workspace/NetLLM-source.
# - Does not install packages or download models.
# - Preserves the failed Phase 3A runtime.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_project_root() {
    if [[ -n "${NETLLM_PROJECT_ROOT:-}" ]]; then
        printf '%s\n' "$NETLLM_PROJECT_ROOT"
        return
    fi

    local git_root
    git_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "$git_root" ]]; then
        printf '%s\n' "$git_root"
        return
    fi

    for candidate in \
        "/root/NetLLM" \
        "/workspace/NetLLM"
    do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
}

detect_first_dir() {
    local override="$1"
    shift
    if [[ -n "$override" && -d "$override" ]]; then
        printf '%s\n' "$override"
        return
    fi

    local candidate
    for candidate in "$@"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
}

detect_first_executable() {
    local override="$1"
    shift
    if [[ -n "$override" && -x "$override" ]]; then
        printf '%s\n' "$override"
        return
    fi

    local candidate
    for candidate in "$@"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
}

PROJECT_ROOT="$(detect_project_root)"
SOURCE_ROOT="$(detect_first_dir "${NETLLM_SOURCE_ROOT:-}" \
    "${PROJECT_ROOT:-}/../NetLLM-source" \
    "/root/NetLLM-source" \
    "/workspace/NetLLM-source")"
PYTHON_BIN="$(detect_first_executable "${NETLLM_PYTHON_BIN:-}" \
    "/venv/vp_netllm_repro/bin/python" \
    "/root/venv/vp_netllm_repro/bin/python" \
    "${PROJECT_ROOT:-}/.venv/bin/python")"
ARTIFACT_PATH="$(detect_first_dir "${NETLLM_ARTIFACT_PATH:-}" \
    "${PROJECT_ROOT:-}/../NetLLM-artifacts/plms/gpt2/base" \
    "/root/NetLLM-artifacts/plms/gpt2/base" \
    "/workspace/NetLLM-artifacts/plms/gpt2/base")"

SOURCE_ROOT="${SOURCE_ROOT:-__MISSING_SOURCE_ROOT__}"
PYTHON_BIN="${PYTHON_BIN:-__MISSING_PYTHON__}"
ARTIFACT_PATH="${ARTIFACT_PATH:-__MISSING_ARTIFACT__}"
VP_ROOT="$SOURCE_ROOT/viewport_prediction"
ARTIFACT_ROOT="$(dirname "$(dirname "$(dirname "$ARTIFACT_PATH")")")"

ORIGINAL_RUNNER="$PROJECT_ROOT/scripts/experiment_phase/phase3a/run_phase3a_identity_equivalence.py"
RUNTIME_ROOT="$PROJECT_ROOT/experiments/vp/phase3a_retry_runtime"
PATCHED_RUNNER="$RUNTIME_ROOT/run_phase3a_identity_equivalence_retry_patched.py"
OUTPUT_JSON="$RUNTIME_ROOT/identity_equivalence.json"

PREFLIGHT_LOG="$RUNTIME_ROOT/preflight.log"
UNIT_LOG="$RUNTIME_ROOT/identity_unit_tests.log"
RUN_LOG="$RUNTIME_ROOT/identity_equivalence.log"
STATUS_FILE="$RUNTIME_ROOT/run_status.txt"
RESULT_FILE="$RUNTIME_ROOT/MASTER_RESULT.txt"
UPSTREAM_BEFORE="$RUNTIME_ROOT/upstream_before.txt"
UPSTREAM_AFTER="$RUNTIME_ROOT/upstream_after.txt"

EXPECTED_COMMIT="105bcf070f2bec808f7b14f8f5a953de6e4e6e54"
EXPECTED_PYTHON="3.8.10"
EXPECTED_TORCH="2.1.0+cu118"
TOLERANCE="1e-7"

fail() {
    echo
    echo "============================================================"
    echo "FAILED: $1"
    echo "============================================================"
    exit "${2:-2}"
}

require_dir() {
    [[ -d "$1" ]] || fail "필수 폴더가 없습니다: $1"
}

require_file() {
    [[ -f "$1" ]] || fail "필수 파일이 없습니다: $1"
}

require_executable() {
    [[ -x "$1" ]] || fail "실행 가능한 Python이 없습니다: $1"
}

artifact_fingerprint() {
    sha256sum \
        "$ARTIFACT_PATH/config.json" \
        "$ARTIFACT_PATH/model.safetensors" \
        "$ARTIFACT_PATH/tokenizer.json" \
        "$ARTIFACT_PATH/merges.txt" \
        "$ARTIFACT_PATH/vocab.json" \
        | sha256sum | cut -d' ' -f1
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

run_and_capture() {
    local log_path="$1"
    shift
    set +e
    "$@" 2>&1 | tee "$log_path"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

echo "============================================================"
echo "NetLLM Phase 3A Master Retry"
echo "============================================================"
echo "이 스크립트는 설치/다운로드를 하지 않습니다."
echo "필수 경로가 없으면 즉시 중단합니다."
echo
echo "자동 탐색 결과:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  SOURCE_ROOT=$SOURCE_ROOT"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  ARTIFACT_PATH=$ARTIFACT_PATH"
echo
echo "잘못 탐지되면 아래 환경변수로 직접 지정할 수 있습니다:"
echo "  NETLLM_PROJECT_ROOT"
echo "  NETLLM_SOURCE_ROOT"
echo "  NETLLM_PYTHON_BIN"
echo "  NETLLM_ARTIFACT_PATH"
echo

# ------------------------------------------------------------
# 1. Preflight: required paths
# ------------------------------------------------------------

[[ -n "$PROJECT_ROOT" ]] || fail "프로젝트 Git root를 자동 탐색하지 못했습니다."
require_dir "$PROJECT_ROOT"
require_dir "$SOURCE_ROOT"
require_dir "$VP_ROOT"
require_executable "$PYTHON_BIN"
require_dir "$ARTIFACT_PATH"
require_file "$ORIGINAL_RUNNER"
require_file "$PROJECT_ROOT/tests/phase3a/test_identity_selector.py"

for artifact_file in \
    config.json \
    model.safetensors \
    tokenizer.json \
    merges.txt \
    vocab.json
do
    require_file "$ARTIFACT_PATH/$artifact_file"
done

if [[ -e "$RUNTIME_ROOT" ]]; then
    fail "Retry 결과 폴더가 이미 존재합니다. 덮어쓰지 않습니다: $RUNTIME_ROOT"
fi

before_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
before_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
before_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
before_commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"

[[ "$before_commit" == "$EXPECTED_COMMIT" ]] || \
    fail "NetLLM 원본 commit이 다릅니다: $before_commit"

[[ -z "$before_status" ]] || fail "NetLLM 원본 Git status가 clean이 아닙니다."
[[ -z "$before_diff" ]] || fail "NetLLM 원본 Git diff가 존재합니다."
[[ "$before_pycache" -eq 0 ]] || fail "NetLLM 원본에 __pycache__가 존재합니다."

mkdir -p "$RUNTIME_ROOT"
record_upstream "$UPSTREAM_BEFORE"

python_version="$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"
torch_version="$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)')"
python_executable="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
cuda_available="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())')"
gpu_name="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"

[[ "$python_executable" == "$PYTHON_BIN" ]] || \
    fail "실제 Python executable이 다릅니다: $python_executable"

[[ "$python_version" == "$EXPECTED_PYTHON" ]] || \
    fail "Python 버전이 다릅니다: $python_version"

[[ "$torch_version" == "$EXPECTED_TORCH" ]] || \
    fail "Torch 버전이 다릅니다: $torch_version"

[[ "$cuda_available" == "True" ]] || fail "CUDA가 인식되지 않습니다."

before_freeze_hash="$("$PYTHON_BIN" -m pip freeze | sha256sum | cut -d' ' -f1)"
before_artifact_hash="$(artifact_fingerprint)"

{
    echo "project_root=$PROJECT_ROOT"
    echo "source_root=$SOURCE_ROOT"
    echo "python_executable=$python_executable"
    echo "python_version=$python_version"
    echo "torch_version=$torch_version"
    echo "cuda_available=$cuda_available"
    echo "gpu_name=$gpu_name"
    echo "upstream_commit=$before_commit"
    echo "environment_freeze_hash=$before_freeze_hash"
    echo "artifact_fingerprint=$before_artifact_hash"
} | tee "$PREFLIGHT_LOG"

# ------------------------------------------------------------
# 2. Prepare a patched COPY of the runner
#    The original project runner is not modified.
# ------------------------------------------------------------

"$PYTHON_BIN" -B - "$ORIGINAL_RUNNER" "$PATCHED_RUNNER" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text()

old = """    index = list(torch.unravel_index(torch.tensor(flat_index), difference.shape))
    index = [int(item.item()) for item in index]
"""

new = """    index = []
    remaining = flat_index
    for size in reversed(tuple(int(value) for value in difference.shape)):
        index.append(remaining % size)
        remaining //= size
    index = list(reversed(index))
"""

if "torch.unravel_index" in text:
    if old not in text:
        raise RuntimeError(
            "torch.unravel_index가 남아 있지만 예상한 코드 형태와 다릅니다. "
            "자동 수정하지 않습니다."
        )
    text = text.replace(old, new, 1)

if "torch.unravel_index" in text:
    raise RuntimeError("patched runner에 torch.unravel_index가 남아 있습니다.")

if "TOLERANCE = 1e-7" not in text:
    raise RuntimeError("Tolerance가 1e-7인지 확인할 수 없습니다.")

target.write_text(text)
print(f"patched_runner={target}")
PY

# ------------------------------------------------------------
# 3. Fixed offline runtime environment
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 4. IdentitySelector unit tests
# ------------------------------------------------------------

echo
echo "===== IdentitySelector unit tests ====="

if ! run_and_capture \
    "$UNIT_LOG" \
    "$PYTHON_BIN" -B -m unittest discover \
    -s "$PROJECT_ROOT/tests/phase3a" \
    -p "test_identity_selector.py" \
    -v
then
    fail "IdentitySelector unit test가 실패했습니다."
fi

# ------------------------------------------------------------
# 5. Original / Disabled / Identity equivalence
# ------------------------------------------------------------

echo
echo "===== Original / Disabled / Identity equivalence ====="

cd "$VP_ROOT"

if ! run_and_capture \
    "$RUN_LOG" \
    "$PYTHON_BIN" -B "$PATCHED_RUNNER" \
    --output "$OUTPUT_JSON"
then
    fail "Equivalence runner가 실패했습니다. 로그: $RUN_LOG"
fi

require_file "$OUTPUT_JSON"

# ------------------------------------------------------------
# 6. Result validation
#    Kept inside this master script so no separate retry test file is required.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7. Independent JSON validation and concise result
# ------------------------------------------------------------

"$PYTHON_BIN" -B - "$OUTPUT_JSON" "$RESULT_FILE" "$TOLERANCE" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
tolerance = float(sys.argv[3])

data = json.loads(result_path.read_text())

assert data["success"] is True
expected_comparisons = {
    "original_vs_disabled",
    "original_vs_identity",
    "disabled_vs_identity",
}
assert set(data["comparisons"]) == expected_comparisons

for name, comparison in data["comparisons"].items():
    assert comparison["max_absolute_difference"] <= tolerance, (name, comparison)
    assert comparison["within_tolerance"] is True
    assert comparison["rtol"] == 0.0
    assert comparison["atol"] == tolerance

expected_lengths = list(range(10, 30))
for name, path in data["paths"].items():
    assert path["output"]["shape"] == [1, 20, 3], name
    assert path["output"]["finite"] is True, name
    assert path["trace"]["sequence_lengths"] == expected_lengths, name
    assert path["trace"]["plm_forward_count"] == 20, name
    assert not any(path["trace"]["past_key_values_passed"]), name

selection = data["identity_selection_output"]
assert selection["original_length"] == 10
assert selection["selected_length"] == 10
assert selection["selected_indices"] == list(range(10))
assert selection["scores"] is None

lines = [
    "PHASE 3A RETRY: SUCCESS",
    "",
    f"Original vs Disabled max diff: "
    f"{data['comparisons']['original_vs_disabled']['max_absolute_difference']}",
    f"Original vs Identity max diff: "
    f"{data['comparisons']['original_vs_identity']['max_absolute_difference']}",
    f"Disabled vs Identity max diff: "
    f"{data['comparisons']['disabled_vs_identity']['max_absolute_difference']}",
    "",
    "Output shape: [1, 20, 3]",
    "Sequence lengths: 10..29",
    "GPT-2 forward count: 20 per path",
    "Identity selected length: 10",
    "Identity selected indices: 0..9",
    "",
    f"Full JSON: {result_path}",
]
summary_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

# ------------------------------------------------------------
# 8. Integrity checks after execution
# ------------------------------------------------------------

record_upstream "$UPSTREAM_AFTER"

after_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
after_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"
after_pycache="$(find "$SOURCE_ROOT" -type d -name __pycache__ | wc -l)"
after_commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
after_freeze_hash="$("$PYTHON_BIN" -m pip freeze | sha256sum | cut -d' ' -f1)"
after_artifact_hash="$(artifact_fingerprint)"

[[ "$after_commit" == "$before_commit" ]] || fail "실행 후 upstream commit이 달라졌습니다."
[[ -z "$after_status" ]] || fail "실행 후 upstream status가 clean이 아닙니다."
[[ -z "$after_diff" ]] || fail "실행 후 upstream diff가 존재합니다."
[[ "$after_pycache" -eq 0 ]] || fail "실행 후 upstream에 __pycache__가 생성됐습니다."
[[ "$after_freeze_hash" == "$before_freeze_hash" ]] || fail "Python 환경이 변경됐습니다."
[[ "$after_artifact_hash" == "$before_artifact_hash" ]] || fail "GPT-2 artifact가 변경됐습니다."

{
    echo "final_exit_code=0"
    echo "phase3a_complete=yes"
    echo "python_executable=$python_executable"
    echo "python_version=$python_version"
    echo "torch_version=$torch_version"
    echo "cuda_available=$cuda_available"
    echo "gpu_name=$gpu_name"
    echo "upstream_commit=$after_commit"
    echo "upstream_clean_after=yes"
    echo "environment_unchanged=yes"
    echo "artifact_unchanged=yes"
    echo "identity_json=$OUTPUT_JSON"
    echo "master_result=$RESULT_FILE"
} > "$STATUS_FILE"

echo
echo "============================================================"
echo "SUCCESS: Phase 3A Identity equivalence 검증 완료"
echo "============================================================"
echo "결과 요약: $RESULT_FILE"
echo "전체 JSON: $OUTPUT_JSON"
echo "상태 파일: $STATUS_FILE"
echo
echo "다음 파일의 내용을 전달하면 다음 단계 판단이 가능합니다:"
echo "  $RESULT_FILE"