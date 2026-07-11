#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/NetLLM"
ENV_PREFIX="/venv/vp_netllm_repro"
CONDA_SH="/opt/miniforge3/etc/profile.d/conda.sh"
REQUIREMENTS="$PROJECT_ROOT/requirements-vp.txt"
OUTPUT_ROOT="$PROJECT_ROOT/experiments/vp/phase1_5b_runtime"
LOG_DIR="$OUTPUT_ROOT/logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_DIR/${TIMESTAMP}_environment_creation.log"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

echo "PHASE1_5B_ENV_CREATION_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PHASE1_5B_ENV_PREFIX=$ENV_PREFIX"
echo "PHASE1_5B_REQUIREMENTS=$REQUIREMENTS"
df -h /venv "$PROJECT_ROOT"

if [[ -e "$ENV_PREFIX" ]]; then
    echo "ERROR: target environment already exists; it was not modified: $ENV_PREFIX" >&2
    exit 2
fi

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda initialization script not found: $CONDA_SH" >&2
    exit 2
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "ERROR: requirements file not found: $REQUIREMENTS" >&2
    exit 2
fi

# shellcheck source=/dev/null
source "$CONDA_SH"

echo "===== Creating independent Python 3.8.10 environment ====="
conda create --prefix "$ENV_PREFIX" python=3.8.10 -y

PYTHON_BIN="$ENV_PREFIX/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: environment Python was not created: $PYTHON_BIN" >&2
    exit 2
fi

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

echo "===== Upgrading pip inside the new environment ====="
"$PYTHON_BIN" -m pip install --upgrade pip

echo "===== Installing PyTorch 2.1.0 cu118 ====="
"$PYTHON_BIN" -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "===== Installing project VP requirements ====="
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS"

echo "===== Recording independent environment paths ====="
"$PYTHON_BIN" -B - <<'PY'
import site
import sys

print(f"python_version={sys.version.split()[0]}")
print(f"python_executable={sys.executable}")
print(f"sys_prefix={sys.prefix}")
print(f"sys_base_prefix={sys.base_prefix}")
print(f"site_packages={site.getsitepackages()}")
print(f"user_site_enabled={site.ENABLE_USER_SITE}")
PY

echo "===== pip check ====="
"$PYTHON_BIN" -m pip check

echo "===== pip freeze ====="
"$PYTHON_BIN" -m pip freeze
echo "PHASE1_5B_ENV_CREATION_FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PHASE1_5B_ENV_CREATION_LOG=$LOG_PATH"
