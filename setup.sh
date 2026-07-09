#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
NETLLM_DIR="${NETLLM_DIR:-$WORKDIR/NetLLM}"
LOG_DIR="${LOG_DIR:-$WORKDIR/research_logs}"
ENV_NAME="${ENV_NAME:-vp_netllm}"
ENV_PREFIX="${ENV_PREFIX:-/venv/$ENV_NAME}"
NETLLM_REPO="${NETLLM_REPO:-https://github.com/duowuyms/NetLLM.git}"
NETLLM_COMMIT="${NETLLM_COMMIT:-105bcf070f2bec808f7b14f8f5a953de6e4e6e54}"
INSTALL_DEV_TOOLS="${INSTALL_DEV_TOOLS:-1}"
INSTALL_CODEX_CLI="${INSTALL_CODEX_CLI:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORKDIR" "$LOG_DIR"

if [ "$(realpath -m "$SCRIPT_DIR")" = "$(realpath -m "$NETLLM_DIR")" ]; then
  echo "ERROR: wrapper repo path conflicts with NETLLM_DIR: $NETLLM_DIR"
  echo "Clone this wrapper repo into a separate path, for example: /workspace/netllm-vp-setup"
  exit 1
fi

echo "===== Recording Vast.ai environment ====="
{
  echo "===== DATE ====="
  date
  echo ""
  echo "===== WHOAMI / PWD ====="
  whoami
  pwd
  echo ""
  echo "===== GPU / DRIVER ====="
  nvidia-smi || true
  echo ""
  echo "===== CUDA ====="
  nvcc --version || echo "nvcc not found"
  echo ""
  echo "===== PYTHON ====="
  python --version || true
  which python || true
  echo ""
  echo "===== CONDA ====="
  which conda || echo "conda not found"
  conda --version || echo "conda version not available"
  echo ""
  echo "===== DISK ====="
  df -h
  echo ""
  echo "===== MEMORY ====="
  free -h
} | tee "$LOG_DIR/00_vast_env_check.txt"

if [ "$INSTALL_DEV_TOOLS" = "1" ]; then
  echo "===== Installing/checking developer tools ====="
  {
    echo "===== CODEX CLI ====="
    if command -v codex >/dev/null 2>&1; then
      codex --version || true
    elif [ "$INSTALL_CODEX_CLI" = "1" ]; then
      echo "Installing Codex CLI with the official install script."
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://chatgpt.com/codex/install.sh | sh || echo "Codex CLI install failed; continue setup."
      else
        echo "curl not found; skip Codex CLI install."
      fi
      export PATH="$HOME/.local/bin:$PATH"
      codex --version || echo "Codex CLI not available on PATH after install."
    else
      echo "INSTALL_CODEX_CLI=0, skip Codex CLI install."
    fi

    echo ""
    echo "===== JUPYTER ====="
    jupyter --version || echo "jupyter not found on PATH; Vast template may expose it differently."

    echo ""
    echo "===== VS CODE SERVER ====="
    if command -v code >/dev/null 2>&1; then
      code --version || true
    else
      echo "code CLI not found. VS Code Remote usually installs its server when you connect."
    fi
  } | tee "$LOG_DIR/02_dev_tools_check.txt"
else
  echo "===== Skipping developer tools check/install: INSTALL_DEV_TOOLS=$INSTALL_DEV_TOOLS ====="
fi

echo "===== Loading conda ====="
if [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source /opt/miniforge3/etc/profile.d/conda.sh
else
  echo "ERROR: /opt/miniforge3/etc/profile.d/conda.sh not found"
  exit 1
fi

echo "===== Creating or reusing conda env: $ENV_NAME ====="
if [ -d "$ENV_PREFIX" ]; then
  echo "Conda env already exists at: $ENV_PREFIX"
elif conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env already exists by name: $ENV_NAME"
else
  mkdir -p "$(dirname "$ENV_PREFIX")"
  conda create -p "$ENV_PREFIX" python=3.8.10 -y
fi

if [ -d "$ENV_PREFIX" ]; then
  conda activate "$ENV_PREFIX"
else
  conda activate "$ENV_NAME"
fi

echo "===== Python check ====="
python --version
which python

echo "===== Installing PyTorch 2.1.0 cu118 ====="
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "===== Installing VP requirements ====="
python -m pip install -r "$SCRIPT_DIR/requirements-vp.txt"

echo "===== Cloning or updating NetLLM ====="
cd "$WORKDIR"

if [ ! -d "$NETLLM_DIR/.git" ]; then
  git clone "$NETLLM_REPO" "$NETLLM_DIR"
else
  echo "NetLLM already exists: $NETLLM_DIR"
  current_origin="$(git -C "$NETLLM_DIR" remote get-url origin || true)"
  if [ "$current_origin" != "$NETLLM_REPO" ]; then
    echo "ERROR: existing NetLLM origin is '$current_origin', expected '$NETLLM_REPO'"
    exit 1
  fi
fi

cd "$NETLLM_DIR"
git fetch origin
git checkout "$NETLLM_COMMIT"
git rev-parse HEAD | tee "$LOG_DIR/01_netllm_commit.txt"

echo "===== Verifying Python imports and GPU ====="
python - <<'PY'
import torch
import cv2
import yacs

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print("cv2:", cv2.__version__)
print("yacs import: ok")
PY

echo "===== VP directory check ====="
cd "$NETLLM_DIR/viewport_prediction"
ls -la
echo ""
echo "Data files preview:"
if [ -d data ]; then
  find data -maxdepth 3 -type f | sort | head -50
else
  echo "data directory not found. Baseline/adaptation runs may fail until datasets are available."
fi

echo ""
echo "===== Setup complete ====="
echo "Next commands:"
echo "  conda activate $ENV_PREFIX"
echo "  cd $NETLLM_DIR/viewport_prediction"
echo "  bash $SCRIPT_DIR/scripts/check_env.sh"
echo "  bash $SCRIPT_DIR/scripts/run_vp_regression_cpu.sh"
echo "  bash $SCRIPT_DIR/scripts/run_vp_gpt2_adapt_e1.sh"
