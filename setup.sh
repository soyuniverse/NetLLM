#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
NETLLM_DIR="${NETLLM_DIR:-$WORKDIR/NetLLM-source}"
LOG_DIR="${LOG_DIR:-$WORKDIR/research_logs}"
ENV_NAME="${ENV_NAME:-vp_netllm}"
ENV_PREFIX="${ENV_PREFIX:-/venv/$ENV_NAME}"
NETLLM_REPO="${NETLLM_REPO:-https://github.com/duowuyms/NetLLM.git}"
NETLLM_COMMIT="${NETLLM_COMMIT:-105bcf070f2bec808f7b14f8f5a953de6e4e6e54}"
INSTALL_DEV_TOOLS="${INSTALL_DEV_TOOLS:-1}"
INSTALL_CODEX_CLI="${INSTALL_CODEX_CLI:-1}"
VP_TORCH_VERSION="${VP_TORCH_VERSION:-2.2.0}"
VP_TORCHVISION_VERSION="${VP_TORCHVISION_VERSION:-0.17.0}"
VP_TORCHAUDIO_VERSION="${VP_TORCHAUDIO_VERSION:-2.2.0}"
VP_CUDA_TOOLKIT_VERSION="${VP_CUDA_TOOLKIT_VERSION:-12.1}"
VP_TORCH_CUDA_RUNTIME="${VP_TORCH_CUDA_RUNTIME:-12.1}"
VP_TORCH_INDEX_URL="${VP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORKDIR" "$LOG_DIR"

if [ "$(realpath -m "$SCRIPT_DIR")" = "$(realpath -m "$NETLLM_DIR")" ]; then
  echo "ERROR: wrapper repo path conflicts with NETLLM_DIR: $NETLLM_DIR"
  echo "Set NETLLM_DIR to a different path, for example: /workspace/NetLLM-source"
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

echo "===== Verifying CUDA devel toolkit ====="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found; use a CUDA $VP_CUDA_TOOLKIT_VERSION-devel image."
  exit 1
fi
actual_cuda_toolkit="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
if [ "$actual_cuda_toolkit" != "$VP_CUDA_TOOLKIT_VERSION" ]; then
  echo "ERROR: CUDA toolkit mismatch: expected $VP_CUDA_TOOLKIT_VERSION, got ${actual_cuda_toolkit:-unknown}."
  echo "Use pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel or an equivalent CUDA 12.1-devel image."
  exit 1
fi
echo "CUDA toolkit: $actual_cuda_toolkit"

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

echo "===== Installing PyTorch $VP_TORCH_VERSION with CUDA $VP_TORCH_CUDA_RUNTIME wheels ====="
echo "NOTE: The CUDA devel toolkit comes from the Vast/Docker template; pip installs the matching PyTorch runtime wheels only."
python -m pip install --upgrade pip
python -m pip install \
  "torch==$VP_TORCH_VERSION" \
  "torchvision==$VP_TORCHVISION_VERSION" \
  "torchaudio==$VP_TORCHAUDIO_VERSION" \
  --index-url "$VP_TORCH_INDEX_URL"

echo "===== Installing VP requirements ====="
python -m pip install -r "$SCRIPT_DIR/requirements-vp.txt"

echo "===== Recording resolved Python dependencies ====="
python -m pip check | tee "$LOG_DIR/03_pip_check.txt"
python -m pip freeze | tee "$LOG_DIR/04_pip_freeze.txt"

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
python - "$VP_TORCH_VERSION" "$VP_TORCHVISION_VERSION" "$VP_TORCHAUDIO_VERSION" "$VP_TORCH_CUDA_RUNTIME" <<'PY'
import sys
import torch
import torchvision
import torchaudio
import cv2
import yacs

expected_torch, expected_vision, expected_audio, expected_cuda = sys.argv[1:]
actual_torch = torch.__version__.split("+", 1)[0]
if actual_torch != expected_torch:
    raise RuntimeError(f"torch version mismatch: expected {expected_torch}, got {torch.__version__}")
if torchvision.__version__.split("+", 1)[0] != expected_vision:
    raise RuntimeError(f"torchvision version mismatch: expected {expected_vision}, got {torchvision.__version__}")
if torchaudio.__version__.split("+", 1)[0] != expected_audio:
    raise RuntimeError(f"torchaudio version mismatch: expected {expected_audio}, got {torchaudio.__version__}")
if torch.version.cuda != expected_cuda:
    raise RuntimeError(f"PyTorch CUDA runtime mismatch: expected {expected_cuda}, got {torch.version.cuda}")

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
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
