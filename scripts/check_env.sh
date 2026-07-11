#!/usr/bin/env bash
set -euo pipefail

NETLLM_DIR="${NETLLM_DIR:-/workspace/NetLLM-source}"
ENV_NAME="${ENV_NAME:-vp_netllm}"
ENV_PREFIX="${ENV_PREFIX:-/venv/$ENV_NAME}"
EXPECTED_CUDA_TOOLKIT="${EXPECTED_CUDA_TOOLKIT:-12.1}"
EXPECTED_TORCH="${EXPECTED_TORCH:-2.2.0+cu121}"
EXPECTED_TORCHVISION="${EXPECTED_TORCHVISION:-0.17.0+cu121}"
EXPECTED_TORCHAUDIO="${EXPECTED_TORCHAUDIO:-2.2.0+cu121}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.1}"

echo "===== CUDA devel toolkit ====="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found; expected CUDA $EXPECTED_CUDA_TOOLKIT-devel"
  exit 1
fi
actual_cuda_toolkit="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
if [ "$actual_cuda_toolkit" != "$EXPECTED_CUDA_TOOLKIT" ]; then
  echo "ERROR: CUDA toolkit mismatch: expected $EXPECTED_CUDA_TOOLKIT, got ${actual_cuda_toolkit:-unknown}"
  exit 1
fi
nvcc --version

if [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source /opt/miniforge3/etc/profile.d/conda.sh
else
  echo "ERROR: /opt/miniforge3/etc/profile.d/conda.sh not found"
  exit 1
fi

if [ -d "$ENV_PREFIX" ]; then
  conda activate "$ENV_PREFIX"
else
  conda activate "$ENV_NAME"
fi

echo "===== Python / pip ====="
python --version
which python
python -m pip --version

echo ""
echo "===== Python package check ====="
python - "$EXPECTED_TORCH" "$EXPECTED_TORCHVISION" "$EXPECTED_TORCHAUDIO" "$EXPECTED_TORCH_CUDA" <<'PY'
import sys
import torch
import torchvision
import torchaudio
import cv2
import yacs

expected_torch, expected_vision, expected_audio, expected_cuda = sys.argv[1:]
assert torch.__version__ == expected_torch, (torch.__version__, expected_torch)
assert torchvision.__version__ == expected_vision, (torchvision.__version__, expected_vision)
assert torchaudio.__version__ == expected_audio, (torchaudio.__version__, expected_audio)
assert torch.version.cuda == expected_cuda, (torch.version.cuda, expected_cuda)

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print("gpu: unavailable")
print("cv2:", cv2.__version__)
print("yacs import: ok")
PY

echo ""
echo "===== NetLLM commit ====="
if [ -d "$NETLLM_DIR/.git" ]; then
  git -C "$NETLLM_DIR" rev-parse HEAD
  git -C "$NETLLM_DIR" status --short
else
  echo "NetLLM repo not found at $NETLLM_DIR"
fi

echo ""
echo "===== VP directory files ====="
if [ -d "$NETLLM_DIR/viewport_prediction" ]; then
  find "$NETLLM_DIR/viewport_prediction" -maxdepth 2 -type f | sort | sed "s#^$NETLLM_DIR/viewport_prediction/##" | head -120
else
  echo "VP directory not found at $NETLLM_DIR/viewport_prediction"
fi
