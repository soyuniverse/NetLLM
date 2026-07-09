#!/usr/bin/env bash
set -euo pipefail

NETLLM_DIR="${NETLLM_DIR:-/workspace/NetLLM-source}"
ENV_NAME="${ENV_NAME:-vp_netllm}"
ENV_PREFIX="${ENV_PREFIX:-/venv/$ENV_NAME}"

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
