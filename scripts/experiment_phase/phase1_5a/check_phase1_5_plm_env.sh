#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/venv/vp_netllm_plmtest/bin/python"
VP_ROOT="/workspace/NetLLM-source/viewport_prediction"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: test environment Python not found: $PYTHON_BIN" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$VP_ROOT"

echo "===== Python ====="
"$PYTHON_BIN" --version
echo "$PYTHON_BIN"

echo "===== Package versions ====="
"$PYTHON_BIN" -B - <<'PY'
from importlib.metadata import version

for name in (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "huggingface-hub",
    "safetensors",
    "tokenizers",
    "opencv-python-headless",
    "yacs",
):
    print(f"{name}=={version(name)}")
PY

echo "===== pip check ====="
"$PYTHON_BIN" -m pip check

echo "===== Core imports and CUDA visibility ====="
"$PYTHON_BIN" -B - <<'PY'
import accelerate
import cv2
import huggingface_hub
import peft
import torch
import transformers
import yacs

print("torch import: OK")
print("transformers import: OK")
print("peft import: OK")
print("accelerate import: OK")
print("huggingface_hub import: OK")
print("cv2 import: OK")
print("yacs import: OK")
print("torch CUDA runtime:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
PY

echo "===== NetLLM VP imports ====="
cd "$VP_ROOT"
"$PYTHON_BIN" -B - <<'PY'
import run_baseline
import run_plm

print("run_baseline import: OK")
print("run_plm import: OK")
PY

