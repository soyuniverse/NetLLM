#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/venv/vp_netllm/bin/python"
VP_ROOT="/workspace/NetLLM-source/viewport_prediction"
LEGACY_SITE="/venv/vp_netllm_repro/lib/python3.8/site-packages"
OVERLAY_SITE="/venv/vp_netllm_plmtest/lib/python3.8/site-packages"
EXPECTED_CUDA_TOOLKIT="12.1"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found; expected CUDA $EXPECTED_CUDA_TOOLKIT-devel" >&2
    exit 2
fi
actual_cuda_toolkit="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
if [[ "$actual_cuda_toolkit" != "$EXPECTED_CUDA_TOOLKIT" ]]; then
    echo "ERROR: CUDA toolkit mismatch: expected $EXPECTED_CUDA_TOOLKIT, got ${actual_cuda_toolkit:-unknown}" >&2
    exit 2
fi
nvcc --version

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: reproducible environment Python not found: $PYTHON_BIN" >&2
    exit 2
fi

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

echo "===== Python and environment isolation ====="
"$PYTHON_BIN" --version
"$PYTHON_BIN" -B - "$LEGACY_SITE" "$OVERLAY_SITE" <<'PY'
import site
import sys

legacy_site, overlay_site = sys.argv[1:]
assert sys.version_info[:3] == (3, 8, 10), sys.version
assert sys.prefix == "/venv/vp_netllm", sys.prefix
assert sys.base_prefix == "/venv/vp_netllm", sys.base_prefix
assert not site.ENABLE_USER_SITE
assert legacy_site not in sys.path, sys.path
assert overlay_site not in sys.path, sys.path
assert all(path.startswith(sys.prefix) for path in site.getsitepackages())
print(f"python_executable={sys.executable}")
print(f"sys_prefix={sys.prefix}")
print(f"sys_base_prefix={sys.base_prefix}")
print(f"site_packages={site.getsitepackages()}")
print("legacy_site_packages_referenced=no")
print("overlay_site_packages_referenced=no")
PY

echo "===== Exact package versions ====="
"$PYTHON_BIN" -B - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.2.0+cu121",
    "torchvision": "0.17.0+cu121",
    "torchaudio": "2.2.0+cu121",
    "transformers": "4.34.1",
    "peft": "0.6.2",
    "accelerate": "0.24.1",
    "huggingface-hub": "0.17.3",
    "safetensors": "0.5.3",
    "tokenizers": "0.14.1",
    "numpy": "1.24.4",
    "opencv-python-headless": "4.8.1.78",
    "yacs": "0.1.8",
}
for name, wanted in expected.items():
    actual = version(name)
    assert actual == wanted, f"{name}: expected {wanted}, got {actual}"
    print(f"{name}=={actual}")
PY

echo "===== pip check ====="
"$PYTHON_BIN" -m pip check

echo "===== Runtime imports and CUDA visibility ====="
"$PYTHON_BIN" -B - <<'PY'
import accelerate
import cv2
import einops
import huggingface_hub
import numpy
import pandas
import peft
import scipy
import sklearn
import torch
import transformers
import yacs

for name in (
    "torch", "transformers", "peft", "accelerate", "huggingface_hub",
    "cv2", "yacs", "numpy", "pandas", "scipy", "sklearn", "einops",
):
    print(f"{name} import: OK")
print(f"torch.version.cuda={torch.version.cuda}")
available = torch.cuda.is_available()
print(f"torch.cuda.is_available()={available}")
if not available:
    raise RuntimeError("CUDA is not available")
print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
PY

echo "===== NetLLM VP entry-point imports ====="
export PYTHONPATH="$VP_ROOT"
cd "$VP_ROOT"
"$PYTHON_BIN" -B - <<'PY'
import run_baseline
import run_plm

print("run_baseline import: OK")
print("run_plm import: OK")
PY
