#!/usr/bin/env bash
set -euo pipefail

NETLLM_DIR="${NETLLM_DIR:-/workspace/NetLLM-source}"
ENV_NAME="${ENV_NAME:-vp_netllm}"
ENV_PREFIX="${ENV_PREFIX:-/venv/$ENV_NAME}"
VP_DIR="$NETLLM_DIR/viewport_prediction"

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

cd "$VP_DIR"
mkdir -p logs

LOG="logs/$(date +%Y%m%d_%H%M)_vp_regression_cpu_jin2022.log"

set +e
python -u run_baseline.py \
  --model regression \
  --test \
  --device cpu \
  --test-dataset Jin2022 \
  --bs 64 \
  --seed 1 \
  --his-window 10 \
  --fut-window 20 \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo ""
echo "Log: $VP_DIR/$LOG"
exit "$status"
