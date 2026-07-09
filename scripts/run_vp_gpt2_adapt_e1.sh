#!/usr/bin/env bash
set -euo pipefail

NETLLM_DIR="${NETLLM_DIR:-/workspace/NetLLM}"
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

LOG="logs/$(date +%Y%m%d_%H%M)_vp_gpt2_adapt_rank32_e1.log"

nohup python -u run_plm.py \
  --adapt \
  --train-dataset Jin2022 \
  --his-window 10 \
  --fut-window 20 \
  --plm-type gpt2 \
  --plm-size base \
  --epochs 1 \
  --bs 1 \
  --lr 5e-4 \
  --grad-accum-steps 8 \
  --device cuda:0 \
  --steps-per-valid 5000 \
  --save-checkpoint-per-epoch 1 \
  --rank 32 \
  --scheduled-sampling \
  > "$LOG" 2>&1 &

echo $! > logs/vp_gpt2_adapt.pid

echo "Started GPT-2 adaptation."
echo "PID: $(cat logs/vp_gpt2_adapt.pid)"
echo "Log: $VP_DIR/$LOG"
echo ""
echo "Monitor with:"
echo "tail -f $VP_DIR/$LOG"
echo ""
echo "Check GPU with:"
echo "watch -n 2 nvidia-smi"
