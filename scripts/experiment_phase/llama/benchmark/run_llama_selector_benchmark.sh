#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/root/NetLLM/src
exec /root/venvs/vp_netllm_llama/bin/python \
  /root/NetLLM/scripts/experiment_phase/llama/benchmark/run_llama_selector_benchmark.py \
  "$@"
