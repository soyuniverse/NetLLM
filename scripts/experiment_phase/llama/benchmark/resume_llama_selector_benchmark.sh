#!/usr/bin/env bash
set -Eeuo pipefail
exec /root/NetLLM/scripts/experiment_phase/llama/benchmark/run_llama_selector_benchmark.sh \
  --mode full --resume "$@"
