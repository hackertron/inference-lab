#!/usr/bin/env bash
# Start the lab's vLLM server in one command:  ./dev.sh
# Then, from another terminal, send requests to http://127.0.0.1:$PORT/v1
#
# Override defaults with env vars, e.g.  PORT=8001 MODEL=Qwen/Qwen3-0.6B ./dev.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/.venv/bin/activate"
source "$ROOT/env.sh"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 2
