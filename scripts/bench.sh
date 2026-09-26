#!/usr/bin/env bash
# Run the standard lab workload against a running vLLM server and save the JSON.
#
#   scripts/bench.sh <label> [concurrency] [num_prompts]
#   scripts/bench.sh eager-c2 2
#
# Keep the workload fixed (256 in / 128 out, seed 0) so runs are comparable.
# Override the model with MODEL=...; results land in results/<label>.json.
set -euo pipefail

LABEL="${1:?usage: scripts/bench.sh <label> [concurrency] [num_prompts]}"
CONC="${2:-2}"
NUM_PROMPTS="${3:-$(( CONC * 10 > 50 ? CONC * 10 : 50 ))}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

vllm bench serve \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 128 \
  --ignore-eos \
  --seed 0 \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONC" \
  --save-result \
  --result-dir "$ROOT/results" \
  --result-filename "$LABEL.json" \
  --metadata "label=$LABEL"
