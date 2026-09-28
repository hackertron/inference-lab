# 03 — CUDA graphs vs eager mode

*2026-09-26 — in progress*

## Goal

Measure how much of the decode gap from [02](02-first-benchmark.md)
(14.8 ms measured vs 7.7 ms roofline) is kernel-launch overhead.

## Background

In **eager mode** the CPU launches every kernel of every decode step one at a
time. A **CUDA graph** records the whole sequence of kernel launches once, then
replays it with a single launch. vLLM also runs `torch.compile` by default,
which fuses small ops into fewer kernels. `--enforce-eager` turns both off.

Cost: slower startup (compile + graph capture) and some extra VRAM for the
captured graphs.

## Prediction

*Written before running.*

TPOT drops from 14.8 ms to **~9–10 ms**. Not all the way to 7.7 ms, since
attention and KV-cache reads also cost bandwidth, and kernels rarely hit peak.

## Setup

Same model, same workload (`scripts/bench.sh`, concurrency 2), two server configs:

- A: `--enforce-eager` (baseline, rerun so both results are saved as JSON)
- B: default (CUDA graphs + `torch.compile`)

## Commands

```bash
# Terminal 1 — A
vllm serve Qwen/Qwen3-1.7B --max-model-len 2048 --gpu-memory-utilization 0.75 --max-num-seqs 2 --enforce-eager
# Terminal 2
scripts/bench.sh 03-eager-c2 2

# Terminal 1 — Ctrl+C, then B
vllm serve Qwen/Qwen3-1.7B --max-model-len 2048 --gpu-memory-utilization 0.75 --max-num-seqs 2
# Terminal 2
scripts/bench.sh 03-graphs-c2 2
```

## Results

| Config | Startup (s) | Mean TTFT (ms) | Mean TPOT (ms) | Output tok/s |
|---|---|---|---|---|
| A: eager | | | | |
| B: graphs + compile | | | | |

Observed while testing `dev.sh` (graphs + compile, same settings):

- Cold startup ~75 s, of which `torch.compile` took 27 s. Graph capture: ~1 s, 0.02–0.04 GiB.
- **KV cache shrank: 16,320 tokens vs 21,376 in eager mode**, with the same
  `--gpu-memory-utilization 0.75`. The graphs themselves are tiny, so something
  else is eating ~0.55 GiB — find out what before blaming CUDA graphs.

## What we learned

TODO

## Open questions

TODO
