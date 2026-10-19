# 03 — CUDA graphs vs eager mode

*2026-09-26 → 2026-10-19*

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

(B was started with [`./dev.sh`](../../dev.sh), which is the same config.)

## Results

Raw JSON: [`results/03-eager-c2.json`](../../results/03-eager-c2.json),
[`results/03-graphs-c2.json`](../../results/03-graphs-c2.json). Both runs on 2026-10-19.

| Metric | A: eager | B: graphs + compile | Change |
|---|---|---|---|
| **Mean TPOT (ms)** | 14.89 | **10.55** | **−29%** |
| TPOT std dev (ms) | 0.60 | 0.11 | 5× steadier |
| P99 TPOT (ms) | 16.52 | 10.72 | |
| P99 ITL (ms) | 18.64 | 11.13 | |
| **Output throughput (tok/s)** | 131.1 | **182.7** | **+39%** |
| Mean TTFT (ms) | 61.8 | 61.4 | — |
| Median TTFT (ms) | 75.4 | 67.3 | |
| P99 TTFT (ms) | 110.7 | 88.7 | |
| Cold startup (s) | ~36 | ~75 (27 s `torch.compile`) | 2× slower |
| KV cache (tokens) | 21,376 | 16,320 | −24% |
| % of 7.7 ms roofline | 52% | **73%** | |

The eager rerun matches journal 02 (14.77 ms then, 14.89 ms now, three weeks
apart), so the measurement is reproducible to ~1%.

## What we learned

**The prediction held.** We guessed 9–10 ms and measured 10.55 ms.

**About 4.3 ms of every 14.9 ms decode step was CPU overhead, not GPU work.**
CUDA graphs removed it by replacing hundreds of individual kernel launches with
one replay. A 1.7B model is small enough that the GPU finishes each kernel
about as fast as Python can launch the next one. A bigger model would spend more
time per kernel, so the same overhead would matter less.

**Graphs also make latency more predictable.** TPOT spread dropped 5× and P99
ITL fell from 18.6 to 11.1 ms. In eager mode, every step depends on the Python
process launching kernels on time, so any CPU hiccup shows up as a slow token.
A graph replay doesn't wait on Python between kernels.

**Prefill barely changed.** Mean TTFT stayed at ~61 ms. Prefill runs fewer,
larger kernels that are limited by compute, so launch overhead is a small part
of it. The median and P99 did improve. A plausible reason is that when a
prefill shares a step with the other request's decode, the decode part is now
faster. That's unconfirmed.

**The speedup has costs:** startup takes twice as long, and the KV cache holds
24% fewer tokens at the same memory setting. For a dev server that restarts
often, eager mode can be the better trade.

**There's still a 2.85 ms gap to the floor.** Each step also reads the KV cache
for attention and runs sampling and scheduler work. And no real kernel reaches
the full 448 GB/s spec, so the practical floor is above 7.7 ms.

## Open questions

- **Why did the KV cache shrink by ~0.55 GiB?** The captured graphs only use
  0.02–0.04 GiB, so they're not the cause. Suspects: memory `torch.compile`
  allocates during profiling, or more VRAM in use by the desktop that day.
- **What bandwidth can this GPU actually reach?** Measure it with a large
  device-to-device copy, then recompute the floor from that number.
- Does the TTFT median/mean gap (two clusters) have the cause guessed in 02?
- Does the graphs speedup shrink at higher concurrency, where each step does
  more GPU work? → [04](04-concurrency-sweep.md)
