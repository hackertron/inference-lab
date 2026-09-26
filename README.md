# inference-lab

A hands-on lab for learning LLM inference engineering on a single consumer GPU.
Every step is written up in [`docs/journal/`](docs/journal/) as it happens,
including the parts that broke, so it can later become a tutorial or blog series.

## Hardware

| Part   | Spec |
|--------|------|
| GPU    | NVIDIA RTX 3060 Ti, 8 GB GDDR6, 448 GB/s, compute capability 8.6 (Ampere) |
| CPU    | AMD Ryzen 9 3900X, 12 cores |
| RAM    | 32 GB |
| OS     | Ubuntu 24.04, NVIDIA driver 595.84 (CUDA 13.2) |

## Software

vLLM 0.30.0, PyTorch 2.13 (cu132), FlashInfer 0.6.18, Python 3.12, managed with `uv`.
No system CUDA toolkit — the compiler comes from pip. See
[journal 00](docs/journal/00-setting-up.md) for why that matters.

## Quickstart

```bash
source .venv/bin/activate && source env.sh
vllm serve Qwen/Qwen3-1.7B --max-model-len 2048 --gpu-memory-utilization 0.75 --max-num-seqs 2
```

In a second terminal (also `source .venv/bin/activate && source env.sh`):

```bash
scripts/bench.sh my-run 2
```

## Layout

```
env.sh                 CUDA_HOME + PATH for the pip CUDA toolkit (source every session)
requirements.lock.txt  pinned environment
scripts/bench.sh       standard benchmark workload -> results/<label>.json
results/               raw benchmark JSON
docs/journal/          the write-up, one entry per step
```

## Journal

| # | Entry | Status |
|---|-------|--------|
| 00 | [Setting up: why vLLM wouldn't start](docs/journal/00-setting-up.md) | done |
| 01 | [First request: prefill vs decode](docs/journal/01-first-request.md) | done |
| 02 | [First benchmark: reading TTFT and TPOT](docs/journal/02-first-benchmark.md) | done |
| 03 | [CUDA graphs vs eager mode](docs/journal/03-cuda-graphs.md) | in progress |
| 04 | Latency vs throughput: concurrency sweep | planned |

### How each entry is written

Goal → setup → exact commands → raw results → what we learned → open questions.
Write the **prediction before running** the experiment; the misses are the interesting part.
