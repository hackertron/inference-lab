# 04 — Latency vs throughput: concurrency sweep

*2026-10-19 — in progress*

## Goal

Find out how throughput and per-request latency change as more requests are
batched together. This is the core trade-off of LLM serving, and it's chapter
1.4 of the book this lab follows.

## Background

Each decode step reads all 3.46 GB of weights once, *however many* sequences
are in the batch. So adding sequences should be nearly free until something
else becomes the limit:

- **KV cache reads.** Each sequence also reads its own KV cache every step:
  112 KiB per token × ~320 tokens of context on average ≈ 36 MB per sequence.
- **Compute.** Each sequence costs 2 × 1.73 B ≈ 3.5 GFLOP per step. The GPU's
  FP16 tensor-core peak is ~32 TFLOPS, so compute only becomes the limit at a
  batch size far above 16.
- **Prefill interference.** New requests' prefills share steps with running
  decodes, which slows those steps down.

## Prediction

*Written before running.* Server: CUDA graphs, `MAX_NUM_SEQS=16`.

| Concurrency | TPOT (ms) | Output tok/s | Mean TTFT (ms) |
|---|---|---|---|
| 1 | ~10 | ~95 | ~45 |
| 2 | ~10.5 | ~185 | ~60 |
| 4 | ~11 | ~350 | ~100 |
| 8 | ~12 | ~650 | ~200 |
| 16 | ~13–15 | ~1,000–1,200 | ~300–600 |

Reasoning:
- At c=16, KV reads add ~0.58 GB per step on top of 3.46 GB of weights,
  which is ~17% more memory traffic. That should make TPOT only modestly worse.
- Throughput should scale *almost* linearly, roughly 6× from c=2 to c=16.
- TTFT should grow the most. With fixed output lengths, requests tend to finish
  together, so new prompts arrive in waves. Prefilling 16 × 256 tokens is ~14
  TFLOP, which takes roughly half a second.

## Commands

```bash
# Terminal 1
MAX_NUM_SEQS=16 ./dev.sh
```

```bash
# Terminal 2
source .venv/bin/activate && source env.sh
for c in 1 2 4 8 16; do scripts/bench.sh 04-sweep-c$c $c; done
```

## Results

TODO

## What we learned

TODO

## Open questions

TODO
