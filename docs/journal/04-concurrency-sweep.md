# 04 — Latency vs throughput: concurrency sweep

*2026-10-19 — part 1 (c = 1–16) done; part 2 (c > 16) planned*

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

Raw JSON: `results/04-sweep-c{1,2,4,8,16}.json`. Chart: `python3 scripts/plot_sweep.py`.

![TPOT and TTFT vs output throughput for concurrency 1 to 16, measured vs predicted](img/04-sweep.svg)

| c | Output tok/s | vs c=1 | Mean TPOT (ms) | Per-user tok/s | Median ITL | P99 ITL (ms) | Mean TTFT (ms) | P99 TTFT (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 99.9 | 1.0× | 9.77 | 102 | 9.76 | 10.26 | 39.9 | 59.3 |
| 2 | 182.5 | 1.8× | 10.56 | 95 | 10.44 | 11.61 | 61.6 | 82.5 |
| 4 | 330.1 | 3.3× | 10.97 | 91 | 10.85 | 11.55 | 101.1 | 139.9 |
| 8 | 602.4 | 6.0× | 11.76 | 85 | 11.19 | 13.75 | 205.4 | 256.7 |
| 16 | 1,001.9 | **10.0×** | 13.64 | 73 | 12.33 | **39.05** | 309.8 | 459.9 |

The number of prompts scales with concurrency (50, 50, 50, 80, 160) so that
every run lasts long enough to measure. The c=2 run reproduces journal 03's
graphs result (10.56 vs 10.55 ms).

### Prediction scorecard

| c | TPOT predicted → measured | tok/s predicted → measured | TTFT predicted → measured |
|---|---|---|---|
| 1 | ~10 → 9.77 ✓ | ~95 → 100 ✓ | ~45 → 40 ✓ |
| 2 | ~10.5 → 10.56 ✓ | ~185 → 182 ✓ | ~60 → 62 ✓ |
| 4 | ~11 → 10.97 ✓ | ~350 → 330 ✓ | ~100 → 101 ✓ |
| 8 | ~12 → 11.76 ✓ | ~650 → 602 (8% low) | ~200 → 205 ✓ |
| 16 | 13–15 → 13.64 ✓ | 1,000–1,200 → 1,002 (low end) | 300–600 → 310 (low end) |

Bandwidth-based reasoning predicted TPOT within 3% at every point.
Throughput came in a little low at high concurrency, for the reason explained
in the next section.

## What we learned

**Batching is the biggest lever on throughput.** Going from 1 to 16
concurrent requests gave **10× the throughput** while each user's speed only
dropped from 102 to 73 tok/s (−29%). Every decode step reads the 3.46 GB of
weights once, however many sequences share it, so extra sequences are close
to free.

**Throughput follows from the two latencies.** Each request takes
`TTFT + 127 × TPOT`, so with c requests in flight:

```
throughput ≈ c × 128 / (TTFT + 127 × TPOT)
c=16:  16 × 128 / (0.310 + 127 × 0.01364) ≈ 1,003 tok/s   (measured 1,002)
```

This formula reproduces every measured point, which makes it a good tool for
explaining why throughput falls short of linear:
- **Slower TPOT** costs the most. At c=16 it's 40% slower than at c=1.
- **Longer TTFT** costs the rest. If TTFT had stayed at 40 ms, c=16 would
  reach ~1,156 tok/s instead of 1,002, so the extra prefill wait cost ~13%.

**Why TPOT grows.** Each sequence also reads its own KV cache every step. At
c=16 that's ~0.59 GB more traffic per step (16 × ~320 tokens × 112 KiB),
which is about +1.3 ms at 448 GB/s. Median ITL grew by 2.6 ms from c=1 to
c=16, so KV reads explain about half of it. The other half is unexplained so
far. Possible causes: larger matrix multiplies that start to need real
compute, attention kernel efficiency, or sampling 16 sequences.

**c=1 runs at 79% of the bandwidth floor** (7.7 / 9.77 ms). That's close to
what a real kernel can achieve, so most of the remaining gap is probably the
difference between spec bandwidth and achievable bandwidth.

**Prefill stalls show up in the tail at c=16.** P99 ITL jumps to 39 ms while
the median stays at 12 ms. A 256-token prefill takes about 30 ms of GPU time
(c=1 TTFT is 40 ms including overhead). When a new request's prefill runs
in the same step as 15 decodes, all 15 users wait for it. This is
**prefill–decode interference**. Chunked prefill keeps the stall bounded, and
disaggregation (book §5.5) removes it by running prefill and decode on
different GPUs.

**Prefill is compute-bound and close to peak.** 256 tokens × ~3.5 GFLOP per
token ≈ 0.89 TFLOP in under 40 ms, which is at least 22 TFLOPS. The FP16
tensor-core peak is ~32 TFLOPS, so prefill already uses most of the
compute while decode uses most of the bandwidth. Prefill and decode hit
different limits, which is the main idea of the book's chapter 2.

**We haven't found the limit yet.** Throughput still grew 1.66× from c=8 to
c=16, so a plateau is further out.

**Ignore**: "Peak concurrent requests" reports 2× the concurrency, the same
counting artifact as in journal 02.

## Open questions

- **Where is the plateau?** The KV cache holds 16,320 tokens. With 384
  tokens per request, that's at most ~42 requests in memory at once. Prediction
  below.
- What's the other ~1.3 ms of TPOT growth at c=16? A profiler
  (`vllm bench serve --profile` or Nsight Systems) would show it.
- How much does chunked prefill's token budget change the P99 ITL spike?

## Next: past 16 (planned)

**Prediction (written before running):** with `MAX_NUM_SEQS=64`, throughput
keeps climbing to around **2,000 tok/s near c=40** and then flattens. Past
~42 concurrent requests the KV cache is full, so extra requests wait in a
queue: TTFT shoots up while TPOT and throughput stay flat. Decode should
still be memory-bound at c=42 (~5 GB of reads per step vs ~145 GFLOP of
compute).

```bash
# Terminal 1
MAX_NUM_SEQS=64 ./dev.sh
```

```bash
# Terminal 2
for c in 24 32 48 64; do scripts/bench.sh 04-sweep-c$c $c; done
python3 scripts/plot_sweep.py
```
