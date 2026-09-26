# 02 — First benchmark: reading TTFT and TPOT

*2026-09-26*

## Goal

Measure the server under load and check whether the numbers are good,
by comparing against what the hardware can physically do.

## Setup

Same server as [01](01-first-request.md) — **eager mode**, `--max-num-seqs 2`.

Workload: 50 requests, each 256 random input tokens → exactly 128 output
tokens, at most 2 in flight.

## Commands

```bash
vllm bench serve --model Qwen/Qwen3-1.7B --dataset-name random \
  --random-input-len 256 --random-output-len 128 \
  --num-prompts 50 --max-concurrency 2
```

(From here on this workload is wrapped in [`scripts/bench.sh`](../../scripts/bench.sh),
which also saves the JSON.)

## Result

```
Successful requests:                     50
Benchmark duration (s):                  48.41
Request throughput (req/s):              1.03
Output token throughput (tok/s):         132.20
Total token throughput (tok/s):          396.61
Mean TTFT (ms):                          60.52
Median TTFT (ms):                        76.22
P99 TTFT (ms):                           109.13
Mean TPOT (ms):                          14.77
Median TPOT (ms):                        14.78
P99 TPOT (ms):                           15.39
Mean ITL (ms):                           14.77
```

## Reading the metrics

| Metric | Meaning | Dominated by |
|---|---|---|
| **TTFT** — time to first token | Request arrives → first token out | Prefill (+ queueing, HTTP, tokenization) |
| **TPOT** — time per output token | Average gap between tokens after the first | Decode step time |
| **ITL** — inter-token latency | Each individual gap (TPOT is its per-request mean) | Decode step time, plus stalls |
| **Output throughput** | Generated tokens/s across *all* requests | Decode step time × batch size |

Sanity check: TPOT 14.77 ms → ~68 tok/s per request. Two concurrent requests →
~135 tok/s. Measured output throughput: 132 tok/s. ✓

## Is 14.8 ms good? The roofline

At small batch sizes, **decode is memory-bandwidth bound**: every step reads
every weight from VRAM once, and the math is tiny by comparison.

```
weights            = 3.22 GiB ≈ 3.46 GB
bandwidth (3060 Ti) = 448 GB/s
floor per step     = 3.46 / 448 ≈ 7.7 ms
```

We measure 14.8 ms, so we're at **~52% of the bandwidth roofline**.

Likely culprit: **eager mode**. Every decode step launches hundreds of small
kernels one by one from Python (28 layers × ~10 kernels each). At this model
size, the CPU-side launch overhead is comparable to the GPU work itself.

**Batching is nearly free in decode**: two sequences share one read of the
weights per step, so running 2 at once roughly doubles throughput while barely
changing per-request TPOT. This is *the* reason inference servers batch.

**TTFT**: ~60–76 ms for a 256-token prompt. Prefill processes all tokens in one
pass, so it's compute-bound rather than bandwidth-bound. Median > mean hints at
two clusters (e.g. prefill running alone vs sharing a step with the other
request's decode) — unconfirmed.

**Ignore**: "Peak concurrent requests: 4" is a counting artifact at request
boundaries (the server caps at 2), and the temperature warning doesn't matter
because output length is fixed.

## Open questions

- How much of the 7.7 → 14.8 ms gap is launch overhead? → [03](03-cuda-graphs.md)
- Where does throughput stop scaling with batch size? → 04
- What explains the bimodal TTFT?
