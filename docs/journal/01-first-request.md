# 01 — First request: prefill vs decode

*2026-09-26*

## Goal

Send a request to the local server and understand what comes back.

## Setup

Server from [00](00-setting-up.md): Qwen3-1.7B, FP16, `--enforce-eager`,
`--max-num-seqs 2`, `--max-model-len 2048`.

vLLM exposes an OpenAI-compatible API, so any OpenAI client works against
`http://127.0.0.1:8000/v1`.

## Commands

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen3-1.7B",
  "messages": [
    {"role": "user", "content": "Explain the difference between latency and throughput in three sentences."}
  ],
  "max_tokens": 128,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
```

## Result

```json
"content": "Latency is the time it takes for a single request to be processed and responded to, while throughput is the number of requests that can be handled in a given period. Latency is often measured in milliseconds, ...",
"finish_reason": "stop",
"usage": {"prompt_tokens": 24, "completion_tokens": 66, "total_tokens": 90}
```

## What we learned

**`chat_template_kwargs.enable_thinking: false`** turns off Qwen3's reasoning
mode at the chat-template level. The alternative, putting `/no_think` in the
prompt, still emits an empty `<think></think>` block.

**`usage` shows the two phases of inference:**

- **Prefill** — the 24 prompt tokens are processed in *one* forward pass, all
  in parallel. This fills the KV cache and produces the first output token.
- **Decode** — each of the 66 output tokens needs its *own* forward pass,
  because token *n+1* depends on token *n*. Decode is sequential and dominates
  wall-clock time.

**`finish_reason`**: `"stop"` means the model emitted end-of-sequence on its
own; `"length"` would mean it hit `max_tokens`.

**Quality check**: asked for three sentences, got two. A 1.7B model trades
instruction-following for speed and memory.

## Open questions

- How long did prefill take vs each decode step? This response has no
  timing — see [02](02-first-benchmark.md).
