# 00 — Setting up: why vLLM wouldn't start

*2026-09-26*

## Goal

Get vLLM serving a small model on an 8 GB RTX 3060 Ti.

## Setup

A Python 3.12 venv created with `uv`, with `vllm==0.30.0` installed. That pulls
in PyTorch 2.13 built for CUDA 13.2, FlashInfer, Triton and ~190 other packages
(pinned in `requirements.lock.txt`).

There is **no CUDA toolkit installed system-wide** — no `/usr/local/cuda`, no
`nvcc` on `PATH`. Only the NVIDIA driver. That's normal for a pip-based setup,
and it's the root of everything below.

## The failure

```bash
vllm serve Qwen/Qwen3-1.7B --max-model-len 2048 --gpu-memory-utilization 0.75 \
  --max-num-seqs 2 --enforce-eager
```

The model loaded fine (3.22 GiB of weights, 2.28 GiB left for KV cache), then
the engine died during warmup:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

The traceback shows where: vLLM's sampler calls
`flashinfer.sampling.top_k_top_p_sampling_from_logits`, and FlashInfer
**JIT-compiles** that kernel the first time it's used. JIT compiling CUDA needs
`nvcc`.

> Lesson: "the model loaded" does not mean "the engine works". vLLM runs a
> warmup pass through every stage (forward pass *and* sampling) before it
> accepts traffic, precisely to surface this kind of failure at startup.

## Debugging it — three layers

It took three fixes, each revealing the next problem.

### 1. Point FlashInfer at the pip-installed compiler

The venv *does* contain a CUDA compiler — the `nvidia-cuda-nvcc` wheel installs
it under `.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc`. FlashInfer
just didn't know to look there. Setting `CUDA_HOME` to that directory got past
the first error.

### 2. Compiler and headers were different CUDA versions

```
error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

The dependency resolver had picked mismatched wheels:

| Package | Version |
|---|---|
| `nvidia-cuda-nvcc` (compiler) | 13.4.92 |
| `nvidia-cuda-runtime` (headers + libcudart) | 13.2.75 |
| PyTorch | built for 13.2 |
| Driver | supports up to 13.2 |

The compiler was the odd one out. Pinning it back to 13.2 lined everything up:

```bash
uv pip install "nvidia-cuda-nvcc==13.2.86" "nvidia-cuda-crt==13.2.86" "nvidia-nvvm==13.2.86"
```

(And the same pins went into `requirements.lock.txt` so a rebuild doesn't
reintroduce it.)

### 3. Linker couldn't find `libcudart`

```
/usr/bin/ld: cannot find -lcudart: No such file or directory
```

FlashInfer links with `-L$CUDA_HOME/lib64 -lcudart`, assuming the layout of a
normal CUDA install. The pip layout differs: the directory is `lib/`, not
`lib64/`, and it only ships the versioned `libcudart.so.13`, not the plain
`libcudart.so` the linker looks for. Two symlinks fix it.

## The fix, packaged

All of it lives in [`env.sh`](../../env.sh), sourced once per shell:

```bash
source .venv/bin/activate && source env.sh
```

It sets `CUDA_HOME`, creates the two symlinks if missing, puts `nvcc` and
`ninja` on `PATH`, and sets `FLASHINFER_CUDA_ARCH_LIST=8.6` so JIT builds only
target this GPU (faster compiles).

Verified by calling the exact kernel that failed, outside vLLM:

```python
import torch, flashinfer
logits = torch.randn(2, 32000, device="cuda", dtype=torch.float16)
k = torch.tensor([50, 50], device="cuda", dtype=torch.int32)
p = torch.tensor([0.9, 0.9], device="cuda")
print(flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, k, p))
```

First run compiles (~70 s), then it's cached in `~/.cache/flashinfer/`.
After that, `vllm serve` started in ~36 s and answered requests.

## What we learned

- Modern inference stacks aren't just prebuilt binaries — FlashInfer, Triton
  and `torch.compile` all generate and compile GPU code **at runtime**, so a
  working compiler toolchain is a real dependency.
- The CUDA versions that must agree: driver ≥ runtime, and compiler = headers.
  "It installed" doesn't mean the resolver picked a consistent set.
- Reproduce the failing piece in isolation (one FlashInfer call) instead of
  restarting the whole server each iteration — seconds per attempt instead of
  a minute.

## Open questions

- Which other kernels get JIT-compiled, and when? (Try `--attention-backend FLASHINFER`.)
- Could we skip JIT entirely with FlashInfer's prebuilt kernel packages?
