# Source this after activating the venv:  source env.sh
#
# There is no system CUDA toolkit on this machine. nvcc/headers come from pip
# (nvidia-cuda-nvcc etc.) under .venv/.../nvidia/cu13. FlashInfer JIT-compiles
# some kernels (e.g. top-k/top-p sampling) at first use and needs CUDA_HOME
# pointing there. Keep nvcc on 13.2.x to match nvidia-cuda-runtime (13.2),
# torch cu132 and the driver's max CUDA 13.2 -- 13.4 nvcc fails on 13.2 headers.

_LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME="$_LAB_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"

# FlashInfer links with -L$CUDA_HOME/lib64 -lcudart; the pip layout has lib/
# and only the versioned libcudart.so.13.
[ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
[ -e "$CUDA_HOME/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"

export PATH="$_LAB_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST="8.6"   # RTX 3060 Ti; only build kernels for sm_86
unset _LAB_DIR
