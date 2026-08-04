#!/usr/bin/env bash
# Nsight Systems capture that proves monitor kernels actually OVERLAP inference kernels.
#
# The latency benchmark tells you *how much* GUARD costs. It cannot tell you *why*. If the
# monitor kernels were simply serialised after each forward pass, a cheap-enough detector
# would still show a small overhead number — and you would have "near-zero overhead" for
# entirely the wrong reason, with no headroom left the moment a detector grows.
#
# The timeline is the proof. What you are looking for:
#
#   1. Two CUDA streams in the trace, not one. If GUARD's kernels appear on the default
#      stream, the monitor stream was never created (CPU fallback) or the tensors were on
#      the wrong device.
#   2. Monitor-stream kernels (softmax / reduce / gemm, all tiny) starting BEFORE the
#      inference kernels around them finish — overlapping bars, not a staircase.
#   3. No `cudaStreamSynchronize` / `cudaMemcpy` (synchronous variant) between forward
#      passes. Either one means something on the hot path is synchronising, which is a
#      prime-directive #2 violation and will show up as a latency cliff under load.
#   4. The device→host summary copy appearing once every `drain_every_k` steps, on the
#      monitor stream, as an *async* memcpy DtoH from pinned memory.
#
# Usage:
#   ./benchmarks/nsight_overlap.sh                 # synthetic model
#   ./benchmarks/nsight_overlap.sh vit_b_16 32     # model and batch size
#
# Then open guard-overlap.nsys-rep in the Nsight Systems GUI, or read the summary printed
# below. Requires the NVIDIA Nsight Systems CLI (`nsys`) on PATH.

set -euo pipefail

MODEL="${1:-synthetic}"
BATCH="${2:-32}"
OUT="${3:-guard-overlap}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v nsys >/dev/null 2>&1; then
  echo "error: nsys not found. Install NVIDIA Nsight Systems and put nsys on PATH." >&2
  exit 127
fi

echo "Capturing ${MODEL} at batch=${BATCH} -> ${OUT}.nsys-rep"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=none \
  --force-overwrite=true \
  --output="${OUT}" \
  python "${ROOT}/benchmarks/ab_latency.py" \
    --model "${MODEL}" \
    --batch "${BATCH}" \
    --iters 120 \
    --blocks 2 \
    --warmup 40 \
    --slo 1000

echo
echo "=== kernel/stream summary (look for two distinct streams) ==="
nsys stats --report cuda_gpu_kern_sum --format table "${OUT}.nsys-rep" | head -40 || true

echo
echo "=== memory operations (the DtoH drain should be async, from pinned memory) ==="
nsys stats --report cuda_gpu_mem_time_sum --format table "${OUT}.nsys-rep" | head -20 || true

echo
echo "Open ${OUT}.nsys-rep in the Nsight Systems GUI and check the four points listed at"
echo "the top of this script. Attach the screenshot to docs/GPU_VERIFICATION.md results."
