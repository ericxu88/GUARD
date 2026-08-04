# GPU verification checklist

**Status: not yet executed.** GUARD was developed and verified on CPU. The engine's logic
is covered by 444 passing tests, but the CUDA-specific behaviour it depends on — kernel
overlap, capture ordering against a live inference stream, and the p99 overhead SLO — can
only be *confirmed*
on hardware. This document is the procedure for doing that, and the place to record the
results.

Until every box below is ticked with a real number next to it, the correct description of
GUARD's Phase-2 status is **"implemented, awaiting hardware verification"** — not "done".
That distinction is the whole point of `CLAUDE.md`'s GPU availability rule.

## Requirements

- NVIDIA GPU, compute capability ≥ 8.0 (Ampere or newer). Lower is likely to work but the
  stream-priority behaviour and concurrent-kernel scheduling are what the design assumes,
  and they are best on Ampere+.
- PyTorch built for that CUDA version. **Pin the exact build your serving cluster runs.**
  `record_stream` semantics and caching-allocator behaviour are version-sensitive, and a
  dev/prod mismatch is precisely how a memory-safety bug hides.
- NVIDIA Nsight Systems (`nsys`) for the overlap proof.

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

---

## 1. Run the GPU test suite

```bash
pytest -m gpu -v
```

Eighteen tests in `tests/test_gpu_engine.py`. They are not smoke tests; each one asserts a
prime directive.

| Test | What it proves | Directive |
|---|---|---|
| `test_observe_performs_no_host_synchronisation` | `observe()` triggers no implicit device→host sync, asserted with `torch.cuda.set_sync_debug_mode("error")` | #2 |
| `test_observe_returns_while_the_gpu_is_still_busy` | An event recorded before `observe()` is still pending when it returns — the call really is fire-and-forget | #2 |
| `test_capture_survives_allocator_reuse` | 256 steps of hostile allocator churn; every drained value is exactly what was written | #3 |
| `test_capture_survives_a_static_output_buffer` | 256 steps against one buffer rewritten immediately after each call — the CUDA-graph-replay / `torch.compile(reduce-overhead)` / TensorRT case that `record_stream` cannot cover | #3 |
| `test_device_memory_does_not_grow_with_steps` | < 1 MiB growth over 2000 steps | #5 |
| `test_pinned_host_buffers_are_page_locked` | The async D2H drain is genuinely async | #2 |
| `test_detectors_match_cpu_reference_on_cuda` | CUDA results match CPU within `DEFAULT_RTOL`/`DEFAULT_ATOL` | #4 |
| `test_half_precision_serving_path_produces_finite_metrics` | fp16/bf16 autocast serving yields finite metrics, not NaN | — |
| `test_monitor_stream_is_real_separate_and_lowest_priority` | A distinct stream at priority 0 exists | — |
| `test_observation_is_dropped_during_cuda_graph_capture` | Graph capture is detected and skipped, not corrupted | #1 |
| `test_full_pipeline_detects_injected_drift_on_device` | Clean traffic is quiet; injected shift alerts | — |

**Record the result here:**

```
date:              ____________________
GPU / driver:      ____________________
torch / CUDA:      ____________________
pytest -m gpu:     ____ passed, ____ failed
```

If either `test_capture_survives_*` test fails, stop. That is a memory-safety failure, and
every metric the system produces is untrustworthy until it is fixed. Do not work around it
by making the ring larger — a deeper ring changes the odds, not the ordering.

---

## 2. Prove kernel overlap with Nsight

The latency benchmark tells you *how much* GUARD costs. Only the timeline tells you *why*.
If monitor kernels were simply serialised after each forward pass, a cheap enough detector
would still report a small overhead — the right number for entirely the wrong reason, with
no headroom left the moment a detector grows.

```bash
./benchmarks/nsight_overlap.sh synthetic 32
# or against the reference model:
./benchmarks/nsight_overlap.sh vit_b_16 32
```

Open `guard-overlap.nsys-rep` and confirm all four:

- [ ] **Two CUDA streams appear**, not one. Monitor kernels on the default stream mean the
      monitor stream was never created (CPU fallback) or the tensors were on another device.
- [ ] **Monitor-stream kernels start before the surrounding inference kernels finish** —
      overlapping bars, not a staircase.
- [ ] **No `cudaStreamSynchronize` or synchronous `cudaMemcpy` between forward passes.**
      Either one is a directive-#2 violation and will show up as a latency cliff under load.
- [ ] **The summary copy appears once every `drain_every_k` steps**, on the monitor stream,
      as an *async* DtoH memcpy from pinned memory.

Attach the screenshot and note anything surprising:

```
overlap confirmed:  yes / no
notes:              ____________________
```

---

## 3. Measure the overhead SLO

The design doc sets **p99 latency increase < 2%** and **throughput drop < 2%**. Nothing in
the test suite asserts those numbers, deliberately: a figure measured on a shared CI GPU
against a synthetic model is not evidence about your deployment.

```bash
python benchmarks/ab_latency.py --model vit_b_16 --batch 32 --iters 500 --json ab.json
```

Exit status is non-zero when the SLO is missed, so this can gate a GPU CI job.

```
p50 overhead:      _______%
p95 overhead:      _______%
p99 overhead:      _______%   (SLO: < 2%)
throughput drop:   _______%   (SLO: < 2%)
enqueue cost:      _______ µs/step on the inference thread
ring buffer:       _______ MiB
```

### Then measure where it stops being free

Overhead is a function of spare SM capacity, so a single number is a half-truth. Publish
the curve:

```bash
python benchmarks/saturation.py --model vit_b_16 --batches 1,2,4,8,16,32,64,128 --json sweep.json
```

```
batch at which p99 overhead exceeds 2%:  ____________
worst observed p99 overhead:             _______%  at batch ____
```

Put this table in the README. Quoting only the best point is how "near-zero overhead"
claims lose their credibility.

---

## 4. Tune, if the numbers demand it

In rough order of effect:

| Symptom | Knob | Effect |
|---|---|---|
| p99 overhead high at large batch | `cov_every` (embedding detector) | The only `O(D³)` work GUARD does. Raise it, or set `0` to drop `embedding_cov_drift`. |
| p99 overhead high, `ring_utilization` near 1.0 | fewer detectors, or larger `drain_every_k` | The monitor stream is falling behind inference. |
| `guard_summaries_dropped_total` climbing | `max_pending_drains`, or a faster sink | The export thread cannot keep up; metrics are being dropped (correctly) instead of blocking inference. |
| `guard_monitor_overhead_us` high | fewer detectors | This is CPU time on the inference thread — enqueue cost, not GPU cost. |
| Memory pressure | `ring_slots` | Correctness does not depend on ring depth (stream ordering does); extra slots only buy pipelining. |

Elevating the *inference* stream to `priority=-1` widens the gap further, but GUARD cannot
do that for you — it does not own the stream your model runs on.

---

## 5. What is still not covered

Honest gaps, so nobody mistakes a green checklist for more than it is:

- **Multi-GPU.** One `GuardMonitor` per device is the documented topology and each monitor
  resolves its own device index, but no multi-GPU test exists. Verify per-device metric
  labels before running it on more than one card.
- **CUDA Graphs.** Observations during capture are dropped by design (§6.4). Capturing
  detector kernels *into* the graph is not implemented.
- **Long soak.** The memory test runs 2000 steps. A multi-day soak under production traffic
  is the only way to catch slow leaks in the pinned-buffer pool or the allocator.
- **Triton / TorchServe adapters.** Not shipped; `GuardModule` plus a handler call is the
  documented path.
