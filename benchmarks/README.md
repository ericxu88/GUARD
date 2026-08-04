# GUARD benchmarks

Three tools, answering three different questions:

| Script | Question | Gates CI? |
|---|---|---|
| `ab_latency.py` | *How much does GUARD cost?* p50/p95/p99 latency and throughput, GUARD off vs on. | Yes — exits non-zero on SLO breach. |
| `saturation.py` | *Where does it stop being free?* The same A/B repeated across a batch-size sweep. | No — always exits 0. |
| `nsight_overlap.sh` | *Why is it cheap?* Nsight Systems timeline proving monitor kernels overlap inference kernels. | No — manual inspection. |

You need all three. `ab_latency.py` alone can report a good number for the wrong reason: if
the monitor kernels were simply serialised after each forward pass, a cheap enough detector
would still look cheap — with no headroom left the moment a detector grows. The timeline is
what distinguishes "overlapped" from "small".

> **No result has been recorded yet.** GUARD was developed without CUDA hardware, so every
> number below is a target, not a measurement. The design doc's p99 < 2% is an unverified
> SLO. `docs/GPU_VERIFICATION.md` is the procedure and the place to write the results down.

---

## `ab_latency.py` — the A/B

```bash
python benchmarks/ab_latency.py                                    # synthetic model, quick
python benchmarks/ab_latency.py --model vit_b_16 --batch 32 --iters 500
python benchmarks/ab_latency.py --json ab.json                     # machine-readable
```

### Flags

| Flag | Default | What it does |
|---|---|---|
| `--model` | `synthetic` | `synthetic` (a 12-layer `TransformerEncoder` with a tappable `norm` layer) or `vit_b_16` (torchvision; needs the `demo` extra). |
| `--device` | `cuda` if available, else `cpu` | Where model, monitor, and ring buffer live. |
| `--batch` | `32` | Batch size per forward pass. Also sets `window_size` and the monitor's `max_batch`. |
| `--iters` | `300` | Measured iterations **per arm**, warm-up excluded. |
| `--blocks` | `6` | Number of interleaved A/B blocks. Each block measures `iters // blocks` iterations off, then the same number on. |
| `--warmup` | `30` | Discarded iterations per arm before measurement begins. |
| `--num-classes` | `1000` | Logits width `C`. **Synthetic model only** — `vit_b_16` ignores it. |
| `--embed-dim` | `768` | Embedding width `D`, and the synthetic model's `d_model`. Synthetic only. |
| `--depth` | `12` | Synthetic model layer count. Lower it to make the model cheap — which makes GUARD look expensive, since overhead is a *ratio*. |
| `--drain-every-k` | `32` | Steps between device→host summary copies. Larger = fewer, bigger copies and staler metrics. |
| `--cov-every` | `32` | Steps between recomputations of `embedding_cov_drift`, the only `O(D³)` work GUARD does. Set `0` to drop the metric entirely. |
| `--slo` | `2.0` | Pass/fail threshold in percent. Applied to **both** p99 overhead and throughput drop, despite the flag's help text naming only p99. |
| `--json` | — | Write the full `Report` dataclass to this path as JSON. |

Note that `--iters` is rounded down: the harness measures `blocks × (iters // blocks)`
iterations per arm and reports the true count in `iterations`.

### Methodology, and why each piece is there

**CUDA-event timing, not a host timer.** Kernel launches are asynchronous. A
`time.perf_counter()` around `forward()` measures how long it took to *enqueue* work, which
on a fast host is nearly independent of how long the GPU then spends on it. `_time_block`
records a `torch.cuda.Event` before and after each iteration — in the stream, where the work
actually is — and calls `torch.cuda.synchronize()` exactly once, after the loop, at the
measurement boundary. Synchronising inside the loop would serialise the pipeline and destroy
the very overlap being measured.

**Warm-up is discarded.** The first iterations pay for caching-allocator growth, cuDNN
autotuning, and lazy module initialisation. `--warmup` iterations run through *both* arms
before anything is recorded. Left in, they would flatter or wreck the result depending on
which arm happened to run first.

**Interleaved A/B blocks.** Off and on alternate (`--blocks` times) instead of running as
two long halves. A GPU that thermally throttles, a noisy co-tenant, or a clock that ramps
during the run then contributes to both arms roughly equally, instead of landing entirely on
whichever arm ran second and masquerading as monitoring overhead.

**Percentiles, not means.** Serving is judged on tails, and a mean hides exactly the
intermittent stalls a monitoring hook is most likely to introduce. Percentiles are
nearest-rank, not interpolated, so a bimodal latency distribution is not smoothed into a
value that never occurred.

**Both arms are realistic.** The "off" arm calls the bare model; the "on" arm calls the
`GuardModule` wrapper, so the forward hook and the `observe()` enqueue are inside the
measurement, not excluded from it. The monitor is toggled with `disable()`/`enable()` rather
than rebuilt, so allocator state is identical across arms. A short calibration run builds a
real baseline first, so the detectors do real arithmetic against real reference statistics —
that baseline is only four batches, which is fine for a cost measurement and far too small
for a detection-quality one.

### Reading the output

```
GUARD A/B latency — synthetic on cuda, batch=32
==============================================================================
arm             p50 ms    p95 ms    p99 ms   mean ms         qps
guard-off        ...       ...       ...       ...          ...
guard-on         ...       ...       ...       ...          ...
------------------------------------------------------------------------------
overhead   p50 +x.xx%   p95 +x.xx%   p99 +x.xx%   throughput +x.xx%
monitor    enqueue xx.x us/step (CPU cost to the inference thread)   ring x.x MiB
SLO        p99 and throughput within 2.0%  ->  PASS
```

- **overhead** — percentage increase of each latency percentile, on minus off. `throughput`
  is the *drop* in qps (positive = GUARD reduced throughput).
- **monitor enqueue** — mean CPU microseconds `observe()` adds to the inference thread (an
  EMA over ~100 steps). This is the cost that is unavoidably serial: it happens on the
  calling thread and cannot be overlapped. If it is large, the problem is Python-side, not
  GPU-side.
- **ring** — the ring buffer's device footprint. Budget it against free VRAM (§14.4).
- **note** lines appear only when something needs attention: dropped observations (a shape
  mismatch), dropped summaries (export backpressure — raise `max_pending_drains` or
  `drain_every_k`), a detector disabled by the firewall, or the CPU-fallback warning below.

**PASS/FAIL** means p99 overhead **and** throughput drop are both within `--slo`. The
process exit status is `0` on PASS and `1` on FAIL, which is what lets the GPU CI job
(`.github/workflows/ci.yml`, `workflow_dispatch` only) gate on it. A FAIL is a real signal:
start from the tuning table in `docs/GPU_VERIFICATION.md` §4 — `cov_every` first, since it is
the only cubic-in-`D` work in the system.

---

## `saturation.py` — the honest curve

```bash
python benchmarks/saturation.py --batches 1,2,4,8,16,32,64,128
python benchmarks/saturation.py --model vit_b_16 --json sweep.json
```

Overhead is not a constant. At small batch sizes a transformer forward pass is launch- and
latency-bound, plenty of SM capacity is idle, and the monitor kernels genuinely disappear
into the gaps. At large batch sizes inference saturates the SMs, the monitor's kernels queue
behind it, and the cost becomes a real serial tail. One number quoted at its best point is
therefore a half-truth; the curve is the truth.

Flags are the same as `ab_latency.py` (`--model`, `--device`, `--iters` default `200`,
`--blocks` default `4`, `--warmup` default `20`, `--num-classes`, `--embed-dim`, `--depth`,
`--drain-every-k`, `--cov-every`, `--json`) with two differences:

- `--batches` (default `1,2,4,8,16,32,64`) — comma-separated batch sizes to sweep, replacing
  `--batch`.
- There is no `--slo`. The sweep sets it to infinity internally: it reports, it does not
  gate, and it always exits `0`.

Output is one row per batch size — off/on p99, p99 overhead %, off/on qps, throughput drop %,
enqueue µs — followed by the worst point. Read it for the **knee**: the batch size where p99
overhead crosses your SLO. That number, not the best-case one, is what belongs in a capacity
plan. Publish the whole table.

---

## `nsight_overlap.sh` — the proof

```bash
./benchmarks/nsight_overlap.sh                    # synthetic, batch 32
./benchmarks/nsight_overlap.sh vit_b_16 32        # model, batch
./benchmarks/nsight_overlap.sh vit_b_16 32 myrun  # + output basename (default guard-overlap)
```

Three positional arguments — model, batch, output basename — not flags. Requires the NVIDIA
Nsight Systems CLI (`nsys`) on `PATH`; it exits `127` with a clear message if that is
missing. It runs `ab_latency.py` under `nsys profile` with a short, fixed configuration
(`--iters 120 --blocks 2 --warmup 40 --slo 1000`) — the SLO is deliberately neutered because
profiling inflates latency and a pass/fail verdict from a traced run would be meaningless.

It writes `<out>.nsys-rep` and prints two `nsys stats` summaries (kernel-by-stream, and
memory operations). The report is the real artifact; open it in the Nsight Systems GUI and
confirm four things:

1. **Two CUDA streams, not one.** GUARD's kernels on the default stream mean the monitor
   stream was never created (CPU fallback) or the tensors were on another device.
2. **Monitor-stream kernels start before the surrounding inference kernels finish** —
   overlapping bars, not a staircase. This is the whole claim.
3. **No `cudaStreamSynchronize` or synchronous `cudaMemcpy` between forward passes.** Either
   one means something on the hot path is synchronising — a prime-directive #2 violation that
   shows up as a latency cliff under load, long before it shows up as an average.
4. **One async DtoH memcpy from pinned memory every `drain_every_k` steps**, on the monitor
   stream. If it is synchronous or per-step, the drain is not doing its job.

Record the outcome in `docs/GPU_VERIFICATION.md` §2.

---

## Running these on CPU

Every script runs without CUDA, and the results mean something quite different.

Off CUDA the stream/event facade in `guard/engine/streams.py` degrades to no-ops and the
detectors execute **inline on the calling thread** (design doc §6.5). There is no overlap to
measure. What you get is the **worst-case fully serial cost** of the detector math — an upper
bound GUARD is specifically designed never to pay in production — measured with
`time.perf_counter()` rather than CUDA events. `ab_latency.py` prints a note saying exactly
this whenever `--device` is not CUDA.

Treat a CPU run as a smoke test of the harness itself: that the model builds, the baseline
builds, the monitor attaches, the drain completes, and nothing raises. That is precisely what
the CPU CI job uses it for (with `--slo 100000`, so it cannot fail on a number a shared
runner has no business producing). It is not evidence about overhead, and a CPU result should
never be quoted as one.
