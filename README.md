# GUARD

**GPU-native unsupervised drift & anomaly monitoring for ML inference.**

[![CI](https://github.com/ericxu88/GUARD/actions/workflows/ci.yml/badge.svg)](https://github.com/ericxu88/GUARD/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Models in production fail quietly. Inputs shift, the predicted-class mix drifts, confidence
collapses — and the server keeps returning HTTP 200 with confident-looking predictions.
Nothing errors. You find out from a business metric, weeks later.

GUARD watches for that **on the GPU, on every request, concurrently with inference**.
Detector kernels run on a separate low-priority CUDA stream and fill the gaps inference
leaves behind, so monitoring costs a few tiny kernels instead of the 10–40% you pay for
copy-to-CPU-and-compute monitoring. No output leaves the GPU on the hot path; a handful of
floats are drained asynchronously every *K* steps.

```python
from guard import Config, GuardModule, PrometheusExporter, build_monitor, load

baseline = load("baselines/vit-b16.guard")            # versioned, checksummed
monitor  = build_monitor(baseline, Config.from_yaml("guard.yaml"),
                         max_batch=64, sink=PrometheusExporter())
model    = GuardModule(model, monitor, embed_layer="encoder.ln", embed_reduce="cls")

logits = model(images)     # monitored. Same output, same code path, no added latency.
```

That is the entire integration. No model surgery, no changes to your serving loop, and
nothing GUARD does can raise into it.

---

## What it detects

| Failure mode | What actually happens | What GUARD measures |
|---|---|---|
| **Covariate shift** | The input distribution moves; the model still looks confident | Mahalanobis distance to the reference Gaussian; streaming mean & covariance drift |
| **Prior / label shift** | The mix of predicted classes drifts | KL and JS divergence against the baseline class distribution |
| **Confidence collapse** | Predictions get uncertain, or individual inputs go out-of-distribution | Per-sample Shannon entropy — window mean, tail quantile, and distance to the baseline entropy histogram |

Per-batch scores are noisy, so every metric stream feeds an online change-point test
(Page-Hinkley or CUSUM) with debounce and cooldown. An alert fires when a change is
statistically real, not when one batch looks odd.

## How the overlap works

```
inference stream   ──[ forward() ]──[copy→ring]──●────[ next forward() ]────────────►
                                     ~µs          │ record event: "slot N is captured"
                                                  ▼
monitor stream     ──────────────────────────────wait──[detectors]──[async D2H every K]──►
  (low priority)                                       tiny kernels, in inference's gaps
```

Three things have to be true, and each is a test in the suite:

1. **The monitor must not start early.** A `torch.cuda.Event` recorded on the inference
   stream gates it; the monitor stream `wait_event`s on that. It is a GPU-side dependency,
   so the calling thread never blocks.
2. **The monitor must not block inference.** `observe()` only *enqueues*. No `.item()`, no
   `.cpu()`, no `synchronize()` anywhere on the path — one of those would serialise the
   inference stream and the whole premise collapses. The GPU suite asserts this with
   `torch.cuda.set_sync_debug_mode("error")`.
3. **The captured tensors must survive.** This is the one that took a redesign. The obvious
   arrangement — queue the ring copy on the monitor stream and rely on `record_stream` —
   only protects memory the caching allocator manages. CUDA-graph replay,
   `torch.compile(mode="reduce-overhead")`, TensorRT bindings and any pre-allocated `out=`
   buffer never free their outputs; they rewrite the same addresses every iteration, and
   nothing would order that against a capture still queued behind the monitor's backlog.
   The failure wouldn't be a crash — it would be metrics computed on a torn mix of two
   batches and exported as if they were real. So the copy runs in the **caller's** stream,
   ordered against both the forward pass and whatever overwrites it next. Inference pays
   two small D2D copies (~450 KB at B=64, C=1000, D=768) for correctness against every kind
   of memory. Two 256-step stress tests prove it: one with hostile allocator churn, one
   with a static buffer rewritten immediately after each call.

## Monitoring must never break serving

This is the constraint everything else bends around:

- **Exception firewall.** Every detector call is wrapped. A detector that raises is logged
  once, disabled after N failures, and its metric column goes to NaN. `forward()` returns
  normally. Same for the exporter, the aggregator, and the enqueue path itself.
- **Bounded everything.** Fixed ring buffer, fixed baselines, a fixed pool of pinned host
  buffers. Nothing allocates per request.
- **Backpressure drops, never blocks.** If export falls behind, summaries are dropped and
  counted in `guard_summaries_dropped_total`. Applying backpressure to inference is never
  the answer.
- **Kill switch.** `GUARD_DISABLED=1` in the environment, or `monitor.disable()` at
  runtime. No redeploy.
- **Malformed input is dropped, not raised.** Wrong shape, wrong device, oversized batch,
  active CUDA-graph capture — all counted in `guard_observations_dropped_total`.

## Install

```bash
pip install -e ".[dev]"       # or: uv sync --extra dev
pip install -e ".[demo]"      # + torchvision, for the ViT-B/16 example
pip install -e ".[otel]"      # + OpenTelemetry export
```

Pin `torch` to the **exact** build your serving cluster runs. CUDA stream priorities and
caching-allocator semantics are version-sensitive, and a dev/prod mismatch is exactly how a
memory-safety bug hides.

## Try it

```bash
python examples/quickstart.py      # ~2s, CPU, no downloads
```

It builds a baseline from clean traffic, persists it with a checksum, attaches the monitor,
serves in-distribution data (silence), injects a gradual then abrupt covariate shift (alerts
fire), lets traffic recover (alerts clear), and prints the Prometheus scrape:

```
  clean        entropy 2.988  JS 0.0000  mahalanobis   5.80  mean-shift   0.31  alerts —
  gradual      entropy 2.973  JS 0.0056  mahalanobis  21.90  mean-shift   4.32  alerts ['embedding']
  abrupt       entropy 2.894  JS 0.0288  mahalanobis  51.36  mean-shift  23.24  alerts ['divergence']
  recovered    entropy 2.988  JS 0.0000  mahalanobis   5.67  mean-shift   0.94  alerts —
```

The model never errored once during any of that. Breaking that silence is the point.

`examples/vit_demo.py` runs the same story against torchvision's ViT-B/16.

## Metrics

Exported to Prometheus, OpenTelemetry, or both. Detector signals: `guard_entropy_mean`,
`guard_entropy_quantile`, `guard_entropy_pop_shift`, `guard_kl_divergence`,
`guard_js_divergence`, `guard_embedding_mahalanobis_mean`, `guard_embedding_mean_shift`,
`guard_embedding_cov_drift`, `guard_drift_alert{detector}`.

GUARD also watches itself, which matters more than it sounds — a monitor you cannot tell is
broken is worse than no monitor: `guard_monitor_overhead_us` (CPU cost added to the
inference thread), `guard_ring_utilization`, `guard_detectors_disabled`,
`guard_detector_failures_total{detector}`, `guard_summaries_dropped_total`,
`guard_observations_dropped_total`, `guard_steps_observed_total`.

A Grafana dashboard and Alertmanager rules are in [`deploy/`](deploy/).

## Performance: what is claimed, and what is measured

The design target is **p99 latency increase < 2%** and **throughput drop < 2%**.

**That number has not been measured.** GUARD was developed on a machine without a CUDA
device. The engine's logic is covered by 444 passing tests and every CUDA-specific
behaviour has a `@pytest.mark.gpu` test written and waiting — but a performance claim you
have not run is marketing, not engineering, so this README does not make one.

[`docs/GPU_VERIFICATION.md`](docs/GPU_VERIFICATION.md) is the procedure, with blanks to fill
in: run the hardware suite, capture the Nsight timeline that proves kernels genuinely
overlap rather than merely being cheap, then measure the A/B curve.

```bash
pytest -m gpu                                              # 18 hardware tests
./benchmarks/nsight_overlap.sh vit_b_16 32                 # the overlap proof
python benchmarks/ab_latency.py --model vit_b_16 --batch 32 --iters 500
python benchmarks/saturation.py --batches 1,2,4,8,16,32,64 # publish the whole curve
```

Overhead is a function of spare SM capacity: near-free when inference is launch-bound,
growing as the GPU saturates. Publish the saturation curve, not the best point on it.

## Testing

```bash
pytest -m "not gpu"              # 444 tests, runs anywhere — the CI default
pytest -m gpu                    # 18 hardware tests (needs CUDA)
ruff check . && ruff format --check . && mypy guard
```

The CPU suite covers far more than the maths. Because the stream/event layer degrades to
no-ops off CUDA, the *engine itself* — slot rotation, drain batching, pinned-buffer
exhaustion, NaN columns for failed detectors, the kill switch, concurrent observers — runs
one code path that CI exercises without a GPU. What is left for hardware is genuinely only
the CUDA primitives.

Every detector is checked against an independent NumPy reference, property-tested with
Hypothesis (entropy ∈ [0, log C], JS ∈ [0, log 2]), and exercised end-to-end by a
drift-injection harness that runs clean → gradual → abrupt → recovery and asserts both that
alerts fire and that they clear.

## Layout

```
guard/
  engine/       monitor.py (observe/drain), ring_buffer.py, streams.py (CUDA facade),
                memory.py (pinned pool), aggregator.py (windows → change-point → export)
  detectors/    entropy, KL/JS divergence, embedding drift, + the Detector protocol
  baseline/     schema, offline builder, versioned checksummed store
  temporal/     Page-Hinkley, CUSUM, debounce/cooldown gate
  safety/       exception firewall, circuit breaker, kill switch
  integration/  GuardModule — nn.Module wrapper + forward hooks
  export/       Prometheus, OpenTelemetry, alert sinks, fan-out
benchmarks/     A/B latency, saturation sweep, Nsight harness
deploy/         Dockerfile, k8s manifests, Grafana dashboard, Alertmanager rules
examples/       quickstart (CPU, no downloads), ViT-B/16 demo, config
docs/           design doc, phase plan, runbook, GPU verification checklist
```

## Documentation

- **[Design & architecture](docs/GUARD_design.md)** — the full rationale, including the
  parts that are hard.
- **[Operational runbook](docs/RUNBOOK.md)** — threshold calibration, baseline lifecycle,
  what to do when each alert fires.
- **[GPU verification](docs/GPU_VERIFICATION.md)** — the hardware checklist.
- **[Phase 1 plan](docs/PHASE_1_PLAN.md)** — the CPU-verifiable slice.

## Limitations, stated plainly

- **Drift is not accuracy loss.** GUARD detects *behavioural change*, which usually but not
  always implies degradation. It is a leading indicator, not a verdict — pair it with
  periodic labelled evaluation.
- **Overlap depends on SM headroom.** If inference saturates every SM, monitor kernels
  queue and add a small serial tail. Benchmark your own workload.
- **Thresholds are per-model.** Shipped defaults are illustrative. Calibrate against your
  own clean traffic; the runbook has a runnable procedure.
- **Baseline validity is a process problem, not a maths one.** A stale baseline alarms on
  legitimate evolution; a too-fresh one masks real drift. Never auto-adopt a new baseline —
  you will calibrate the alarm to the fire.
- **The change-point tests are one-sided (upward).** They catch metrics rising, which is the
  direction drift moves them. A shift that *reduces* entropy will not trip the entropy test.
- **CUDA Graphs.** Observations during graph capture are dropped by design; per-step hooks
  cannot fire inside a captured graph.
- **Not yet built:** Triton/TorchServe adapters, ADWIN, MMD with random Fourier features,
  multi-GPU validation.

## License

Apache 2.0 — see [LICENSE](LICENSE).
