# GUARD

GPU-native, unsupervised drift and anomaly monitoring for PyTorch inference.

[![CI](https://github.com/ericxu88/GUARD/actions/workflows/ci.yml/badge.svg)](https://github.com/ericxu88/GUARD/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

GUARD watches a serving model for distribution shift without labels and without taking
outputs off the GPU. Detector kernels run on a separate low-priority CUDA stream, so they
execute in whatever SM capacity inference leaves idle. Only a few floats per step cross to
the host, batched and asynchronous.

It compares live behaviour against a baseline computed offline from known-good data, and
raises an alert when an online change-point test decides the departure is real rather than
noise.

## Install

```bash
pip install -e ".[dev]"       # or: uv sync --extra dev
pip install -e ".[demo]"      # + torchvision, for the ViT-B/16 example
pip install -e ".[otel]"      # + OpenTelemetry export
```

Pin `torch` to the exact build your serving cluster runs. CUDA stream priorities and
caching-allocator behaviour are version-sensitive.

## Quickstart

```bash
python examples/quickstart.py
```

Runs on CPU in a couple of seconds with no downloads. It builds a baseline, attaches the
monitor, serves clean traffic, injects a covariate shift, and prints the Prometheus scrape.

## Usage

```python
from guard import Config, GuardModule, PrometheusExporter, build_monitor, load

baseline = load("baselines/vit-b16.guard", expected_model_version="vit-b16@abc123")
config   = Config.from_yaml("guard.yaml")
exporter = PrometheusExporter()

monitor = build_monitor(baseline, config, max_batch=64, device="cuda", sink=exporter)
exporter.start_http_server(config.export.prometheus_port)

model = GuardModule(model, monitor, embed_layer="encoder.ln", embed_reduce="cls")

logits = model(images)     # monitored; same output, same code path
```

`GuardModule` registers a forward hook on the named layer to capture the penultimate
embedding and reads logits from the return value. Model weights are not touched. Use
`candidate_embed_layers(model)` to find the layer name, and check the captured shape on one
batch before committing to it.

If you drive the loop yourself, skip the wrapper and call `monitor.observe(logits, embeddings)`
directly after `forward()`.

### Building a baseline

```python
from guard import build_baseline, save

baseline = build_baseline(
    calibration_batches,          # iterable of (logits [B, C], embeddings [B, D])
    model_version="vit-b16@abc123",
    num_classes=1000,
    embed_dim=768,
)
save(baseline, "baselines/vit-b16.guard")
```

Baselines are versioned artifacts bound to a model version, checksummed on write and
verified on load. Calibrate on substantially more samples than the embedding dimension, or
the reference covariance is ill-conditioned. See [docs/RUNBOOK.md](docs/RUNBOOK.md) §2.

### Configuration

`examples/guard.yaml` is a documented starting point. Thresholds are keyed by **metric**
name, not detector name, and the shipped values are illustrative. Calibrate them against
your own clean traffic before enabling alerts; [docs/RUNBOOK.md](docs/RUNBOOK.md) §1 has a
runnable procedure.

```yaml
enabled: true
window_size: 32
drain_every_k: 32

detectors:
  - name: entropy
    params: { quantile: 0.99 }
  - name: divergence
  - name: embedding
    params: { cov_every: 32 }

thresholds:
  js_divergence:
    method: page_hinkley
    threshold: 0.05
    delta: 0.002
    debounce: 3
    cooldown: 20
```

## What it detects

| Signal | Failure mode |
|---|---|
| Shannon entropy — window mean, tail quantile, distance to the baseline histogram | Confidence collapse, per-sample OOD |
| KL and JS divergence against the baseline class distribution | Prior / label shift |
| Mahalanobis distance, streaming mean and covariance drift | Covariate shift |

Each metric stream feeds an online change-point test (Page-Hinkley or CUSUM) with debounce
and cooldown, so a single unusual batch does not page anyone.

## Metrics

Prometheus by default; OpenTelemetry and multi-sink fan-out are available.

**Detectors:** `guard_entropy_mean`, `guard_entropy_quantile`, `guard_entropy_pop_shift`,
`guard_kl_divergence`, `guard_js_divergence`, `guard_embedding_mahalanobis_mean`,
`guard_embedding_mean_shift`, `guard_embedding_cov_drift`, `guard_drift_alert{detector}`

**Self-telemetry:** `guard_monitor_overhead_us`, `guard_ring_utilization`,
`guard_detectors_disabled`, `guard_detector_failures_total{detector}`,
`guard_steps_observed_total`, `guard_observations_dropped_total`,
`guard_summaries_dropped_total`

A Grafana dashboard and Alertmanager rules are in [`deploy/`](deploy/).

## How it works

```
inference stream   ──[ forward() ]──[copy→ring]──●────[ next forward() ]────────────►
                                                 │ event: slot captured
                                                 ▼
monitor stream     ──────────────────────────────wait──[detectors]──[async D2H every K]──►
  (low priority)
```

`observe()` runs on the inference thread and only enqueues work. It contains no `.item()`,
`.cpu()`, or `synchronize()` call, so it never serialises the inference stream.

The ring-buffer copy is issued on the **inference** stream rather than the monitor stream.
`record_stream` alone is not sufficient: it defers reuse of memory the caching allocator
frees, but CUDA-graph replay, `torch.compile(mode="reduce-overhead")`, TensorRT bindings and
pre-allocated `out=` buffers rewrite the same addresses every iteration without the
allocator's involvement. Copying in the caller's stream orders the capture against both the
forward pass and the next overwrite. Inference pays two small device-to-device copies.

Slot reuse is handled by making the inference stream wait on the slot's completion event
before overwriting it. At the default ring depth that event has long since completed.

Full rationale: [docs/GUARD_design.md](docs/GUARD_design.md) §6.

## Failure isolation

Monitoring is best-effort and cannot affect serving.

- Every detector call runs behind an exception firewall. A detector that raises is logged
  once and disabled after N failures; its metric column becomes NaN and the aggregator skips
  it. `forward()` returns normally.
- Fixed ring buffer, fixed baselines, fixed pool of pinned host buffers. Nothing allocates
  per request.
- When the export path falls behind, summaries are dropped and counted in
  `guard_summaries_dropped_total`. Backpressure is never applied to inference.
- Malformed input (wrong shape, wrong device, oversized batch, active graph capture) is
  dropped and counted in `guard_observations_dropped_total`.
- Kill switch: `GUARD_DISABLED=1` in the environment, or `monitor.disable()` at runtime.

## Testing

```bash
pytest -m "not gpu"                              # 480 tests, no GPU required
pytest -m gpu                                    # 18 hardware tests
ruff check . && ruff format --check . && mypy guard
```

The stream and event layer degrades to no-ops off CUDA, so slot rotation, drain batching,
buffer exhaustion, failure isolation and concurrency are all covered by the CPU suite.
`tests/test_stream_contract.py` drives the engine through a recording stand-in for
`torch.cuda` and asserts the ordering contract the memory-safety argument depends on.
`tests/test_accelerator.py` runs the detectors and the full pipeline on CUDA or MPS,
whichever is present.

Detectors are checked against independent NumPy references and property-tested with
Hypothesis. `tests/test_drift_injection.py` runs clean → gradual → abrupt → recovery and
asserts that alerts both fire and clear.

## Benchmarks

Overhead depends on how much SM capacity inference leaves idle, so it has to be measured on
your model and your hardware.

```bash
python benchmarks/ab_latency.py --model vit_b_16 --batch 32 --iters 500
python benchmarks/saturation.py --batches 1,2,4,8,16,32,64
./benchmarks/nsight_overlap.sh vit_b_16 32
```

`ab_latency.py` reports p50/p95/p99 deltas with GUARD off and on, timed with CUDA events and
interleaved A/B blocks, and exits non-zero if the measured overhead exceeds `--slo`, so it
can gate CI. `saturation.py` sweeps batch size to show where overlap stops being free.
`nsight_overlap.sh` captures the timeline that confirms monitor kernels interleave with
inference rather than serialising after it.

[docs/GPU_VERIFICATION.md](docs/GPU_VERIFICATION.md) is the full hardware checklist.

## Deployment

GUARD runs in-process with the inference server; it needs the same CUDA context and the live
tensors. There is no GUARD service and nothing of GUARD's on the request path. One monitor
per device; every metric is per-process, so scrape per-pod endpoints.

[`deploy/`](deploy/) has a reference Dockerfile, Kubernetes manifests, the Grafana dashboard,
and Alertmanager rules.

## Layout

```
guard/
  engine/       observe/drain, ring buffer, CUDA stream facade, pinned pool, aggregator
  detectors/    entropy, KL/JS divergence, embedding drift, Detector protocol
  baseline/     schema, offline builder, versioned checksummed store
  temporal/     Page-Hinkley, CUSUM, debounce/cooldown gate
  safety/       exception firewall, circuit breaker, kill switch
  integration/  GuardModule wrapper and forward hooks
  export/       Prometheus, OpenTelemetry, alert sinks, fan-out
benchmarks/     A/B latency, saturation sweep, Nsight harness
deploy/         Dockerfile, k8s manifests, Grafana dashboard, alert rules
examples/       quickstart, ViT-B/16 demo, annotated config
docs/           design doc, runbook, GPU verification checklist
```

## Documentation

- [Design and architecture](docs/GUARD_design.md)
- [Operational runbook](docs/RUNBOOK.md) — threshold calibration, baseline lifecycle, alert response
- [GPU verification](docs/GPU_VERIFICATION.md)
- [Contributing](CONTRIBUTING.md)

## Limitations

- Drift is not accuracy loss. GUARD detects behavioural change, which usually but not always
  implies degradation. Pair it with periodic labelled evaluation.
- Thresholds are per-model and must be calibrated against your own clean traffic.
- A stale baseline alarms on legitimate evolution; a too-fresh one masks real drift.
  Re-baselining is a human-approved workflow, not an automatic one.
- The change-point tests are one-sided. They catch metrics rising, which is the direction
  drift moves them; a shift that reduces entropy will not trip the entropy test.
- Observations during CUDA graph capture are dropped. Per-step hooks cannot fire inside a
  captured graph.
- Not implemented: Triton and TorchServe adapters, ADWIN, MMD with random Fourier features,
  multi-GPU validation.

## License

Apache 2.0. See [LICENSE](LICENSE).
