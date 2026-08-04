# GUARD Grafana dashboard

`guard-dashboard.json` — a single dashboard covering drift signals, per-detector detail,
and GUARD's own health. Built for Grafana 10/11 (`schemaVersion: 39`), UID
`guard-overview`.

Every query in it uses a metric name that `guard/export/prometheus.py` actually registers.
Nothing is invented; if a panel says "No data" the metric exists but has not been fed yet.

## Prerequisites

GUARD does not bind a port on its own — a library that opens a socket behind your back is
a library you cannot deploy. Your startup code has to serve `/metrics`:

```python
from guard.export.prometheus import PrometheusExporter
from guard.engine.monitor import build_monitor

exporter = PrometheusExporter(namespace=config.export.namespace)   # "guard"
monitor = build_monitor(baseline, config, max_batch=64, sink=exporter)
if config.export.enabled:
    exporter.start_http_server(config.export.prometheus_port)      # default 9300
```

Then point Prometheus at that port. The dashboard's `job` and `instance` variables are
populated from `label_values(guard_steps_observed_total, …)`, so they come from whatever
your scrape config labels the target as — GUARD adds no labels of its own.

**The namespace must be `guard`.** Every query is written against `guard_*`. If you set
`export.namespace` to something else, the dashboard shows nothing until you find-and-replace
the prefix in the JSON.

## Import

1. Grafana → **Dashboards** → **New** → **Import**.
2. Upload `guard-dashboard.json` (or paste its contents).
3. Pick your Prometheus datasource when prompted, then **Import**.

The dashboard has a `datasource` template variable, so it is not pinned to a datasource UID
and imports cleanly into any stack. Re-importing overwrites the previous copy because the
UID (`guard-overview`) is fixed; change the UID before importing if you want to keep two
variants side by side.

For file-based provisioning, drop the JSON into your provider's path and let Grafana pick
it up:

```yaml
# /etc/grafana/provisioning/dashboards/guard.yaml
apiVersion: 1
providers:
  - name: guard
    type: file
    options:
      path: /var/lib/grafana/dashboards/guard
```

### Variables

| Variable     | What it does                                                              |
| ------------ | ------------------------------------------------------------------------- |
| `datasource` | Which Prometheus datasource every panel queries.                          |
| `job`        | Multi-select over `job`, from `guard_steps_observed_total`. Defaults to All. |
| `instance`   | Multi-select over `instance`, filtered by the current `job`. Defaults to All. |

With All selected on both, each panel draws one series per replica. That is deliberate:
drift is usually uniform across replicas, so a single replica diverging from the rest is
itself a finding (a bad rollout, a stale baseline on one pod).

There is also an annotation query — `max by (detector) (guard_drift_alert) == 1` — which
paints a red marker on every panel while any detector is latched in alarm, so you can line
a firing up against the raw signals without switching views.

## What each row tells you

### 1. Drift signals

The four numbers you look at first, plus a count of currently latched alerts.

- **Mean output entropy** (`guard_entropy_mean`) — how confident the model is on average.
  Rising = the model is hedging on inputs it used to be sure about. Falling can be worse:
  confidently wrong is the failure mode nobody catches without a baseline.
- **JS divergence** (`guard_js_divergence`) — the preferred alerting signal for output
  drift. Bounded in `[0, ln 2]` with the natural-log convention, symmetric, and finite even
  when a class disappears entirely, which is exactly where KL blows up.
- **Embedding Mahalanobis mean** (`guard_embedding_mahalanobis_mean`) — mean per-sample
  distance to the reference Gaussian in feature space. This one moves *before* the output
  metrics do, because inputs can wander a long way into unfamiliar feature territory while
  the classifier still produces a plausible-looking distribution.
- **Embedding mean shift** (`guard_embedding_mean_shift`) — the same distance applied to
  the running batch mean rather than to individual samples. A large mean shift with a flat
  per-sample distance means the whole population moved together (a new camera, a new
  region, a preprocessing change) rather than a few odd samples arriving.
- **Active drift alerts** (`guard_drift_alert{detector}`) — how many detectors' change-point
  tests are latched. 0 is healthy. This gauge has no series at all until the first alert
  transition in a process, so the query falls back to `vector(0)`; a reading of 0 means
  either "no drift" or "nothing has happened yet". Confirm against the steps-observed panel
  in row 3 before you trust it.

### 2. Detector detail

Where you go once row 1 has told you *something* moved, to work out *what*.

- **KL divergence** (`guard_kl_divergence`) — same comparison as JS, unbounded and
  asymmetric. Useful for magnitude; a poor thing to page on.
- **Entropy tail quantile** (`guard_entropy_quantile`) — the configured upper quantile of
  per-sample entropy. Catches a growing tail of genuinely confusing inputs that the mean
  dilutes away. (The exporter also registers `guard_entropy_p99`, but it is only fed when
  the detector's configured quantile really is 0.99, so the dashboard reads the generic
  gauge instead.)
- **Entropy population shift** (`guard_entropy_pop_shift`) — total-variation distance
  between the live entropy histogram and the baseline's. Bounded in `[0, 1]`, and sensitive
  to shape changes that leave the mean and the quantile alone — e.g. a bimodal split into
  "very confident" and "very confused".
- **Embedding covariance drift** (`guard_embedding_cov_drift`) — `‖Σ_run·Σ_ref⁻¹ − I‖_F`.
  Moves when the *shape* of the feature cloud changes rather than its location. Rising here
  while the mean shift stays flat usually means the input mix got more (or less) diverse.

Row 3 of the runbook (`docs/RUNBOOK.md` §3) has a per-metric triage procedure and a table
for reading combinations of these signals together.

### 3. GUARD self-telemetry

GUARD watching itself. Read this row **first** during an incident, because a quiet drift
panel means nothing if the monitor stopped running.

- **`observe()` overhead** (`guard_monitor_overhead_us`) — an EMA over roughly 100 steps of
  the CPU microseconds `observe()` adds to the serving thread. This is enqueue cost only;
  detector math runs on the monitor stream. A step change after a deploy is the signature
  of a host synchronisation (`.item()`, `.cpu()`, printing a CUDA tensor) having crept back
  into the hot path. **The threshold line at 200 µs is a placeholder** — 2% of a
  hypothetical 10 ms step. Set it to 2% of *your* measured step time.
- **Ring utilisation** (`guard_ring_utilization`) — fraction of ring slots whose monitor
  work has not finished on the GPU. Near 0 is healthy. Persistently near 1.0 means the
  monitor stream is falling behind inference, which is the honest signal that overlap has
  stopped being free on this workload.
- **Summaries dropped /s** (`rate(guard_summaries_dropped_total[5m])`) — scalar summaries
  discarded because every pinned host buffer was still in flight. Bounded-memory
  backpressure working as designed, but a sustained rate means you are losing metrics:
  raise `max_pending_drains` or `drain_every_k`.
- **Observations dropped /s** (`rate(guard_observations_dropped_total[5m])`) — batches
  `observe()` rejected before doing any work: wrong shape, wrong device, batch above
  `max_batch`, or an active CUDA graph capture. A steady non-zero rate is a wiring bug, not
  a drift signal. Only the first rejection is logged (per monitor), so this counter is the
  durable evidence.
- **Detector failures /s** (`rate(guard_detector_failures_total[5m])`) — exceptions the
  safety firewall caught, by detector. The exception never reaches the serving path, so
  this is the only early warning. A burst followed by silence is a *tripped breaker*, not a
  recovery.
- **Detectors disabled** (`guard_detectors_disabled`) — detectors the breaker switched off
  after `max_detector_failures` (default 3). Inference is unaffected; the drift panels above
  just go quiet, which looks reassuring and is not.
- **Steps observed /s** (`rate(guard_steps_observed_total[5m])`) — the liveness signal. Flat
  at 0 while traffic flows means GUARD is not seeing it: check the global kill-switch
  (`enabled: false` in config, or `GUARD_DISABLED=1` in the environment) and the engine
  breaker. Every other panel is meaningless while this one is 0.

Two registered metrics are deliberately not on the dashboard: `guard_entropy_p99` (an alias
covered by the quantile panel) and `guard_monitor_gpu_time_us`, which is only populated when
you construct the monitor with `enable_gpu_timing=True`. Add a panel for the latter if you
turn timing on.

## The thresholds here are placeholders

This is the part to read before you wire any of it to a pager.

Every coloured threshold and every dashed line in this JSON is a **shipped default with no
knowledge of your model**. Absolute values of entropy, KL/JS, and Mahalanobis distance are
not comparable across models, datasets, or even across baselines built from different
calibration windows. A `guard_js_divergence` of 0.05 can be a routine daily cycle for one
model and a serious regression for another. There is no universal number, and any dashboard
that claims one is lying to you.

Calibrate against your own deployment's clean traffic before you trust a colour:
**`docs/RUNBOOK.md` §1 (Calibrating thresholds)** has the procedure, including the
distinction between `delta` (metric units, per-step slack) and `threshold` (accumulated
evidence, unitless) that is the most common way to get this wrong. Do not eyeball it from
the dashboard; the runbook derives the numbers from a recorded clean window.

The `observe()` overhead threshold deserves its own warning. The 200 µs line encodes a
guess at your step time, and the project's p99-overhead-under-2% figure is an **SLO to be
verified on hardware, not a measured result** — see `docs/GPU_VERIFICATION.md` for how to
measure it (Nsight overlap proof plus an A/B latency run) on your own GPU.

## Validating a local edit

```bash
python -c "import json; d=json.load(open('deploy/grafana/guard-dashboard.json')); print(len(d['panels']))"
```

19 panels: 3 row headers plus 16 visualisations.
