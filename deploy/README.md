# Deploying GUARD

Reference deployment artifacts: a container image, Kubernetes manifests, Prometheus
alerting rules, and a Grafana dashboard.

**None of this is built, applied, or fired in CI.** It is the shape of a correct
deployment, verified against the code (every metric name, config key, port, and environment
variable below was read out of the source, not remembered), but it has not run on a
cluster. Pin the image tags to digests, calibrate the thresholds, and scan the image before
any of it goes near production.

## What you are deploying

GUARD is a **library that runs in-process with your inference server**, not a service.
It needs the same CUDA context and the live output tensors, so there is no GUARD pod, no
sidecar, and nothing of GUARD's on the request path. `observe()` is called on the inference
thread right after `forward()` and returns in microseconds having only *enqueued* work onto
a low-priority CUDA stream.

What that means for deployment:

- The manifests here describe **your serving Deployment** with GUARD linked into it. The
  only GUARD-specific additions are one extra port, one environment variable, two mounted
  volumes, and the scrape configuration.
- Scaling GUARD means scaling your serving replicas. There is nothing else to scale.
- Every GUARD metric is **per-process**. `guard_ring_utilization`,
  `guard_monitor_overhead_us`, and each detector's change-point state live inside one
  replica's engine. Scrape per-pod endpoints (which is what the `ServiceMonitor` here does)
  or an unhealthy replica averages away into a healthy fleet.

## Files

| Path | What it is |
|---|---|
| `docker/Dockerfile` | Multi-stage CUDA image: builder venv + slim runtime, non-root UID 10001, `EXPOSE 9300` |
| `docker/.dockerignore` | Context exclusions — read the header, Docker does **not** find this file automatically |
| `k8s/deployment.yaml` | `ConfigMap` (guard.yaml) + `PersistentVolumeClaim` (baselines) + serving `Deployment` |
| `k8s/servicemonitor.yaml` | `Service` + Prometheus Operator `ServiceMonitor` |
| `prometheus/guard-alerts.yml` | Alerting rules: drift, GUARD's own health, overhead regression |
| `grafana/guard-dashboard.json` | Dashboard: drift signals, per-detector detail, monitor health |

## 1. Build

```bash
# From the REPOSITORY ROOT — the Dockerfile COPYs pyproject.toml, README.md, and guard/.
docker build -f deploy/docker/Dockerfile -t guard-monitor:0.2.0 .
```

`.dockerignore` needs one of these first, or the whole repo (including `.git` and `.venv`)
is uploaded to the daemon:

```bash
ln -s deploy/docker/.dockerignore .dockerignore                        # context root
# or, with BuildKit:
cp deploy/docker/.dockerignore deploy/docker/Dockerfile.dockerignore
```

### Pin torch

The single most important build argument:

```bash
docker build -f deploy/docker/Dockerfile \
  --build-arg TORCH_VERSION=2.5.1 \
  --build-arg TORCH_CUDA=cu124 \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  -t guard-monitor:0.2.0 .
```

`pyproject.toml` declares `torch>=2.5`, which is a **floor for library consumers, not a
deployment target**. `record_stream` semantics, stream priorities, and caching-allocator
reuse are version-sensitive, and GUARD's cross-stream memory safety (prime directive #3)
rests on all three. A monitor validated against 2.5.1+cu124 and deployed against a
different build is not the thing you tested. An image is a deployment artifact and gets an
`==`; the build fails loudly if dependency resolution replaces the torch you asked for.

If your serving stack already standardises on `nvcr.io/nvidia/pytorch:<yy.mm>-py3`, build
on that instead and **drop the torch install** — do not layer a pip torch over NGC's.

### In practice

You probably will not deploy this image as-is. It packages the `guard` library plus a
CUDA-capable Python runtime; the realistic path is either building your serving image
`FROM guard-monitor:0.2.0`, or just adding `guard-monitor` to your existing serving image's
requirements with torch already pinned there. The Dockerfile is worth reading either way —
the base-image, non-root, and `PYTORCH_CUDA_ALLOC_CONF` decisions apply to your image too.

## 2. Run

The image's default command runs `examples/quickstart.py`, which is a genuine smoke test:
it builds a baseline, serves, injects drift, alerts, recovers, prints the Prometheus
exposition, and exits non-zero if any of that breaks.

```bash
# CPU: verifies the package imports and the whole pipeline works. No GPU needed.
docker run --rm guard-monitor:0.2.0

# GPU: same run, but detectors move onto the low-priority monitor stream and overlap.
docker run --rm --gpus all guard-monitor:0.2.0 \
  python /app/examples/quickstart.py --device cuda
```

If you run the container with a read-only root filesystem, redirect the artifact it writes:
`--baseline /tmp/quickstart.guard`.

## 3. Wire it up

### 3.1 In your serving code

GUARD does not start itself. Three things belong in your startup path:

```python
from guard import Config, GuardModule, build_monitor, load

config   = Config.from_yaml(os.environ["GUARD_CONFIG_PATH"])     # /etc/guard/guard.yaml
baseline = load(
    os.environ["GUARD_BASELINE_PATH"],                           # verifies the checksum
    expected_model_version=os.environ["GUARD_MODEL_VERSION"],    # and the version binding
)
monitor  = build_monitor(baseline, config, max_batch=64)

# Assert your thresholds actually match metrics. `thresholds` is an open mapping, so a
# misspelled metric name validates cleanly and then alerts on nothing, silently, forever.
ignored = set(config.thresholds) - set(monitor.metric_names)
assert not ignored, f"these threshold entries alert on nothing: {sorted(ignored)}"

# build_monitor builds the exporter but deliberately does not bind a port — a library
# should not open a socket behind your back. This is the line that makes /metrics exist.
monitor.sink.start_http_server(config.export.prometheus_port)    # 9300

model = GuardModule(model, monitor, embed_layer="backbone")      # every forward is monitored
```

Set `max_batch` to your server's **true** maximum, not its nominal one. A batch outside
`[1, max_batch]` is dropped entirely, never truncated.

### 3.2 The manifests

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/servicemonitor.yaml
```

Before you do, replace: the image, the namespace (`inference`), `model-version` (in the pod
label, the `subPath`, and `GUARD_BASELINE_PATH` — all three), the resource numbers, the
`nodeSelector`/`tolerations`, the storage class, and the `release:` label on the
ServiceMonitor (it must match your Prometheus CR's `serviceMonitorSelector`, or the Operator
ignores the object with no error anywhere).

Populate the baselines PVC with a directory per model version:

```
/etc/guard/baselines/v1.4.2/baseline.guard
```

Baselines are versioned, checksummed artifacts **bound to a model version** — `load()`
recomputes a SHA-256 over the tensor content and refuses a file whose bytes moved. Binding
the mount `subPath` to the version means a model that ships without its matching baseline
fails to start, instead of running for a week and paging you at 3am with drift alerts on a
perfectly healthy model. That mismatch is the most common false page there is.

### 3.3 Prometheus and Grafana

```bash
# Rules: mount into your Prometheus, or reference from a PrometheusRule CR.
promtool check rules deploy/prometheus/guard-alerts.yml

# Dashboard: import deploy/grafana/guard-dashboard.json (uid: guard-overview).
```

Confirm the target is actually up before trusting any of it:

```bash
kubectl port-forward deploy/model-server 9300:9300
curl -s localhost:9300/metrics | grep '^guard_'
```

A newly started process exports zeros and, for `guard_drift_alert`, **no series at all** —
that gauge is labelled per detector and only exists after the first alert state transition.
A reading of 0 there means "no alert" *or* "no alert yet"; cross-check
`guard_steps_observed_total` to confirm GUARD is seeing traffic.

### 3.4 Probes stay off GUARD

Every probe in `deployment.yaml` targets the serving port. **None** touches 9300 or any
GUARD state, and that is deliberate.

Prime directive #1 — monitoring must never block, stall, or crash inference — is enforced
in-process by the exception firewall. That guarantee is worth nothing if the orchestrator
then kills the pod because the monitor went quiet. A probe on `/metrics` would mean a
tripped detector breaker restarts a healthy server, and `GUARD_DISABLED=1` — the documented
way to turn monitoring off during an incident — becomes a fleet-wide CrashLoopBackOff.

When GUARD is unhealthy you want an **alert**, not a restart. Alerts wake a human who can
read a traceback; restarts destroy the state that traceback came from. That is what the
`guard-self-health` rule group is for.

## 4. When an alert fires: what to check first

### The rule that matters most

**A firing `guard_drift_alert` means investigate. It does not mean the model has
degraded.** GUARD is unsupervised — no labels, no ground truth, no accuracy measurement.
It reports behavioural change relative to a baseline. That is a leading indicator, and it
is valuable precisely because it arrives before any accuracy metric could; it is not a
verdict. Drift without degradation is common and legitimate. Do not roll back a model on
this signal alone.

### Four questions, in this order

1. **Is the baseline the right one?** Compare the `model_version` label and the alert
   payload's `baseline_version` against what is actually deployed. A model rollout that
   shipped without its baseline produces drift alerts on a healthy model. Check this first
   every time — it is the most common cause and the cheapest to rule out.
2. **What else changed at the same time?** Deploys, preprocessing, a new client, a routing
   change, a batch-size change. Correlate the firing timestamp with the deploy log before
   you look at the model at all.
3. **Which other metrics moved?** One metric moving alone means something different from
   all of them moving together. The dashboard shows them side by side; the combination
   table in `docs/RUNBOOK.md` §3.9 distinguishes covariate shift from label shift from
   confidence collapse.
4. **Is GUARD itself healthy?** `guard_detectors_disabled`,
   `rate(guard_observations_dropped_total[5m])`, `rate(guard_summaries_dropped_total[5m])`.
   A partially broken monitor produces metric streams with holes, and a holed stream can
   page. Rule this out before you spend an afternoon on the model.

Note the `guard_drift_alert` label is the **detector**, not the metric: `entropy_mean` and
`entropy_pop_shift` both raise `detector="entropy"`. Only the `Alert` payload in your alert
sink names the metric that actually fired.

### If GUARD itself is the thing alerting

| Alert | First thing to check |
|---|---|
| `GuardDetectorDisabled` | The `guard.engine` logger on that pod. The firewall logs the **first** failure at ERROR with a traceback and the trip at WARNING; later identical failures are counted but not logged, so do not wait for a fresh one. |
| `GuardDetectorFailing` | Same log. Catches an intermittent detector *before* its breaker trips. |
| `GuardSummariesDropped` | `stats().drains_enqueued - stats().drains_completed` sitting at `max_pending_drains`. Fix by raising `drain_every_k` first. |
| `GuardObservationsDropped` | The single WARNING at pod startup naming the reason — that one line is the whole diagnosis. Usually dynamic batching exceeding `max_batch`. |
| `GuardNotObservingTraffic` | Whether the pod is serving at all. If it is, check `GUARD_DISABLED`, then look for a `guard.engine` ERROR naming component `"engine"` — that means all monitoring stopped. |
| `GuardMonitorOverheadRegression` | A host synchronisation (`.item()`, `.cpu()`, `.tolist()`, `synchronize()`) that reached a detector or the enqueue path. It serialises the inference stream. |
| `GuardRingBufferSaturated` | Raise `cov_every` or disable the `embedding` detector. Raising `ring_slots` buys pipelining depth, not throughput. |

`docs/RUNBOOK.md` §4 has the full diagnosis and recovery procedure for each.

### Turning it off

Three mechanisms, increasing blast radius (`docs/RUNBOOK.md` §5):

```bash
# 1. Kill-switch, fleet-wide. GUARD_DISABLE_ENV in guard/safety/firewall.py. Truthy values
#    are 1/true/yes/on, case-insensitive; anything else (including "0") leaves it ON.
#    Already wired in deployment.yaml set to "0", so this is one command under pressure.
kubectl set env deployment/model-server GUARD_DISABLED=1

# 2. In-process, no restart:  monitor.disable()
# 3. Config:                  enabled: false in guard.yaml (needs a rollout to take effect)
```

`GuardNotObservingTraffic` will fire while GUARD is intentionally off. That is correct —
"monitoring is disabled" should be visible, not silent. Silence it in Alertmanager for the
duration.

## 5. Calibrate before this pages anyone

Two sets of numbers ship as placeholders and **must** be replaced:

- **`thresholds` in `guard.yaml`.** The values in the ConfigMap are the illustrative ones
  from `examples/guard.yaml` — chosen so the example runs, not so they fit your model.
  `delta` is in the metric's own units; `threshold` bounds the change-point test's
  *cumulative statistic*, which is a different quantity. Mixing them up is the most common
  calibration mistake. `docs/RUNBOOK.md` §1 has a runnable procedure.
- **`GuardMonitorOverheadHigh` (250us) and `GuardRingBufferSaturated` (0.9).** Both are
  workload- and hardware-dependent, with no universal good value. Record yours from a
  healthy deployment on day one. The companion `GuardMonitorOverheadRegression` rule is
  self-relative (versus the same pod yesterday) and needs no calibration, which is why it
  is the one to trust first.

### On the overhead SLO

The design targets **p99 inference overhead < 2%**. That target is **unverified** — it has
not been measured on hardware (`docs/GPU_VERIFICATION.md` is the procedure and the place to
record the result). It also **cannot be evaluated from these metrics**: GUARD exports its
own enqueue cost, not your inference latency. Alerting on the real SLO means comparing your
serving latency histogram against a `GUARD_DISABLED=1` control, which is what
`benchmarks/ab_latency.py` measures offline. The rules in `guard-alerts.yml` are the leading
indicators available from GUARD alone, and they are not a substitute for that measurement.
