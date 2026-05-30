"""Prometheus export scaffolding (P1-09).

Defines the Prometheus metric objects whose names are locked by the design doc (§10) and
exposes an :func:`update` entry point that the Phase-2 drain thread will call.

**Phase-1 scope:** registry + metric definitions + ``update()`` + self-telemetry stubs.
No background thread, no live wiring to ``observe()`` — those are Phase 2.

Metric names (from design doc §10, all prefixed ``guard_``):

  Detector metrics
  ----------------
  guard_entropy_mean          Gauge    — windowed mean Shannon entropy
  guard_entropy_p99           Gauge    — windowed 99th-percentile entropy
  guard_js_divergence         Gauge    — JS(P‖Q) vs baseline class distribution
  guard_embedding_mahalanobis_mean  Gauge — mean Mahalanobis distance to ref Gaussian
  guard_drift_alert           Gauge    — 1.0 when a change-point test is in alarm, 0 otherwise
                                        (labelled by detector name)

  Self-telemetry
  --------------
  guard_monitor_overhead_us   Gauge    — monitor-stream wall time per step (μs); Phase 2 fills this
  guard_ring_utilization      Gauge    — fraction of ring buffer slots occupied; Phase 2 fills this
  guard_summaries_dropped_total Counter — scalar summaries dropped due to export backpressure

``update(summary)`` accepts a flat ``dict[str, float]`` mapping the metric names above
(without the ``guard_`` prefix) to their current values.  Unknown keys are silently ignored
so callers can pass the full ``DetectorResult.scores`` dict without filtering.
"""

from __future__ import annotations

import prometheus_client as prom
from prometheus_client import CollectorRegistry

# ─── module-level registry (isolated from the default; avoids collision in tests) ──

REGISTRY: CollectorRegistry = CollectorRegistry(auto_describe=True)

# ─── detector metrics ─────────────────────────────────────────────────────────

entropy_mean = prom.Gauge(
    "guard_entropy_mean",
    "Windowed mean Shannon entropy of softmax outputs",
    registry=REGISTRY,
)

entropy_p99 = prom.Gauge(
    "guard_entropy_p99",
    "Windowed 99th-percentile Shannon entropy of softmax outputs",
    registry=REGISTRY,
)

js_divergence = prom.Gauge(
    "guard_js_divergence",
    "Jensen-Shannon divergence JS(P||Q) between live and baseline class distributions",
    registry=REGISTRY,
)

embedding_mahalanobis_mean = prom.Gauge(
    "guard_embedding_mahalanobis_mean",
    "Windowed mean per-sample Mahalanobis distance to the reference Gaussian",
    registry=REGISTRY,
)

drift_alert = prom.Gauge(
    "guard_drift_alert",
    "1 when the change-point test for this detector is in alarm, 0 otherwise",
    labelnames=["detector"],
    registry=REGISTRY,
)

# ─── self-telemetry stubs (filled by Phase-2 overlap engine) ─────────────────

monitor_overhead_us = prom.Gauge(
    "guard_monitor_overhead_us",
    "Monitor-stream wall time per inference step in microseconds (Phase-2 filled)",
    registry=REGISTRY,
)

ring_utilization = prom.Gauge(
    "guard_ring_utilization",
    "Fraction of on-GPU ring buffer slots currently occupied (Phase-2 filled)",
    registry=REGISTRY,
)

# prometheus_client auto-appends "_total" in the text-format output; name without suffix.
summaries_dropped_total = prom.Counter(
    "guard_summaries_dropped",
    "Scalar summaries dropped due to export-thread backpressure",
    registry=REGISTRY,
)

# ─── key → metric setter map ─────────────────────────────────────────────────

# Maps the un-prefixed key (as it appears in DetectorResult.scores) to the setter
# function that writes the value into the Prometheus metric.
_SETTERS: dict[str, prom.Gauge] = {
    "entropy_mean": entropy_mean,
    "entropy_quantile": entropy_p99,  # quantile key produced by EntropyDetector
    "entropy_p99": entropy_p99,  # alias for direct use
    "js_divergence": js_divergence,
    "embedding_mahalanobis_mean": embedding_mahalanobis_mean,
    "monitor_overhead_us": monitor_overhead_us,
    "ring_utilization": ring_utilization,
}

_DRIFT_ALERT_DETECTORS = frozenset({"entropy", "divergence", "embedding"})


def update(summary: dict[str, float]) -> None:
    """Write scalar values into the Prometheus metric objects.

    Args:
        summary: flat mapping of un-prefixed metric name → value. Unknown keys are
            silently ignored. ``drift_alert`` entries follow the convention
            ``"drift_alert.<detector>"`` (e.g. ``"drift_alert.entropy"``).
    """
    for key, value in summary.items():
        if key in _SETTERS:
            _SETTERS[key].set(value)
        elif key.startswith("drift_alert."):
            detector = key[len("drift_alert.") :]
            drift_alert.labels(detector=detector).set(value)
        elif key == "summaries_dropped":
            summaries_dropped_total.inc(max(0.0, value))
