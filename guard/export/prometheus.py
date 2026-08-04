"""Prometheus export (P1-09, completed in Phase 2).

Two layers:

* :class:`PrometheusExporter` — the real thing. Owns its own ``CollectorRegistry``, honours
  :class:`~guard.config.ExportConfig`'s ``namespace``, creates a gauge for every metric the
  configured detectors actually emit, and implements :class:`~guard.export.base.MetricSink`
  so the drain thread can write to it.
* A module-level default instance and the ``REGISTRY`` / ``update()`` functions, kept as
  the stable Phase-1 surface. They delegate to a default exporter in the ``guard``
  namespace.

Metric names (design doc §10, prefixed with the configured namespace, ``guard`` by default):

  Detector metrics — one gauge per metric a detector emits
  -------------------------------------------------------
  guard_entropy_mean                 windowed mean Shannon entropy
  guard_entropy_quantile             windowed tail quantile of entropy
  guard_entropy_p99                  alias of the above, fed only when the detector's
                                     configured quantile really is 0.99
  guard_entropy_pop_shift            TV distance vs the baseline entropy histogram
  guard_kl_divergence                KL(P‖Q) vs the baseline class distribution
  guard_js_divergence                JS(P‖Q), the preferred alerting signal
  guard_embedding_mahalanobis_mean   mean per-sample Mahalanobis distance
  guard_embedding_mean_shift         Mahalanobis distance of the running mean
  guard_embedding_cov_drift          ‖Σ_run·Σ_ref⁻¹ − I‖_F
  guard_drift_alert{detector}        1 while a detector's change-point test is latched

  Self-telemetry — GUARD watching itself
  --------------------------------------
  guard_monitor_overhead_us          CPU time ``observe()`` adds to the inference thread
  guard_monitor_gpu_time_us          monitor-stream GPU time per step (opt-in timing)
  guard_ring_utilization             fraction of ring slots with work still in flight
  guard_steps_observed_total         batches successfully enqueued
  guard_observations_dropped_total   batches rejected (bad shape, oversized, graph capture)
  guard_summaries_dropped_total      scalar summaries dropped under export backpressure
  guard_detector_failures_total{detector}  exceptions caught by the firewall
  guard_detectors_disabled           detectors currently disabled by a tripped breaker

``guard_entropy_p99`` deserves a word. The design doc locks that name, but the detector's
quantile is configurable, and publishing a p75 under a name that says p99 is the kind of
quiet lie that makes an on-call engineer distrust the whole dashboard. So the generic
``guard_entropy_quantile`` gauge always carries the true value, and ``guard_entropy_p99``
is fed only when the configured quantile is 0.99.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import prometheus_client as prom
from prometheus_client import CollectorRegistry

DEFAULT_NAMESPACE = "guard"

# Metrics every build registers, whether or not the matching detector is enabled, so a
# dashboard never breaks with "no data" on a panel that simply has not been fed yet.
_CORE_GAUGES: tuple[tuple[str, str], ...] = (
    ("entropy_mean", "Windowed mean Shannon entropy of softmax outputs"),
    ("entropy_quantile", "Windowed tail quantile of per-sample Shannon entropy"),
    ("entropy_p99", "Windowed 99th-percentile Shannon entropy (fed only when quantile==0.99)"),
    ("entropy_pop_shift", "Total-variation distance to the baseline entropy histogram"),
    ("kl_divergence", "KL(P||Q) between live and baseline class distributions"),
    (
        "js_divergence",
        "Jensen-Shannon divergence JS(P||Q) between live and baseline class distributions",
    ),
    (
        "embedding_mahalanobis_mean",
        "Windowed mean per-sample Mahalanobis distance to the reference Gaussian",
    ),
    ("embedding_mean_shift", "Mahalanobis distance of the running embedding mean to reference"),
    ("embedding_cov_drift", "Frobenius norm of (Sigma_run * Sigma_ref^-1 - I)"),
)

_TELEMETRY_GAUGES: tuple[tuple[str, str], ...] = (
    ("monitor_overhead_us", "CPU microseconds observe() adds to the inference thread"),
    ("monitor_gpu_time_us", "Monitor-stream GPU microseconds per step (requires opt-in timing)"),
    ("ring_utilization", "Fraction of on-GPU ring buffer slots with work still in flight"),
    ("detectors_disabled", "Number of detectors currently disabled by a tripped circuit breaker"),
)

_COUNTERS: tuple[tuple[str, str], ...] = (
    ("steps_observed", "Batches successfully enqueued onto the monitor stream"),
    ("observations_dropped", "Batches rejected by observe() (bad shape, oversized, graph capture)"),
    ("summaries_dropped", "Scalar summaries dropped due to export-thread backpressure"),
)


class PrometheusExporter:
    """Prometheus-backed :class:`~guard.export.base.MetricSink`.

    Args:
        namespace: metric name prefix; ``"guard"`` reproduces the names in the design doc.
        registry: registry to register into. Defaults to a fresh private registry, which
            is what keeps two exporters (or two test cases) from colliding.
        extra_metrics: additional metric names to create gauges for up front. The engine
            passes the metric names its detectors declare, so a custom detector's metrics
            are exported without any change here.
        emit_p99_alias: feed ``<ns>_entropy_p99`` from ``entropy_quantile``. The engine
            sets this only when the entropy detector's quantile is 0.99.
    """

    def __init__(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        registry: CollectorRegistry | None = None,
        extra_metrics: Iterable[str] = (),
        emit_p99_alias: bool = True,
    ) -> None:
        self.namespace = namespace
        self.registry = registry if registry is not None else CollectorRegistry(auto_describe=True)
        self.emit_p99_alias = emit_p99_alias

        self._gauges: dict[str, prom.Gauge] = {}
        self._counters: dict[str, prom.Counter] = {}

        for name, doc in _CORE_GAUGES + _TELEMETRY_GAUGES:
            self._make_gauge(name, doc)
        for name, doc in _COUNTERS:
            self._counters[name] = prom.Counter(f"{namespace}_{name}", doc, registry=self.registry)

        self.drift_alert = prom.Gauge(
            f"{namespace}_drift_alert",
            "1 while this detector's change-point test is in alarm, 0 otherwise",
            labelnames=["detector"],
            registry=self.registry,
        )
        self.detector_failures = prom.Counter(
            f"{namespace}_detector_failures",
            "Detector exceptions caught by the GUARD firewall",
            labelnames=["detector"],
            registry=self.registry,
        )

        self.ensure_metrics(extra_metrics)

    # ── registration ─────────────────────────────────────────────────────────

    def _make_gauge(self, name: str, doc: str) -> prom.Gauge:
        gauge = prom.Gauge(f"{self.namespace}_{name}", doc, registry=self.registry)
        self._gauges[name] = gauge
        return gauge

    def ensure_metrics(self, names: Iterable[str]) -> None:
        """Create a gauge for each name that does not have one yet. Idempotent.

        Called once at engine startup with the detectors' declared metric names, never per
        step — creating collectors is not a hot-path operation.
        """
        for name in names:
            if name not in self._gauges:
                self._make_gauge(name, f"GUARD detector metric {name!r}")

    def gauge(self, name: str) -> prom.Gauge:
        """The gauge for ``name``, creating it on first use."""
        if name not in self._gauges:
            self._make_gauge(name, f"GUARD detector metric {name!r}")
        return self._gauges[name]

    def counter(self, name: str) -> prom.Counter:
        """The counter registered under ``name``. Raises ``KeyError`` if unknown."""
        return self._counters[name]

    # ── MetricSink ───────────────────────────────────────────────────────────

    def update(self, values: Mapping[str, float]) -> None:
        """Write metric values.

        Recognised key forms:

        * ``"<metric>"`` — sets the matching gauge; unknown names create a gauge on first
          use so a custom detector needs no registration step.
        * ``"drift_alert.<detector>"`` — sets the labelled alert gauge.
        * ``"summaries_dropped"`` — **increments** the counter by the value (a delta, not a
          cumulative total; passing a running total would double-count).

        Non-finite values are skipped **before** any routing decision: a NaN in the summary
        buffer means "this detector did not report this step", and publishing it would
        poison Grafana's rate/avg panels. On a counter it is worse than cosmetic — a counter
        incremented by ``inf`` is permanently ruined, since a counter can never go down.
        """
        for key, value in values.items():
            if not math.isfinite(value):
                continue
            if key.startswith("drift_alert."):
                self.drift_alert.labels(detector=key[len("drift_alert.") :]).set(value)
                continue
            if key in self._counters:
                self._counters[key].inc(max(0.0, float(value)))
                continue
            if key == "entropy_quantile" and self.emit_p99_alias:
                self._gauges["entropy_p99"].set(value)
            self.gauge(key).set(value)

    def set_alert(self, detector: str, active: bool) -> None:
        """Set the labelled ``drift_alert`` gauge for one detector."""
        self.drift_alert.labels(detector=detector).set(1.0 if active else 0.0)

    def add_dropped(self, count: int) -> None:
        """Increment the dropped-summaries counter by ``count``."""
        if count > 0:
            self._counters["summaries_dropped"].inc(count)

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        """Write GUARD's self-telemetry: gauges are set, counters are incremented."""
        for key, value in values.items():
            if not math.isfinite(value):
                continue
            if key in self._counters:
                self._counters[key].inc(max(0.0, float(value)))
            elif key in self._gauges:
                self._gauges[key].set(value)

    def record_detector_failure(self, detector: str, count: int = 1) -> None:
        """Increment the per-detector firewall failure counter."""
        if count > 0:
            self.detector_failures.labels(detector=detector).inc(count)

    # ── serving ──────────────────────────────────────────────────────────────

    def start_http_server(self, port: int, addr: str = "0.0.0.0") -> None:
        """Expose ``/metrics`` on ``port`` in a daemon thread.

        Binds all interfaces by default because the intended deployment is a container
        scraped by Prometheus over the pod network; pass ``addr="127.0.0.1"`` to restrict
        it to localhost.
        """
        prom.start_http_server(port, addr=addr, registry=self.registry)

    def generate_latest(self) -> bytes:
        """Render the registry in Prometheus text exposition format."""
        return prom.generate_latest(self.registry)


# ─── module-level default instance (stable Phase-1 surface) ───────────────────

_DEFAULT = PrometheusExporter()

REGISTRY: CollectorRegistry = _DEFAULT.registry

entropy_mean = _DEFAULT.gauge("entropy_mean")
entropy_p99 = _DEFAULT.gauge("entropy_p99")
entropy_quantile = _DEFAULT.gauge("entropy_quantile")
js_divergence = _DEFAULT.gauge("js_divergence")
kl_divergence = _DEFAULT.gauge("kl_divergence")
embedding_mahalanobis_mean = _DEFAULT.gauge("embedding_mahalanobis_mean")
drift_alert = _DEFAULT.drift_alert
monitor_overhead_us = _DEFAULT.gauge("monitor_overhead_us")
ring_utilization = _DEFAULT.gauge("ring_utilization")
summaries_dropped_total = _DEFAULT.counter("summaries_dropped")


def update(summary: Mapping[str, float]) -> None:
    """Write scalar values into the default exporter. See :meth:`PrometheusExporter.update`."""
    _DEFAULT.update(summary)


def default_exporter() -> PrometheusExporter:
    """The process-wide default exporter backing :func:`update` and :data:`REGISTRY`."""
    return _DEFAULT
