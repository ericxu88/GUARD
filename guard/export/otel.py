"""OpenTelemetry metric bridge (optional).

Prometheus is the default because it is what most GPU serving stacks already scrape. Shops
standardised on OTel collectors want the same numbers on their own pipeline, so this module
mirrors every drained metric onto an OTel meter.

OpenTelemetry is an **optional** dependency (``pip install "guard-monitor[otel]"``). The
import is deferred to construction so that neither ``import guard`` nor the Prometheus path
pays for it, and a missing package produces one clear error at the point of configuration
rather than an obscure failure inside the drain thread.

Use :class:`~guard.export.multi.MultiSink` to export to Prometheus and OTel simultaneously.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_NAMESPACE = "guard"


class OTelSink:
    """Mirrors GUARD metrics onto an OpenTelemetry meter.

    Args:
        meter: an ``opentelemetry.metrics.Meter``. Defaults to one obtained from the global
            meter provider, so whatever the host application already configured (OTLP
            exporter, collector endpoint, resource attributes) applies unchanged.
        namespace: instrument name prefix, matching the Prometheus namespace.

    Raises:
        ImportError: if the ``opentelemetry-api`` package is not installed.
    """

    def __init__(self, meter: Any = None, *, namespace: str = DEFAULT_NAMESPACE) -> None:
        try:
            from opentelemetry import metrics as otel_metrics
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "OTelSink requires OpenTelemetry. Install it with: "
                'pip install "guard-monitor[otel]"'
            ) from exc

        self.namespace = namespace
        self._meter = meter if meter is not None else otel_metrics.get_meter("guard")
        self._gauges: dict[str, Any] = {}
        self._counters: dict[str, Any] = {}

    # ── instrument management ────────────────────────────────────────────────

    def _gauge(self, name: str) -> Any:
        """Get or create a synchronous gauge instrument for ``name``."""
        if name not in self._gauges:
            self._gauges[name] = self._meter.create_gauge(f"{self.namespace}.{name}")
        return self._gauges[name]

    def _counter(self, name: str) -> Any:
        """Get or create a counter instrument for ``name``."""
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(f"{self.namespace}.{name}")
        return self._counters[name]

    # ── MetricSink ───────────────────────────────────────────────────────────

    def update(self, values: Mapping[str, float]) -> None:
        """Record metric values, skipping non-finite entries."""
        for key, value in values.items():
            if value != value or value in (float("inf"), float("-inf")):
                continue
            if key.startswith("drift_alert."):
                detector = key[len("drift_alert.") :]
                self._gauge("drift_alert").set(value, {"detector": detector})
                continue
            self._gauge(key).set(value)

    def set_alert(self, detector: str, active: bool) -> None:
        """Record a detector's alert state as a labelled gauge."""
        self._gauge("drift_alert").set(1.0 if active else 0.0, {"detector": detector})

    def add_dropped(self, count: int) -> None:
        """Add to the dropped-summaries counter."""
        if count > 0:
            self._counter("summaries_dropped").add(count)

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        """Record GUARD self-telemetry values."""
        self.update(values)

    def record_detector_failure(self, detector: str, count: int = 1) -> None:
        """Add to the per-detector firewall failure counter."""
        if count > 0:
            self._counter("detector_failures").add(count, {"detector": detector})
