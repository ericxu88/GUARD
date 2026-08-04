"""The sink contract every exporter implements.

The aggregation thread does not know or care whether metrics end up in Prometheus, an
OpenTelemetry meter, a log line, or a test double — it only calls the four methods below.
Keeping the surface this small is what lets :class:`guard.engine.GuardMonitor` be tested
without a metrics backend, and what lets an operator add one without touching the engine.

Every method must be **total and non-raising in spirit**: a sink is called from the drain
thread, and although that thread runs behind the firewall, an exporter that throws on every
call trips its breaker and silently ends metric export. Prefer clamping or dropping a bad
value over raising.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricSink(Protocol):
    """Destination for drained scalar metrics, alert states, and self-telemetry."""

    def update(self, values: Mapping[str, float]) -> None:
        """Record the current value of each named metric (un-prefixed names)."""
        ...

    def set_alert(self, detector: str, active: bool) -> None:
        """Record whether ``detector``'s change-point test is currently in alarm."""
        ...

    def add_dropped(self, count: int) -> None:
        """Add to the count of scalar summaries dropped under export backpressure."""
        ...

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        """Record GUARD's own health metrics (overhead, ring utilization, …)."""
        ...

    def record_detector_failure(self, detector: str, count: int = 1) -> None:
        """Add to the count of exceptions the firewall caught for ``detector``."""
        ...


class NullSink:
    """A sink that discards everything. The default when no exporter is configured.

    Useful in tests and in deployments that read metrics off
    :meth:`guard.engine.GuardMonitor.stats` directly instead of scraping.
    """

    def update(self, values: Mapping[str, float]) -> None:
        """Discard the values."""

    def set_alert(self, detector: str, active: bool) -> None:
        """Discard the alert state."""

    def add_dropped(self, count: int) -> None:
        """Discard the drop count."""

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        """Discard the telemetry."""

    def record_detector_failure(self, detector: str, count: int = 1) -> None:
        """Discard the failure count."""
