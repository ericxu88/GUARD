"""Fan-out sink: export the same metrics to several backends at once."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from guard.export.base import MetricSink

logger = logging.getLogger("guard.export")


class MultiSink:
    """Forwards every call to each wrapped sink, isolating failures.

    One backend being down (an OTel collector restart, a Pushgateway timeout) must not stop
    the others from receiving metrics — and must not raise into the drain thread. Each
    forwarded call is individually contained; a sink that keeps failing is logged, not
    disabled, because :class:`~guard.safety.Firewall` already owns the disable decision one
    level up.
    """

    def __init__(self, *sinks: MetricSink) -> None:
        self.sinks = list(sinks)

    def _fan(self, method: str, *args: object) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, method)(*args)
            except Exception:
                logger.exception("GUARD sink %r failed in %s()", type(sink).__name__, method)

    def update(self, values: Mapping[str, float]) -> None:
        """Forward metric values to every sink."""
        self._fan("update", values)

    def set_alert(self, detector: str, active: bool) -> None:
        """Forward an alert state change to every sink."""
        self._fan("set_alert", detector, active)

    def add_dropped(self, count: int) -> None:
        """Forward a drop count to every sink."""
        self._fan("add_dropped", count)

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        """Forward self-telemetry to every sink."""
        self._fan("set_telemetry", values)

    def record_detector_failure(self, detector: str, count: int = 1) -> None:
        """Forward a detector failure count to every sink."""
        self._fan("record_detector_failure", detector, count)
