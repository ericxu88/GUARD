"""Alert model and dispatchers.

A change-point firing is the only output of GUARD a human is ever paged for, so an alert
carries everything the responder needs to triage without opening a notebook: which
detector and metric, the value that fired, the test statistic and threshold that decided
it, and the model + baseline versions the verdict is relative to. That last pair matters
more than it looks — "drift" against a stale baseline is a baseline problem, not a model
problem, and the alert should make that distinguishable at a glance.

Two dispatchers ship:

* :class:`LoggingAlertSink` — writes a structured line through ``logging``. Zero
  dependencies, always safe, the sensible default.
* :class:`WebhookAlertSink` — POSTs JSON to a URL (Alertmanager, Slack relay, PagerDuty
  events API) from its **own** worker thread behind a bounded queue. The drain thread never
  waits on network I/O, and a wedged endpoint drops alerts rather than backing up the
  metric pipeline.

Both are best-effort by construction: a dispatcher that raises is contained by the
firewall in the drain thread, and neither can propagate anything toward inference.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("guard.alerts")


@dataclass(frozen=True)
class Alert:
    """One change-point firing.

    Attributes:
        metric: metric stream that fired (e.g. ``"js_divergence"``).
        detector: detector that produced the metric (e.g. ``"divergence"``).
        step: monitor step index the firing was attributed to.
        value: the windowed metric value at the firing.
        statistic: the change-point test statistic at the firing.
        threshold: the boundary the statistic crossed.
        method: change-point method name (``"page_hinkley"`` / ``"cusum"``).
        model_version: model the baseline is bound to, when known.
        baseline_version: baseline checksum/identifier, when known.
        labels: free-form extra context (gpu index, pod name, …).
    """

    metric: str
    detector: str
    step: int
    value: float
    statistic: float
    threshold: float
    method: str
    model_version: str = ""
    baseline_version: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation, suitable for a webhook body."""
        return asdict(self)

    def summary(self) -> str:
        """One-line human-readable rendering for logs."""
        return (
            f"GUARD drift alert: detector={self.detector} metric={self.metric} "
            f"value={self.value:.6g} statistic={self.statistic:.6g} "
            f"threshold={self.threshold:.6g} method={self.method} step={self.step} "
            f"model={self.model_version or '?'} baseline={self.baseline_version or '?'}"
        )


@runtime_checkable
class AlertSink(Protocol):
    """Destination for fired alerts."""

    def dispatch(self, alert: Alert) -> None:
        """Deliver one alert. Must not block for long and must not raise on transient errors."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


class LoggingAlertSink:
    """Writes each alert through ``logging`` at WARNING. The default sink."""

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log if log is not None else logger

    def dispatch(self, alert: Alert) -> None:
        """Log the alert."""
        self._log.warning("%s", alert.summary())

    def close(self) -> None:
        """Nothing to release."""


class WebhookAlertSink:
    """POSTs alerts as JSON from a background worker behind a bounded queue.

    Args:
        url: endpoint to POST to.
        timeout: per-request timeout in seconds. Keep it short; alerting is not the place
            to wait on a slow endpoint.
        max_queue: alerts buffered before new ones are dropped (and counted in
            :attr:`dropped`).
        headers: extra HTTP headers, e.g. an authorization token.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 2.0,
        max_queue: int = 64,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.dropped = 0
        self.delivered = 0
        self._counter_lock = threading.Lock()
        self._closed = False
        self._queue: queue.Queue[Alert | None] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="guard-alert-webhook", daemon=True)
        self._thread.start()

    def dispatch(self, alert: Alert) -> None:
        """Queue the alert for delivery; count it as dropped if it cannot be delivered.

        An alert queued after ``close()``, or after the worker has died, would sit in the
        queue forever and never be counted — the one thing an alert sink must not do is
        lose a firing silently. Both cases increment :attr:`dropped`, which is the honest
        record of what was not delivered.
        """
        if self._closed or not self._thread.is_alive():
            self._drop("sink is closed or its worker is not running")
            return
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            self._drop("queue is full")

    def _drop(self, reason: str) -> None:
        """Count an undelivered alert, logging the first one only."""
        with self._counter_lock:
            self.dropped += 1
            first = self.dropped == 1
        if first:
            logger.warning(
                "GUARD alert dropped (%s); further drops are counted in "
                "WebhookAlertSink.dropped but not logged.",
                reason,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                alert = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if alert is None:
                break
            self._post(alert)

    def _post(self, alert: Alert) -> None:
        # Everything is inside the try, including building the Request: a malformed URL
        # (no scheme, say) raises from the constructor, and an unguarded raise here would
        # kill the worker thread and turn every later alert into a silent no-op.
        try:
            body = json.dumps(alert.to_dict()).encode("utf-8")
            request = urllib.request.Request(
                self.url, data=body, headers=self.headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=self.timeout), self._counter_lock:
                self.delivered += 1
        except Exception as exc:
            logger.warning("GUARD alert webhook delivery failed (%s): %s", self.url, exc)
            self._drop(f"delivery failed: {exc}")

    def close(self) -> None:
        """Stop the worker thread, waiting briefly for an in-flight request."""
        self._closed = True
        self._stop.set()
        # A full queue means the worker is busy and will exit on the stop flag anyway.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self._thread.join(timeout=self.timeout + 1.0)


class MultiAlertSink:
    """Fans one alert out to several sinks; a failing sink never stops the others."""

    def __init__(self, *sinks: AlertSink) -> None:
        self.sinks = list(sinks)

    def dispatch(self, alert: Alert) -> None:
        """Deliver to every configured sink."""
        for sink in self.sinks:
            try:
                sink.dispatch(alert)
            except Exception:
                logger.exception("GUARD alert sink %r failed", type(sink).__name__)

    def close(self) -> None:
        """Close every configured sink."""
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                logger.exception("GUARD alert sink %r failed to close", type(sink).__name__)
