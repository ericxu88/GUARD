"""Windowing, change-point testing, and export for drained scalars (Phase 2).

This is the CPU half of the split the design doc describes: heavy per-sample math on the
GPU, tiny per-window temporal logic on the host. By the time a value reaches this module it
is already a single float — one number per detector metric per inference step — so
everything here costs nothing and runs on the drain thread, never on the serving path.

Two windows, deliberately different:

* a **rolling** window (a ``deque`` of the last ``window_size`` values) whose mean is what
  gets exported. Dashboards want a smooth line that updates every step.
* a **tumbling** window (a running sum reset every ``window_size`` values) whose mean feeds
  the change-point test. Change-point tests assume independent observations; feeding them
  overlapping rolling means would make consecutive inputs correlated and inflate the
  false-positive rate — the exact failure the temporal layer exists to prevent.

Alert latching. Page-Hinkley and CUSUM fire *edges*: one True on the step the change is
declared, then silence. A gauge that is 1 for a single scrape interval is invisible in
Grafana and unusable in an Alertmanager ``for:`` clause. So a firing latches
``guard_drift_alert{detector}`` at 1 for ``alert_hold`` windows, and it decays to 0 only
after the stream has been quiet that long — which is what makes an alert *clear* on
recovery rather than needing a manual reset.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Mapping, Sequence

from guard.config import ThresholdConfig
from guard.export.alerts import Alert, AlertSink
from guard.export.base import MetricSink, NullSink
from guard.temporal import from_threshold_config
from guard.temporal.base import ChangePointTest

logger = logging.getLogger("guard.engine")

DEFAULT_ALERT_HOLD = 10


class MetricAggregator:
    """Turns a stream of per-step metric dicts into exported gauges and alerts.

    Args:
        metric_names: every metric this aggregator may receive, in a stable order.
        metric_detectors: metric name → owning detector name, used to label
            ``guard_drift_alert`` and to attribute alerts.
        window_size: number of steps per window (>= 1). ``1`` disables smoothing and tests
            every step.
        thresholds: metric name → change-point configuration. Metrics without an entry are
            exported but never alert — which is the right default for diagnostic metrics
            like ``entropy_pop_shift`` that you want on a dashboard but not on a pager.
        sink: where metrics go. Defaults to :class:`~guard.export.base.NullSink`.
        alert_sink: where firings go. ``None`` disables alert dispatch (the latched gauge
            still moves).
        alert_hold: windows a firing keeps the alert gauge latched at 1.
        model_version / baseline_version: stamped onto every alert for triage.
    """

    def __init__(
        self,
        metric_names: Sequence[str],
        metric_detectors: Mapping[str, str],
        *,
        window_size: int = 1,
        thresholds: Mapping[str, ThresholdConfig] | None = None,
        sink: MetricSink | None = None,
        alert_sink: AlertSink | None = None,
        alert_hold: int = DEFAULT_ALERT_HOLD,
        model_version: str = "",
        baseline_version: str = "",
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if alert_hold < 0:
            raise ValueError(f"alert_hold must be >= 0, got {alert_hold}")

        self.metric_names = tuple(metric_names)
        self.metric_detectors = dict(metric_detectors)
        self.window_size = window_size
        self.alert_hold = alert_hold
        self.model_version = model_version
        self.baseline_version = baseline_version
        self.sink: MetricSink = sink if sink is not None else NullSink()
        self.alert_sink = alert_sink

        thresholds = thresholds or {}
        self._configs: dict[str, ThresholdConfig] = {
            name: cfg for name, cfg in thresholds.items() if name in self.metric_names
        }
        # `thresholds` is keyed by METRIC name, and the config schema cannot validate the
        # keys because it does not know which detectors will be built. A typo therefore
        # produces a threshold that is silently never applied — an alert an operator
        # believes is armed and is not. Say so, loudly, at startup.
        self.unconfigured_thresholds = tuple(
            sorted(name for name in thresholds if name not in self.metric_names)
        )
        if self.unconfigured_thresholds:
            logger.warning(
                "GUARD: %d threshold(s) match no metric and will NEVER fire: %s. "
                "Valid metric names for the configured detectors: %s. "
                "(thresholds are keyed by metric name, not detector name.)",
                len(self.unconfigured_thresholds),
                list(self.unconfigured_thresholds),
                list(self.metric_names),
            )
        self._tests: dict[str, ChangePointTest] = {
            name: from_threshold_config(cfg) for name, cfg in self._configs.items()
        }

        self._rolling: dict[str, deque[float]] = {
            name: deque(maxlen=window_size) for name in self.metric_names
        }
        self._acc_sum: dict[str, float] = dict.fromkeys(self.metric_names, 0.0)
        self._acc_n: dict[str, int] = dict.fromkeys(self.metric_names, 0)
        self._hold: dict[str, int] = dict.fromkeys(self.metric_names, 0)
        self._steps_in_window = 0

        self.detectors = tuple(dict.fromkeys(self.metric_detectors.values()))
        self._alert_state: dict[str, bool] = dict.fromkeys(self.detectors, False)

        self.steps_seen = 0
        self.windows_closed = 0
        self.alerts_fired = 0
        self.values_skipped = 0

    # ── ingestion ────────────────────────────────────────────────────────────

    def push(self, step: int, values: Mapping[str, float]) -> list[Alert]:
        """Fold one step's metrics in; return any alerts that fired on this step.

        Non-finite values are skipped and counted: a NaN in the summary buffer means the
        detector did not report that step (it was disabled, or its slice was never
        written), and letting it into a running mean would silently poison every subsequent
        window.
        """
        self.steps_seen += 1
        alerts: list[Alert] = []
        exported: dict[str, float] = {}

        # Age every latch on elapsed steps, not on closed windows. Decaying only inside
        # _test_window ties the latch to the metric still reporting — so a detector that
        # fires an alarm and then trips its circuit breaker (or simply stops producing
        # finite values) leaves guard_drift_alert pinned at 1 until the process restarts.
        # An alert that cannot clear is one an operator learns to ignore.
        self._steps_in_window += 1
        if self._steps_in_window >= self.window_size:
            self._steps_in_window = 0
            self._decay_latches()

        for name in self.metric_names:
            value = values.get(name)
            if value is None:
                continue
            if not math.isfinite(value):
                self.values_skipped += 1
                continue

            window = self._rolling[name]
            window.append(value)
            exported[name] = sum(window) / len(window)

            self._acc_sum[name] += value
            self._acc_n[name] += 1
            if self._acc_n[name] < self.window_size:
                continue

            window_mean = self._acc_sum[name] / self._acc_n[name]
            self._acc_sum[name] = 0.0
            self._acc_n[name] = 0
            alert = self._test_window(step, name, window_mean)
            if alert is not None:
                alerts.append(alert)

        self._publish(exported, alerts)
        return alerts

    def _test_window(self, step: int, name: str, window_mean: float) -> Alert | None:
        """Feed one closed window to the metric's change-point test and latch the gauge."""
        self.windows_closed += 1
        test = self._tests.get(name)
        if test is None:
            return None

        result = test.update(window_mean)
        if not result.alarm:
            return None

        self._hold[name] = self.alert_hold
        self.alerts_fired += 1
        cfg = self._configs[name]
        return Alert(
            metric=name,
            detector=self.metric_detectors.get(name, name),
            step=step,
            value=window_mean,
            statistic=result.statistic,
            threshold=cfg.threshold,
            method=cfg.method,
            model_version=self.model_version,
            baseline_version=self.baseline_version,
        )

    def _decay_latches(self) -> None:
        """Age every alert latch by one window, independent of whether its metric reported."""
        for name, held in self._hold.items():
            if held > 0:
                self._hold[name] = held - 1

    def _publish(self, exported: Mapping[str, float], alerts: Sequence[Alert]) -> None:
        """Push exported values, alert-state changes, and alert payloads outward."""
        if exported:
            self.sink.update(exported)

        for detector in self.detectors:
            active = any(
                self._hold[name] > 0
                for name, det in self.metric_detectors.items()
                if det == detector and name in self._hold
            )
            if active != self._alert_state[detector]:
                self._alert_state[detector] = active
                self.sink.set_alert(detector, active)

        if self.alert_sink is not None:
            for alert in alerts:
                self.alert_sink.dispatch(alert)

    # ── introspection ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, float]:
        """Current rolling-window mean of every metric that has seen data."""
        return {
            name: sum(window) / len(window)
            for name, window in self._rolling.items()
            if len(window) > 0
        }

    def alert_states(self) -> dict[str, bool]:
        """Per-detector latched alert state."""
        return dict(self._alert_state)

    def reset(self) -> None:
        """Clear all windows, test state, and latches. For re-baselining and tests."""
        for name in self.metric_names:
            self._rolling[name].clear()
            self._acc_sum[name] = 0.0
            self._acc_n[name] = 0
            self._hold[name] = 0
        self._steps_in_window = 0
        for test in self._tests.values():
            test.reset()
        for detector in self.detectors:
            if self._alert_state[detector]:
                self._alert_state[detector] = False
                self.sink.set_alert(detector, False)
        self.steps_seen = 0
        self.windows_closed = 0
        self.alerts_fired = 0
        self.values_skipped = 0
