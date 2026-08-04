"""Phase 2: MetricAggregator — rolling vs tumbling windows, latching, and reset semantics.

The aggregator is the seam between the GPU side (one float per detector metric per step)
and the outside world (gauges + alerts). Everything asserted here is cadence and
bookkeeping: *which* number reaches the exporter on *which* push, and how many times a
sink method is called. Those counts are the whole contract — a rolling mean fed to a
change-point test inflates false positives, and a gauge re-written every window makes the
"alert cleared" edge unusable in Alertmanager.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from guard.config import ThresholdConfig
from guard.engine.aggregator import MetricAggregator
from guard.export.alerts import Alert, AlertSink
from guard.export.base import MetricSink
from guard.temporal.base import ChangePointResult, ChangePointTest


# --------------------------------------------------------------------------------------
# Recording doubles
# --------------------------------------------------------------------------------------
class RecordingSink:
    """A MetricSink that remembers every call, in order.

    Ordering matters as much as content: the alert gauge is an edge-triggered signal, so
    "was set_alert called at all on this push" is the assertion with teeth, not the final
    value of the gauge.
    """

    def __init__(self) -> None:
        self.updates: list[dict[str, float]] = []
        self.alerts: list[tuple[str, bool]] = []
        self.dropped: list[int] = []
        self.telemetry: list[dict[str, float]] = []
        self.detector_failures: list[tuple[str, int]] = []

    def update(self, values: Mapping[str, float]) -> None:
        # Copy: the aggregator owns the dict it passes and may reuse it.
        self.updates.append(dict(values))

    def set_alert(self, detector: str, active: bool) -> None:
        self.alerts.append((detector, active))

    def add_dropped(self, count: int) -> None:
        self.dropped.append(count)

    def record_detector_failure(self, detector: str, count: int = 1) -> None:
        self.detector_failures.append((detector, count))

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        self.telemetry.append(dict(values))

    def values_for(self, metric: str) -> list[float]:
        """Every exported value of one metric, in push order."""
        return [u[metric] for u in self.updates if metric in u]


class RecordingTest:
    """A ChangePointTest that records what it is fed and alarms on chosen window indices.

    Substituted for the real Page-Hinkley/CUSUM instance so window cadence and latching can
    be asserted exactly, without depending on any test's statistic arithmetic.
    """

    def __init__(self, alarm_at: frozenset[int] = frozenset()) -> None:
        self.received: list[float] = []
        self.resets = 0
        self._alarm_at = alarm_at

    def update(self, x: float) -> ChangePointResult:
        self.received.append(x)
        # alarm_at is 1-indexed by window number, counting from the last reset.
        alarm = len(self.received) in self._alarm_at
        return ChangePointResult(alarm=alarm, statistic=float(len(self.received)))

    def reset(self) -> None:
        self.resets += 1
        self.received.clear()


class RecordingAlertSink:
    """An AlertSink that keeps every dispatched alert."""

    def __init__(self) -> None:
        self.dispatched: list[Alert] = []
        self.closed = 0

    def dispatch(self, alert: Alert) -> None:
        self.dispatched.append(alert)

    def close(self) -> None:
        self.closed += 1


CFG = ThresholdConfig(method="page_hinkley", threshold=0.5, delta=0.0)


def _install_test(agg: MetricAggregator, metric: str, fake: RecordingTest) -> None:
    """Swap in a recording change-point test for ``metric``.

    Reaches into a private dict deliberately: the aggregator builds its tests from config in
    ``__init__``, and there is no injection point. The alternative — driving a real
    Page-Hinkley into alarm — would couple every latching assertion to that test's math.
    """
    agg._tests[metric] = fake


def test_doubles_satisfy_the_protocols() -> None:
    """If the doubles drift from the real interfaces the rest of this file proves nothing."""
    assert isinstance(RecordingSink(), MetricSink)
    assert isinstance(RecordingTest(), ChangePointTest)
    assert isinstance(RecordingAlertSink(), AlertSink)


# --------------------------------------------------------------------------------------
# Rolling (exported) vs tumbling (tested) windows
# --------------------------------------------------------------------------------------
def test_rolling_mean_is_exported_on_every_push() -> None:
    sink = RecordingSink()
    agg = MetricAggregator(["m"], {"m": "d"}, window_size=4, sink=sink)

    for step, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], start=1):
        agg.push(step, {"m": value})

    # One export per push, each the mean of the last <= 4 values — a dashboard line that
    # moves every step, not a staircase that only moves every window.
    assert sink.values_for("m") == [1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 6.5]
    assert agg.steps_seen == 8


def test_change_point_test_sees_one_tumbling_mean_per_window() -> None:
    sink = RecordingSink()
    agg = MetricAggregator(["m"], {"m": "d"}, window_size=4, thresholds={"m": CFG}, sink=sink)
    fake = RecordingTest()
    _install_test(agg, "m", fake)

    for step, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], start=1):
        agg.push(step, {"m": value})

    # Two non-overlapping windows: mean(1..4) and mean(5..8). Consecutive rolling means
    # would share three of four samples, which is exactly the correlation that inflates a
    # change-point test's false-positive rate.
    assert fake.received == [2.5, 6.5]
    assert agg.windows_closed == 2

    # The intermediate rolling values were exported but must never have been tested.
    for rolling_only in (1.0, 1.5, 2.0, 3.5, 4.5, 5.5):
        assert rolling_only not in fake.received
    assert len(sink.values_for("m")) == 8


def test_window_size_one_collapses_rolling_and_tumbling() -> None:
    """window_size=1 is documented as "no smoothing"; both streams must be the raw values."""
    sink = RecordingSink()
    agg = MetricAggregator(["m"], {"m": "d"}, window_size=1, thresholds={"m": CFG}, sink=sink)
    fake = RecordingTest()
    _install_test(agg, "m", fake)

    values = [3.0, 1.0, 4.0, 1.0, 5.0]
    for step, value in enumerate(values, start=1):
        agg.push(step, {"m": value})

    assert sink.values_for("m") == values
    assert fake.received == values


def test_snapshot_reports_the_rolling_mean() -> None:
    agg = MetricAggregator(["m", "other"], {"m": "d", "other": "d"}, window_size=4)
    for step, value in enumerate([2.0, 4.0, 6.0], start=1):
        agg.push(step, {"m": value})

    # "other" never received data, so it must be absent rather than reported as 0.0 —
    # a zero would look like a healthy metric on a dashboard.
    assert agg.snapshot() == {"m": 4.0}


# --------------------------------------------------------------------------------------
# Non-finite handling
# --------------------------------------------------------------------------------------
def test_non_finite_values_are_skipped_counted_and_never_averaged() -> None:
    sink = RecordingSink()
    agg = MetricAggregator(["m"], {"m": "d"}, window_size=3, thresholds={"m": CFG}, sink=sink)
    fake = RecordingTest()
    _install_test(agg, "m", fake)

    # A NaN means "the detector did not report this step" (disabled, or its summary slice
    # was never written), not "the metric was zero".
    stream = [1.0, float("nan"), 2.0, 3.0, float("inf"), float("-inf"), 4.0]
    for step, value in enumerate(stream, start=1):
        agg.push(step, {"m": value})

    assert agg.values_skipped == 3
    assert agg.steps_seen == 7

    # Skipped pushes export nothing at all: 7 pushes, 3 skipped, 4 updates.
    assert sink.values_for("m") == [1.0, 1.5, 2.0, 3.0]
    assert len(sink.updates) == 4

    # The single closed window is mean(1, 2, 3) — the skips neither poisoned it nor
    # advanced it, so the window closed on the third *finite* value.
    assert fake.received == [2.0]
    assert agg.windows_closed == 1


def test_absent_metric_is_not_counted_as_skipped() -> None:
    """A metric missing from the dict is "not reported this step" and is not an error."""
    sink = RecordingSink()
    agg = MetricAggregator(["a", "b"], {"a": "d", "b": "d"}, window_size=2, sink=sink)

    agg.push(1, {"a": 1.0})
    agg.push(2, {"a": 3.0, "b": 10.0, "unknown": 99.0})

    assert agg.values_skipped == 0
    assert sink.updates == [{"a": 1.0}, {"a": 2.0, "b": 10.0}]
    # An unmodelled key in the summary dict must not leak into the exporter.
    assert all("unknown" not in u for u in sink.updates)


# --------------------------------------------------------------------------------------
# Metrics without a threshold config
# --------------------------------------------------------------------------------------
def test_metric_without_threshold_is_exported_but_never_alerts() -> None:
    sink = RecordingSink()
    alert_sink = RecordingAlertSink()
    agg = MetricAggregator(
        ["js_divergence", "entropy_pop_shift"],
        {"js_divergence": "divergence", "entropy_pop_shift": "entropy"},
        window_size=1,
        thresholds={"js_divergence": CFG},
        sink=sink,
        alert_sink=alert_sink,
    )
    fake = RecordingTest(alarm_at=frozenset({2}))
    _install_test(agg, "js_divergence", fake)

    # entropy_pop_shift is driven far harder than the configured metric; if the aggregator
    # silently defaulted a test onto it, it would be the first thing to fire.
    for step in range(1, 5):
        agg.push(step, {"js_divergence": 0.1, "entropy_pop_shift": 100.0 * step})

    assert sink.values_for("entropy_pop_shift") == [100.0, 200.0, 300.0, 400.0]
    assert agg.alerts_fired == 1
    assert [a.metric for a in alert_sink.dispatched] == ["js_divergence"]
    # The unconfigured metric's detector must never touch the alert gauge.
    assert all(detector == "divergence" for detector, _ in sink.alerts)
    assert agg.alert_states()["entropy"] is False


# --------------------------------------------------------------------------------------
# Alert payload and dispatch
# --------------------------------------------------------------------------------------
def test_alert_carries_full_triage_context() -> None:
    """Built on a real Page-Hinkley so from_threshold_config wiring is exercised too."""
    sink = RecordingSink()
    alert_sink = RecordingAlertSink()
    cfg = ThresholdConfig(method="page_hinkley", threshold=0.5, delta=0.0)
    agg = MetricAggregator(
        ["js_divergence"],
        {"js_divergence": "divergence"},
        window_size=1,
        thresholds={"js_divergence": cfg},
        sink=sink,
        alert_sink=alert_sink,
        model_version="vit_b_16@f00d",
        baseline_version="baseline-2026-01-01",
    )

    fired: list[Alert] = []
    for step, value in enumerate([0.0, 0.0, 0.0, 0.0, 10.0], start=1):
        fired += agg.push(step, {"js_divergence": value})

    assert len(fired) == 1
    alert = fired[0]
    assert alert.metric == "js_divergence"
    assert alert.detector == "divergence"
    assert alert.step == 5
    assert alert.value == pytest.approx(10.0)
    # PH statistic on the jump: cum = x - running_mean = 10 - 2.
    assert alert.statistic == pytest.approx(8.0)
    assert alert.threshold == pytest.approx(0.5)
    assert alert.method == "page_hinkley"
    # Without these two, "drift" against a stale baseline is indistinguishable from a
    # genuine model regression.
    assert alert.model_version == "vit_b_16@f00d"
    assert alert.baseline_version == "baseline-2026-01-01"

    # push() returns exactly what it dispatched, once each.
    assert alert_sink.dispatched == [alert]


def test_each_alert_is_dispatched_exactly_once() -> None:
    alert_sink = RecordingAlertSink()
    agg = MetricAggregator(
        ["m"],
        {"m": "d"},
        window_size=1,
        thresholds={"m": CFG},
        alert_sink=alert_sink,
        alert_hold=0,
    )
    fake = RecordingTest(alarm_at=frozenset({2, 5}))
    _install_test(agg, "m", fake)

    for step in range(1, 8):
        agg.push(step, {"m": 1.0})

    assert agg.alerts_fired == 2
    assert len(alert_sink.dispatched) == 2
    assert [a.step for a in alert_sink.dispatched] == [2, 5]


def test_no_alert_sink_still_moves_the_latched_gauge() -> None:
    """alert_sink=None disables dispatch only; the gauge is the aggregator's own job."""
    sink = RecordingSink()
    agg = MetricAggregator(
        ["m"], {"m": "d"}, window_size=1, thresholds={"m": CFG}, sink=sink, alert_hold=1
    )
    _install_test(agg, "m", RecordingTest(alarm_at=frozenset({1})))

    agg.push(1, {"m": 1.0})
    assert sink.alerts == [("d", True)]


# --------------------------------------------------------------------------------------
# Latching
# --------------------------------------------------------------------------------------
def test_alert_latches_for_alert_hold_windows_then_clears_once() -> None:
    sink = RecordingSink()
    agg = MetricAggregator(
        ["m"],
        {"m": "d"},
        window_size=2,
        thresholds={"m": CFG},
        sink=sink,
        alert_hold=3,
    )
    # Fire on the third closed window, i.e. on push 6 with window_size=2.
    _install_test(agg, "m", RecordingTest(alarm_at=frozenset({3})))

    for step in range(1, 6):
        agg.push(step, {"m": 1.0})
    assert sink.alerts == []  # nothing has fired yet

    agg.push(6, {"m": 1.0})
    assert sink.alerts == [("d", True)]
    assert agg.alert_states() == {"d": True}

    # Pushes 7..11 span only two quiet *windows* (8 and 10). A hold decremented per push
    # instead of per window would have expired by now.
    for step in range(7, 12):
        agg.push(step, {"m": 1.0})
    assert sink.alerts == [("d", True)], "gauge re-set or cleared early"
    assert agg.alert_states() == {"d": True}

    # Push 12 closes the third quiet window and the latch decays.
    agg.push(12, {"m": 1.0})
    assert sink.alerts == [("d", True), ("d", False)]
    assert agg.alert_states() == {"d": False}

    # Steady state must be silent: an edge-triggered gauge re-written every window would
    # make an Alertmanager `for:` clause fire on a healthy stream.
    for step in range(13, 25):
        agg.push(step, {"m": 1.0})
    assert sink.alerts == [("d", True), ("d", False)]


def test_refiring_inside_the_hold_extends_it_without_re_setting_the_gauge() -> None:
    sink = RecordingSink()
    agg = MetricAggregator(
        ["m"], {"m": "d"}, window_size=1, thresholds={"m": CFG}, sink=sink, alert_hold=2
    )
    # Second firing lands while the first firing's hold is still counting down.
    _install_test(agg, "m", RecordingTest(alarm_at=frozenset({1, 2})))

    for step in range(1, 5):
        agg.push(step, {"m": 1.0})

    assert agg.alerts_fired == 2
    # Two firings, still one rising edge; the hold restarts from the second one, so it
    # clears on window 4 rather than window 3.
    assert sink.alerts == [("d", True), ("d", False)]
    assert agg.steps_seen == 4


def test_two_metrics_on_one_detector_share_a_single_latch() -> None:
    """The gauge is labelled by detector, so either metric holds it up on its own."""
    sink = RecordingSink()
    agg = MetricAggregator(
        ["a", "b"],
        {"a": "d", "b": "d"},
        window_size=1,
        thresholds={"a": CFG, "b": CFG},
        sink=sink,
        alert_hold=2,
    )
    _install_test(agg, "a", RecordingTest(alarm_at=frozenset({1})))
    _install_test(agg, "b", RecordingTest(alarm_at=frozenset({2})))

    agg.push(1, {"a": 1.0, "b": 1.0})  # a fires: hold(a)=2
    agg.push(2, {"a": 1.0, "b": 1.0})  # b fires: hold(a)=1, hold(b)=2
    agg.push(3, {"a": 1.0, "b": 1.0})  # hold(a)=0, hold(b)=1 — b keeps it latched
    assert sink.alerts == [("d", True)]

    agg.push(4, {"a": 1.0, "b": 1.0})  # hold(b)=0 — now the detector goes quiet
    assert sink.alerts == [("d", True), ("d", False)]


# --------------------------------------------------------------------------------------
# reset()
# --------------------------------------------------------------------------------------
def test_reset_clears_windows_tests_latches_and_counters() -> None:
    sink = RecordingSink()
    agg = MetricAggregator(
        ["m"],
        {"m": "d"},
        window_size=4,
        thresholds={"m": CFG},
        sink=sink,
        alert_hold=2,
    )
    fake = RecordingTest(alarm_at=frozenset({1}))
    _install_test(agg, "m", fake)

    for step, value in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
        agg.push(step, {"m": value})
    for step, value in enumerate([5.0, 6.0, 7.0], start=5):
        agg.push(step, {"m": value})
    agg.push(8, {"m": float("nan")})

    assert fake.received == [2.5]
    assert sink.alerts == [("d", True)]
    assert agg.snapshot() != {}

    agg.reset()

    # A latched gauge must be released, or a re-baselined monitor starts life in alarm.
    assert sink.alerts == [("d", True), ("d", False)]
    assert agg.alert_states() == {"d": False}
    assert agg.snapshot() == {}
    assert fake.resets == 1
    assert fake.received == []
    counters = (agg.steps_seen, agg.windows_closed, agg.alerts_fired, agg.values_skipped)
    assert counters == (0, 0, 0, 0)

    # Resetting a quiet aggregator must not emit a redundant clear.
    agg.reset()
    assert sink.alerts == [("d", True), ("d", False)]


def test_reset_clears_the_partial_tumbling_accumulator() -> None:
    sink = RecordingSink()
    agg = MetricAggregator(["m"], {"m": "d"}, window_size=4, thresholds={"m": CFG}, sink=sink)
    fake = RecordingTest()
    _install_test(agg, "m", fake)

    for step, value in enumerate([1.0, 2.0, 3.0], start=1):
        agg.push(step, {"m": value})
    agg.reset()

    for step in range(4, 8):
        agg.push(step, {"m": 10.0})

    # A surviving accumulator would have closed a window after one post-reset push with
    # mean (1+2+3+10)/4 = 4.0 — pre-reset data leaking across a re-baseline.
    assert fake.received == [10.0]
    # A surviving rolling deque would have exported (1+2+3+10)/4 = 4.0 here too.
    assert sink.values_for("m")[3] == pytest.approx(10.0)


# --------------------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("window_size", [0, -1])
def test_window_size_below_one_rejected(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size"):
        MetricAggregator(["m"], {"m": "d"}, window_size=window_size)


def test_negative_alert_hold_rejected() -> None:
    with pytest.raises(ValueError, match="alert_hold"):
        MetricAggregator(["m"], {"m": "d"}, alert_hold=-1)


def test_alert_hold_zero_is_allowed_and_clears_immediately() -> None:
    """hold=0 is a legal "edge only" mode: the gauge never latches above the firing push."""
    sink = RecordingSink()
    agg = MetricAggregator(
        ["m"], {"m": "d"}, window_size=1, thresholds={"m": CFG}, sink=sink, alert_hold=0
    )
    _install_test(agg, "m", RecordingTest(alarm_at=frozenset({1})))

    agg.push(1, {"m": 1.0})
    agg.push(2, {"m": 1.0})

    assert agg.alerts_fired == 1
    assert sink.alerts == []
    assert agg.alert_states() == {"d": False}


def test_threshold_for_unknown_metric_is_ignored() -> None:
    """A stale metric name in config must not create a phantom test or crash the monitor."""
    agg = MetricAggregator(
        ["m"], {"m": "d"}, window_size=1, thresholds={"m": CFG, "gone_metric": CFG}
    )
    agg.push(1, {"m": 1.0})
    assert set(agg._tests) == {"m"}
