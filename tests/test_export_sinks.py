"""Phase-2 export layer: PrometheusExporter instances, MultiSink fan-out, alert sinks.

`tests/test_export.py` covers the process-wide default exporter (the stable Phase-1
surface). This module covers everything built on top of it in Phase 2: per-instance
exporters with their own registry and namespace, the fan-out sinks, and
`guard/export/alerts.py`. Nothing here may touch the module-level default registry —
every test builds its own exporter so the suite stays order-independent.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
from collections.abc import Mapping
from typing import Any

import pytest

from guard.export.alerts import (
    Alert,
    AlertSink,
    LoggingAlertSink,
    MultiAlertSink,
    WebhookAlertSink,
)
from guard.export.multi import MultiSink
from guard.export.otel import OTelSink
from guard.export.prometheus import PrometheusExporter

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _sample(
    exporter: PrometheusExporter, name: str, labels: dict[str, str] | None = None
) -> float | None:
    """Value of the first sample called `name` in `exporter`'s own registry.

    Counters are exposed with a `_total` suffix in the sample stream but registered under
    the bare family name, so both spellings are accepted.
    """
    for family in exporter.registry.collect():
        for s in family.samples:
            if s.name not in (name, f"{name}_total"):
                continue
            if labels is None or all(s.labels.get(k) == v for k, v in labels.items()):
                return float(s.value)
    return None


class RecordingSink:
    """MetricSink double that remembers every call it received."""

    def __init__(self) -> None:
        self.updates: list[dict[str, float]] = []
        self.alerts: list[tuple[str, bool]] = []
        self.dropped: list[int] = []
        self.telemetry: list[dict[str, float]] = []

    def update(self, values: Mapping[str, float]) -> None:
        self.updates.append(dict(values))

    def set_alert(self, detector: str, active: bool) -> None:
        self.alerts.append((detector, active))

    def add_dropped(self, count: int) -> None:
        self.dropped.append(count)

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        self.telemetry.append(dict(values))


class ExplodingSink:
    """MetricSink double that fails on every call — stands in for a down backend."""

    def update(self, values: Mapping[str, float]) -> None:
        raise RuntimeError("collector unreachable")

    def set_alert(self, detector: str, active: bool) -> None:
        raise RuntimeError("collector unreachable")

    def add_dropped(self, count: int) -> None:
        raise RuntimeError("collector unreachable")

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        raise RuntimeError("collector unreachable")


def _alert(**overrides: Any) -> Alert:
    """A fully-populated alert; every field fixed so renderings are deterministic."""
    kwargs: dict[str, Any] = {
        "metric": "js_divergence",
        "detector": "divergence",
        "step": 4096,
        "value": 0.375,
        "statistic": 12.5,
        "threshold": 8.0,
        "method": "page_hinkley",
        "model_version": "vit_b_16@1",
        "baseline_version": "sha256:abcd",
        "labels": {"gpu": "0"},
    }
    kwargs.update(overrides)
    return Alert(**kwargs)


# --------------------------------------------------------------------------------------
# PrometheusExporter: namespace and registry isolation
# --------------------------------------------------------------------------------------


def test_custom_namespace_prefixes_every_metric() -> None:
    exporter = PrometheusExporter(namespace="acme")
    names = {family.name for family in exporter.registry.collect()}

    assert "acme_entropy_mean" in names
    assert "acme_js_divergence" in names
    assert "acme_drift_alert" in names
    assert "acme_summaries_dropped" in names
    # The default prefix must be gone entirely, not merely supplemented.
    assert not any(n.startswith("guard_") for n in names)


def test_two_exporters_same_namespace_do_not_collide() -> None:
    """Back-to-back construction is the realistic case (a test, then the engine's own).

    A shared default registry would raise `Duplicated timeseries` on the second build; this
    is the regression guard for the exporter silently reverting to `prom.REGISTRY`.
    """
    first = PrometheusExporter(namespace="acme")
    second = PrometheusExporter(namespace="acme")

    assert first.registry is not second.registry

    first.update({"entropy_mean": 1.5})
    assert _sample(first, "acme_entropy_mean") == 1.5
    # Writes to one exporter must be invisible in the other's registry.
    assert _sample(second, "acme_entropy_mean") == 0.0


def test_registry_is_independent_of_the_process_default() -> None:
    import guard.export.prometheus as exp

    exporter = PrometheusExporter(namespace="acme")
    assert exporter.registry is not exp.REGISTRY


# --------------------------------------------------------------------------------------
# ensure_metrics / gauge creation
# --------------------------------------------------------------------------------------


def test_ensure_metrics_creates_gauges_for_unknown_names() -> None:
    # A third-party detector declares metrics the exporter has never heard of.
    exporter = PrometheusExporter(namespace="acme", extra_metrics=("custom_score",))
    exporter.ensure_metrics(["another_score"])

    names = {family.name for family in exporter.registry.collect()}
    assert "acme_custom_score" in names
    assert "acme_another_score" in names


def test_ensure_metrics_is_idempotent_and_preserves_values() -> None:
    """Re-registering the same name would raise; re-creating it would silently zero it."""
    exporter = PrometheusExporter(namespace="acme")
    exporter.ensure_metrics(["custom_score"])
    gauge = exporter.gauge("custom_score")
    exporter.update({"custom_score": 4.25})

    exporter.ensure_metrics(["custom_score", "custom_score"])
    exporter.ensure_metrics(["custom_score"])

    assert exporter.gauge("custom_score") is gauge
    assert _sample(exporter, "acme_custom_score") == 4.25


def test_update_creates_a_gauge_for_a_never_declared_metric() -> None:
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"never_declared": 2.0})
    assert _sample(exporter, "acme_never_declared") == 2.0


# --------------------------------------------------------------------------------------
# update(): non-finite handling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_update_skips_non_finite_and_keeps_previous_value(bad: float) -> None:
    """A NaN means "not reported this step"; publishing it poisons every Grafana avg/rate.

    The assertion is on the *retained* value, not on `isnan(...) is False` — a gauge that
    silently reset to 0.0 would be just as wrong as one holding NaN.
    """
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"entropy_mean": 1.75})
    exporter.update({"entropy_mean": bad})

    assert _sample(exporter, "acme_entropy_mean") == 1.75


def test_update_non_finite_does_not_block_the_other_keys() -> None:
    # One dead detector must not suppress the metrics reported alongside it.
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"entropy_mean": float("nan"), "js_divergence": 0.5})
    assert _sample(exporter, "acme_js_divergence") == 0.5


# --------------------------------------------------------------------------------------
# update(): entropy_p99 alias
# --------------------------------------------------------------------------------------


def test_p99_alias_fed_when_enabled() -> None:
    exporter = PrometheusExporter(namespace="acme", emit_p99_alias=True)
    exporter.update({"entropy_quantile": 1.99})

    assert _sample(exporter, "acme_entropy_quantile") == 1.99
    assert _sample(exporter, "acme_entropy_p99") == 1.99


def test_p99_alias_not_fed_when_quantile_is_not_p99() -> None:
    """Publishing a p75 under a name that says p99 is the quiet lie the flag exists to stop."""
    exporter = PrometheusExporter(namespace="acme", emit_p99_alias=False)
    exporter.update({"entropy_quantile": 1.99})

    assert _sample(exporter, "acme_entropy_quantile") == 1.99
    assert _sample(exporter, "acme_entropy_p99") == 0.0  # never written → still at its default


def test_explicit_entropy_p99_key_is_honoured_regardless_of_alias() -> None:
    # The alias flag gates the *derivation*, not a caller that names the metric directly.
    exporter = PrometheusExporter(namespace="acme", emit_p99_alias=False)
    exporter.update({"entropy_p99": 3.5})
    assert _sample(exporter, "acme_entropy_p99") == 3.5


# --------------------------------------------------------------------------------------
# update(): counters, alerts, telemetry
# --------------------------------------------------------------------------------------


def test_summaries_dropped_key_increments_by_the_delta() -> None:
    """The value is a per-drain delta; treating it as a `set` would make the counter
    non-monotonic, and treating a cumulative total as a delta would double-count."""
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"summaries_dropped": 3.0})
    exporter.update({"summaries_dropped": 2.0})

    assert _sample(exporter, "acme_summaries_dropped") == 5.0


def test_summaries_dropped_key_is_not_exported_as_a_gauge() -> None:
    # If the counter branch were missed, `update()` would create an `acme_summaries_dropped`
    # *gauge* alongside the counter and the registry would carry two conflicting families.
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"summaries_dropped": 1.0})

    families = [f for f in exporter.registry.collect() if f.name == "acme_summaries_dropped"]
    assert len(families) == 1
    assert families[0].type == "counter"


def test_add_dropped_ignores_non_positive_counts() -> None:
    exporter = PrometheusExporter(namespace="acme")
    exporter.add_dropped(4)
    exporter.add_dropped(0)
    exporter.add_dropped(-7)  # a negative delta would raise inside prometheus_client
    assert _sample(exporter, "acme_summaries_dropped") == 4.0


def test_set_alert_writes_the_labelled_drift_alert_gauge() -> None:
    exporter = PrometheusExporter(namespace="acme")
    exporter.set_alert("entropy", True)
    exporter.set_alert("embedding", False)

    assert _sample(exporter, "acme_drift_alert", {"detector": "entropy"}) == 1.0
    assert _sample(exporter, "acme_drift_alert", {"detector": "embedding"}) == 0.0


def test_set_alert_clears_a_previously_latched_detector() -> None:
    # A latched alert that never clears is worse than no alert at all.
    exporter = PrometheusExporter(namespace="acme")
    exporter.set_alert("entropy", True)
    exporter.set_alert("entropy", False)
    assert _sample(exporter, "acme_drift_alert", {"detector": "entropy"}) == 0.0


def test_update_drift_alert_prefix_routes_to_the_labelled_gauge() -> None:
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"drift_alert.divergence": 1.0})

    assert _sample(exporter, "acme_drift_alert", {"detector": "divergence"}) == 1.0
    # The dotted key must not also leak out as a metric of its own.
    assert "acme_drift_alert.divergence" not in {f.name for f in exporter.registry.collect()}


def test_set_telemetry_increments_counters_and_sets_gauges() -> None:
    """Telemetry mixes both kinds: `steps_observed` is monotonic, `ring_utilization` is a
    point-in-time reading, and confusing the two makes both useless."""
    exporter = PrometheusExporter(namespace="acme")
    exporter.set_telemetry({"steps_observed": 5.0, "ring_utilization": 0.25})
    exporter.set_telemetry({"steps_observed": 2.0, "ring_utilization": 0.5})

    assert _sample(exporter, "acme_steps_observed") == 7.0
    assert _sample(exporter, "acme_ring_utilization") == 0.5


def test_record_detector_failure_increments_per_detector() -> None:
    exporter = PrometheusExporter(namespace="acme")
    exporter.record_detector_failure("entropy")
    exporter.record_detector_failure("entropy", 2)
    exporter.record_detector_failure("embedding")

    assert _sample(exporter, "acme_detector_failures", {"detector": "entropy"}) == 3.0
    assert _sample(exporter, "acme_detector_failures", {"detector": "embedding"}) == 1.0


def test_update_skips_non_finite_counter_delta() -> None:
    # A counter incremented by inf is ruined permanently — counters cannot go down — so the
    # non-finite guard has to run before the routing decision, not after it.
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"summaries_dropped": 1.0})
    exporter.update({"summaries_dropped": float("inf")})

    value = _sample(exporter, "acme_summaries_dropped")
    assert value is not None and math.isfinite(value)


def test_update_skips_non_finite_drift_alert() -> None:
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"drift_alert.entropy": 1.0})
    exporter.update({"drift_alert.entropy": float("nan")})

    assert _sample(exporter, "acme_drift_alert", {"detector": "entropy"}) == 1.0


# --------------------------------------------------------------------------------------
# generate_latest(): text exposition format
# --------------------------------------------------------------------------------------


def test_generate_latest_renders_namespaced_text_format() -> None:
    exporter = PrometheusExporter(namespace="acme")
    exporter.update({"entropy_mean": 1.25})
    exporter.set_alert("entropy", True)
    exporter.add_dropped(2)

    text = exporter.generate_latest().decode("utf-8")

    assert "# TYPE acme_entropy_mean gauge" in text
    assert "acme_entropy_mean 1.25" in text
    assert "# TYPE acme_drift_alert gauge" in text
    assert 'acme_drift_alert{detector="entropy"} 1.0' in text
    # prometheus_client renders counters with the `_total` suffix mandated by the format.
    assert "# TYPE acme_summaries_dropped_total counter" in text
    assert "acme_summaries_dropped_total 2.0" in text


def test_generate_latest_carries_help_text_for_ad_hoc_metrics() -> None:
    # A metric with no HELP line shows up bare in Grafana; ensure_metrics must supply one.
    exporter = PrometheusExporter(namespace="acme", extra_metrics=("custom_score",))
    text = exporter.generate_latest().decode("utf-8")
    assert "# HELP acme_custom_score" in text


# --------------------------------------------------------------------------------------
# MultiSink fan-out
# --------------------------------------------------------------------------------------


def test_multisink_fans_every_method_out_to_all_sinks() -> None:
    a, b = RecordingSink(), RecordingSink()
    multi = MultiSink(a, b)

    multi.update({"entropy_mean": 1.0})
    multi.set_alert("entropy", True)
    multi.add_dropped(3)
    multi.set_telemetry({"ring_utilization": 0.5})

    for sink in (a, b):
        assert sink.updates == [{"entropy_mean": 1.0}]
        assert sink.alerts == [("entropy", True)]
        assert sink.dropped == [3]
        assert sink.telemetry == [{"ring_utilization": 0.5}]


def test_multisink_raising_sink_does_not_stop_the_others() -> None:
    """The raiser is listed FIRST so the test proves iteration continues past a failure,
    not merely that the exception is swallowed after everyone was already served."""
    good = RecordingSink()
    multi = MultiSink(ExplodingSink(), good, ExplodingSink())

    multi.update({"entropy_mean": 1.0})
    multi.set_alert("entropy", False)
    multi.add_dropped(1)
    multi.set_telemetry({"ring_utilization": 0.25})

    assert good.updates == [{"entropy_mean": 1.0}]
    assert good.alerts == [("entropy", False)]
    assert good.dropped == [1]
    assert good.telemetry == [{"ring_utilization": 0.25}]


def test_multisink_logs_the_failure_it_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    # Silent swallowing would make a permanently-broken exporter undiagnosable.
    multi = MultiSink(ExplodingSink())
    with caplog.at_level(logging.ERROR, logger="guard.export"):
        multi.update({"entropy_mean": 1.0})

    assert any("ExplodingSink" in record.getMessage() for record in caplog.records)


def test_multisink_with_a_real_exporter_writes_through() -> None:
    exporter = PrometheusExporter(namespace="acme")
    recorder = RecordingSink()
    multi = MultiSink(ExplodingSink(), exporter, recorder)

    multi.update({"entropy_mean": 0.75})
    multi.set_alert("entropy", True)

    assert _sample(exporter, "acme_entropy_mean") == 0.75
    assert _sample(exporter, "acme_drift_alert", {"detector": "entropy"}) == 1.0
    assert recorder.updates == [{"entropy_mean": 0.75}]


def test_multisink_with_no_sinks_is_a_no_op() -> None:
    MultiSink().update({"entropy_mean": 1.0})  # must not raise


# --------------------------------------------------------------------------------------
# Alert model
# --------------------------------------------------------------------------------------


def test_alert_to_dict_round_trips_through_json() -> None:
    """The webhook body is `json.dumps(alert.to_dict())`; any non-serialisable field would
    only blow up inside the delivery thread, where nobody is watching."""
    alert = _alert()
    restored = json.loads(json.dumps(alert.to_dict()))

    assert restored == {
        "metric": "js_divergence",
        "detector": "divergence",
        "step": 4096,
        "value": 0.375,
        "statistic": 12.5,
        "threshold": 8.0,
        "method": "page_hinkley",
        "model_version": "vit_b_16@1",
        "baseline_version": "sha256:abcd",
        "labels": {"gpu": "0"},
    }
    assert Alert(**restored) == alert


def test_alert_to_dict_deep_copies_labels() -> None:
    # Mutating the exported dict must not reach back into the frozen alert.
    alert = _alert()
    alert.to_dict()["labels"]["gpu"] = "7"
    assert alert.labels == {"gpu": "0"}


def test_alert_summary_names_detector_metric_and_value() -> None:
    summary = _alert().summary()

    assert "detector=divergence" in summary
    assert "metric=js_divergence" in summary
    assert "value=0.375" in summary
    assert "statistic=12.5" in summary
    assert "threshold=8" in summary
    assert "step=4096" in summary


def test_alert_summary_marks_unknown_versions() -> None:
    """Drift against a stale baseline is a baseline problem; an empty version string that
    rendered as `model=` would read as a value rather than as 'unknown'."""
    summary = _alert(model_version="", baseline_version="").summary()
    assert "model=?" in summary
    assert "baseline=?" in summary


def test_alert_is_frozen() -> None:
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
        _alert().metric = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# LoggingAlertSink
# --------------------------------------------------------------------------------------


def test_logging_alert_sink_logs_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    sink = LoggingAlertSink()
    with caplog.at_level(logging.WARNING, logger="guard.alerts"):
        sink.dispatch(_alert())

    records = [r for r in caplog.records if r.name == "guard.alerts"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "detector=divergence" in records[0].getMessage()


def test_logging_alert_sink_honours_an_injected_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = LoggingAlertSink(logging.getLogger("guard.alerts.custom"))
    with caplog.at_level(logging.WARNING, logger="guard.alerts.custom"):
        sink.dispatch(_alert())

    assert [r.name for r in caplog.records] == ["guard.alerts.custom"]


def test_logging_alert_sink_close_is_safe() -> None:
    sink = LoggingAlertSink()
    sink.close()
    sink.close()


def test_logging_alert_sink_satisfies_the_protocol() -> None:
    assert isinstance(LoggingAlertSink(), AlertSink)


# --------------------------------------------------------------------------------------
# MultiAlertSink
# --------------------------------------------------------------------------------------


class RecordingAlertSink:
    """AlertSink double that records dispatches and closes."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []
        self.closed = 0

    def dispatch(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def close(self) -> None:
        self.closed += 1


class ExplodingAlertSink:
    """AlertSink double that fails on dispatch and on close."""

    def dispatch(self, alert: Alert) -> None:
        raise RuntimeError("pager down")

    def close(self) -> None:
        raise RuntimeError("pager down")


def test_multi_alert_sink_fans_out_and_isolates_failures() -> None:
    good = RecordingAlertSink()
    alert = _alert()
    multi = MultiAlertSink(ExplodingAlertSink(), good)

    multi.dispatch(alert)

    assert good.alerts == [alert]


def test_multi_alert_sink_close_isolates_failures() -> None:
    # A sink that throws on shutdown must not strand the webhook thread of the next one.
    good = RecordingAlertSink()
    multi = MultiAlertSink(ExplodingAlertSink(), good)

    multi.close()

    assert good.closed == 1


def test_multi_alert_sink_logs_the_swallowed_dispatch_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    multi = MultiAlertSink(ExplodingAlertSink())
    with caplog.at_level(logging.ERROR, logger="guard.alerts"):
        multi.dispatch(_alert())

    assert any("ExplodingAlertSink" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------------------
# WebhookAlertSink
# --------------------------------------------------------------------------------------


def _saturated_webhook_sink(**kwargs: Any) -> WebhookAlertSink:
    """A sink whose worker is alive but consumes nothing, with its queue already full.

    Determinism is the whole point. The real worker would race the producer for the queue
    slots, so it is stopped and replaced by an idle placeholder thread: `is_alive()` stays
    True — which is what routes `dispatch` down the queue-full branch under test rather
    than the dead-worker branch — while nothing ever drains the queue. Port 1 on loopback
    is never listening, so no request could succeed either way.

    Release it with `_release(sink)`, not `close()` alone, or the placeholder outlives the
    join timeout.
    """
    sink = WebhookAlertSink("http://127.0.0.1:1/", timeout=0.05, **kwargs)
    sink._stop.set()
    sink._thread.join(timeout=2.0)

    idle = threading.Event()
    sink._thread = threading.Thread(target=idle.wait, name="idle-placeholder", daemon=True)
    sink._thread.start()
    sink._idle_gate = idle  # type: ignore[attr-defined]

    while True:
        try:
            sink._queue.put_nowait(_alert(step=-1))
        except queue.Full:
            break
    return sink


def _release(sink: WebhookAlertSink) -> None:
    """Let the idle placeholder exit, then close the sink."""
    gate: threading.Event = sink._idle_gate  # type: ignore[attr-defined]
    gate.set()
    sink.close()


def _stopped_webhook_sink(**kwargs: Any) -> WebhookAlertSink:
    """A webhook sink whose worker thread has been shut down before any dispatch."""
    sink = WebhookAlertSink("http://127.0.0.1:1/", timeout=0.05, **kwargs)
    sink._stop.set()
    sink._thread.join(timeout=2.0)
    assert not sink._thread.is_alive(), "webhook worker did not honour the stop flag"
    return sink


def test_webhook_sink_drops_and_counts_when_the_queue_is_full() -> None:
    sink = _saturated_webhook_sink(max_queue=1)
    try:
        sink.dispatch(_alert(step=2))  # dropped: queue already full
        sink.dispatch(_alert(step=3))  # dropped

        assert sink.dropped == 2
        assert sink.delivered == 0
    finally:
        _release(sink)


def test_webhook_sink_counts_alerts_dispatched_after_close() -> None:
    # An alert handed to a closed sink used to be queued to a worker that would never run
    # it: silently lost, and not visible in `dropped`. Losing a firing is acceptable under
    # backpressure; losing it without saying so is not.
    sink = WebhookAlertSink("http://127.0.0.1:1/", timeout=0.05, max_queue=8)
    sink.close()
    sink.dispatch(_alert(step=1))
    sink.dispatch(_alert(step=2))
    assert sink.dropped == 2
    assert sink.delivered == 0


def test_webhook_sink_counts_drops_when_its_worker_has_died() -> None:
    # Same contract when the worker is gone for any other reason.
    sink = _stopped_webhook_sink(max_queue=8)
    try:
        for step in range(5):
            sink.dispatch(_alert(step=step))
        assert sink.dropped == 5
    finally:
        sink.close()


def test_webhook_sink_dispatch_never_raises_on_backpressure() -> None:
    """`dispatch` runs on the drain thread; raising there trips the firewall and kills
    alerting entirely, which is a far worse outcome than losing one alert."""
    sink = _saturated_webhook_sink(max_queue=1)
    try:
        for step in range(50):
            sink.dispatch(_alert(step=step))
        assert sink.dropped == 50
    finally:
        _release(sink)


def test_webhook_sink_close_is_idempotent() -> None:
    sink = _stopped_webhook_sink(max_queue=1)
    sink.dispatch(_alert())  # leaves the queue full so close()'s sentinel put must not block
    sink.close()
    sink.close()
    assert not sink._thread.is_alive()


def test_webhook_sink_sets_json_content_type_and_merges_headers() -> None:
    sink = _stopped_webhook_sink(max_queue=1, headers={"Authorization": "Bearer t"})
    try:
        assert sink.headers["Content-Type"] == "application/json"
        assert sink.headers["Authorization"] == "Bearer t"
    finally:
        sink.close()


def test_webhook_sink_worker_thread_is_a_daemon() -> None:
    # A non-daemon worker would keep the serving process alive at shutdown.
    sink = WebhookAlertSink("http://127.0.0.1:1/", timeout=0.05, max_queue=1)
    try:
        assert sink._thread.daemon
    finally:
        sink.close()


def test_webhook_sink_satisfies_the_protocol() -> None:
    sink = _stopped_webhook_sink(max_queue=1)
    try:
        assert isinstance(sink, AlertSink)
    finally:
        sink.close()


# --------------------------------------------------------------------------------------
# OTelSink (optional extra)
# --------------------------------------------------------------------------------------


class FakeGauge:
    def __init__(self) -> None:
        self.points: list[tuple[float, dict[str, str] | None]] = []

    def set(self, value: float, attributes: dict[str, str] | None = None) -> None:
        self.points.append((value, attributes))


class FakeCounter:
    def __init__(self) -> None:
        self.adds: list[float] = []

    def add(self, value: float) -> None:
        self.adds.append(value)


class FakeMeter:
    """Stands in for an `opentelemetry.metrics.Meter`, recording instrument creation."""

    def __init__(self) -> None:
        self.gauges: dict[str, FakeGauge] = {}
        self.counters: dict[str, FakeCounter] = {}

    def create_gauge(self, name: str) -> FakeGauge:
        assert name not in self.gauges, f"instrument {name!r} created twice"
        self.gauges[name] = FakeGauge()
        return self.gauges[name]

    def create_counter(self, name: str) -> FakeCounter:
        assert name not in self.counters, f"instrument {name!r} created twice"
        self.counters[name] = FakeCounter()
        return self.counters[name]


def test_otel_sink_records_namespaced_gauges() -> None:
    pytest.importorskip("opentelemetry")
    meter = FakeMeter()
    sink = OTelSink(meter, namespace="acme")

    sink.update({"entropy_mean": 1.5, "js_divergence": 0.25})

    assert meter.gauges["acme.entropy_mean"].points == [(1.5, None)]
    assert meter.gauges["acme.js_divergence"].points == [(0.25, None)]


def test_otel_sink_reuses_instruments_across_calls() -> None:
    """Creating a new instrument per drain step leaks one per step; the `create_gauge`
    assertion in FakeMeter is what catches it."""
    pytest.importorskip("opentelemetry")
    meter = FakeMeter()
    sink = OTelSink(meter, namespace="acme")

    sink.update({"entropy_mean": 1.0})
    sink.update({"entropy_mean": 2.0})

    assert meter.gauges["acme.entropy_mean"].points == [(1.0, None), (2.0, None)]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_otel_sink_skips_non_finite(bad: float) -> None:
    pytest.importorskip("opentelemetry")
    meter = FakeMeter()
    sink = OTelSink(meter, namespace="acme")

    sink.update({"entropy_mean": bad})

    assert meter.gauges == {}


def test_otel_sink_alerts_use_a_detector_attribute() -> None:
    pytest.importorskip("opentelemetry")
    meter = FakeMeter()
    sink = OTelSink(meter, namespace="acme")

    sink.set_alert("entropy", True)
    sink.update({"drift_alert.divergence": 0.0})

    assert meter.gauges["acme.drift_alert"].points == [
        (1.0, {"detector": "entropy"}),
        (0.0, {"detector": "divergence"}),
    ]
    # The dotted key must be consumed by the alert branch, never become its own instrument.
    assert "acme.drift_alert.divergence" not in meter.gauges


def test_otel_sink_add_dropped_ignores_non_positive() -> None:
    pytest.importorskip("opentelemetry")
    meter = FakeMeter()
    sink = OTelSink(meter, namespace="acme")

    sink.add_dropped(3)
    sink.add_dropped(0)
    sink.add_dropped(-1)

    assert meter.counters["acme.summaries_dropped"].adds == [3]


def test_otel_sink_is_a_metric_sink() -> None:
    pytest.importorskip("opentelemetry")
    from guard.export.base import MetricSink

    assert isinstance(OTelSink(FakeMeter(), namespace="acme"), MetricSink)
