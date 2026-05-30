"""P1-09: Prometheus export scaffolding — registry, update(), metric names."""

from __future__ import annotations

import guard.export.prometheus as exp


def _families() -> dict[str, object]:
    """Collect all metric families from the GUARD registry, keyed by name."""
    return {m.name: m for m in exp.REGISTRY.collect()}


def _sample_value(metric_name: str, labels: dict[str, str] | None = None) -> float | None:
    """Return the value of the first sample matching name + labels from the registry."""
    for family in exp.REGISTRY.collect():
        for sample in family.samples:
            if sample.name != metric_name and sample.name != metric_name + "_total":
                continue
            if labels is None or all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


# ─── registration ─────────────────────────────────────────────────────────────


def test_all_design_doc_metrics_are_registered() -> None:
    families = _families()
    required = {
        "guard_entropy_mean",
        "guard_entropy_p99",
        "guard_js_divergence",
        "guard_embedding_mahalanobis_mean",
        "guard_drift_alert",
        "guard_monitor_overhead_us",
        "guard_ring_utilization",
        "guard_summaries_dropped",  # prometheus_client appends _total in text output
    }
    missing = required - families.keys()
    assert not missing, f"missing metric families: {sorted(missing)}"


def test_no_duplicate_metric_names() -> None:
    names = [m.name for m in exp.REGISTRY.collect()]
    assert len(names) == len(set(names)), f"duplicate metric names: {sorted(names)}"


# ─── update() sets values ─────────────────────────────────────────────────────


def test_update_sets_gauge_values() -> None:
    exp.update(
        {
            "entropy_mean": 1.23,
            "entropy_p99": 2.34,
            "js_divergence": 0.42,
            "embedding_mahalanobis_mean": 3.14,
        }
    )
    assert abs(_sample_value("guard_entropy_mean") - 1.23) < 1e-6  # type: ignore[operator]
    assert abs(_sample_value("guard_entropy_p99") - 2.34) < 1e-6  # type: ignore[operator]
    assert abs(_sample_value("guard_js_divergence") - 0.42) < 1e-6  # type: ignore[operator]
    assert abs(_sample_value("guard_embedding_mahalanobis_mean") - 3.14) < 1e-6  # type: ignore[operator]


def test_update_sets_drift_alert_by_detector() -> None:
    exp.update({"drift_alert.entropy": 1.0, "drift_alert.embedding": 0.0})
    assert _sample_value("guard_drift_alert", {"detector": "entropy"}) == 1.0
    assert _sample_value("guard_drift_alert", {"detector": "embedding"}) == 0.0


def test_update_entropy_quantile_alias() -> None:
    # EntropyDetector emits "entropy_quantile"; the exporter maps it to guard_entropy_p99.
    exp.update({"entropy_quantile": 1.99})
    assert abs(_sample_value("guard_entropy_p99") - 1.99) < 1e-6  # type: ignore[operator]


def test_update_unknown_keys_are_silently_ignored() -> None:
    # Should not raise; extra keys from future detector versions are safe to pass through.
    exp.update({"completely_unknown_metric": 9.9, "entropy_mean": 0.5})
    assert abs(_sample_value("guard_entropy_mean") - 0.5) < 1e-6  # type: ignore[operator]


def test_summaries_dropped_counter_increments() -> None:
    before = _sample_value("guard_summaries_dropped_total") or 0.0
    exp.update({"summaries_dropped": 3.0})
    after = _sample_value("guard_summaries_dropped_total") or 0.0
    assert after - before == 3.0


# ─── metric type contract ─────────────────────────────────────────────────────


def test_drift_alert_is_labelled_gauge() -> None:
    assert _families()["guard_drift_alert"].type == "gauge"


def test_summaries_dropped_is_counter() -> None:
    assert _families()["guard_summaries_dropped"].type == "counter"


def test_detector_gauges_are_gauges() -> None:
    families = _families()
    for name in (
        "guard_entropy_mean",
        "guard_entropy_p99",
        "guard_js_divergence",
        "guard_embedding_mahalanobis_mean",
        "guard_monitor_overhead_us",
        "guard_ring_utilization",
    ):
        assert families[name].type == "gauge", f"{name} should be a gauge"
