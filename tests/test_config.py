"""P1-01: config schema validation and YAML round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from guard.config import Config, DetectorConfig, ExportConfig, ThresholdConfig


def _valid_config() -> Config:
    return Config(
        enabled=True,
        window_size=512,
        drain_every_k=16,
        detectors=[
            DetectorConfig(name="entropy", enabled=True, params={"quantile": 0.99}),
            DetectorConfig(name="divergence", enabled=False),
        ],
        thresholds={
            "entropy_mean": ThresholdConfig(method="page_hinkley", threshold=2.0, delta=0.1),
            "js_divergence": ThresholdConfig(method="cusum", threshold=1.5),
        },
        export=ExportConfig(prometheus_port=9300, namespace="guard"),
    )


def test_dict_round_trip() -> None:
    cfg = _valid_config()
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = _valid_config()
    p = tmp_path / "guard.yaml"
    cfg.to_yaml(p)
    loaded = Config.from_yaml(p)
    assert loaded == cfg


def test_defaults_apply() -> None:
    cfg = Config(window_size=128, drain_every_k=8)
    assert cfg.enabled is True
    assert cfg.detectors == []
    assert cfg.export.namespace == "guard"
    assert cfg.export.enabled is True


def test_negative_window_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(window_size=-1, drain_every_k=8)


def test_zero_drain_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(window_size=128, drain_every_k=0)


def test_unknown_detector_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown detector"):
        Config(window_size=128, drain_every_k=8, detectors=[DetectorConfig(name="bogus")])


def test_duplicate_detector_names_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate detector names"):
        Config(
            window_size=128,
            drain_every_k=8,
            detectors=[DetectorConfig(name="entropy"), DetectorConfig(name="entropy")],
        )


def test_non_positive_threshold_rejected() -> None:
    with pytest.raises(ValidationError):
        ThresholdConfig(threshold=0.0)


def test_unknown_change_point_method_rejected() -> None:
    with pytest.raises(ValidationError):
        ThresholdConfig(method="adwin", threshold=1.0)  # type: ignore[arg-type]


def test_extra_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.from_dict({"window_size": 128, "drain_every_k": 8, "typo_field": 1})


def test_from_yaml_empty_file_uses_defaults_then_fails_on_required(tmp_path: Path) -> None:
    # window_size/drain_every_k are required; an empty config must fail clearly.
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.from_yaml(p)


def test_from_yaml_non_mapping_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        Config.from_yaml(p)
