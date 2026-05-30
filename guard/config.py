"""Runtime configuration schema (LOCKED field names for Phase 1).

GUARD is configured declaratively (YAML in production, dicts in tests). This module is the
single source of truth for what a valid configuration looks like; it validates eagerly and
fails loudly so a bad config can never silently disable monitoring.

The locked top-level field names (do not rename without updating every caller and the
design doc):
  * ``detectors: list[DetectorConfig]``
  * ``window_size: int``
  * ``drain_every_k: int``
  * ``thresholds: dict[str, ThresholdConfig]``
  * ``export: ExportConfig``
  * ``enabled: bool``  (global kill-switch)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Detector names GUARD knows how to build. A config naming anything else is rejected at
# load time rather than failing later when the engine tries to instantiate it.
KNOWN_DETECTORS: frozenset[str] = frozenset({"entropy", "divergence", "embedding"})

# Change-point methods the temporal layer (P1-06) implements.
ChangePointMethod = Literal["page_hinkley", "cusum"]


class _Strict(BaseModel):
    """Base model that forbids unknown keys, so typos in YAML surface as errors."""

    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_Strict):
    """One detector entry: which detector, whether it runs, and its parameters.

    ``params`` is intentionally an open mapping — each detector validates its own
    parameters when constructed. Keeping it loose here lets new detectors add knobs
    without changing this schema.
    """

    name: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known_name(cls, v: str) -> str:
        if v not in KNOWN_DETECTORS:
            raise ValueError(f"unknown detector {v!r}; known detectors: {sorted(KNOWN_DETECTORS)}")
        return v


class ThresholdConfig(_Strict):
    """Change-point test configuration for a single scalar metric stream.

    Maps onto the online tests in :mod:`guard.temporal` (P1-06). ``method`` selects the
    test; ``delta`` is its allowed-slack / magnitude parameter; ``threshold`` (lambda) is
    the decision boundary; ``debounce`` requires N consecutive flagged windows before an
    alert fires, and ``cooldown`` suppresses re-firing for N windows after an alert.
    """

    method: ChangePointMethod = "page_hinkley"
    threshold: float = Field(gt=0.0)
    delta: float = Field(default=0.0, ge=0.0)
    debounce: int = Field(default=1, ge=1)
    cooldown: int = Field(default=0, ge=0)


class ExportConfig(_Strict):
    """Where and whether scalar summaries are exported (Prometheus in Phase 1)."""

    enabled: bool = True
    prometheus_port: int = Field(default=9300, ge=1, le=65535)
    namespace: str = "guard"
    # Optional Pushgateway URL for batch/offline jobs; None = pull-based scraping only.
    push_gateway_url: str | None = None


class Config(_Strict):
    """Top-level GUARD runtime configuration."""

    enabled: bool = True
    window_size: int = Field(gt=0)
    drain_every_k: int = Field(gt=0)
    detectors: list[DetectorConfig] = Field(default_factory=list)
    thresholds: dict[str, ThresholdConfig] = Field(default_factory=dict)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @field_validator("detectors")
    @classmethod
    def _unique_detector_names(cls, v: list[DetectorConfig]) -> list[DetectorConfig]:
        names = [d.name for d in v]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate detector names: {dupes}")
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Validate a plain dict (e.g. parsed YAML) into a Config."""
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load and validate configuration from a YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for YAML dumping or round-tripping."""
        return self.model_dump(mode="python")

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
