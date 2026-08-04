"""Detectors: one batch of model outputs → a handful of device-resident scalars."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from guard.detectors.base import Detector, DetectorResult, DeviceCache
from guard.detectors.divergence import (
    DivergenceDetector,
    js_divergence,
    kl_divergence,
    mean_softmax,
)
from guard.detectors.embedding import (
    EmbeddingDriftDetector,
    WelfordCovariance,
    mahalanobis,
)
from guard.detectors.entropy import EntropyDetector, softmax_entropy

if TYPE_CHECKING:  # import-cycle safety: guard.baseline imports guard.detectors.entropy
    from guard.baseline.schema import Baseline
    from guard.config import Config


def build_detectors(config: Config, baseline: Baseline) -> list[Detector]:
    """Instantiate the detectors a :class:`~guard.config.Config` enables.

    Each entry's ``params`` is passed straight to the detector's constructor, so a bad knob
    surfaces here — at startup, with the offending detector named — instead of as a
    ``TypeError`` inside the first ``observe()`` call, where the firewall would swallow it
    and the operator would see nothing but a metric that never appears.

    Args:
        config: validated runtime configuration.
        baseline: the reference every detector compares against.

    Returns:
        Detector instances in config order. Entries with ``enabled: false`` are skipped.
    """
    builders: dict[str, Callable[..., Detector]] = {
        "entropy": lambda **kw: EntropyDetector(baseline=baseline, **kw),
        "divergence": lambda **kw: DivergenceDetector(baseline, **kw),
        "embedding": lambda **kw: EmbeddingDriftDetector(baseline, **kw),
    }
    detectors: list[Detector] = []
    for entry in config.detectors:
        if not entry.enabled:
            continue
        try:
            detectors.append(builders[entry.name](**entry.params))
        except TypeError as exc:
            raise ValueError(
                f"detector {entry.name!r} rejected its configured params {entry.params!r}: {exc}"
            ) from exc
    return detectors


__all__ = [
    "Detector",
    "DetectorResult",
    "DeviceCache",
    "DivergenceDetector",
    "EmbeddingDriftDetector",
    "EntropyDetector",
    "WelfordCovariance",
    "build_detectors",
    "js_divergence",
    "kl_divergence",
    "mahalanobis",
    "mean_softmax",
    "softmax_entropy",
]
