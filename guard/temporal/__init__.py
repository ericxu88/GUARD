from guard.config import ThresholdConfig
from guard.temporal.base import AlertGate, ChangePointResult, ChangePointTest
from guard.temporal.cusum import Cusum
from guard.temporal.page_hinkley import PageHinkley


def from_threshold_config(cfg: ThresholdConfig) -> ChangePointTest:
    """Construct the change-point test named by ``cfg.method``."""
    if cfg.method == "page_hinkley":
        return PageHinkley.from_config(cfg)
    if cfg.method == "cusum":
        return Cusum.from_config(cfg)
    raise ValueError(f"unknown change-point method: {cfg.method!r}")


__all__ = [
    "AlertGate",
    "ChangePointResult",
    "ChangePointTest",
    "Cusum",
    "PageHinkley",
    "from_threshold_config",
]
