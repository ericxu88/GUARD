"""Page-Hinkley change-point test (P1-06).

Detects an abrupt *upward* shift in the mean of a scalar stream — the relevant direction for
GUARD's drift metrics (entropy, divergence, Mahalanobis), which rise when the model drifts.

The cumulative statistic tracks how far the stream has run above its own running mean
(minus an allowed slack ``delta``):

    x̄_t  = running mean
    m_t  = Σ_{i≤t} (x_i − x̄_i − delta)
    PH_t = m_t − min_{i≤t} m_i

An alarm is raised when ``PH_t`` exceeds ``threshold`` (subject to debounce/cooldown). One
scalar of cumulative state ⇒ O(1) per step, deterministic given the input sequence.
"""

from __future__ import annotations

from guard.config import ThresholdConfig
from guard.temporal.base import AlertGate, ChangePointResult


class PageHinkley:
    """Online one-sided (upward) Page-Hinkley test with debounce + cooldown.

    Args:
        threshold: alarm boundary ``λ`` for the Page-Hinkley statistic (must be > 0).
        delta: allowed slack / magnitude tolerance subtracted each step (>= 0).
        debounce: consecutive crossings required before an alarm fires (>= 1).
        cooldown: steps to suppress re-alarming after a firing (>= 0).
    """

    def __init__(
        self,
        threshold: float,
        delta: float = 0.0,
        debounce: int = 1,
        cooldown: int = 0,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        self.threshold = threshold
        self.delta = delta
        self._gate = AlertGate(debounce, cooldown)
        self.reset()

    @classmethod
    def from_config(cls, cfg: ThresholdConfig) -> PageHinkley:
        """Build from a :class:`ThresholdConfig` (``method`` is assumed 'page_hinkley')."""
        return cls(
            threshold=cfg.threshold,
            delta=cfg.delta,
            debounce=cfg.debounce,
            cooldown=cfg.cooldown,
        )

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._n = 0
        self._x_mean = 0.0
        self._cum = 0.0
        self._cum_min = 0.0
        self._gate.reset()

    def update(self, x: float) -> ChangePointResult:
        """Feed one scalar; return the statistic and whether an alarm fires this step."""
        self._n += 1
        self._x_mean += (x - self._x_mean) / self._n
        self._cum += x - self._x_mean - self.delta
        self._cum_min = min(self._cum_min, self._cum)
        statistic = self._cum - self._cum_min

        alarm = self._gate.update(statistic > self.threshold)
        if alarm:
            # Reset so evidence must re-accumulate; this is what lets an alert clear.
            self.reset()
        return ChangePointResult(alarm=alarm, statistic=statistic)
