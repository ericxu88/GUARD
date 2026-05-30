"""CUSUM change-point test (P1-06).

A one-sided upper cumulative-sum control chart over a scalar stream. It accumulates the
amount by which the stream runs above a reference level plus a slack ``delta`` (the CUSUM
``k``), floored at zero so quiet periods decay back toward 0:

    S_t = max(0, S_{t-1} + (x_t − target_t − delta))

``target`` is either a fixed in-control level (textbook CUSUM) or, when unset, the running
mean of the stream so far (unsupervised, self-calibrating). An alarm fires when ``S_t``
exceeds ``threshold`` (the CUSUM ``h``), subject to debounce/cooldown. O(1) state per step,
deterministic given the input sequence.

**One alarm per regime transition (target=None).** When the target is the running mean the
detector fires once when the shift first accumulates enough evidence. After the alarm the
statistic resets to 0, but the running mean continues to track the stream — so if the
stream remains elevated the running mean converges to the new level and the CUSUM increments
become non-positive again, producing no further alarms. This is by design: the unsupervised
mode signals that a level change occurred; it does not sustain an alert during a plateau.
For continuous alarming during a sustained shift, supply a fixed ``target``.
"""

from __future__ import annotations

from guard.config import ThresholdConfig
from guard.temporal.base import AlertGate, ChangePointResult


class Cusum:
    """Online one-sided (upper) CUSUM test with debounce + cooldown.

    Args:
        threshold: decision interval ``h`` (must be > 0).
        delta: reference slack ``k`` subtracted each step (>= 0).
        target: fixed in-control mean; if ``None``, the running mean is used instead.
        debounce: consecutive crossings required before an alarm fires (>= 1).
        cooldown: steps to suppress re-alarming after a firing (>= 0).
    """

    def __init__(
        self,
        threshold: float,
        delta: float = 0.0,
        target: float | None = None,
        debounce: int = 1,
        cooldown: int = 0,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        self.threshold = threshold
        self.delta = delta
        self.target = target
        self._gate = AlertGate(debounce, cooldown)
        self.reset()

    @classmethod
    def from_config(cls, cfg: ThresholdConfig) -> Cusum:
        """Build from a :class:`ThresholdConfig` (``method`` is assumed 'cusum')."""
        return cls(
            threshold=cfg.threshold,
            delta=cfg.delta,
            debounce=cfg.debounce,
            cooldown=cfg.cooldown,
        )

    def _reset_statistic(self) -> None:
        """Reset the CUSUM statistic only; running mean and gate cooldown are preserved.

        Preserving the running mean means the detector re-evaluates future observations
        relative to the history it has seen, not from a cold start. Gate cooldown set on
        this alarm step is not erased (see ``update``).
        """
        self._s = 0.0

    def reset(self) -> None:
        """Clear all accumulated state including running mean and alert gate."""
        self._n = 0
        self._x_mean = 0.0
        self._s = 0.0
        self._gate.reset()

    def update(self, x: float) -> ChangePointResult:
        """Feed one scalar; return the statistic and whether an alarm fires this step."""
        self._n += 1
        self._x_mean += (x - self._x_mean) / self._n
        target = self._x_mean if self.target is None else self.target

        self._s = max(0.0, self._s + (x - target - self.delta))
        statistic = self._s

        alarm = self._gate.update(statistic > self.threshold)
        if alarm:
            # Only reset the statistic — _gate already set its cooldown on this step;
            # calling reset() would erase it before the next update() can honour it.
            self._reset_statistic()
        return ChangePointResult(alarm=alarm, statistic=statistic)
