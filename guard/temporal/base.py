"""Shared scaffolding for online change-point tests (P1-06).

A change-point test consumes one scalar per window (a detector metric) and decides when a
sustained change is statistically real rather than noise. These run on the CPU aggregation
thread, one scalar at a time, so they are plain Python floats with O(1) state — no torch,
no allocation growth.

Every test wraps its raw threshold crossing in an :class:`AlertGate` that adds **debounce**
(require N consecutive crossings before alarming, to suppress single-sample spikes) and
**cooldown** (suppress re-alarming for M steps after a firing, to prevent alert storms). On
a firing the test resets its statistic so it must re-accumulate evidence — which is what
makes an alert *clear* once the stream returns to baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChangePointResult:
    """Outcome of feeding one scalar to a change-point test.

    Attributes:
        alarm: True exactly on the step a (debounced, non-cooled-down) change is declared.
        statistic: the test statistic at this step (for export / debugging).
    """

    alarm: bool
    statistic: float


class AlertGate:
    """Debounce + cooldown state machine over a raw boolean crossing stream.

    O(1) state. Deterministic. Returns True only on the step an alarm *event* fires.
    """

    def __init__(self, debounce: int = 1, cooldown: int = 0) -> None:
        if debounce < 1:
            raise ValueError(f"debounce must be >= 1, got {debounce}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {cooldown}")
        self.debounce = debounce
        self.cooldown = cooldown
        self._consecutive = 0
        self._cooldown_left = 0

    def reset(self) -> None:
        self._consecutive = 0
        self._cooldown_left = 0

    def update(self, raw: bool) -> bool:
        """Advance one step with the raw crossing flag; return whether an alarm fires."""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._consecutive = 0
            return False
        self._consecutive = self._consecutive + 1 if raw else 0
        if self._consecutive >= self.debounce:
            self._consecutive = 0
            self._cooldown_left = self.cooldown
            return True
        return False


@runtime_checkable
class ChangePointTest(Protocol):
    """Structural interface shared by Page-Hinkley and CUSUM."""

    def update(self, x: float) -> ChangePointResult: ...

    def reset(self) -> None: ...
