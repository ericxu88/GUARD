"""Exception firewall, circuit breaker, and kill-switch (prime directive #1).

Monitoring is best-effort observability, never a hard dependency of serving. This module
is the single mechanism that enforces that: **no exception raised by a detector, an
export path, or the engine itself may ever reach the inference caller.**

Three layers, in order of escalation:

1. :class:`Firewall` — runs a callable and swallows anything it raises, logging the first
   failure per component with a traceback and staying quiet afterwards so a persistently
   broken detector cannot flood the serving logs.
2. :class:`CircuitBreaker` — after ``max_failures`` failures a component is *disabled*:
   the firewall stops calling it entirely, so a detector that fails on every batch costs
   one dict lookup per step instead of an exception round-trip.
3. :class:`KillSwitch` — a process-wide, thread-safe on/off flag that also honours the
   ``GUARD_DISABLED`` environment variable, so an operator can neutralise GUARD in a
   running deployment (via env + restart) or at runtime (via the flag) without redeploying
   a new image.

The breaker counts *total* failures rather than consecutive ones. A detector that fails
intermittently is as untrustworthy as one that fails always — its metric stream has holes,
and a drift verdict computed from a holed stream is worse than no verdict. Call
:meth:`Firewall.reset` after fixing the underlying cause to re-arm a component.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("guard.safety")

# Set to "1"/"true"/"yes"/"on" (case-insensitive) to disable GUARD process-wide.
GUARD_DISABLE_ENV = "GUARD_DISABLED"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

T = TypeVar("T")


def env_kill_switch_engaged() -> bool:
    """True when the ``GUARD_DISABLED`` environment variable is set to a truthy value."""
    return os.environ.get(GUARD_DISABLE_ENV, "").strip().lower() in _TRUTHY


class KillSwitch:
    """Thread-safe global enable/disable flag for a GUARD instance.

    The environment variable is checked on every read, not cached, so a test (or an
    operator with a debugger) can flip it at runtime. The runtime flag and the environment
    variable are ANDed: GUARD runs only when the flag is set *and* the env kill-switch is
    not engaged.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._lock = threading.Lock()
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """True when monitoring should run."""
        with self._lock:
            flag = self._enabled
        return flag and not env_kill_switch_engaged()

    def enable(self) -> None:
        """Re-enable monitoring (still subject to the environment kill-switch)."""
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable monitoring for this instance. Takes effect on the next ``observe``."""
        with self._lock:
            self._enabled = False


class CircuitBreaker:
    """Failure counter for one named component; trips permanently at ``max_failures``.

    Args:
        name: component identifier used in log messages (e.g. a detector name).
        max_failures: number of failures tolerated before the component is disabled.
            Must be >= 1.
    """

    def __init__(self, name: str, max_failures: int = 3) -> None:
        if max_failures < 1:
            raise ValueError(f"max_failures must be >= 1, got {max_failures}")
        self.name = name
        self.max_failures = max_failures
        self.failures = 0
        self.tripped = False
        self.last_error: BaseException | None = None

    def record_failure(self, exc: BaseException) -> None:
        """Count a failure and trip the breaker once the budget is exhausted."""
        self.failures += 1
        self.last_error = exc
        if self.failures >= self.max_failures:
            self.tripped = True

    def reset(self) -> None:
        """Clear the failure count and re-arm the component."""
        self.failures = 0
        self.tripped = False
        self.last_error = None


class Firewall:
    """Runs untrusted callables so that no exception escapes to the inference path.

    Usage::

        fw = Firewall(max_failures=3)
        ok, result = fw.call("entropy", detector.compute, logits, embeddings)
        if not ok:
            ...  # metric unavailable this step; inference already returned

    Args:
        max_failures: failures tolerated per component before it is disabled.
        log: logger to report through; defaults to ``guard.safety``.
    """

    def __init__(self, max_failures: int = 3, log: logging.Logger | None = None) -> None:
        # Validated here, not lazily in CircuitBreaker: an invalid budget would otherwise
        # raise from inside `call`'s except handler, so the very mechanism that exists to
        # stop exceptions reaching inference would be the thing that raised into it.
        if max_failures < 1:
            raise ValueError(f"max_failures must be >= 1, got {max_failures}")
        self.max_failures = max_failures
        self._log = log if log is not None else logger
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    # ── introspection ────────────────────────────────────────────────────────

    def is_disabled(self, component: str) -> bool:
        """True when ``component`` has tripped its breaker and is no longer called."""
        with self._lock:
            breaker = self._breakers.get(component)
            return breaker is not None and breaker.tripped

    @property
    def disabled(self) -> frozenset[str]:
        """Names of all components currently disabled by a tripped breaker."""
        with self._lock:
            return frozenset(name for name, b in self._breakers.items() if b.tripped)

    def failure_count(self, component: str) -> int:
        """Number of failures recorded for ``component``."""
        with self._lock:
            breaker = self._breakers.get(component)
            return breaker.failures if breaker is not None else 0

    def last_error(self, component: str) -> BaseException | None:
        """The most recent exception raised by ``component``, if any."""
        with self._lock:
            breaker = self._breakers.get(component)
            return breaker.last_error if breaker is not None else None

    def reset(self, component: str | None = None) -> None:
        """Re-arm one component, or all of them when ``component`` is ``None``."""
        with self._lock:
            if component is None:
                for breaker in self._breakers.values():
                    breaker.reset()
            elif component in self._breakers:
                self._breakers[component].reset()

    # ── the firewall itself ──────────────────────────────────────────────────

    def call(
        self,
        component: str,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[bool, T | None]:
        """Invoke ``fn`` behind the firewall.

        Returns ``(True, result)`` on success and ``(False, None)`` if the component is
        disabled or the call raised. Never raises — that is the whole point.

        ``BaseException`` subclasses that are not ``Exception`` (``KeyboardInterrupt``,
        ``SystemExit``) are deliberately **not** caught: those signal that the whole
        process is going down, and swallowing them inside a monitoring hook would make the
        server unkillable.
        """
        with self._lock:
            breaker = self._breakers.get(component)
            if breaker is not None and breaker.tripped:
                return False, None

        try:
            return True, fn(*args, **kwargs)
        except Exception as exc:
            self._record(component, exc)
            return False, None

    def _record(self, component: str, exc: Exception) -> None:
        """Count the failure, and log the first one and the trip event only."""
        with self._lock:
            breaker = self._breakers.get(component)
            if breaker is None:
                breaker = CircuitBreaker(component, self.max_failures)
                self._breakers[component] = breaker
            first = breaker.failures == 0
            breaker.record_failure(exc)
            tripped_now = breaker.tripped and breaker.failures == breaker.max_failures

        # `log` is caller-supplied, and a structured-logging adapter or a custom filter can
        # itself raise. Reporting a failure must never become a new way to fail into the
        # inference path, so the reporting is contained too.
        try:
            if first:
                self._log.error(
                    "GUARD component %r failed; monitoring degraded, inference unaffected. "
                    "Further identical failures are logged only on circuit-breaker trip.",
                    component,
                    exc_info=exc,
                )
            if tripped_now:
                self._log.warning(
                    "GUARD disabled component %r after %d failures (last error: %r). "
                    "Call Firewall.reset(%r) to re-arm.",
                    component,
                    self.max_failures,
                    exc,
                    component,
                )
        except Exception:
            pass  # a broken logger must not defeat the firewall
