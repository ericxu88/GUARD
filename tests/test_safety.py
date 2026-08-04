"""Phase 2 safety: exception firewall, circuit breaker, kill-switch (prime directive #1).

Everything here defends one invariant: monitoring is best-effort, so a broken detector must
degrade into *missing metrics*, never into a failed inference request. The tests are written
adversarially — each one names a way the serving path could be harmed (an escaped exception,
a detector that keeps burning CPU after it was disabled, a log flood, a swallowed
KeyboardInterrupt that makes the server unkillable) and asserts it cannot happen.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from typing import Any, NoReturn

import pytest

from guard.safety import (
    GUARD_DISABLE_ENV,
    CircuitBreaker,
    Firewall,
    KillSwitch,
    env_kill_switch_engaged,
)

LOGGER_NAME = "guard.safety"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class Boom(Exception):
    """A detector-specific failure; nothing about it may reach the caller."""


class CountingFailure:
    """Callable that raises every time and records how often it was actually invoked.

    The invocation count is the only way to prove a tripped breaker *stops calling* the
    component rather than merely re-swallowing its exception.
    """

    def __init__(self, exc: BaseException | None = None) -> None:
        self._lock = threading.Lock()  # the firewall calls fn outside its own lock
        self._calls = 0
        self._exc = exc if exc is not None else Boom("detector exploded")

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        with self._lock:
            self._calls += 1
        raise self._exc


class _CollectingHandler(logging.Handler):
    """Appends each record's level to a list — caplog is not safe to read mid-flight."""

    def __init__(self, sink: list[int]) -> None:
        super().__init__(level=logging.DEBUG)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record.levelno)


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Only records emitted by the firewall, so unrelated loggers cannot skew the counts."""
    return [r for r in caplog.records if r.name == LOGGER_NAME]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GUARD_DISABLED inherited from the developer's shell would silently pass tests."""
    monkeypatch.delenv(GUARD_DISABLE_ENV, raising=False)


@pytest.fixture
def _preempt_often() -> Iterator[None]:
    """Force the interpreter to switch threads mid-bytecode.

    ``self.failures += 1`` is three bytecodes; with the default 5 ms switch interval a
    missing lock almost never loses an update, so an unsynchronised counter would sail
    through a threaded test. Shrinking the interval makes the race actually happen.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


# --------------------------------------------------------------------------------------
# Firewall.call — the (ok, result) contract
# --------------------------------------------------------------------------------------
def test_call_returns_true_and_the_exact_result() -> None:
    fw = Firewall()
    sentinel = object()

    ok, result = fw.call("detector", lambda: sentinel)

    assert ok is True
    assert result is sentinel


def test_call_forwards_positional_and_keyword_arguments() -> None:
    fw = Firewall()

    def fn(a: int, b: int, *, scale: int = 1) -> int:
        return (a + b) * scale

    ok, result = fw.call("detector", fn, 2, 3, scale=10)

    assert (ok, result) == (True, 50)


def test_successful_call_records_no_failure_state() -> None:
    fw = Firewall(max_failures=1)

    fw.call("detector", lambda: 1)

    # A single tolerated failure would trip instantly, so a success must not be miscounted.
    assert fw.failure_count("detector") == 0
    assert fw.last_error("detector") is None
    assert fw.is_disabled("detector") is False
    assert fw.disabled == frozenset()


def test_call_returning_none_is_still_reported_as_success() -> None:
    fw = Firewall()

    ok, result = fw.call("detector", lambda: None)

    # (True, None) and (False, None) must be distinguishable via the flag alone.
    assert ok is True
    assert result is None
    assert fw.failure_count("detector") == 0


@pytest.mark.parametrize(
    "exc",
    [
        Boom("custom"),
        RuntimeError("cuda error"),
        ValueError("bad shape"),
        ZeroDivisionError("empty batch"),
        MemoryError("out of memory"),
        AssertionError("internal invariant"),
        StopIteration("exhausted"),
    ],
)
def test_call_swallows_every_exception_subclass(exc: Exception) -> None:
    """Detectors fail in creative ways; the serving path must see the same (False, None)."""
    fw = Firewall(max_failures=100)
    fn = CountingFailure(exc)

    ok, result = fw.call("detector", fn)

    assert ok is False
    assert result is None
    assert fn.calls == 1
    assert fw.last_error("detector") is exc


def test_call_swallows_an_exception_raised_in_a_finally_block() -> None:
    """A failure while unwinding a detector's teardown escapes as normally as any other."""
    fw = Firewall()

    def fn() -> int:
        try:
            return 1
        finally:
            raise Boom("raised during unwind")

    assert fw.call("detector", fn) == (False, None)


# --------------------------------------------------------------------------------------
# BaseException must propagate — a monitoring hook may not make the server unkillable
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, BaseException])
def test_non_exception_base_exceptions_propagate(exc_type: type[BaseException]) -> None:
    fw = Firewall()
    fn = CountingFailure(exc_type("shutting down"))

    with pytest.raises(exc_type):
        fw.call("detector", fn)

    assert fn.calls == 1


def test_propagating_base_exception_does_not_count_as_a_component_failure() -> None:
    """Ctrl-C during inference is the operator's fault, not the detector's — do not punish it."""
    fw = Firewall(max_failures=2)
    fn = CountingFailure(KeyboardInterrupt())

    for _ in range(3):
        with pytest.raises(KeyboardInterrupt):
            fw.call("detector", fn)

    assert fw.failure_count("detector") == 0
    assert fw.last_error("detector") is None
    assert fw.is_disabled("detector") is False
    assert fn.calls == 3


# --------------------------------------------------------------------------------------
# Circuit breaker — trips at exactly max_failures and then stops calling
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("max_failures", [1, 2, 3, 5])
def test_breaker_trips_at_exactly_max_failures(max_failures: int) -> None:
    fw = Firewall(max_failures=max_failures)
    fn = CountingFailure()

    for i in range(1, max_failures):
        fw.call("detector", fn)
        assert fw.is_disabled("detector") is False, f"tripped early after {i} failures"
        assert fw.failure_count("detector") == i

    fw.call("detector", fn)  # the failure that exhausts the budget
    assert fw.is_disabled("detector") is True
    assert fw.failure_count("detector") == max_failures
    assert fn.calls == max_failures


def test_tripped_component_is_never_invoked_again() -> None:
    fw = Firewall(max_failures=2)
    fn = CountingFailure()

    for _ in range(2):
        fw.call("detector", fn)
    assert fn.calls == 2

    # The whole point of the breaker: a permanently broken detector costs a dict lookup,
    # not an exception round-trip, on every subsequent inference step.
    for _ in range(50):
        assert fw.call("detector", fn) == (False, None)
    assert fn.calls == 2
    assert fw.failure_count("detector") == 2


def test_tripped_component_does_not_run_even_when_it_would_succeed() -> None:
    fw = Firewall(max_failures=1)
    calls: list[str] = []

    fw.call("detector", CountingFailure())
    ok, result = fw.call("detector", lambda: calls.append("ran"))

    assert (ok, result) == (False, None)
    assert calls == []


def test_breakers_are_independent_per_component() -> None:
    """One broken detector must not disable the healthy ones sharing the firewall."""
    fw = Firewall(max_failures=2)
    broken = CountingFailure()
    healthy_calls = 0

    def healthy() -> int:
        nonlocal healthy_calls
        healthy_calls += 1
        return 7

    for _ in range(4):
        fw.call("broken", broken)
        assert fw.call("healthy", healthy) == (True, 7)

    assert fw.disabled == frozenset({"broken"})
    assert fw.failure_count("healthy") == 0
    assert healthy_calls == 4
    assert broken.calls == 2


def test_disabled_set_lists_every_tripped_component() -> None:
    fw = Firewall(max_failures=1)

    fw.call("a", CountingFailure())
    fw.call("b", CountingFailure())
    fw.call("c", lambda: 0)

    assert fw.disabled == frozenset({"a", "b"})
    assert isinstance(fw.disabled, frozenset)


def test_last_error_tracks_the_most_recent_exception_identity() -> None:
    fw = Firewall(max_failures=10)
    first, second = Boom("first"), Boom("second")

    fw.call("detector", CountingFailure(first))
    assert fw.last_error("detector") is first

    fw.call("detector", CountingFailure(second))
    assert fw.last_error("detector") is second


def test_introspection_of_an_unknown_component_is_inert() -> None:
    fw = Firewall()

    assert fw.failure_count("never-seen") == 0
    assert fw.last_error("never-seen") is None
    assert fw.is_disabled("never-seen") is False
    assert fw.disabled == frozenset()


# --------------------------------------------------------------------------------------
# CircuitBreaker in isolation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, -100])
def test_circuit_breaker_rejects_non_positive_budget(bad: int) -> None:
    # max_failures=0 would mean "disabled before the first call", which is never intended.
    with pytest.raises(ValueError, match="max_failures"):
        CircuitBreaker("detector", max_failures=bad)


def test_circuit_breaker_counts_and_trips() -> None:
    cb = CircuitBreaker("detector", max_failures=2)
    exc = Boom("x")

    assert (cb.failures, cb.tripped, cb.last_error) == (0, False, None)
    cb.record_failure(exc)
    assert (cb.failures, cb.tripped, cb.last_error) == (1, False, exc)
    cb.record_failure(exc)
    assert (cb.failures, cb.tripped) == (2, True)


def test_circuit_breaker_stays_tripped_past_the_budget() -> None:
    cb = CircuitBreaker("detector", max_failures=1)
    for _ in range(3):
        cb.record_failure(Boom("x"))

    assert cb.tripped is True
    assert cb.failures == 3


def test_circuit_breaker_reset_clears_all_state() -> None:
    cb = CircuitBreaker("detector", max_failures=1)
    cb.record_failure(Boom("x"))
    cb.reset()

    assert (cb.failures, cb.tripped, cb.last_error) == (0, False, None)


# --------------------------------------------------------------------------------------
# Reset — re-arming after the operator fixed the cause
# --------------------------------------------------------------------------------------
def test_reset_component_rearms_and_calls_resume() -> None:
    fw = Firewall(max_failures=2)
    fn = CountingFailure()
    for _ in range(2):
        fw.call("detector", fn)
    assert fw.is_disabled("detector") is True

    fw.reset("detector")

    assert fw.is_disabled("detector") is False
    assert fw.failure_count("detector") == 0
    assert fw.last_error("detector") is None
    assert fw.disabled == frozenset()
    # A re-armed component is genuinely invoked again, and gets a fresh failure budget.
    assert fw.call("detector", lambda: 42) == (True, 42)
    fw.call("detector", fn)
    assert fn.calls == 3
    assert fw.is_disabled("detector") is False


def test_reset_one_component_leaves_the_others_disabled() -> None:
    fw = Firewall(max_failures=1)
    fw.call("a", CountingFailure())
    fw.call("b", CountingFailure())

    fw.reset("a")

    assert fw.disabled == frozenset({"b"})
    assert fw.failure_count("b") == 1


def test_reset_all_rearms_every_component() -> None:
    fw = Firewall(max_failures=1)
    a, b = CountingFailure(), CountingFailure()
    fw.call("a", a)
    fw.call("b", b)

    fw.reset()

    assert fw.disabled == frozenset()
    assert fw.failure_count("a") == 0
    assert fw.failure_count("b") == 0
    fw.call("a", a)
    fw.call("b", b)
    assert (a.calls, b.calls) == (2, 2)


def test_reset_of_an_unknown_component_is_a_noop() -> None:
    fw = Firewall(max_failures=1)
    fw.call("a", CountingFailure())

    fw.reset("never-seen")  # must not raise and must not resurrect "a"

    assert fw.disabled == frozenset({"a"})
    assert fw.failure_count("never-seen") == 0


# --------------------------------------------------------------------------------------
# Logging — loud once, then quiet, so a broken detector cannot flood the serving logs
# --------------------------------------------------------------------------------------
def test_first_failure_logs_exactly_one_error_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fw = Firewall(max_failures=3)
    exc = Boom("first failure")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        fw.call("detector", CountingFailure(exc))

    records = _records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert "detector" in records[0].getMessage()
    # Without exc_info the operator gets a name and no way to find the broken line.
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is exc


def test_intermediate_failures_are_silent(caplog: pytest.LogCaptureFixture) -> None:
    """Failures 2..max-1 must log nothing at any level — one per batch would flood the logs."""
    fw = Firewall(max_failures=5)
    fn = CountingFailure()

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        fw.call("detector", fn)  # failure 1 → ERROR
        caplog.clear()
        for _ in range(3):  # failures 2, 3, 4 → silence
            fw.call("detector", fn)
        assert _records(caplog) == []

        fw.call("detector", fn)  # failure 5 → trip → WARNING
        records = _records(caplog)

    assert len(records) == 1
    assert records[0].levelno == logging.WARNING


def test_trip_logs_a_single_warning_naming_the_component(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fw = Firewall(max_failures=2)
    fn = CountingFailure()

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        for _ in range(2):
            fw.call("detector", fn)
        caplog.clear()
        for _ in range(20):  # already tripped: not called, so nothing more to say
            fw.call("detector", fn)
        assert _records(caplog) == []

    assert fw.is_disabled("detector") is True


def test_trip_warning_mentions_reset_and_the_last_error(caplog: pytest.LogCaptureFixture) -> None:
    fw = Firewall(max_failures=1)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        fw.call("entropy", CountingFailure(Boom("cuda oom")))

    warnings = [r for r in _records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    # The operator must be able to act on the message alone: what broke and how to re-arm.
    assert "entropy" in message
    assert "reset" in message.lower()
    assert "cuda oom" in message


def test_max_failures_one_logs_both_the_error_and_the_trip(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With a budget of one, the first failure is also the trip — neither log may be dropped."""
    fw = Firewall(max_failures=1)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        fw.call("detector", CountingFailure())

    levels = [r.levelno for r in _records(caplog)]
    assert levels == [logging.ERROR, logging.WARNING]


def test_reset_rearms_the_first_failure_log(caplog: pytest.LogCaptureFixture) -> None:
    """After a reset the next failure is 'first' again — otherwise a recurrence goes unseen."""
    fw = Firewall(max_failures=3)
    fn = CountingFailure()

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        fw.call("detector", fn)
        fw.reset("detector")
        caplog.clear()
        fw.call("detector", fn)
        records = _records(caplog)

    assert len(records) == 1
    assert records[0].levelno == logging.ERROR


def test_failures_of_distinct_components_each_log_once(caplog: pytest.LogCaptureFixture) -> None:
    fw = Firewall(max_failures=3)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        fw.call("a", CountingFailure())
        fw.call("b", CountingFailure())
        records = _records(caplog)

    assert len(records) == 2
    assert all(r.levelno == logging.ERROR for r in records)


def test_custom_logger_is_used_instead_of_the_module_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Embedders route GUARD's noise into their own logger; the default must not leak."""
    custom = logging.getLogger("test.safety.custom")
    fw = Firewall(max_failures=1, log=custom)

    with caplog.at_level(logging.DEBUG, logger=custom.name):
        fw.call("detector", CountingFailure())

    assert _records(caplog) == []
    assert [r.levelno for r in caplog.records if r.name == custom.name] == [
        logging.ERROR,
        logging.WARNING,
    ]


def test_a_logger_that_itself_raises_does_not_break_inference() -> None:
    """The firewall accepts an arbitrary caller-supplied logger; it must not trust it."""

    class ExplodingLogger(logging.Logger):
        def error(self, *args: Any, **kwargs: Any) -> None:
            raise Boom("logging adapter is broken")

    fw = Firewall(max_failures=3, log=ExplodingLogger("test.safety.exploding"))

    assert fw.call("detector", CountingFailure()) == (False, None)


def test_firewall_rejects_an_impossible_failure_budget_at_construction() -> None:
    # A zero budget used to be accepted and then blow up inside CircuitBreaker.__init__ —
    # from within `call`'s except handler, so the mechanism that exists to keep exceptions
    # away from inference was itself the thing raising into it. Reject it up front, where a
    # misconfiguration is a startup error rather than a per-request one.
    with pytest.raises(ValueError, match="max_failures"):
        Firewall(max_failures=0)
    with pytest.raises(ValueError, match="max_failures"):
        Firewall(max_failures=-1)
    # And the smallest legal budget still contains everything it is given.
    fw = Firewall(max_failures=1)
    assert fw.call("detector", CountingFailure()) == (False, None)
    assert fw.is_disabled("detector") is True


# --------------------------------------------------------------------------------------
# KillSwitch
# --------------------------------------------------------------------------------------
def test_kill_switch_defaults_to_enabled() -> None:
    assert KillSwitch().enabled is True


def test_kill_switch_can_start_disabled() -> None:
    assert KillSwitch(enabled=False).enabled is False


def test_kill_switch_enable_disable_round_trip() -> None:
    ks = KillSwitch()

    ks.disable()
    assert ks.enabled is False
    ks.disable()  # idempotent
    assert ks.enabled is False
    ks.enable()
    assert ks.enabled is True
    ks.enable()
    assert ks.enabled is True


@pytest.mark.parametrize("value", ["1", "true", "YES", "on", "True", " on ", "ON", "Yes"])
def test_env_var_truthy_values_force_disabled(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GUARD_DISABLE_ENV, value)
    ks = KillSwitch(enabled=True)

    assert env_kill_switch_engaged() is True
    # The env var wins over the runtime flag: an operator must be able to neutralise GUARD
    # without redeploying, even if application code calls enable() on startup.
    assert ks.enabled is False
    ks.enable()
    assert ks.enabled is False


@pytest.mark.parametrize("value", ["0", "", "false", "no", "off", "2", "disabled", " "])
def test_env_var_falsy_values_leave_the_flag_in_charge(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GUARD_DISABLE_ENV, value)

    assert env_kill_switch_engaged() is False
    assert KillSwitch(enabled=True).enabled is True
    assert KillSwitch(enabled=False).enabled is False


def test_env_var_absent_means_not_engaged() -> None:
    assert env_kill_switch_engaged() is False


def test_env_var_is_read_on_every_access_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caching would mean an operator's flip only takes effect after a process restart."""
    ks = KillSwitch(enabled=True)
    assert ks.enabled is True

    monkeypatch.setenv(GUARD_DISABLE_ENV, "1")
    assert ks.enabled is False

    monkeypatch.delenv(GUARD_DISABLE_ENV)
    assert ks.enabled is True


def test_env_var_does_not_mutate_the_runtime_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env kill-switch masks the flag; clearing it must restore the operator's setting."""
    ks = KillSwitch(enabled=False)
    monkeypatch.setenv(GUARD_DISABLE_ENV, "yes")
    assert ks.enabled is False

    monkeypatch.delenv(GUARD_DISABLE_ENV)
    assert ks.enabled is False  # still disabled by the flag, not re-enabled by the env change


def test_kill_switch_survives_concurrent_flips() -> None:
    """Reads and flips race on every request; they must not deadlock, raise, or return junk."""
    ks = KillSwitch()
    errors: list[BaseException] = []
    seen: list[bool] = []
    seen_lock = threading.Lock()
    n_threads, iterations = 8, 200
    barrier = threading.Barrier(n_threads)

    def worker(index: int) -> None:
        local: list[bool] = []
        try:
            barrier.wait()
            for i in range(iterations):
                if (index + i) % 2 == 0:
                    ks.enable()
                else:
                    ks.disable()
                local.append(ks.enabled)
        except BaseException as exc:  # re-asserted on the main thread
            errors.append(exc)
        with seen_lock:
            seen.extend(local)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        # A non-reentrant lock taken twice (e.g. `enabled` calling `disable`) hangs here.
        assert not t.is_alive(), "KillSwitch deadlocked under concurrent access"

    assert errors == []
    assert len(seen) == n_threads * iterations
    assert all(isinstance(v, bool) for v in seen)
    ks.enable()
    assert ks.enabled is True


# --------------------------------------------------------------------------------------
# Firewall under concurrency — a shared firewall is called from every serving thread
# --------------------------------------------------------------------------------------
@pytest.mark.usefixtures("_preempt_often")
def test_concurrent_failures_are_counted_exactly() -> None:
    """A lost increment would let a broken detector run forever without ever tripping."""
    n_threads, per_thread = 8, 200
    fw = Firewall(max_failures=10**9)  # effectively untrippable: every call reaches the counter
    fn = CountingFailure()
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(per_thread):
                assert fw.call("detector", fn) == (False, None)
        except BaseException as exc:  # re-asserted on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive(), "Firewall deadlocked under concurrent access"

    assert errors == []
    assert fn.calls == n_threads * per_thread
    assert fw.failure_count("detector") == n_threads * per_thread
    assert fw.is_disabled("detector") is False


@pytest.mark.usefixtures("_preempt_often")
def test_concurrent_trip_logs_the_warning_exactly_once(caplog: pytest.LogCaptureFixture) -> None:
    """Racing threads must not each emit a trip banner — that is the log flood we prevent."""
    n_threads, per_thread = 8, 40
    fw = Firewall(max_failures=3)
    fn = CountingFailure()
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        for _ in range(per_thread):
            fw.call("detector", fn)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
            assert not t.is_alive()

    records = _records(caplog)
    assert [r.levelno for r in records].count(logging.ERROR) == 1
    assert [r.levelno for r in records].count(logging.WARNING) == 1
    assert fw.is_disabled("detector") is True
    # Once tripped the component is quarantined. Only calls that passed the tripped-check
    # before the trip was recorded can run: at most one in flight per thread, plus the
    # max_failures that did the tripping — nowhere near the n_threads * per_thread attempts.
    assert 3 <= fn.calls <= n_threads + 3


@pytest.mark.usefixtures("_preempt_often")
def test_simultaneous_first_failure_is_recorded_and_logged_exactly_once() -> None:
    """Every serving thread hits an unseen component at the same instant on the first bad batch.

    Without the lock around breaker creation two threads can both find no breaker, both build
    one, and both consider themselves the "first" failure — losing failure counts (so the
    breaker never trips) and duplicating the ERROR banner. One round is a coin flip; the
    rounds make the race certain to show up.
    """
    n_threads, rounds = 8, 150

    for _ in range(rounds):
        log = logging.Logger("test.safety.simultaneous")
        seen: list[int] = []
        log.addHandler(_CollectingHandler(seen))
        log.setLevel(logging.DEBUG)
        fw = Firewall(max_failures=n_threads * 10, log=log)  # far too high to trip here
        gate = threading.Barrier(n_threads)

        def fn(gate: threading.Barrier = gate) -> NoReturn:
            gate.wait(timeout=10.0)  # all threads raise in the same instant
            raise Boom("simultaneous")

        threads = [
            threading.Thread(target=fw.call, args=("detector", fn)) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive()

        assert fw.failure_count("detector") == n_threads
        assert seen.count(logging.ERROR) == 1


@pytest.mark.usefixtures("_preempt_often")
def test_concurrent_mixed_components_keep_independent_state() -> None:
    """Interleaved healthy and broken components must not corrupt each other's counters."""
    n_threads, per_thread = 6, 150
    fw = Firewall(max_failures=10**9)
    broken = CountingFailure()
    healthy_hits = 0
    healthy_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def healthy() -> int:
        nonlocal healthy_hits
        with healthy_lock:
            healthy_hits += 1
        return 1

    def worker() -> None:
        barrier.wait()
        for _ in range(per_thread):
            fw.call("broken", broken)
            assert fw.call("healthy", healthy) == (True, 1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive()

    assert fw.failure_count("broken") == n_threads * per_thread
    assert fw.failure_count("healthy") == 0
    assert healthy_hits == n_threads * per_thread
    assert fw.disabled == frozenset()
