"""P1-06: Page-Hinkley & CUSUM — false-positive guard, detection lag, recovery, debounce."""

from __future__ import annotations

import numpy as np
import pytest

from guard.config import ThresholdConfig
from guard.temporal import Cusum, PageHinkley, from_threshold_config
from guard.temporal.base import AlertGate, ChangePointResult

# Documented detection lag budget: an alarm must fire within this many steps of a shift.
DETECTION_LAG = 60

# Calibration: slack `delta` gives the statistic a negative per-step drift on stationary
# noise (σ=1), so upward excursions to THRESH are exponentially rare — the false-positive
# guard — while an injected shift of +4 still nets ~3/step and crosses quickly.
THRESH = 10.0
DELTA = 1.0


def _stationary(n: int, seed: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=n)


def _first_alarm(test: PageHinkley | Cusum, stream: np.ndarray) -> int | None:
    for i, x in enumerate(stream):
        if test.update(float(x)).alarm:
            return i
    return None


def _alarm_indices(test: PageHinkley | Cusum, stream: np.ndarray) -> list[int]:
    out = []
    for i, x in enumerate(stream):
        if test.update(float(x)).alarm:
            out.append(i)
    return out


# --------------------------------------------------------------------------------------
# False-positive guard on stationary noise
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_page_hinkley_no_false_positive_on_stationary(seed: int) -> None:
    ph = PageHinkley(threshold=THRESH, delta=DELTA)
    assert _first_alarm(ph, _stationary(2000, seed)) is None


@pytest.mark.parametrize("seed", range(8))
def test_cusum_no_false_positive_on_stationary(seed: int) -> None:
    cs = Cusum(threshold=THRESH, delta=DELTA)
    assert _first_alarm(cs, _stationary(2000, seed)) is None


# --------------------------------------------------------------------------------------
# Detection of an injected mean shift, within the documented lag
# --------------------------------------------------------------------------------------
def _shift_stream(seed: int, pre: int = 300, post: int = 300, shift: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=pre)
    b = rng.normal(shift, 1.0, size=post)
    return np.concatenate([a, b])


@pytest.mark.parametrize("seed", range(5))
def test_page_hinkley_detects_shift_within_lag(seed: int) -> None:
    pre = 300
    ph = PageHinkley(threshold=THRESH, delta=DELTA)
    idx = _first_alarm(ph, _shift_stream(seed, pre=pre))
    assert idx is not None
    assert pre <= idx <= pre + DETECTION_LAG


@pytest.mark.parametrize("seed", range(5))
def test_cusum_detects_shift_within_lag(seed: int) -> None:
    pre = 300
    cs = Cusum(threshold=THRESH, delta=DELTA)
    idx = _first_alarm(cs, _shift_stream(seed, pre=pre))
    assert idx is not None
    assert pre <= idx <= pre + DETECTION_LAG


# --------------------------------------------------------------------------------------
# Alert clears after recovery
# --------------------------------------------------------------------------------------
def _shift_then_recover(seed: int) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, size=300)
    shifted = rng.normal(5.0, 1.0, size=300)
    recovered = rng.normal(0.0, 1.0, size=400)
    stream = np.concatenate([clean, shifted, recovered])
    recovery_start = 600
    return stream, len(clean), recovery_start


@pytest.mark.parametrize("cls", [PageHinkley, Cusum])
def test_alarm_clears_after_recovery(cls: type[PageHinkley | Cusum]) -> None:
    stream, shift_start, recovery_start = _shift_then_recover(seed=0)
    test = cls(threshold=THRESH, delta=DELTA, cooldown=20)
    alarms = _alarm_indices(test, stream)

    # Fires during the shifted segment...
    assert any(shift_start <= i < recovery_start for i in alarms)
    # ...and goes quiet well after recovery (allow a margin for cooldown + decay).
    late = [i for i in alarms if i >= recovery_start + 100]
    assert late == []


# --------------------------------------------------------------------------------------
# Debounce, cooldown, determinism
# --------------------------------------------------------------------------------------
def test_debounce_requires_consecutive_crossings() -> None:
    gate = AlertGate(debounce=3, cooldown=0)
    # Isolated crossings never reach 3-in-a-row.
    assert [gate.update(b) for b in [True, False, True, False, True]] == [False] * 5
    gate.reset()
    # Three consecutive crossings fire exactly once, on the third.
    assert [gate.update(True) for _ in range(3)] == [False, False, True]


def test_cooldown_suppresses_refiring() -> None:
    gate = AlertGate(debounce=1, cooldown=5)
    out = [gate.update(True) for _ in range(8)]
    # Step 0 fires; next 5 are cooled down; then it can fire again.
    assert out == [True, False, False, False, False, False, True, False]


def test_deterministic_given_seed() -> None:
    stream = _shift_stream(seed=42)
    a = _alarm_indices(PageHinkley(threshold=THRESH, delta=DELTA), stream)
    b = _alarm_indices(PageHinkley(threshold=THRESH, delta=DELTA), stream)
    assert a == b


def test_o1_state_does_not_grow() -> None:
    # State is a fixed set of scalar attributes regardless of stream length.
    ph = PageHinkley(threshold=THRESH, delta=DELTA)
    for x in _stationary(5000, seed=1):
        ph.update(float(x))
    assert set(vars(ph)) == {"threshold", "delta", "_gate", "_n", "_x_mean", "_cum", "_cum_min"}


# --------------------------------------------------------------------------------------
# Config integration + result type
# --------------------------------------------------------------------------------------
def test_from_threshold_config_dispatches_method() -> None:
    ph = from_threshold_config(ThresholdConfig(method="page_hinkley", threshold=8.0, delta=0.5))
    cs = from_threshold_config(ThresholdConfig(method="cusum", threshold=8.0, delta=0.5))
    assert isinstance(ph, PageHinkley)
    assert isinstance(cs, Cusum)
    assert ph.threshold == 8.0 and ph.delta == 0.5


def test_update_returns_changepoint_result() -> None:
    ph = PageHinkley(threshold=8.0)
    r = ph.update(0.0)
    assert isinstance(r, ChangePointResult)
    assert r.alarm is False
    assert isinstance(r.statistic, float)


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        PageHinkley(threshold=0.0)
    with pytest.raises(ValueError):
        Cusum(threshold=-1.0)
    with pytest.raises(ValueError):
        AlertGate(debounce=0)


# --------------------------------------------------------------------------------------
# Regression tests for confirmed bugs
# --------------------------------------------------------------------------------------
def test_cooldown_honoured_through_page_hinkley() -> None:
    # Bug: reset() called on alarm wiped the gate's _cooldown_left before the next step,
    # making cooldown silently ignored. Fix: _reset_statistic() preserves gate cooldown.
    # Use alternating 0/large values so PH running mean lags and statistic accumulates.
    ph = PageHinkley(threshold=1.0, delta=0.0, cooldown=5)
    alarm_fired = False
    for i in range(200):
        x = 0.0 if i % 2 == 0 else 8.0  # alternating; running mean lags → PH grows
        if ph.update(x).alarm:
            alarm_fired = True
            # Immediately after alarm the gate should still have cooldown set,
            # not wiped to 0 by the old buggy reset().
            assert ph._gate._cooldown_left == 5, (
                f"gate cooldown was wiped to {ph._gate._cooldown_left} "
                f"(expected 5); cooldown is being erased by reset()"
            )
            break
    assert alarm_fired, "alarm never fired — check threshold/stream calibration"


def test_cooldown_honoured_through_cusum() -> None:
    # Same bug in Cusum: reset() wiped gate cooldown immediately after it was set.
    cs = Cusum(threshold=1.0, delta=0.0, cooldown=5, target=0.0)
    alarm_fired = False
    for _ in range(200):
        if cs.update(5.0).alarm:  # fixed target=0 so large x accumulates S reliably
            alarm_fired = True
            assert cs._gate._cooldown_left == 5, (
                f"gate cooldown was wiped to {cs._gate._cooldown_left} "
                f"(expected 5); cooldown is being erased by reset()"
            )
            break
    assert alarm_fired, "alarm never fired"


def test_cusum_running_mean_preserved_across_alarm() -> None:
    # Bug: reset() zeroed _n and _x_mean on alarm, losing the running mean.
    # Fix: _reset_statistic() only resets _s; _x_mean and _n persist across alarms.
    cs = Cusum(threshold=0.01, delta=0.0, target=None)
    # Build a running mean near 0 over 100 stationary steps.
    for _ in range(100):
        cs.update(0.0)
    assert abs(cs._x_mean) < 0.01
    assert cs._n == 100
    # Inject a large spike to trigger an alarm.
    cs.update(10.0)
    # After alarm: _n and _x_mean must NOT be zeroed.
    assert cs._n > 0, "_n was reset to 0 on alarm (running mean lost)"
    assert abs(cs._x_mean) < 2.0, f"_x_mean was reset to near-0 on alarm; got {cs._x_mean}"
