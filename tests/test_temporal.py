"""P1-06: Page-Hinkley & CUSUM — false-positive guard, detection lag, recovery, debounce."""

from __future__ import annotations

import math

import numpy as np
import pytest

from guard.config import ChangePointMethod, ThresholdConfig
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


def _stat_trace(test: PageHinkley | Cusum, stream: np.ndarray) -> np.ndarray:
    return np.array([test.update(float(x)).statistic for x in stream])


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
    assert set(vars(ph)) == {
        "threshold",
        "delta",
        "_gate",
        "_n",
        "_x_mean",
        "_cum",
        "_cum_min",
        "skipped",
    }


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


# --------------------------------------------------------------------------------------
# (A) Recovery genuinely re-arms Page-Hinkley
#
# `test_alarm_clears_after_recovery` only asserts that alarms *stop*. Page-Hinkley also
# stops alarming when the shift persists (it resets its accumulator and its cold-started
# running mean immediately re-centres on the elevated level), so "no late alarms" passes
# for a stream that never recovered at all. The tests below pin the property that actually
# matters operationally: after a real recovery the detector is armed again and a fresh
# shift of the SAME magnitude re-fires, whereas an un-recovered stream has no second
# change-point to find and stays silent.
# --------------------------------------------------------------------------------------
# Segment boundaries of the four-part stream built by `_shift_recover_reshift`.
SEG_SHIFT1 = 300
SEG_MIDDLE = 600
SEG_SHIFT2 = 1000
SEG_END = 1300


def _shift_recover_reshift(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two streams that differ *only* in their middle segment.

    Both are: 300 clean / 300 shifted(+5) / 400 middle / 300 shifted(+5). The recovering
    variant's middle returns to the clean level; the persistent variant's middle stays
    elevated. The final shifted segment is byte-identical between the two, so any
    difference in behaviour there is attributable to the recovery alone.
    """
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, size=SEG_SHIFT1)
    shift1 = rng.normal(5.0, 1.0, size=SEG_MIDDLE - SEG_SHIFT1)
    recovered = rng.normal(0.0, 1.0, size=SEG_SHIFT2 - SEG_MIDDLE)
    shift2 = rng.normal(5.0, 1.0, size=SEG_END - SEG_SHIFT2)
    persisted = rng.normal(5.0, 1.0, size=SEG_SHIFT2 - SEG_MIDDLE)
    recovering = np.concatenate([clean, shift1, recovered, shift2])
    persistent = np.concatenate([clean, shift1, persisted, shift2])
    return recovering, persistent


def test_page_hinkley_rearms_and_refires_after_recovery() -> None:
    recovering, _ = _shift_recover_reshift()
    alarms = _alarm_indices(PageHinkley(threshold=THRESH, delta=DELTA, cooldown=20), recovering)

    first = [i for i in alarms if SEG_SHIFT1 <= i < SEG_MIDDLE]
    assert first, f"no alarm in the first shifted segment; alarms={alarms}"
    assert first[0] <= SEG_SHIFT1 + DETECTION_LAG

    # Quiet through the tail of the recovery (margin for cooldown + statistic decay).
    assert [i for i in alarms if SEG_MIDDLE + 100 <= i < SEG_SHIFT2] == []

    # The load-bearing assertion: an identical shift after recovery must be detected again.
    second = [i for i in alarms if i >= SEG_SHIFT2]
    assert second, f"detector never re-armed — no alarm after the second shift; alarms={alarms}"
    assert second[0] <= SEG_SHIFT2 + DETECTION_LAG


def test_page_hinkley_stays_silent_when_shift_persists_instead_of_recovering() -> None:
    # Paired with the test above and using the identical final segment. Without a recovery
    # there is no second change-point, so re-firing here would mean the detector is
    # alarming on the level rather than on the change — and would mean the re-fire
    # assertion above proves nothing.
    _, persistent = _shift_recover_reshift()
    alarms = _alarm_indices(PageHinkley(threshold=THRESH, delta=DELTA, cooldown=20), persistent)

    assert any(SEG_SHIFT1 <= i < SEG_MIDDLE for i in alarms), f"alarms={alarms}"
    assert [i for i in alarms if i >= SEG_SHIFT2] == ([]), (
        f"fired on a non-existent change-point; alarms={alarms}"
    )


def test_page_hinkley_statistic_decays_on_recovery_but_not_on_persistence() -> None:
    # Probe with an unreachable threshold: on a real alarm the detector resets its
    # accumulator, which would mask the statistic's own dynamics. With no reset in the way,
    # recovery drains the statistic to zero (every step pushes the cumulative sum to a new
    # minimum) while a persisting shift keeps compounding it.
    recovering, persistent = _shift_recover_reshift()
    unreachable = 1e9
    rec = _stat_trace(PageHinkley(threshold=unreachable, delta=DELTA), recovering)
    per = _stat_trace(PageHinkley(threshold=unreachable, delta=DELTA), persistent)

    tail = slice(SEG_SHIFT2 - 100, SEG_SHIFT2)
    assert rec[tail].max() < THRESH / 2, f"statistic did not decay on recovery: {rec[tail].max()}"
    assert per[tail].min() > THRESH / 2, f"statistic decayed without recovery: {per[tail].min()}"


# --------------------------------------------------------------------------------------
# (B) ThresholdConfig -> AlertGate wiring
#
# `from_config` is on the live path via guard/engine/aggregator.py; a version that
# hardcoded debounce=1/cooldown=0 (i.e. dropped the operator's alert-storm controls) used
# to leave the suite green.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["page_hinkley", "cusum"])
def test_from_config_propagates_every_gate_parameter(method: ChangePointMethod) -> None:
    cfg = ThresholdConfig(method=method, threshold=8.0, delta=0.5, debounce=4, cooldown=7)
    test = from_threshold_config(cfg)

    assert isinstance(test, PageHinkley | Cusum)
    assert test.threshold == cfg.threshold
    assert test.delta == cfg.delta
    assert test._gate.debounce == cfg.debounce, "debounce dropped between config and gate"
    assert test._gate.cooldown == cfg.cooldown, "cooldown dropped between config and gate"


# Fixed, RNG-free step input: ten quiet samples then a sustained jump to +4.
_STEP_STREAM = np.array([0.0] * 10 + [4.0] * 40)


@pytest.mark.parametrize("method", ["page_hinkley", "cusum"])
def test_config_delta_shifts_first_alarm_index(method: ChangePointMethod) -> None:
    # Behavioural counterpart to the attribute check: `delta` is the per-step slack, so a
    # larger slack must measurably postpone the crossing on identical input.
    def first(delta: float) -> int | None:
        cfg = ThresholdConfig(method=method, threshold=8.0, delta=delta)
        test = from_threshold_config(cfg)
        return _first_alarm(test, _STEP_STREAM)  # type: ignore[arg-type]

    assert first(0.0) == 12
    assert first(1.0) == 13


@pytest.mark.parametrize("method", ["page_hinkley", "cusum"])
def test_config_debounce_postpones_first_alarm(method: ChangePointMethod) -> None:
    # debounce=5 needs four extra consecutive crossings after the raw one at step 12.
    def first(debounce: int) -> int | None:
        cfg = ThresholdConfig(method=method, threshold=8.0, delta=0.0, debounce=debounce)
        test = from_threshold_config(cfg)
        return _first_alarm(test, _STEP_STREAM)  # type: ignore[arg-type]

    assert first(1) == 12
    assert first(5) == 16


def test_config_cooldown_thins_repeat_alarms() -> None:
    # CUSUM with a fixed-slack ramp re-crosses repeatedly, which is exactly the alert storm
    # cooldown exists to thin — so it is the method that can observe cooldown end-to-end.
    def alarms(cooldown: int) -> list[int]:
        cfg = ThresholdConfig(method="cusum", threshold=8.0, delta=0.0, cooldown=cooldown)
        test = from_threshold_config(cfg)
        return _alarm_indices(test, _STEP_STREAM)  # type: ignore[arg-type]

    assert alarms(0) == [12, 15, 19, 24, 30, 37, 46]
    assert alarms(7) == [12, 20, 28, 36, 45]


# --------------------------------------------------------------------------------------
# (C) Non-finite input is skipped, not folded in
#
# A single NaN metric (one inf logit, one degenerate batch) folded into the running mean
# pins the statistic at NaN forever; `NaN > threshold` is False, so the test would never
# alarm again — a permanent, silent monitoring outage.
# --------------------------------------------------------------------------------------
def _clean_and_poisoned(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Identical streams except two non-finite samples spliced in just before the shift."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, size=300)
    shifted = rng.normal(4.0, 1.0, size=300)
    return (
        np.concatenate([clean, shifted]),
        np.concatenate([clean, [math.nan, math.inf], shifted]),
    )


@pytest.mark.parametrize("cls", [PageHinkley, Cusum])
def test_non_finite_samples_are_skipped_not_absorbed(cls: type[PageHinkley | Cusum]) -> None:
    good, poisoned = _clean_and_poisoned()
    baseline_test = cls(threshold=THRESH, delta=DELTA)
    poisoned_test = cls(threshold=THRESH, delta=DELTA)

    baseline_idx = _first_alarm(baseline_test, good)
    poisoned_idx = _first_alarm(poisoned_test, poisoned)

    assert baseline_idx is not None
    assert poisoned_idx is not None, "NaN/inf poisoned the state — the shift was never detected"
    # Skipping must be a true no-op on the statistic: the alarm lands on the same *sample*,
    # only shifted by the two spliced-in positions. A guard that merely counted the bad
    # samples while still folding them in would fail here, not just the alarm assertion.
    assert poisoned_idx == baseline_idx + 2
    assert poisoned_test.skipped == 2
    assert baseline_test.skipped == 0


@pytest.mark.parametrize("cls", [PageHinkley, Cusum])
def test_statistic_stays_finite_across_non_finite_samples(cls: type[PageHinkley | Cusum]) -> None:
    test = cls(threshold=THRESH, delta=DELTA)
    for x in _stationary(200, seed=3):
        test.update(float(x))

    for bad in (math.nan, math.inf, -math.inf):
        result = test.update(bad)
        assert result.alarm is False
        assert math.isfinite(result.statistic), f"{bad} leaked into the statistic"

    # And the very next good sample must still produce a finite statistic.
    assert math.isfinite(test.update(0.5).statistic)
    assert test.skipped == 3


@pytest.mark.parametrize("cls", [PageHinkley, Cusum])
def test_poisoned_running_mean_would_fail_silently(cls: type[PageHinkley | Cusum]) -> None:
    # Characterisation of the failure mode the guard prevents: with NaN forced into the
    # running mean by hand, an unmistakable +50 shift produces no alarm and no error —
    # nothing observable distinguishes it from a healthy quiet stream. This is why the
    # guard has to sit in `update()` rather than being left to the caller.
    test = cls(threshold=THRESH, delta=DELTA)
    test._x_mean = math.nan
    test._n = 1
    assert _first_alarm(test, np.full(500, 50.0)) is None


# --------------------------------------------------------------------------------------
# (C) AlertGate edge semantics: exact cooldown length and consecutive-run debounce
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cooldown", [0, 1, 2, 5])
def test_cooldown_suppresses_exactly_cooldown_steps(cooldown: int) -> None:
    # Off-by-one pin: with a permanently-raised raw flag the gate must fire on step 0 and
    # then every `cooldown + 1` steps — suppressing `cooldown` steps, never one more or
    # one fewer.
    gate = AlertGate(debounce=1, cooldown=cooldown)
    fired = [i for i in range(24) if gate.update(True)]
    assert fired == list(range(0, 24, cooldown + 1))


@pytest.mark.parametrize("debounce", [2, 3, 5])
def test_debounce_requires_exactly_n_consecutive_crossings(debounce: int) -> None:
    # A run one short of `debounce`, broken by a single non-crossing, must reset the
    # counter to zero — otherwise sporadic spikes would accumulate into a false alarm.
    gate = AlertGate(debounce=debounce, cooldown=0)
    raw = [True] * (debounce - 1) + [False] + [True] * debounce
    out = [gate.update(b) for b in raw]
    assert out == [False] * (len(raw) - 1) + [True]


@pytest.mark.parametrize(("debounce", "cooldown"), [(2, 2), (3, 0), (3, 4)])
def test_debounce_and_cooldown_compose(debounce: int, cooldown: int) -> None:
    # Cooldown also clears the consecutive counter, so after it expires the gate needs a
    # fresh full run: firings settle into a period of `debounce + cooldown`.
    gate = AlertGate(debounce=debounce, cooldown=cooldown)
    fired = [i for i in range(30) if gate.update(True)]
    assert fired == list(range(debounce - 1, 30, debounce + cooldown))
