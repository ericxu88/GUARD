"""P1-08: end-to-end drift injection harness.

Four-phase synthetic stream: clean → gradual covariate shift → abrupt label shift → recovery.
Each detector's metric is fed into a Page-Hinkley change-point test.

Acceptance criteria (PHASE_1_PLAN.md §P1-08):
  * Each detector's change-point test fires within DETECTION_LAG steps of the injected shift.
  * Metrics return to baseline after recovery: the alert clears within SETTLE steps *and*
    the metric value itself falls back to its clean level.
  * False-positive rate on the clean segment is below FP_RATE_BOUND.

Stream design:
  * Clean:   logit_scale=4.0, embed_bias=0    → confident model, normal embeddings
  * Gradual: logit_scale ramps 4→0.2 (model becomes uncertain → entropy rises),
             embed_bias ramps 0→5 (covariate shift → Mahalanobis rises).
  * Abrupt:  logit_scale=0.5, embed_bias=5, strong class_bias → JS divergence rises.
  * Recover: logit_scale=4.0, embed_bias=0    → back to baseline conditions.

The recovery assertions are run against a control stream (``recover=False``) whose fourth
phase stays drifted. Any recovery check that also passes on the control is measuring
nothing, so the control is asserted to fail the same comparison.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

from guard.baseline import Baseline, build_baseline
from guard.detectors.divergence import DivergenceDetector
from guard.detectors.embedding import EmbeddingDriftDetector
from guard.detectors.entropy import EntropyDetector
from guard.temporal import PageHinkley

# ─── stream parameters ────────────────────────────────────────────────────────

C = 10  # number of output classes
D = 16  # embedding dimension
B = 64  # batch size per step

CLEAN_STEPS = 300
GRADUAL_STEPS = 100
ABRUPT_STEPS = 120
RECOVER_STEPS = 200

DETECTION_LAG = 80  # steps after shift onset within which alarm must fire
FP_RATE_BOUND = 0.05  # max fraction of clean steps that may alarm


# ─── stream generator ─────────────────────────────────────────────────────────


@dataclass
class Step:
    logits: torch.Tensor  # [B, C]
    embeddings: torch.Tensor  # [B, D]
    phase: str


def _stream(seed: int, *, recover: bool = True) -> Iterator[Step]:
    """Deterministic four-phase (logits, embeddings) stream.

    ``recover=False`` produces the control stream: identical for the first three phases,
    but the fourth keeps the drifted conditions instead of returning to clean ones. It is
    still labelled ``"recover"`` and draws the same number of random batches, so the two
    streams differ only in the thing under test.
    """
    g = torch.Generator().manual_seed(seed)

    def _batch(logit_scale: float, embed_bias: float, class_bias: int | None) -> Step:
        emb = torch.randn(B, D, generator=g, dtype=torch.float32) + embed_bias
        raw = torch.randn(B, C, generator=g, dtype=torch.float32) * logit_scale
        if class_bias is not None:
            raw[:, class_bias] += 8.0
        return Step(logits=raw, embeddings=emb, phase="")

    # Clean — confident model, centred embeddings.
    for _ in range(CLEAN_STEPS):
        s = _batch(4.0, 0.0, None)
        yield Step(s.logits, s.embeddings, "clean")

    # Gradual — logit scale collapses (entropy rises) + embedding mean drifts.
    for i in range(GRADUAL_STEPS):
        frac = (i + 1) / GRADUAL_STEPS
        scale = 4.0 - 3.8 * frac  # 4.0 → 0.2
        bias = 5.0 * frac  # 0 → 5
        s = _batch(scale, bias, None)
        yield Step(s.logits, s.embeddings, "gradual")

    # Abrupt — strong label shift onto class 0, embedding bias held at 5.
    for _ in range(ABRUPT_STEPS):
        s = _batch(0.5, 5.0, class_bias=0)
        yield Step(s.logits, s.embeddings, "abrupt")

    # Recovery — identical to clean (or, for the control stream, still drifted).
    for _ in range(RECOVER_STEPS):
        s = _batch(4.0, 0.0, None) if recover else _batch(0.5, 5.0, class_bias=0)
        yield Step(s.logits, s.embeddings, "recover")


_BASELINE_SEED = 99


def _make_baseline() -> Baseline:
    """Build the reference baseline from 512 clean batches (separate seed)."""
    g = torch.Generator().manual_seed(_BASELINE_SEED)
    data = [
        (
            torch.randn(B, C, generator=g, dtype=torch.float32) * 4.0,
            torch.randn(B, D, generator=g, dtype=torch.float32),
        )
        for _ in range(8)  # 8 × 64 = 512 samples
    ]
    return build_baseline(data, model_version="harness-v0", num_classes=C, embed_dim=D)


# ─── helper: run detector + PH through the full stream ───────────────────────


_PHASES = ("clean", "gradual", "abrupt", "recover")

# The three concrete detectors this harness drives; named once so the config tables below
# type-check without per-call-site ignores.
AnyDetector = EntropyDetector | DivergenceDetector | EmbeddingDriftDetector


@dataclass
class Trace:
    """Everything one detector produced over one stream, kept per phase and in order."""

    scalars: dict[str, list[float]]  # phase → the metric value at each step of that phase
    alarms: dict[str, list[int]]  # phase → global step indices where PH fired


def _trace(
    detector: AnyDetector,
    metric_key: str,
    ph: PageHinkley,
    seed: int,
    *,
    recover: bool = True,
) -> Trace:
    scalars: dict[str, list[float]] = {p: [] for p in _PHASES}
    alarms: dict[str, list[int]] = {p: [] for p in _PHASES}
    for step, s in enumerate(_stream(seed, recover=recover)):
        scalar = float(detector.compute(s.logits, s.embeddings).scores[metric_key])
        scalars[s.phase].append(scalar)
        if ph.update(scalar).alarm:
            alarms[s.phase].append(step)
    return Trace(scalars=scalars, alarms=alarms)


def _run(
    detector: AnyDetector,
    metric_key: str,
    ph: PageHinkley,
    seed: int,
) -> dict[str, list[int]]:
    return _trace(detector, metric_key, ph, seed).alarms


# ─── step index boundaries (for assertion messages) ──────────────────────────

_SHIFT_ONSET = CLEAN_STEPS
_ABRUPT_ONSET = CLEAN_STEPS + GRADUAL_STEPS
_RECOVER_ONSET = _ABRUPT_ONSET + ABRUPT_STEPS

_SEED = 0


def _build() -> Baseline:
    return _make_baseline()


# ─── tests ───────────────────────────────────────────────────────────────────


def test_entropy_fires_within_lag_on_gradual_shift() -> None:
    bl = _build()
    det = EntropyDetector(quantile=0.99, baseline=bl)
    ph = PageHinkley(threshold=5.0, delta=0.3)
    alarms = _run(det, "entropy_mean", ph, _SEED)

    drift_alarms = alarms["gradual"] + alarms["abrupt"]
    assert drift_alarms, "entropy detector never fired during gradual/abrupt shift"
    first = min(drift_alarms)
    assert first - _SHIFT_ONSET <= DETECTION_LAG, (
        f"entropy first alarm step {first}, {first - _SHIFT_ONSET} steps after shift onset; "
        f"exceeds lag budget {DETECTION_LAG}"
    )


def test_divergence_fires_within_lag_on_abrupt_shift() -> None:
    # JS divergence measures class-distribution shift. The gradual phase only changes
    # logit scale and embed bias — class mix stays roughly uniform. The abrupt phase
    # introduces a strong class-0 bias, which is the intended trigger.
    bl = _build()
    det = DivergenceDetector(bl)
    ph = PageHinkley(threshold=5.0, delta=0.05)
    alarms = _run(det, "js_divergence", ph, _SEED)

    assert alarms["abrupt"], "divergence detector never fired during abrupt label shift"
    first = min(alarms["abrupt"])
    assert first - _ABRUPT_ONSET <= DETECTION_LAG, (
        f"divergence first alarm step {first}, {first - _ABRUPT_ONSET} steps after "
        f"abrupt onset {_ABRUPT_ONSET}; exceeds lag budget {DETECTION_LAG}"
    )


def test_embedding_fires_within_lag_on_gradual_shift() -> None:
    bl = _build()
    det = EmbeddingDriftDetector(bl)
    ph = PageHinkley(threshold=5.0, delta=0.5)
    alarms = _run(det, "embedding_mahalanobis_mean", ph, _SEED)

    drift_alarms = alarms["gradual"] + alarms["abrupt"]
    assert drift_alarms, "embedding detector never fired during shift"
    first = min(drift_alarms)
    assert first - _SHIFT_ONSET <= DETECTION_LAG, (
        f"embedding first alarm step {first}, {first - _SHIFT_ONSET} steps after onset"
    )


def test_false_positive_rate_below_bound() -> None:
    bl = _build()
    configs: list[tuple[AnyDetector, str, float, float]] = [
        (EntropyDetector(quantile=0.99, baseline=bl), "entropy_mean", 5.0, 0.3),
        (DivergenceDetector(bl), "js_divergence", 5.0, 0.05),
        (EmbeddingDriftDetector(bl), "embedding_mahalanobis_mean", 5.0, 0.5),
    ]
    for det, key, thresh, delta in configs:
        ph = PageHinkley(threshold=thresh, delta=delta)
        alarms = _run(det, key, ph, _SEED)
        fp_rate = len(alarms["clean"]) / CLEAN_STEPS
        assert fp_rate <= FP_RATE_BOUND, (
            f"{det.__class__.__name__}/{key}: FP rate {fp_rate:.3f} > bound {FP_RATE_BOUND} "
            f"({len(alarms['clean'])} alarms in {CLEAN_STEPS} clean steps)"
        )


# Number of trailing steps of a phase averaged when comparing metric levels. The clean
# phase has 300 steps and recovery 200, so 100 fits in both with room for PH to settle.
_TAIL = 100

# A recovered metric must land within this many *clean-phase* standard deviations of the
# clean mean. Expressing the budget in units of the metric's own step-to-step noise keeps
# one threshold meaningful across entropy (~0.7 nats), JS divergence (~0.01) and
# Mahalanobis distance (~4.0), whose absolute scales differ by three orders of magnitude.
_RECOVERY_SIGMA = 0.75

# The control stream must miss by at least this multiple of the same budget. Without it,
# a budget loose enough to pass anything would still look like a passing test.
_CONTROL_MARGIN = 5.0


def _tail_mean(values: list[float]) -> float:
    return statistics.fmean(values[-_TAIL:])


def test_metrics_clear_after_recovery() -> None:
    """Recovery must show up in the metric *values*, not just in the alarm log.

    This test used to assert only that no alarm fired late in the recovery phase.
    Page-Hinkley is one-sided and latches onto the drifted mean, so it goes quiet late in
    the run whether or not the stream recovered — the assertion produced identical output
    for a recovering and a never-recovering stream, and therefore proved nothing.

    It now asserts three things per metric: the recovered level matches the clean level,
    the alarm has cleared within SETTLE steps, and — the part that supplies the teeth — a
    control stream that never recovers *fails* the level comparison by a wide margin.
    """
    bl = _build()
    settle = DETECTION_LAG
    # Factories, not instances: EmbeddingDriftDetector carries running Welford/covariance
    # state across calls, so the control run must start from a fresh detector rather than
    # inheriting whatever the recovery run left behind.
    configs: list[tuple[Callable[[], AnyDetector], str, float, float]] = [
        (lambda: EntropyDetector(quantile=0.99, baseline=bl), "entropy_mean", 5.0, 0.3),
        (lambda: EntropyDetector(quantile=0.99, baseline=bl), "entropy_pop_shift", 5.0, 0.3),
        (lambda: DivergenceDetector(bl), "js_divergence", 5.0, 0.05),
        (lambda: EmbeddingDriftDetector(bl), "embedding_mahalanobis_mean", 5.0, 0.5),
    ]
    for make_det, key, thresh, delta in configs:
        recovered = _trace(make_det(), key, PageHinkley(threshold=thresh, delta=delta), _SEED)
        control = _trace(
            make_det(),
            key,
            PageHinkley(threshold=thresh, delta=delta),
            _SEED,
            recover=False,
        )
        name = f"{make_det().__class__.__name__}/{key}"

        clean = recovered.scalars["clean"]
        clean_mean = _tail_mean(clean)
        # Sample noise of the metric itself sets the scale of "unchanged".
        budget = _RECOVERY_SIGMA * statistics.stdev(clean[-_TAIL:])
        assert budget > 0.0, f"{name}: clean metric is constant; the budget would be vacuous"

        drift_recovered = abs(_tail_mean(recovered.scalars["recover"]) - clean_mean)
        assert drift_recovered <= budget, (
            f"{name}: recovered level {_tail_mean(recovered.scalars['recover']):.6f} differs "
            f"from clean level {clean_mean:.6f} by {drift_recovered:.6f} > budget {budget:.6f}"
        )

        # Teeth: the identical comparison on a stream that never recovers must fail, by a
        # wide margin. If this passes too, the budget above is measuring nothing.
        drift_control = abs(_tail_mean(control.scalars["recover"]) - clean_mean)
        assert drift_control > _CONTROL_MARGIN * budget, (
            f"{name}: never-recovering control is only {drift_control:.6f} from the clean "
            f"level (budget {budget:.6f}) — this test cannot distinguish recovery from drift"
        )

        # The original acceptance criterion: the alert itself has to stop.
        late = [i for i in recovered.alarms["recover"] if i >= _RECOVER_ONSET + settle]
        assert not late, f"{name}: alarms still firing after recovery+settle: {late}"
