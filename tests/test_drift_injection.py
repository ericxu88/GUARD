"""P1-08: end-to-end drift injection harness.

Four-phase synthetic stream: clean → gradual covariate shift → abrupt label shift → recovery.
Each detector's metric is fed into a Page-Hinkley change-point test.

Acceptance criteria (PHASE_1_PLAN.md §P1-08):
  * Each detector's change-point test fires within DETECTION_LAG steps of the injected shift.
  * Metrics return to baseline after recovery (alert clears within SETTLE steps).
  * False-positive rate on the clean segment is below FP_RATE_BOUND.

Stream design:
  * Clean:   logit_scale=4.0, embed_bias=0    → confident model, normal embeddings
  * Gradual: logit_scale ramps 4→0.2 (model becomes uncertain → entropy rises),
             embed_bias ramps 0→5 (covariate shift → Mahalanobis rises).
  * Abrupt:  logit_scale=0.5, embed_bias=5, strong class_bias → JS divergence rises.
  * Recover: logit_scale=4.0, embed_bias=0    → back to baseline conditions.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from guard.baseline import build_baseline
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


def _stream(seed: int) -> Iterator[Step]:
    """Deterministic four-phase (logits, embeddings) stream."""
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

    # Recovery — identical to clean.
    for _ in range(RECOVER_STEPS):
        s = _batch(4.0, 0.0, None)
        yield Step(s.logits, s.embeddings, "recover")


_BASELINE_SEED = 99


def _make_baseline() -> object:
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


def _run(
    detector: EntropyDetector | DivergenceDetector | EmbeddingDriftDetector,
    metric_key: str,
    ph: PageHinkley,
    seed: int,
) -> dict[str, list[int]]:
    alarms: dict[str, list[int]] = {p: [] for p in ("clean", "gradual", "abrupt", "recover")}
    for step, s in enumerate(_stream(seed)):
        scalar = float(detector.compute(s.logits, s.embeddings).scores[metric_key])
        if ph.update(scalar).alarm:
            alarms[s.phase].append(step)
    return alarms


# ─── step index boundaries (for assertion messages) ──────────────────────────

_SHIFT_ONSET = CLEAN_STEPS
_ABRUPT_ONSET = CLEAN_STEPS + GRADUAL_STEPS
_RECOVER_ONSET = _ABRUPT_ONSET + ABRUPT_STEPS

_SEED = 0


def _build() -> object:
    return _make_baseline()


# ─── tests ───────────────────────────────────────────────────────────────────


def test_entropy_fires_within_lag_on_gradual_shift() -> None:
    bl = _build()
    det = EntropyDetector(quantile=0.99, baseline=bl)  # type: ignore[arg-type]
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
    det = DivergenceDetector(bl)  # type: ignore[arg-type]
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
    det = EmbeddingDriftDetector(bl)  # type: ignore[arg-type]
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
    configs: list[tuple[object, str, float, float]] = [
        (EntropyDetector(quantile=0.99, baseline=bl), "entropy_mean", 5.0, 0.3),
        (DivergenceDetector(bl), "js_divergence", 5.0, 0.05),
        (EmbeddingDriftDetector(bl), "embedding_mahalanobis_mean", 5.0, 0.5),
    ]
    for det, key, thresh, delta in configs:
        ph = PageHinkley(threshold=thresh, delta=delta)
        alarms = _run(det, key, ph, _SEED)  # type: ignore[arg-type]
        fp_rate = len(alarms["clean"]) / CLEAN_STEPS
        assert fp_rate <= FP_RATE_BOUND, (
            f"{det.__class__.__name__}/{key}: FP rate {fp_rate:.3f} > bound {FP_RATE_BOUND} "
            f"({len(alarms['clean'])} alarms in {CLEAN_STEPS} clean steps)"
        )


def test_metrics_clear_after_recovery() -> None:
    """After the stream returns to clean conditions, alerts should stop within SETTLE steps."""
    bl = _build()
    settle = DETECTION_LAG
    configs: list[tuple[object, str, float, float]] = [
        (EntropyDetector(quantile=0.99, baseline=bl), "entropy_mean", 5.0, 0.3),
        (DivergenceDetector(bl), "js_divergence", 5.0, 0.05),
        (EmbeddingDriftDetector(bl), "embedding_mahalanobis_mean", 5.0, 0.5),
    ]
    for det, key, thresh, delta in configs:
        ph = PageHinkley(threshold=thresh, delta=delta)
        alarms = _run(det, key, ph, _SEED)  # type: ignore[arg-type]
        late = [i for i in alarms["recover"] if i >= _RECOVER_ONSET + settle]
        assert not late, (
            f"{det.__class__.__name__}/{key}: alarms still firing after recovery+settle: {late}"
        )
