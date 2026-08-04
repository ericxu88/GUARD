"""P1-03: entropy detector — NumPy parity, property bounds, no-host-sync contract."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from guard.baseline.schema import DEFAULT_ATOL, DEFAULT_RTOL, Baseline
from guard.detectors.entropy import EntropyDetector, softmax_entropy

EPS = 1e-12


# --------------------------------------------------------------------------------------
# NumPy CPU reference
# --------------------------------------------------------------------------------------
def _np_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _np_entropy(logits: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = _np_softmax(logits)
    # The floor goes INSIDE the log (max(p, eps)), not added to p. `np.log(p + eps)` is the
    # old buggy formula: log(1 + eps) > 0 makes H come out at about -1e-12 for a saturated
    # row, so a reference written that way would happily "confirm" a negative entropy.
    # Numerically the two agree to ~1e-12, so every existing parity assertion still holds.
    return np.maximum(-(p * np.log(np.maximum(p, eps))).sum(axis=1), 0.0)


# --------------------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------------------
def test_matches_numpy_reference() -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 10, dtype=torch.float64)

    h_torch = softmax_entropy(logits, EPS)
    h_np = _np_entropy(logits.numpy(), EPS)

    assert np.allclose(h_torch.numpy(), h_np, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_detector_mean_and_quantile_match_numpy() -> None:
    torch.manual_seed(1)
    logits = torch.randn(128, 7, dtype=torch.float64)
    det = EntropyDetector(quantile=0.99)

    result = det.compute(logits, embeddings=torch.empty(0))

    h_np = _np_entropy(logits.numpy(), EPS)
    assert np.allclose(
        result.scores["entropy_mean"].numpy(), h_np.mean(), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )
    assert np.allclose(
        result.scores["entropy_quantile"].numpy(),
        np.quantile(h_np, 0.99),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


# --------------------------------------------------------------------------------------
# Property tests
# --------------------------------------------------------------------------------------
@settings(max_examples=200, deadline=None)
@given(
    arr=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 16), st.integers(2, 32)),
        elements=st.floats(-30.0, 30.0, allow_nan=False, allow_infinity=False),
    )
)
def test_entropy_bounds(arr: np.ndarray) -> None:
    logits = torch.from_numpy(arr)
    c = logits.shape[1]
    h = softmax_entropy(logits, EPS)
    log_c = math.log(c)
    # 0 <= H <= log C, within numerical tolerance (eps inside log can push H by ~eps).
    assert torch.all(h >= -DEFAULT_ATOL)
    assert torch.all(h <= log_c + DEFAULT_ATOL)


@given(c=st.integers(2, 64), b=st.integers(1, 8))
def test_one_hot_entropy_is_zero(c: int, b: int) -> None:
    # A dominant logit drives softmax to ~one-hot, so H ≈ 0.
    logits = torch.zeros(b, c, dtype=torch.float64)
    logits[:, 0] = 1e4
    h = softmax_entropy(logits, EPS)
    assert torch.all(h.abs() < 1e-6)


@given(c=st.integers(2, 64), b=st.integers(1, 8))
def test_uniform_entropy_is_log_c(c: int, b: int) -> None:
    # Equal logits → uniform distribution → H = log C exactly.
    logits = torch.zeros(b, c, dtype=torch.float64)
    h = softmax_entropy(logits, EPS)
    assert torch.allclose(h, torch.full_like(h, math.log(c)), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


# --------------------------------------------------------------------------------------
# Population shift vs baseline
# --------------------------------------------------------------------------------------
def _baseline_from_logits(logits: torch.Tensor, c: int, n_bins: int = 16) -> Baseline:
    """Build a baseline whose entropy histogram is taken from `logits` itself."""
    h = softmax_entropy(logits, EPS)
    log_c = math.log(c)
    edges = torch.linspace(0.0, log_c, n_bins + 1, dtype=torch.float64)
    hist = torch.histogram(h, bins=edges).hist
    return Baseline(
        model_version="t",
        created_at="2026-01-01T00:00:00Z",
        sample_count=int(logits.shape[0]),
        num_classes=c,
        embed_dim=2,
        class_distribution=torch.full((c,), 1.0 / c, dtype=torch.float64),
        entropy_histogram=hist,
        entropy_bin_edges=edges,
        embed_mean=torch.zeros(2, dtype=torch.float64),
        embed_precision=torch.eye(2, dtype=torch.float64),
    )


def test_pop_shift_zero_against_self() -> None:
    torch.manual_seed(2)
    c = 10
    logits = torch.randn(2048, c, dtype=torch.float64)
    base = _baseline_from_logits(logits, c)
    det = EntropyDetector(quantile=0.99, baseline=base)

    shift = det.compute(logits, embeddings=torch.empty(0)).scores["entropy_pop_shift"]
    # Same data binned the same way ⇒ near-zero total-variation distance.
    assert float(shift) < 1e-6


def test_pop_shift_detects_distribution_change() -> None:
    torch.manual_seed(3)
    c = 10
    # Baseline from low-entropy (confident) logits.
    confident = torch.randn(2048, c, dtype=torch.float64) * 6.0
    base = _baseline_from_logits(confident, c)
    det = EntropyDetector(quantile=0.99, baseline=base)

    # Live batch is near-uniform (high entropy) — a clear population shift.
    uniform_like = torch.randn(2048, c, dtype=torch.float64) * 0.01
    shift = det.compute(uniform_like, embeddings=torch.empty(0)).scores["entropy_pop_shift"]
    assert float(shift) > 0.5  # TV in [0, 1]; large, clearly-shifted distributions


def test_pop_shift_is_bounded() -> None:
    torch.manual_seed(4)
    c = 8
    base = _baseline_from_logits(torch.randn(512, c, dtype=torch.float64), c)
    det = EntropyDetector(quantile=0.95, baseline=base)
    shift = det.compute(torch.randn(512, c, dtype=torch.float64), embeddings=torch.empty(0)).scores[
        "entropy_pop_shift"
    ]
    assert 0.0 <= float(shift) <= 1.0 + DEFAULT_ATOL


# --------------------------------------------------------------------------------------
# Contract: outputs and no host sync
# --------------------------------------------------------------------------------------
def test_scores_are_zero_dim_tensors_on_input_device() -> None:
    logits = torch.randn(32, 5)
    result = EntropyDetector().compute(logits, embeddings=torch.empty(0))
    for name, value in result.scores.items():
        assert isinstance(value, torch.Tensor), name
        assert value.ndim == 0, name
        assert value.device == logits.device, name


def test_compute_has_no_host_sync_calls() -> None:
    """Static guard for prime directive #2: no host-blocking calls inside the hot path."""
    forbidden = (".item(", ".cpu(", ".numpy(", ".tolist(", "synchronize(")
    for fn in (EntropyDetector.compute, EntropyDetector._population_shift, softmax_entropy):
        src = inspect.getsource(fn)
        for token in forbidden:
            assert token not in src, (
                f"{fn.__qualname__} contains forbidden host-sync call {token!r}"
            )


def test_invalid_quantile_rejected() -> None:
    with pytest.raises(ValueError, match="quantile"):
        EntropyDetector(quantile=1.5)


def test_satisfies_detector_protocol() -> None:
    from guard.detectors.base import Detector

    assert isinstance(EntropyDetector(), Detector)


# --------------------------------------------------------------------------------------
# Regression (A): saturated softmax must never yield a negative entropy
#
# The old implementation was `-(p * (p + eps).log())`. For a confident prediction p≈1 that
# is `-(1 * log(1 + 1e-12)) = -1e-12` — negative. It looks like noise but is not: a negative
# H falls below the first entropy-histogram bin edge (0.0) and `torch.histogram` *silently
# discards* out-of-range samples, so every confident sample disappeared from the baseline.
# On a well-trained model that empties the histogram entirely and pins `entropy_pop_shift`
# at a constant 0.5 against the exact data it was calibrated on.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("c", [2, 10, 1000])
def test_saturated_logits_entropy_is_nonnegative(c: int) -> None:
    # One logit at +60, the rest at -30: a 90-nat gap saturates softmax to p_max == 1.0 in
    # float64 while the remaining probabilities (~1e-39) stay below eps, which is exactly
    # the regime where the `p + eps` form goes negative.
    logits = torch.full((8, c), -30.0, dtype=torch.float64)
    logits[:, 0] = 60.0

    h = softmax_entropy(logits, EPS)

    assert torch.all(h >= 0.0), f"negative entropy for C={c}: {h.min()}"
    assert torch.all(h <= math.log(c) + 1e-9), f"entropy above log C for C={c}: {h.max()}"


def test_saturated_entropy_survives_histogram_binning() -> None:
    """The consequence the sign bug actually caused: samples dropped from the baseline."""
    c = 10
    logits = torch.full((256, c), -30.0, dtype=torch.float64)
    logits[:, 0] = 60.0

    base = _baseline_from_logits(logits, c)

    # With H = -1e-12 every one of the 256 samples lands outside [0, log C] and is dropped.
    assert float(base.entropy_histogram.sum()) == 256.0


@settings(max_examples=300, deadline=None)
@given(
    arr=arrays(
        dtype=np.float64,
        # Magnitudes up to 1e3 push softmax fully into the saturated/underflow regime where
        # some probabilities are exactly 0 and others are exactly 1.
        shape=st.tuples(st.integers(1, 8), st.integers(2, 64)),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False, width=64),
    )
)
def test_entropy_in_zero_to_log_c_for_extreme_logits(arr: np.ndarray) -> None:
    logits = torch.from_numpy(arr)
    c = logits.shape[1]

    h = softmax_entropy(logits, EPS)

    assert torch.all(torch.isfinite(h))
    # Exact 0 lower bound (not -atol): H is a Shannon entropy, it cannot be negative, and
    # downstream histogram binning treats anything below 0 as out of range.
    assert torch.all(h >= 0.0)
    assert torch.all(h <= math.log(c) + 1e-9)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_one_hot_row_entropy_is_exactly_zero(dtype: torch.dtype) -> None:
    # exp(-800) underflows to exactly 0.0 in both float32 and float64, so p is a true
    # one-hot vector and the 0·log 0 terms must contribute exactly 0 (not NaN, not -0-ish).
    logits = torch.zeros(4, 16, dtype=dtype)
    logits[:, 0] = 800.0

    h = softmax_entropy(logits, EPS)

    assert torch.equal(h, torch.zeros_like(h)), f"{dtype}: expected exact 0, got {h}"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("c", [2, 10, 1000])
def test_uniform_row_entropy_is_log_c(dtype: torch.dtype, c: int) -> None:
    # The opposite extreme of the one-hot case: H must reach the upper bound, not fall
    # short of it because of the eps floor.
    h = softmax_entropy(torch.zeros(4, c, dtype=dtype), EPS)

    expected = torch.full_like(h, math.log(c))
    assert torch.allclose(h, expected, rtol=1e-6, atol=1e-6), f"{dtype}, C={c}: {h}"


# --------------------------------------------------------------------------------------
# Regression (B): half precision must not produce NaN or raise
#
# eps = 1e-12 underflows to exactly 0 in float16, so the log floor stopped working and a
# saturated row produced log(0) = -inf -> NaN. Separately, `torch.quantile` raises
# RuntimeError ("quantile() input tensor must be either float or double dtype") on half
# inputs. Autocast serving emits exactly these dtypes, so both failures are the default
# production path, and a NaN score never crosses an alert threshold.
# --------------------------------------------------------------------------------------
_HALF_DTYPES = [torch.float16, torch.bfloat16]


def _realistic_logits() -> torch.Tensor:
    """B=32, C=1000 ImageNet-shaped logits, scaled so some softmax entries are exactly 0."""
    torch.manual_seed(20)
    return torch.randn(32, 1000, dtype=torch.float32) * 20.0


def test_realistic_half_logits_contain_exact_softmax_zeros() -> None:
    """Guard the premise of the tests below: this input really does hit the log(0) path."""
    p = torch.softmax(_realistic_logits().half().float(), dim=1)
    assert int((p == 0).sum()) > 0


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_detector_half_precision_scores_are_finite(dtype: torch.dtype) -> None:
    det = EntropyDetector(quantile=0.99)

    # Must not raise (torch.quantile on half) and must not be NaN (log(0) via eps underflow).
    scores = det.compute(_realistic_logits().to(dtype), embeddings=torch.empty(0)).scores

    assert set(scores) == {"entropy_mean", "entropy_quantile"}
    for name, value in scores.items():
        assert torch.isfinite(value), f"{dtype} {name} is not finite: {value}"
        assert float(value) >= 0.0, f"{dtype} {name} is negative: {value}"
        assert float(value) <= math.log(1000) + 1e-5, f"{dtype} {name} above log C: {value}"


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_half_precision_matches_float32(dtype: torch.dtype) -> None:
    det = EntropyDetector(quantile=0.99)
    logits32 = _realistic_logits()

    ref = det.compute(logits32, embeddings=torch.empty(0)).scores
    got = det.compute(logits32.to(dtype), embeddings=torch.empty(0)).scores

    for name in ref:
        a, b = float(ref[name]), float(got[name])
        # Loose bound on purpose: the input itself is quantized before we ever see it —
        # bfloat16 keeps only ~8 mantissa bits, so rounding 1000 logits perturbs the softmax
        # and hence the true entropy by ~1e-2 relative. The point of the assertion is that
        # the half path computes the *same quantity*, not that it is bit-accurate.
        assert abs(a - b) / max(abs(a), 1e-12) < 5e-2, f"{dtype} {name}: fp32={a} half={b}"


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_half_precision_entropy_promotes_to_float32(dtype: torch.dtype) -> None:
    # The promotion is the fix; asserting it keeps a future "optimization" back to native
    # half from silently reintroducing both failures.
    h = softmax_entropy(_realistic_logits().to(dtype), EPS)
    assert h.dtype == torch.float32
    assert torch.all(torch.isfinite(h))
