"""P1-04: KL/JS divergence detector — NumPy parity, bounds, zero-prob stability."""

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
from guard.detectors.divergence import (
    DivergenceDetector,
    js_divergence,
    kl_divergence,
    mean_softmax,
)

EPS = 1e-12


# --------------------------------------------------------------------------------------
# NumPy reference
# --------------------------------------------------------------------------------------
def _np_norm(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.clip(x, 0.0, None) + eps
    return x / x.sum()


def _np_kl(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    pc, qc = _np_norm(p), _np_norm(q)
    return float((pc * (np.log(pc + eps) - np.log(qc + eps))).sum())


def _np_js(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    pc, qc = _np_norm(p), _np_norm(q)
    m = 0.5 * (pc + qc)
    return 0.5 * _np_kl(pc, m, eps) + 0.5 * _np_kl(qc, m, eps)


def _baseline_with_q(q: torch.Tensor) -> Baseline:
    c = q.numel()
    return Baseline(
        model_version="t",
        created_at="2026-01-01T00:00:00Z",
        sample_count=1000,
        num_classes=c,
        embed_dim=2,
        class_distribution=q,
        entropy_histogram=torch.zeros(8),
        entropy_bin_edges=torch.linspace(0.0, 1.0, 9),
        embed_mean=torch.zeros(2),
        embed_precision=torch.eye(2),
    )


# --------------------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------------------
def test_kl_matches_numpy() -> None:
    torch.manual_seed(0)
    p = torch.rand(12, dtype=torch.float64)
    q = torch.rand(12, dtype=torch.float64)
    got = kl_divergence(p, q, EPS)
    assert np.allclose(
        got.numpy(), _np_kl(p.numpy(), q.numpy()), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


def test_js_matches_numpy() -> None:
    torch.manual_seed(1)
    p = torch.rand(20, dtype=torch.float64)
    q = torch.rand(20, dtype=torch.float64)
    got = js_divergence(p, q, EPS)
    assert np.allclose(
        got.numpy(), _np_js(p.numpy(), q.numpy()), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


def test_detector_matches_numpy_end_to_end() -> None:
    torch.manual_seed(2)
    c = 8
    logits = torch.randn(64, c, dtype=torch.float64)
    q = torch.rand(c, dtype=torch.float64)
    q = q / q.sum()
    det = DivergenceDetector(_baseline_with_q(q))

    result = det.compute(logits, embeddings=torch.empty(0))

    p_np = torch.softmax(logits, dim=1).mean(0).numpy()
    assert np.allclose(
        result.scores["kl_divergence"].numpy(),
        _np_kl(p_np, q.numpy()),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    assert np.allclose(
        result.scores["js_divergence"].numpy(),
        _np_js(p_np, q.numpy()),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


# --------------------------------------------------------------------------------------
# Mathematical properties
# --------------------------------------------------------------------------------------
def test_kl_self_is_zero() -> None:
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    assert torch.allclose(
        kl_divergence(p, p, EPS), torch.zeros((), dtype=torch.float64), atol=DEFAULT_ATOL
    )


def test_js_self_is_zero() -> None:
    p = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    assert torch.allclose(
        js_divergence(p, p, EPS), torch.zeros((), dtype=torch.float64), atol=DEFAULT_ATOL
    )


def test_js_symmetric() -> None:
    torch.manual_seed(3)
    p = torch.rand(16, dtype=torch.float64)
    q = torch.rand(16, dtype=torch.float64)
    assert torch.allclose(js_divergence(p, q, EPS), js_divergence(q, p, EPS), atol=DEFAULT_ATOL)


@settings(max_examples=200, deadline=None)
@given(
    p=arrays(np.float64, st.integers(2, 32), elements=st.floats(0.0, 10.0, allow_nan=False)),
    q=arrays(np.float64, st.integers(2, 32), elements=st.floats(0.0, 10.0, allow_nan=False)),
)
def test_kl_nonnegative_and_js_bounded(p: np.ndarray, q: np.ndarray) -> None:
    # Align lengths (hypothesis draws them independently). Degenerate all-zero inputs are
    # handled by the detector's eps-smoothing, so no special-casing is needed here.
    n = min(p.shape[0], q.shape[0])
    pt, qt = torch.from_numpy(p[:n]).clone(), torch.from_numpy(q[:n]).clone()
    kl = kl_divergence(pt, qt, EPS)
    js = js_divergence(pt, qt, EPS)
    assert torch.isfinite(kl) and torch.isfinite(js)
    assert float(kl) >= -DEFAULT_ATOL
    assert -DEFAULT_ATOL <= float(js) <= math.log(2.0) + 1e-6


def test_disjoint_support_is_stable_and_near_log2() -> None:
    # P and Q with disjoint support: KL would be +inf without eps; JS -> log 2.
    p = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    kl = kl_divergence(p, q, EPS)
    js = js_divergence(p, q, EPS)
    assert torch.isfinite(kl) and torch.isfinite(js)
    assert abs(float(js) - math.log(2.0)) < 1e-3


def test_zero_probability_classes_no_nan() -> None:
    torch.manual_seed(4)
    # Logits that softmax to a distribution with several near-zero classes.
    logits = torch.tensor([[50.0, -50.0, -50.0, 0.0]], dtype=torch.float64)
    q = torch.tensor([0.0, 0.5, 0.5, 0.0], dtype=torch.float64)
    det = DivergenceDetector(_baseline_with_q(q))
    scores = det.compute(logits, embeddings=torch.empty(0)).scores
    assert torch.isfinite(scores["kl_divergence"])
    assert torch.isfinite(scores["js_divergence"])


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------
def test_scores_are_zero_dim_tensors_on_input_device() -> None:
    logits = torch.randn(16, 6)
    q = torch.full((6,), 1.0 / 6)
    result = DivergenceDetector(_baseline_with_q(q)).compute(logits, embeddings=torch.empty(0))
    for name, value in result.scores.items():
        assert isinstance(value, torch.Tensor) and value.ndim == 0, name
        assert value.device == logits.device, name


def test_compute_has_no_host_sync_calls() -> None:
    forbidden = (".item(", ".cpu(", ".numpy(", ".tolist(", "synchronize(")
    for fn in (DivergenceDetector.compute, kl_divergence, js_divergence, mean_softmax):
        src = inspect.getsource(fn)
        for token in forbidden:
            assert token not in src, (
                f"{fn.__qualname__} contains forbidden host-sync call {token!r}"
            )


def test_satisfies_detector_protocol() -> None:
    from guard.detectors.base import Detector

    q = torch.full((4,), 0.25)
    assert isinstance(DivergenceDetector(_baseline_with_q(q)), Detector)


# --------------------------------------------------------------------------------------
# Regression (C): half precision must not produce NaN
#
# eps = 1e-12 underflows to exactly 0 in float16, so `x.clamp_min(0) + eps` no longer
# smooths anything and `(pc + eps).log()` evaluates log(0) = -inf. The 0·(-inf) product for
# a zero-probability class then yields NaN. Autocast serving hands the monitor exactly these
# dtypes, and a peaked batch-mean softmax has hard zeros routinely — so this was the default
# production path. A NaN silently never crosses an alert threshold: the monitor goes blind
# with nothing logged anywhere. The fix promotes half inputs to float32 before smoothing.
# --------------------------------------------------------------------------------------
_HALF_DTYPES = [torch.float16, torch.bfloat16]


def _realistic_logits() -> torch.Tensor:
    """B=16, C=1000 ImageNet-shaped logits, scaled so the mean softmax has exact zeros."""
    torch.manual_seed(11)
    return torch.randn(16, 1000, dtype=torch.float32) * 20.0


def _reference_q(c: int = 1000) -> torch.Tensor:
    torch.manual_seed(12)
    q = torch.rand(c, dtype=torch.float32)
    return q / q.sum()


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_half_precision_divergences_are_finite_and_bounded(dtype: torch.dtype) -> None:
    p = mean_softmax(_realistic_logits().to(dtype))
    q = _reference_q().to(dtype)

    kl = kl_divergence(p, q, EPS)
    js = js_divergence(p, q, EPS)

    assert torch.isfinite(kl), f"{dtype}: KL is {kl}"
    assert torch.isfinite(js), f"{dtype}: JS is {js}"
    assert float(kl) >= 0.0, f"{dtype}: KL negative: {kl}"
    # JS is bounded in [0, log 2] by construction; the eps smoothing can only pull it in.
    assert 0.0 <= float(js) <= math.log(2.0) + 1e-6, f"{dtype}: JS out of range: {js}"


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_half_precision_divergences_match_float32(dtype: torch.dtype) -> None:
    logits = _realistic_logits()
    q32 = _reference_q()

    for fn in (kl_divergence, js_divergence):
        ref = float(fn(mean_softmax(logits), q32, EPS))
        got = float(fn(mean_softmax(logits.to(dtype)), q32.to(dtype), EPS))
        # Loose bound on purpose: the logits and Q are quantized before the detector ever
        # sees them (bfloat16 keeps ~8 mantissa bits), so the *inputs* differ by ~1e-2
        # relative. The assertion checks the half path computes the same quantity, not that
        # it is bit-accurate.
        assert abs(ref - got) / max(abs(ref), 1e-12) < 5e-2, f"{dtype} {fn.__name__}: {ref} {got}"


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_detector_half_precision_scores_are_finite(dtype: torch.dtype) -> None:
    det = DivergenceDetector(_baseline_with_q(_reference_q()))

    scores = det.compute(_realistic_logits().to(dtype), embeddings=torch.empty(0)).scores

    assert set(scores) == {"kl_divergence", "js_divergence"}
    assert torch.isfinite(scores["kl_divergence"]), f"{dtype}: {scores['kl_divergence']}"
    assert torch.isfinite(scores["js_divergence"]), f"{dtype}: {scores['js_divergence']}"
    assert float(scores["kl_divergence"]) >= 0.0
    assert 0.0 <= float(scores["js_divergence"]) <= math.log(2.0) + 1e-6


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_detector_half_precision_matches_float32(dtype: torch.dtype) -> None:
    det = DivergenceDetector(_baseline_with_q(_reference_q()))
    logits = _realistic_logits()

    ref = det.compute(logits, embeddings=torch.empty(0)).scores
    got = det.compute(logits.to(dtype), embeddings=torch.empty(0)).scores

    for name in ref:
        a, b = float(ref[name]), float(got[name])
        assert abs(a - b) / max(abs(a), 1e-12) < 5e-2, f"{dtype} {name}: fp32={a} half={b}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hard_zero_probabilities_do_not_produce_nan(dtype: torch.dtype) -> None:
    """A peaked P with *exact* zeros — the input that turned KL/JS into NaN in half."""
    logits = torch.zeros(4, 512, dtype=dtype)
    logits[:, 0] = 800.0  # exp(-800) underflows to exactly 0.0 in every float width here
    p = mean_softmax(logits)
    # Guard the premise: without real zeros this test would pass vacuously.
    assert int((p == 0).sum()) == 511, f"{dtype}: expected a one-hot P, got {int((p == 0).sum())}"

    q = torch.full((512,), 1.0 / 512, dtype=dtype)
    kl = kl_divergence(p, q, EPS)
    js = js_divergence(p, q, EPS)

    assert torch.isfinite(kl) and torch.isfinite(js), f"{dtype}: KL={kl} JS={js}"
    assert float(kl) >= 0.0
    assert 0.0 <= float(js) <= math.log(2.0) + 1e-6


@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_half_precision_results_are_promoted_to_float32(dtype: torch.dtype) -> None:
    # Promotion is the fix; asserting it stops a future "keep it in half" optimization from
    # silently reintroducing the NaN.
    p = mean_softmax(_realistic_logits().to(dtype))
    assert kl_divergence(p, _reference_q().to(dtype), EPS).dtype == torch.float32
    assert js_divergence(p, _reference_q().to(dtype), EPS).dtype == torch.float32
