"""P1-04: KL/JS divergence detector — NumPy parity, bounds, zero-prob stability."""

from __future__ import annotations

import inspect
import math

import numpy as np
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
