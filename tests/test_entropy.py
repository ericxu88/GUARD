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
    return -(p * np.log(p + eps)).sum(axis=1)


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
