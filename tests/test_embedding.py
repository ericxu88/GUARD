"""P1-05: embedding-drift detector — Mahalanobis parity, Welford covariance, degeneracy."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from guard.baseline.schema import DEFAULT_ATOL, DEFAULT_RTOL, Baseline
from guard.detectors.embedding import EmbeddingDriftDetector, WelfordCovariance, mahalanobis


def _baseline(mu: torch.Tensor, precision: torch.Tensor) -> Baseline:
    d = mu.numel()
    return Baseline(
        model_version="t",
        created_at="2026-01-01T00:00:00Z",
        sample_count=1000,
        num_classes=3,
        embed_dim=d,
        class_distribution=torch.full((3,), 1.0 / 3),
        entropy_histogram=torch.zeros(8),
        entropy_bin_edges=torch.linspace(0.0, 1.0, 9),
        embed_mean=mu,
        embed_precision=precision,
    )


# --------------------------------------------------------------------------------------
# Mahalanobis
# --------------------------------------------------------------------------------------
def test_mahalanobis_matches_numpy_reference() -> None:
    torch.manual_seed(0)
    d = 6
    z = torch.randn(20, d, dtype=torch.float64)
    mu = torch.randn(d, dtype=torch.float64)
    a = torch.randn(d, d, dtype=torch.float64)
    precision = a @ a.T + torch.eye(d, dtype=torch.float64)

    got = mahalanobis(z, mu, precision).numpy()

    diff = (z - mu).numpy()
    ref = np.sqrt(np.einsum("bi,ij,bj->b", diff, precision.numpy(), diff))
    assert np.allclose(got, ref, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_mahalanobis_identity_precision_is_euclidean() -> None:
    torch.manual_seed(1)
    d = 5
    z = torch.randn(10, d, dtype=torch.float64)
    mu = torch.randn(d, dtype=torch.float64)
    got = mahalanobis(z, mu, torch.eye(d, dtype=torch.float64))
    ref = torch.linalg.norm(z - mu, dim=1)
    assert torch.allclose(got, ref, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_mahalanobis_at_mean_is_zero() -> None:
    d = 4
    mu = torch.arange(d, dtype=torch.float64)
    z = mu.unsqueeze(0)
    got = mahalanobis(z, mu, torch.eye(d, dtype=torch.float64))
    assert torch.allclose(got, torch.zeros(1, dtype=torch.float64), atol=DEFAULT_ATOL)


# --------------------------------------------------------------------------------------
# Welford streaming covariance
# --------------------------------------------------------------------------------------
def test_welford_covariance_matches_torch_cov_single_batch() -> None:
    torch.manual_seed(2)
    d = 7
    x = torch.randn(500, d, dtype=torch.float64)
    w = WelfordCovariance(d, device=x.device, dtype=x.dtype)
    w.update(x)
    # torch.cov expects [variables, observations].
    expected = torch.cov(x.T)
    assert torch.allclose(w.covariance(), expected, rtol=1e-6, atol=1e-8)
    assert torch.allclose(w.mean, x.mean(0), rtol=1e-6, atol=1e-8)


def test_welford_covariance_matches_torch_cov_across_batches() -> None:
    torch.manual_seed(3)
    d = 5
    full = torch.randn(900, d, dtype=torch.float64)
    w = WelfordCovariance(d, device=full.device, dtype=full.dtype)
    for batch in full.split(37):  # uneven trailing batch on purpose
        w.update(batch)
    expected = torch.cov(full.T)
    assert torch.allclose(w.covariance(), expected, rtol=1e-6, atol=1e-8)
    assert torch.allclose(w.mean, full.mean(0), rtol=1e-6, atol=1e-8)
    assert w.n == 900


def test_welford_constant_batch_zero_covariance_no_nan() -> None:
    d = 4
    x = torch.ones(50, d, dtype=torch.float64) * 3.0
    w = WelfordCovariance(d, device=x.device, dtype=x.dtype)
    w.update(x)
    cov = w.covariance()
    assert torch.all(torch.isfinite(cov))
    assert torch.allclose(cov, torch.zeros(d, d, dtype=torch.float64), atol=DEFAULT_ATOL)


# --------------------------------------------------------------------------------------
# Detector end-to-end + degeneracy
# --------------------------------------------------------------------------------------
def test_detector_scores_finite_and_shaped() -> None:
    torch.manual_seed(4)
    d = 8
    mu = torch.zeros(d, dtype=torch.float64)
    det = EmbeddingDriftDetector(_baseline(mu, torch.eye(d, dtype=torch.float64)))
    z = torch.randn(64, d, dtype=torch.float64)
    scores = det.compute(torch.empty(0), z).scores
    assert set(scores) == {
        "embedding_mahalanobis_mean",
        "embedding_mean_shift",
        "embedding_cov_drift",
    }
    for name, v in scores.items():
        assert isinstance(v, torch.Tensor) and v.ndim == 0, name
        assert torch.isfinite(v), name
        assert v.device == z.device, name


def test_no_drift_when_batch_matches_reference() -> None:
    torch.manual_seed(5)
    d = 6
    base_data = torch.randn(4000, d, dtype=torch.float64)
    mu = base_data.mean(0)
    cov = torch.cov(base_data.T)
    precision = torch.linalg.inv(cov)
    det = EmbeddingDriftDetector(_baseline(mu, precision))

    scores = det.compute(torch.empty(0), base_data).scores
    # Running mean ≈ μ_ref and Σ_run ≈ Σ_ref ⇒ both population drift scores ≈ 0.
    assert float(scores["embedding_mean_shift"]) < 1e-3
    assert float(scores["embedding_cov_drift"]) < 1e-3


def test_drift_detected_on_shifted_batch() -> None:
    torch.manual_seed(6)
    d = 6
    base_data = torch.randn(4000, d, dtype=torch.float64)
    mu = base_data.mean(0)
    precision = torch.linalg.inv(torch.cov(base_data.T))
    det = EmbeddingDriftDetector(_baseline(mu, precision))

    shifted = torch.randn(4000, d, dtype=torch.float64) + 5.0  # clear mean shift
    scores = det.compute(torch.empty(0), shifted).scores
    assert float(scores["embedding_mahalanobis_mean"]) > 1.0
    assert float(scores["embedding_mean_shift"]) > 1.0


def test_constant_batch_no_nan() -> None:
    d = 5
    det = EmbeddingDriftDetector(_baseline(torch.zeros(d), torch.eye(d)))
    z = torch.full((32, d), 2.0)
    scores = det.compute(torch.empty(0), z).scores
    for v in scores.values():
        assert torch.isfinite(v)


def test_single_sample_batch_no_nan() -> None:
    # n < 2 ⇒ covariance undefined; detector must still return finite scores.
    d = 4
    det = EmbeddingDriftDetector(_baseline(torch.zeros(d), torch.eye(d)))
    scores = det.compute(torch.empty(0), torch.randn(1, d)).scores
    for v in scores.values():
        assert torch.isfinite(v)


def test_cov_drift_is_zero_when_welford_has_fewer_than_two_samples() -> None:
    # Bug: covariance() returns zeros when n < 2, so ‖0·Σ⁻¹ − I‖_F = sqrt(D) — a large
    # spurious drift signal on the very first call. Fix: guard returns 0 instead.
    d = 6
    det = EmbeddingDriftDetector(_baseline(torch.zeros(d), torch.eye(d, dtype=torch.float64)))
    scores = det.compute(torch.empty(0), torch.randn(1, d, dtype=torch.float64)).scores
    assert float(scores["embedding_cov_drift"]) == 0.0, (
        f"expected 0, got {float(scores['embedding_cov_drift']):.4g} "
        f"(was sqrt(D)={d**0.5:.4g} before fix)"
    )


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------
def test_compute_has_no_host_sync_calls() -> None:
    forbidden = (".item(", ".cpu(", ".numpy(", ".tolist(", "synchronize(")
    for fn in (
        EmbeddingDriftDetector.compute,
        WelfordCovariance.update,
        WelfordCovariance.covariance,
        mahalanobis,
    ):
        src = inspect.getsource(fn)
        for token in forbidden:
            assert token not in src, (
                f"{fn.__qualname__} contains forbidden host-sync call {token!r}"
            )


def test_satisfies_detector_protocol() -> None:
    from guard.detectors.base import Detector

    d = 4
    assert isinstance(EmbeddingDriftDetector(_baseline(torch.zeros(d), torch.eye(d))), Detector)
