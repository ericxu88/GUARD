"""P1-07: baseline builder — NumPy parity, validate(), determinism, regularisation, round-trip."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from guard.baseline import Baseline, build_baseline, load, save
from guard.baseline.schema import DEFAULT_ATOL, DEFAULT_RTOL
from guard.detectors.entropy import EntropyDetector

# Mirrors guard.baseline.compute._DEFAULT_REG_FACTOR / _EPS. Deliberately re-declared
# rather than imported: the parity reference must not share constants with the code it
# checks, otherwise a silent change to the regularisation strength would go unnoticed.
REG_FACTOR = 1e-4
ENTROPY_EPS = 1e-12
N_BINS = 64

# ─── helpers ──────────────────────────────────────────────────────────────────


def _synthetic_data(
    n_batches: int,
    batch_size: int,
    num_classes: int,
    embed_dim: int,
    seed: int,
    *,
    logit_scale: float = 3.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Reproducible (logits, embeddings) pairs drawn from a fixed generator."""
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(batch_size, num_classes, generator=g, dtype=torch.float64) * logit_scale,
            torch.randn(batch_size, embed_dim, generator=g, dtype=torch.float64),
        )
        for _ in range(n_batches)
    ]


def _build(seed: int = 0, c: int = 10, d: int = 8, n: int = 20, b: int = 32) -> Baseline:
    data = _synthetic_data(n, b, c, d, seed)
    return build_baseline(data, model_version=f"test@seed{seed}", num_classes=c, embed_dim=d)


# ─── independent NumPy reference ──────────────────────────────────────────────
# Nothing below imports a guard function to produce an expected value: the point is to
# re-derive every baseline statistic from the raw batches with a second implementation.


def _np_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _np_entropy(logits: np.ndarray, eps: float = ENTROPY_EPS) -> np.ndarray:
    p = _np_softmax(logits)
    return np.maximum(-(p * np.log(np.maximum(p, eps))).sum(axis=1), 0.0)


def _np_precision(emb: np.ndarray, reg_factor: float = REG_FACTOR) -> np.ndarray:
    """Regularised inverse covariance: ``(Σ + λI)⁻¹`` with ``λ = reg_factor·max(diag Σ)``."""
    cov = np.cov(emb.T, ddof=1)
    lam = reg_factor * max(float(np.diag(cov).max()), 1e-12)
    return np.linalg.inv(cov + lam * np.eye(emb.shape[1]))


# ─── parity against the independent reference ─────────────────────────────────


def test_statistics_match_numpy_reference() -> None:
    """Every stored statistic equals a second, independent computation over the same rows.

    The other tests here only assert invariants the builder enforces by construction
    (shapes, sums-to-1, positive-definiteness), which a wholly wrong value can still
    satisfy — e.g. a Welford update that mixes up batch weights still yields a symmetric
    PD precision matrix. This is the only test that pins the actual numbers.
    """
    c, d, n, b = 10, 8, 20, 32
    data = _synthetic_data(n, b, c, d, seed=11)
    # float64 output so the comparison is limited by the algorithms (streaming Welford vs
    # one-shot np.cov), not by float32 storage rounding.
    baseline = build_baseline(
        data,
        model_version="parity",
        num_classes=c,
        embed_dim=d,
        n_bins=N_BINS,
        reg_factor=REG_FACTOR,
        dtype=torch.float64,
    )

    all_logits = np.concatenate([lg.numpy() for lg, _ in data], axis=0)
    all_emb = np.concatenate([em.numpy() for _, em in data], axis=0)
    assert all_logits.shape == (n * b, c)

    assert baseline.sample_count == n * b

    # Q — mean softmax over every row, not a mean of per-batch means.
    expected_q = _np_softmax(all_logits).mean(axis=0)
    assert np.allclose(
        baseline.class_distribution.numpy(), expected_q, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )

    # μ_ref — plain column mean; catches a Welford update that weights batches wrongly.
    assert np.allclose(
        baseline.embed_mean.numpy(), all_emb.mean(axis=0), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )

    # Σ⁻¹ — the streaming M2 accumulation must reproduce a one-shot sample covariance
    # (ddof=1) before regularisation and inversion.
    assert np.allclose(
        baseline.embed_precision.numpy(),
        _np_precision(all_emb),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )

    # Entropy histogram — bin edges span [0, log C] uniformly, counts are exact integers.
    edges = np.linspace(0.0, math.log(c), N_BINS + 1)
    assert np.allclose(
        baseline.entropy_bin_edges.numpy(), edges, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )
    h_np = _np_entropy(all_logits)
    # Precondition: no sample sits outside the support, so the builder's clamp into
    # [0, log C] is a no-op here and np.histogram discards nothing.
    assert h_np.min() >= 0.0 and h_np.max() <= edges[-1]
    expected_hist, _ = np.histogram(h_np, bins=edges)
    assert np.array_equal(baseline.entropy_histogram.numpy(), expected_hist.astype(np.float64))
    assert float(baseline.entropy_histogram.sum()) == float(n * b)


def test_parity_holds_for_ragged_batches() -> None:
    """Unequal batch sizes — the case that separates a true pooled mean from a mean of means.

    With uniform batches those two are identical, so an accumulator that averages per-batch
    statistics instead of summing per-sample ones looks correct. A real DataLoader emits a
    short final batch, and then the difference is real.
    """
    c, d = 6, 5
    g = torch.Generator().manual_seed(23)
    data = [
        (
            torch.randn(size, c, generator=g, dtype=torch.float64) * 1.5,
            torch.randn(size, d, generator=g, dtype=torch.float64),
        )
        for size in (64, 3, 40, 1, 17)
    ]
    baseline = build_baseline(
        data, model_version="parity2", num_classes=c, embed_dim=d, dtype=torch.float64
    )
    assert baseline.sample_count == 125
    all_logits = np.concatenate([lg.numpy() for lg, _ in data], axis=0)
    all_emb = np.concatenate([em.numpy() for _, em in data], axis=0)

    assert np.allclose(
        baseline.class_distribution.numpy(),
        _np_softmax(all_logits).mean(axis=0),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    assert np.allclose(
        baseline.embed_mean.numpy(), all_emb.mean(axis=0), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )
    assert np.allclose(
        baseline.embed_precision.numpy(),
        _np_precision(all_emb),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


# ─── validate() passes ────────────────────────────────────────────────────────


def test_output_passes_validate() -> None:
    _build().validate()  # must not raise


# ─── shapes and ranges ────────────────────────────────────────────────────────


def test_class_distribution_shape_and_normalised() -> None:
    b = _build(c=5)
    assert b.class_distribution.shape == (5,)
    assert torch.isclose(b.class_distribution.sum(), torch.tensor(1.0), atol=1e-5)
    assert (b.class_distribution >= 0).all()


def test_entropy_histogram_shape() -> None:
    b = _build()
    assert b.entropy_histogram.shape == (64,)
    assert b.entropy_bin_edges.shape == (65,)
    assert (b.entropy_histogram >= 0).all()


def test_embed_mean_shape() -> None:
    b = _build(d=16)
    assert b.embed_mean.shape == (16,)


def test_precision_symmetric_and_pd() -> None:
    b = _build()
    d = b.embed_dim
    assert b.embed_precision.shape == (d, d)
    assert torch.allclose(b.embed_precision, b.embed_precision.T, atol=1e-5)
    eigvals = torch.linalg.eigvalsh(b.embed_precision)
    assert (eigvals > 0).all(), f"precision not PD; min eigval = {eigvals.min():.4g}"


# ─── determinism ──────────────────────────────────────────────────────────────


def test_deterministic_given_seed() -> None:
    b1 = _build(seed=42)
    b2 = _build(seed=42)
    assert torch.allclose(b1.class_distribution, b2.class_distribution)
    assert torch.allclose(b1.embed_mean, b2.embed_mean)
    assert torch.allclose(b1.embed_precision, b2.embed_precision)
    assert torch.allclose(b1.entropy_histogram, b2.entropy_histogram)


def test_different_seeds_differ() -> None:
    b1 = _build(seed=0)
    b2 = _build(seed=1)
    assert not torch.allclose(b1.embed_mean, b2.embed_mean)


# ─── regularisation (low-rank / singular input) ───────────────────────────────


def test_no_crash_on_low_rank_embeddings() -> None:
    # Embeddings confined to a 2-D subspace of R^8 → covariance is rank-deficient.
    d, c = 8, 5
    g = torch.Generator().manual_seed(7)
    basis = torch.randn(d, 2, generator=g, dtype=torch.float64)
    coords = torch.randn(200, 2, generator=g, dtype=torch.float64)
    emb = coords @ basis.T  # shape [200, 8], rank 2
    logits = torch.randn(200, c, generator=g, dtype=torch.float64)
    data = [(logits, emb)]
    b = build_baseline(data, model_version="low-rank", num_classes=c, embed_dim=d)
    b.validate()
    eigvals = torch.linalg.eigvalsh(b.embed_precision)
    assert (eigvals > 0).all()


def test_no_crash_on_single_batch() -> None:
    # n < 2 → covariance undefined; builder must still produce a valid baseline.
    data = _synthetic_data(1, 16, 4, 6, seed=3)
    b = build_baseline(data, model_version="one-batch", num_classes=4, embed_dim=6)
    b.validate()


def test_no_crash_on_constant_embeddings() -> None:
    # All embeddings identical → zero variance; regularisation must prevent singular inversion.
    c, d = 4, 6
    emb = torch.ones(50, d, dtype=torch.float64) * 3.14
    logits = torch.zeros(50, c, dtype=torch.float64)
    b = build_baseline([(logits, emb)], model_version="const", num_classes=c, embed_dim=d)
    b.validate()


# ─── regression: saturated logits must not empty the entropy histogram ────────


def _saturated_logits(n: int, c: int, dominant: int = 0) -> torch.Tensor:
    """Logits so peaked that float64 softmax rounds the winning probability to exactly 1.0."""
    logits = torch.full((n, c), -30.0, dtype=torch.float64)
    logits[:, dominant] = 60.0
    return logits


def test_saturated_logits_are_counted_in_histogram() -> None:
    """A well-trained (confident) model must not produce an empty entropy histogram.

    Regression for the ``-(p * (p + eps).log())`` formulation: for a saturated prediction
    ``log(1 + eps) > 0`` made H ≈ -1e-12, one ulp below the first bin edge, and
    ``torch.histogram`` silently *discards* out-of-range samples. Every confident sample
    therefore vanished and the reference histogram came out completely empty.
    """
    n, c, d = 128, 10, 4
    logits = _saturated_logits(n, c)
    g = torch.Generator().manual_seed(31)
    emb = torch.randn(n, d, generator=g, dtype=torch.float64)

    b = build_baseline([(logits, emb)], model_version="saturated", num_classes=c, embed_dim=d)

    assert float(b.entropy_histogram.sum()) == float(n), (
        "saturated samples were dropped from the entropy histogram"
    )
    # All mass belongs in the lowest bin: H = 0 for a one-hot prediction.
    assert float(b.entropy_histogram[0]) == float(n)


def test_saturated_baseline_scores_zero_pop_shift_on_its_own_data() -> None:
    """The calibration data itself must score ~0 population shift, never the empty-hist 0.5.

    An all-zero reference histogram normalises to an all-zero probability vector, against
    which *every* live batch scores a total-variation distance of exactly 0.5 — a
    plausible-looking mid-range number that never moves. That silent failure mode is what
    this pins down.
    """
    n, c, d = 128, 10, 4
    logits = _saturated_logits(n, c)
    g = torch.Generator().manual_seed(31)
    emb = torch.randn(n, d, generator=g, dtype=torch.float64)

    b = build_baseline([(logits, emb)], model_version="saturated", num_classes=c, embed_dim=d)
    shift = float(EntropyDetector(baseline=b).compute(logits, emb).scores["entropy_pop_shift"])

    assert shift < 1e-9, f"pop shift on the calibration data itself is {shift}, expected ~0"
    assert abs(shift - 0.5) > 0.4, "pop shift pinned at the empty-histogram value of 0.5"


# ─── device handling ──────────────────────────────────────────────────────────


def test_grad_tracking_input_is_detached() -> None:
    """Inputs captured from a live model carry autograd history; the builder must drop it.

    Without the ``.detach()`` the accumulators would join the caller's graph — keeping the
    whole activation history alive for the length of the calibration run, and making the
    stored baseline tensors non-leaf. This is the CPU stand-in for the CUDA path, where the
    same call also performs the device transfer.
    """
    c, d = 5, 4
    g = torch.Generator().manual_seed(13)
    logits = (torch.randn(64, c, generator=g, dtype=torch.float64) * 2.0).requires_grad_(True)
    emb = torch.randn(64, d, generator=g, dtype=torch.float64).requires_grad_(True)

    b = build_baseline([(logits, emb)], model_version="grad", num_classes=c, embed_dim=d)

    for name in ("class_distribution", "entropy_histogram", "embed_mean", "embed_precision"):
        t = getattr(b, name)
        assert not t.requires_grad, f"{name} still tracks gradients"
        assert t.grad_fn is None, f"{name} is still attached to the autograd graph"


@pytest.mark.gpu
def test_build_baseline_accepts_cuda_batches() -> None:
    """A CUDA dataloader is the normal source of these tensors; CPU accumulators must not
    trip a device-mismatch RuntimeError, and the result must match the CPU-fed baseline."""
    if not torch.cuda.is_available():  # pragma: no cover - guarded by the gpu marker
        pytest.skip("CUDA not available")
    c, d = 6, 5
    data = _synthetic_data(4, 32, c, d, seed=17)
    cuda_data = [(lg.cuda(), em.cuda()) for lg, em in data]

    on_cpu = build_baseline(
        data, model_version="dev", num_classes=c, embed_dim=d, dtype=torch.float64
    )
    on_cuda = build_baseline(
        cuda_data, model_version="dev", num_classes=c, embed_dim=d, dtype=torch.float64
    )

    # Baselines are stored on CPU regardless of where the calibration data lived.
    assert on_cuda.embed_mean.device.type == "cpu"
    assert torch.allclose(
        on_cuda.class_distribution, on_cpu.class_distribution, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )
    assert torch.allclose(
        on_cuda.embed_mean, on_cpu.embed_mean, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )
    assert torch.allclose(
        on_cuda.embed_precision, on_cpu.embed_precision, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )
    assert torch.equal(on_cuda.entropy_histogram, on_cpu.entropy_histogram)


# ─── error cases ──────────────────────────────────────────────────────────────


def test_empty_data_raises() -> None:
    with pytest.raises(ValueError, match="no batches"):
        build_baseline([], model_version="v0", num_classes=4, embed_dim=4)


def test_wrong_num_classes_raises() -> None:
    data = _synthetic_data(1, 8, 10, 4, seed=0)
    with pytest.raises(ValueError, match="classes"):
        build_baseline(data, model_version="v0", num_classes=5, embed_dim=4)


def test_wrong_embed_dim_raises() -> None:
    data = _synthetic_data(1, 8, 4, 8, seed=0)
    with pytest.raises(ValueError, match="dim"):
        build_baseline(data, model_version="v0", num_classes=4, embed_dim=4)


def test_mismatched_batch_size_raises() -> None:
    logits = torch.randn(8, 4, dtype=torch.float64)
    emb = torch.randn(7, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="batch size"):
        build_baseline([(logits, emb)], model_version="v0", num_classes=4, embed_dim=6)


# ─── save / load round-trip ───────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path: Path) -> None:
    original = _build(seed=5)
    path = tmp_path / "baseline.guard"
    save(original, path)
    loaded = load(path)
    assert torch.allclose(loaded.class_distribution, original.class_distribution, atol=1e-6)
    assert torch.allclose(loaded.embed_mean, original.embed_mean, atol=1e-6)
    assert torch.allclose(loaded.embed_precision, original.embed_precision, atol=1e-6)
    assert loaded.sample_count == original.sample_count
    assert loaded.model_version == original.model_version
