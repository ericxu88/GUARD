"""P1-07: baseline builder — validate(), determinism, regularisation, round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guard.baseline import Baseline, build_baseline, load, save

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
