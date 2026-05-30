"""Baseline builder (P1-07).

Computes the reference statistics used by every detector from a dataset of known-good
(logits, embeddings) pairs. This runs **offline** — never on the inference hot path.

Statistics produced:

* ``class_distribution`` (Q) — mean softmax over the entire dataset; represents the
  expected class-probability vector under normal conditions.
* ``entropy_histogram`` / ``entropy_bin_edges`` — histogram of per-sample Shannon entropy
  values; used by :class:`EntropyDetector` for population-shift scoring.
* ``embed_mean`` (μ_ref) — mean embedding vector.
* ``embed_precision`` (Σ_ref⁻¹) — regularized inverse of the embedding covariance;
  used by :class:`EmbeddingDriftDetector` for Mahalanobis distances.

The builder accepts any iterable of ``(logits, embeddings)`` tensor pairs, so it works
with PyTorch DataLoaders, generators, or hand-crafted lists. All accumulation is done in
float64 for numerical stability; the output tensors are cast back to the dtype of the
first batch (defaulting to float32 when empty).

Covariance regularization: ``Σ_reg = Σ + λ·I`` where ``λ = reg_factor · max(diag(Σ))``.
This prevents singular-matrix crashes on low-rank embeddings (e.g. when the dataset is
smaller than the embedding dimension) and on constant dimensions.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable

import torch

from guard.baseline.schema import Baseline
from guard.detectors.entropy import softmax_entropy

_EPS = 1e-12
_DEFAULT_REG_FACTOR = 1e-4
_DEFAULT_N_BINS = 64


def _welford_update(
    n: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    x: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Batched Welford update; returns (new_n, new_mean, new_M2)."""
    bn = x.shape[0]
    if bn == 0:
        return n, mean, m2
    batch_mean = x.mean(0)
    centered = x - batch_mean
    batch_m2 = centered.T @ centered
    if n == 0:
        return bn, batch_mean.clone(), batch_m2.clone()
    delta = batch_mean - mean
    tot = n + bn
    new_mean = mean + delta * (bn / tot)
    new_m2 = m2 + batch_m2 + torch.outer(delta, delta) * (n * bn / tot)
    return tot, new_mean, new_m2


def build_baseline(
    data: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    model_version: str,
    num_classes: int,
    embed_dim: int,
    n_bins: int = _DEFAULT_N_BINS,
    reg_factor: float = _DEFAULT_REG_FACTOR,
    dtype: torch.dtype = torch.float32,
) -> Baseline:
    """Compute a :class:`Baseline` from an iterable of ``(logits, embeddings)`` batches.

    Args:
        data: iterable of ``(logits [B, C], embeddings [B, D])`` tensor pairs.
        model_version: version string bound to this baseline (e.g. ``"vit-b16@abc123"``).
        num_classes: expected number of output classes C; validated against each batch.
        embed_dim: expected embedding dimension D; validated against each batch.
        n_bins: number of histogram bins for the entropy distribution.
        reg_factor: covariance regularization coefficient (see module docstring).

    Returns:
        A validated :class:`Baseline`. Raises :class:`ValueError` if no data is provided
        or if any batch has the wrong shape.
    """
    # Accumulators — float64 throughout for numerical stability.
    softmax_sum = torch.zeros(num_classes, dtype=torch.float64)
    entropy_vals: list[torch.Tensor] = []
    n_embed = 0
    embed_mean = torch.zeros(embed_dim, dtype=torch.float64)
    embed_m2 = torch.zeros(embed_dim, embed_dim, dtype=torch.float64)
    n_samples = 0

    for logits, embeddings in data:
        if logits.shape[1] != num_classes:
            raise ValueError(
                f"expected logits with {num_classes} classes, got shape {tuple(logits.shape)}"
            )
        if embeddings.shape[1] != embed_dim:
            raise ValueError(
                f"expected embeddings with dim {embed_dim}, got shape {tuple(embeddings.shape)}"
            )
        if logits.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"logits and embeddings must have the same batch size, "
                f"got {logits.shape[0]} vs {embeddings.shape[0]}"
            )
        logits_f64 = logits.detach().double()
        emb_f64 = embeddings.detach().double()

        softmax_sum += torch.softmax(logits_f64, dim=1).sum(0)
        entropy_vals.append(softmax_entropy(logits_f64).cpu())
        n_embed, embed_mean, embed_m2 = _welford_update(n_embed, embed_mean, embed_m2, emb_f64)
        n_samples += logits.shape[0]

    if n_samples == 0:
        raise ValueError("data iterable produced no batches; cannot build a baseline")

    # Class distribution Q — cast to output dtype; renormalise to absorb rounding.
    q = (softmax_sum / n_samples).to(dtype)
    q = q / q.sum()

    # Entropy histogram.
    all_h = torch.cat(entropy_vals)
    log_c = torch.tensor(num_classes, dtype=torch.float64).log()
    edges = torch.linspace(0.0, float(log_c), n_bins + 1, dtype=dtype)
    hist = torch.histogram(all_h.to(dtype), bins=edges).hist

    # Embedding mean and regularised precision (Σ + λ·I)⁻¹.
    mu = embed_mean.to(dtype)
    if n_embed < 2:
        cov = torch.zeros(embed_dim, embed_dim, dtype=torch.float64)
    else:
        cov = embed_m2 / (n_embed - 1)

    diag_max = cov.diagonal().max().clamp_min(_EPS)
    reg = reg_factor * diag_max * torch.eye(embed_dim, dtype=torch.float64)
    cov_reg = cov + reg
    precision = torch.linalg.inv(cov_reg).to(dtype)
    # Symmetrise to absorb float rounding before validate() checks allclose.
    precision = (precision + precision.T) / 2.0

    baseline = Baseline(
        model_version=model_version,
        created_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sample_count=n_samples,
        num_classes=num_classes,
        embed_dim=embed_dim,
        class_distribution=q,
        entropy_histogram=hist,
        entropy_bin_edges=edges,
        embed_mean=mu,
        embed_precision=precision,
    )
    baseline.validate()
    return baseline
