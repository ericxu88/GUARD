"""Embedding-drift detector (P1-05).

Tracks *covariate shift* in the penultimate embedding space ``z ∈ ℝ^{B×D}``. Two signals,
both cheap and always-on:

* **Per-sample Mahalanobis distance** to the reference Gaussian
  ``d_i² = (z_i − μ_ref)ᵀ Σ_ref⁻¹ (z_i − μ_ref)`` — a per-sample OOD score whose batch mean
  tracks covariate shift. With the precision matrix cached this is two matmuls.
* **Streaming population drift** via Welford's algorithm: a numerically stable running
  mean/covariance over all observed embeddings, compared to the reference by a scaled
  mean-shift (Mahalanobis of the running mean) and a covariance drift
  (``‖Σ_run·Σ_ref⁻¹ − I‖_F``).

MMD with random Fourier features is a documented stretch goal, not implemented here.

All math is device-agnostic ``torch`` and host-sync-free; ``compute`` returns 0-dim
device-resident tensors (CLAUDE.md prime directive #2). The batch count is a Python int
derived from tensor *shape* (not from reading device data), so no ``.item()`` is needed.
"""

from __future__ import annotations

import torch

from guard.baseline.schema import Baseline
from guard.detectors.base import DetectorResult


def mahalanobis(z: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    """Per-sample Mahalanobis distance of rows of ``z`` ``[B, D]`` to ``mu`` ``[D]``.

    ``precision`` is ``Σ⁻¹`` ``[D, D]``. Returns ``[B]`` distances (the sqrt of the squared
    form, clamped at 0 for numerical safety). With ``precision == I`` this is the Euclidean
    distance to ``mu``.
    """
    diff = z - mu  # [B, D]
    # (diff @ precision) elementwise* diff, summed over D == diag(diff P diffᵀ).
    sq = ((diff @ precision) * diff).sum(dim=1)  # [B]
    return sq.clamp_min(0.0).sqrt()


class WelfordCovariance:
    """Streaming mean & covariance via Chan's parallel (batched) Welford update.

    Numerically stable and O(D²) state — no growth with the number of batches. The
    covariance is the unbiased (``N-1``) estimator, matching ``torch.cov``'s default.
    """

    def __init__(self, dim: int, *, device: torch.device, dtype: torch.dtype) -> None:
        self.n = 0
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self._m2 = torch.zeros(dim, dim, device=device, dtype=dtype)

    def update(self, x: torch.Tensor) -> None:
        """Fold a batch ``x`` ``[B, D]`` into the running statistics."""
        bn = x.shape[0]
        if bn == 0:
            return
        batch_mean = x.mean(dim=0)
        centered = x - batch_mean
        batch_m2 = centered.T @ centered  # [D, D] sum of deviation outer products

        if self.n == 0:
            self.mean = batch_mean
            self._m2 = batch_m2
            self.n = bn
            return

        delta = batch_mean - self.mean
        tot = self.n + bn
        self.mean = self.mean + delta * (bn / tot)
        self._m2 = self._m2 + batch_m2 + torch.outer(delta, delta) * (self.n * bn / tot)
        self.n = tot

    def covariance(self) -> torch.Tensor:
        """Unbiased covariance ``M2 / (n-1)``; zeros until at least 2 samples are seen."""
        if self.n < 2:
            return torch.zeros_like(self._m2)
        return self._m2 / (self.n - 1)


class EmbeddingDriftDetector:
    """Mahalanobis OOD score + streaming population drift against a reference Gaussian.

    Args:
        baseline: reference providing ``μ_ref`` (``embed_mean``) and ``Σ_ref⁻¹``
            (``embed_precision``).
    """

    name = "embedding"

    def __init__(self, baseline: Baseline) -> None:
        self._baseline = baseline
        self._mu = baseline.embed_mean
        self._precision = baseline.embed_precision
        self._welford: WelfordCovariance | None = None

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Reduce one batch of embeddings to scalar drift metrics."""
        del logits  # embedding drift depends only on the penultimate features
        z = embeddings
        mu = self._mu.to(device=z.device, dtype=z.dtype)
        precision = self._precision.to(device=z.device, dtype=z.dtype)

        # Per-sample OOD: batch-mean Mahalanobis distance.
        maha_mean = mahalanobis(z, mu, precision).mean()

        # Streaming population drift (cumulative across batches).
        if self._welford is None:
            self._welford = WelfordCovariance(z.shape[1], device=z.device, dtype=z.dtype)
        self._welford.update(z)

        # Scaled mean shift: Mahalanobis distance of the running mean to the reference.
        mean_shift = mahalanobis(self._welford.mean.unsqueeze(0), mu, precision).squeeze(0)

        # Covariance drift: ‖Σ_run·Σ_ref⁻¹ − I‖_F (scale-free, 0 when Σ_run == Σ_ref).
        cov = self._welford.covariance()
        d = cov.shape[0]
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        cov_drift = torch.linalg.norm(cov @ precision - eye)

        return DetectorResult(
            scores={
                "embedding_mahalanobis_mean": maha_mean,
                "embedding_mean_shift": mean_shift,
                "embedding_cov_drift": cov_drift,
            }
        )

    def reference(self) -> dict[str, object]:
        """Return the baseline artifacts this detector compares against."""
        return {
            "embed_mean": self._baseline.embed_mean,
            "embed_precision": self._baseline.embed_precision,
        }
