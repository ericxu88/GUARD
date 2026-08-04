"""Embedding-drift detector (P1-05).

Tracks *covariate shift* in the penultimate embedding space ``z ∈ ℝ^{B×D}``. Three signals:

* **Per-sample Mahalanobis distance** to the reference Gaussian
  ``d_i² = (z_i − μ_ref)ᵀ Σ_ref⁻¹ (z_i − μ_ref)`` — a per-sample OOD score whose batch mean
  tracks covariate shift. With the precision matrix cached this is two matmuls, ``O(B·D²)``.
* **Mean shift** — Mahalanobis distance of the streaming (Welford) running mean to the
  reference mean. Free: it reuses the same precision matrix on a single vector.
* **Covariance drift** — ``‖Σ_run·Σ_ref⁻¹ − I‖_F``, scale-free and zero when the running
  covariance matches the reference.

Why covariance drift is not computed every step
-----------------------------------------------
``Σ_run·Σ_ref⁻¹`` is a ``D×D × D×D`` matmul: ``O(D³)``. At D=768 that is 4.5·10⁸ FLOPs —
comparable to a whole transformer layer, and roughly *twenty times* the cost of everything
else this detector does. Running it on every inference step would quietly spend the entire
overhead budget the design doc sets at <2%, on a statistic whose input (a cumulative
running covariance) barely moves between consecutive batches.

So it is recomputed every ``cov_every`` steps and held in a device-resident buffer in
between; the metric reports the most recently computed value, which is what a cumulative
statistic means anyway. Set ``cov_every=1`` to restore per-step computation, or ``0`` to
drop the metric entirely.

MMD with random Fourier features is a documented stretch goal, not implemented here.

All math is device-agnostic ``torch`` and host-sync-free; ``compute`` returns 0-dim
device-resident tensors (CLAUDE.md prime directive #2). Every Python branch is on a shape
or a step counter — never on a tensor value, which would force an implicit device sync.
"""

from __future__ import annotations

import torch

from guard.baseline.schema import Baseline
from guard.detectors.base import DetectorResult, DeviceCache

_DEFAULT_COV_EVERY = 32
# Effective memory of ~100 batches (half-life ~69). Long enough to average out batch noise,
# short enough that a recovered stream produces a recovered metric.
_DEFAULT_DECAY = 0.99


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

    Numerically stable and ``O(D²)`` state — no growth with the number of batches. The
    covariance is the unbiased (``N-1``) estimator, matching ``torch.cov``'s default.

    All ``D×D`` intermediates are pre-allocated once and reused in place. A naive version
    allocates three ``D×D`` temporaries per step; at D=768 that is 7 MB of allocator churn
    on the monitor stream per inference batch, which is exactly the kind of per-request
    growth prime directive #5 forbids.

    Forgetting
    ----------
    ``decay`` multiplies the accumulated weight before each batch is folded in, making this
    an exponentially-weighted estimator over roughly ``1 / (1 - decay)`` recent batches
    (half-life ``ln 2 / ln(1/decay)``).

    ``decay = 1.0`` is the textbook infinite-memory estimator and is what a *batch*
    computation should be compared against — but it is the wrong choice for a live monitor.
    With infinite memory, a drift episode is permanently baked into the running statistics:
    once traffic recovers, the mean-shift and covariance-drift metrics stay elevated
    forever, so the alert they drive can never clear and the operator learns to ignore it.
    Forgetting is what makes "recovered" observable.

    Args:
        dim: embedding dimension ``D``.
        device / dtype: where the state lives; must match the batches fed to ``update``.
        decay: per-batch weight multiplier in ``(0, 1]``.
    """

    def __init__(
        self, dim: int, *, device: torch.device, dtype: torch.dtype, decay: float = 1.0
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.decay = decay
        self.n = 0.0
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self._m2 = torch.zeros(dim, dim, device=device, dtype=dtype)
        self._batch_m2 = torch.empty(dim, dim, device=device, dtype=dtype)
        self._outer = torch.empty(dim, dim, device=device, dtype=dtype)
        self._cov = torch.zeros(dim, dim, device=device, dtype=dtype)
        self._delta = torch.empty(dim, device=device, dtype=dtype)

    def update(self, x: torch.Tensor) -> None:
        """Fold a batch ``x`` ``[B, D]`` into the running statistics."""
        bn = x.shape[0]
        if bn == 0:
            return
        batch_mean = x.mean(dim=0)
        centered = x - batch_mean
        torch.mm(centered.t(), centered, out=self._batch_m2)  # [D, D] deviation products

        if self.n == 0.0:
            self.mean.copy_(batch_mean)
            self._m2.copy_(self._batch_m2)
            self.n = float(bn)
            return

        # Age the accumulated evidence before merging the new batch. With decay == 1.0 this
        # is exactly Chan's parallel update.
        aged = self.n * self.decay
        if self.decay != 1.0:
            self._m2.mul_(self.decay)

        torch.sub(batch_mean, self.mean, out=self._delta)
        tot = aged + bn
        self.mean.add_(self._delta, alpha=bn / tot)
        torch.outer(self._delta, self._delta, out=self._outer)
        self._m2.add_(self._batch_m2).add_(self._outer, alpha=aged * bn / tot)
        self.n = tot

    def covariance(self) -> torch.Tensor:
        """Unbiased covariance ``M2 / (n-1)``; zeros until at least 2 samples are seen.

        Returns an **internal buffer** that the next call overwrites. Clone it if you need
        to keep it — the detector only reads it immediately, which is why it can be reused.
        """
        if self.n < 2.0:
            return self._cov.zero_()
        return torch.div(self._m2, self.n - 1.0, out=self._cov)


class _CovDriftState:
    """Pre-allocated buffers for the rate-limited ``‖Σ_run·Σ_ref⁻¹ − I‖_F`` statistic."""

    def __init__(self, dim: int, *, device: torch.device, dtype: torch.dtype) -> None:
        self.eye = torch.eye(dim, device=device, dtype=dtype)
        self.scratch = torch.empty(dim, dim, device=device, dtype=dtype)
        self.value = torch.zeros((), device=device, dtype=dtype)

    def refresh(self, welford: WelfordCovariance, precision: torch.Tensor) -> None:
        """Recompute the drift statistic in place. The ``O(D³)`` step.

        Guarded on ``n >= 2``: with fewer samples the covariance is undefined and returns
        zeros, for which ``‖0·Σ⁻¹ − I‖_F = √D`` — a large spurious drift signal at exactly
        the moment the detector knows nothing.
        """
        if welford.n < 2:
            self.value.zero_()
            return
        torch.mm(welford.covariance(), precision, out=self.scratch)
        self.scratch.sub_(self.eye)
        self.value.copy_(torch.linalg.norm(self.scratch))


class EmbeddingDriftDetector:
    """Mahalanobis OOD score + streaming population drift against a reference Gaussian.

    Args:
        baseline: reference providing ``μ_ref`` (``embed_mean``) and ``Σ_ref⁻¹``
            (``embed_precision``).
        cov_every: recompute the ``O(D³)`` covariance-drift statistic every N steps.
            ``0`` drops the metric entirely; ``1`` computes it every step. See the module
            docstring for why the default is not 1.
        decay: exponential forgetting factor for the streaming population statistics, in
            ``(0, 1]``. The default keeps roughly the last 100 batches, so ``mean_shift``
            and ``cov_drift`` fall back toward zero once traffic returns to normal. Pass
            ``1.0`` for infinite memory — mathematically the textbook estimator, but its
            alerts can never clear after a drift episode.
    """

    name = "embedding"

    def __init__(
        self,
        baseline: Baseline,
        *,
        cov_every: int = _DEFAULT_COV_EVERY,
        decay: float = _DEFAULT_DECAY,
    ) -> None:
        if cov_every < 0:
            raise ValueError(f"cov_every must be >= 0, got {cov_every}")
        self.decay = decay
        self._baseline = baseline
        self._mu = DeviceCache(baseline.embed_mean)
        self._precision = DeviceCache(baseline.embed_precision)
        self._welford: WelfordCovariance | None = None
        self._cov_state: _CovDriftState | None = None
        self.cov_every = cov_every
        self._steps = 0

        self.metric_names: tuple[str, ...] = (
            "embedding_mahalanobis_mean",
            "embedding_mean_shift",
        )
        if cov_every > 0:
            self.metric_names = (*self.metric_names, "embedding_cov_drift")

    def _ensure_state(self, z: torch.Tensor) -> WelfordCovariance:
        """Lazily allocate the streaming state on the first batch's device/dtype."""
        if self._welford is None:
            dim = z.shape[1]
            self._welford = WelfordCovariance(dim, device=z.device, dtype=z.dtype, decay=self.decay)
            if self.cov_every > 0:
                self._cov_state = _CovDriftState(dim, device=z.device, dtype=z.dtype)
        return self._welford

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Reduce one batch of embeddings to scalar drift metrics."""
        del logits  # embedding drift depends only on the penultimate features
        z = embeddings
        mu = self._mu.like(z)
        precision = self._precision.like(z)

        # Per-sample OOD: batch-mean Mahalanobis distance.
        maha_mean = mahalanobis(z, mu, precision).mean()

        welford = self._ensure_state(z)
        welford.update(z)

        # Scaled mean shift: Mahalanobis distance of the running mean to the reference.
        mean_shift = mahalanobis(welford.mean.unsqueeze(0), mu, precision).squeeze(0)

        scores = {
            "embedding_mahalanobis_mean": maha_mean,
            "embedding_mean_shift": mean_shift,
        }
        cov_state = self._cov_state
        if cov_state is not None:
            if self._steps % self.cov_every == 0:
                cov_state.refresh(welford, precision)
            # Clone so the reported scalar is a stable snapshot: a caller holding the
            # result must not see it change when the next step refreshes the buffer.
            scores["embedding_cov_drift"] = cov_state.value.clone()
        self._steps += 1

        return DetectorResult(scores=scores)

    def reference(self) -> dict[str, object]:
        """Return the baseline artifacts this detector compares against."""
        return {
            "embed_mean": self._baseline.embed_mean,
            "embed_precision": self._baseline.embed_precision,
        }
