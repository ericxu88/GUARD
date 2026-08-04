"""Entropy detector (P1-03).

Per-sample Shannon entropy of the model's softmax output is the cheapest, earliest signal
of confidence collapse / input shift: a sustained rise in mean entropy usually precedes any
visible accuracy drop. This detector reduces a batch of logits to a few device-resident
scalars — window mean, a tail quantile, and (optionally) a population-shift distance against
the baseline entropy histogram.

All math is device-agnostic ``torch`` so the identical code runs on CPU (for the reference
parity test) and on a CUDA monitor stream in production. ``compute`` is host-sync-free: it
never calls ``.item()``/``.cpu()``/``.numpy()``/``synchronize()`` and returns 0-dim tensors
that the engine drains asynchronously (see CLAUDE.md prime directive #2). Every reference
tensor is moved to the serving device once via :class:`~guard.detectors.base.DeviceCache`,
never per batch.
"""

from __future__ import annotations

import torch

from guard.baseline.schema import Baseline
from guard.detectors.base import DetectorResult, DeviceCache

# Floor inside the log so we never evaluate log(0); small enough not to perturb parity.
_DEFAULT_EPS = 1e-12


def softmax_entropy(logits: torch.Tensor, eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """Per-sample Shannon entropy ``H_i = -Σ_c p[i,c]·log(max(p[i,c], eps))`` over softmax probs.

    The ``eps`` floor goes **inside** the clamp, not added to ``p``. Writing it the obvious
    way, ``-(p * (p + eps).log())``, makes ``log(1 + eps) > 0`` for a saturated prediction,
    so ``H`` comes out *negative* (about ``-1e-12``) instead of zero. That looks harmless
    and is not: ``torch.histogram`` silently discards samples below its first bin edge, so
    every confident sample vanished from the baseline entropy histogram — and a baseline
    built on a well-trained model could end up entirely empty, pinning
    ``entropy_pop_shift`` at a constant 0.5 on the very data it was calibrated on.

    Clamping instead keeps ``H ∈ [0, log C]`` exactly: ``p ≤ 1`` implies
    ``log(max(p, eps)) ≤ 0``, and ``p = 0`` contributes ``0`` rather than ``0·log 0 = NaN``.

    Half precision is promoted to float32 first. ``eps = 1e-12`` underflows to zero in
    float16, which would reintroduce ``log(0)``, and a 10-bit mantissa cannot represent a
    1000-class entropy sum with useful accuracy anyway.

    Args:
        logits: raw (pre-softmax) model outputs, shape ``[B, C]``.
        eps: probability floor inside the log for numerical safety.

    Returns:
        Tensor of shape ``[B]``, non-negative, on the same device as ``logits``
        (float32 when ``logits`` is half precision, otherwise ``logits.dtype``).
    """
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    p = torch.softmax(logits, dim=1)
    # Split the negation out: `-(x).clamp_min(0)` would clamp before negating.
    h = -(p * p.clamp_min(eps).log()).sum(dim=1)
    return h.clamp_min(0.0)


def _normalize_hist(hist: torch.Tensor, eps: float) -> torch.Tensor:
    """Turn a (possibly count-valued) histogram into a probability vector that sums to 1."""
    total = hist.sum()
    return hist / total.clamp_min(eps)


class EntropyDetector:
    """Windowed entropy statistics with optional baseline population-shift scoring.

    Args:
        quantile: tail quantile of the per-sample entropy to report (e.g. 0.99).
        baseline: optional reference; when given, ``compute`` adds an ``entropy_pop_shift``
            metric (total-variation distance between the batch entropy histogram, binned on
            the baseline's edges, and the baseline histogram). Bounded in ``[0, 1]``.
        eps: numerical floor used in logs and histogram normalization.
    """

    name = "entropy"

    def __init__(
        self,
        quantile: float = 0.99,
        baseline: Baseline | None = None,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self.eps = eps
        self._baseline = baseline
        self._q_scalar = DeviceCache(torch.tensor(quantile))

        self.metric_names: tuple[str, ...] = ("entropy_mean", "entropy_quantile")
        if baseline is not None:
            # bucketize needs only the interior boundaries; slice once, not per batch.
            self._interior_edges: DeviceCache | None = DeviceCache(
                baseline.entropy_bin_edges[1:-1].contiguous()
            )
            self._ref_hist: DeviceCache | None = DeviceCache(
                _normalize_hist(baseline.entropy_histogram, eps)
            )
            self._n_bins = int(baseline.entropy_histogram.numel())
            self.metric_names = (*self.metric_names, "entropy_pop_shift")
        else:
            self._interior_edges = None
            self._ref_hist = None
            self._n_bins = 0

    def _population_shift(self, h: torch.Tensor) -> torch.Tensor:
        """Total-variation distance between the batch entropy histogram and the baseline.

        Bins ``h`` on the baseline's edges via ``bucketize`` + ``scatter_add`` (all on
        device, no host sync), normalizes, and returns ``0.5 * sum|p - q|`` (0-dim tensor).

        Values outside the baseline's range fall into the first or last bin rather than
        being discarded, so the two histograms always describe the same total mass and the
        distance stays a true TV distance in ``[0, 1]``.
        """
        edges = self._interior_edges
        ref = self._ref_hist
        if edges is None or ref is None:  # unreachable: guarded by the caller
            return torch.zeros((), device=h.device, dtype=h.dtype)

        interior = edges.like(h)
        q = ref.like(h)

        idx = torch.bucketize(h, interior)
        counts = torch.zeros(self._n_bins, device=h.device, dtype=h.dtype)
        counts.scatter_add_(0, idx, torch.ones_like(h))
        p = counts / counts.sum().clamp_min(self.eps)

        return 0.5 * (p - q).abs().sum()

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Reduce one batch of logits to scalar entropy metrics.

        ``embeddings`` is unused by this detector but kept to satisfy the shared contract.
        """
        del embeddings  # entropy depends only on the output distribution
        h = softmax_entropy(logits, self.eps)

        scores: dict[str, torch.Tensor] = {
            "entropy_mean": h.mean(),
            "entropy_quantile": torch.quantile(h, self._q_scalar.like(h)),
        }
        if self._baseline is not None:
            scores["entropy_pop_shift"] = self._population_shift(h)
        return DetectorResult(scores=scores)

    def reference(self) -> dict[str, object]:
        """Return the baseline artifact this detector compares against (empty if none)."""
        if self._baseline is None:
            return {}
        return {
            "entropy_histogram": self._baseline.entropy_histogram,
            "entropy_bin_edges": self._baseline.entropy_bin_edges,
        }
