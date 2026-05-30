"""Entropy detector (P1-03).

Per-sample Shannon entropy of the model's softmax output is the cheapest, earliest signal
of confidence collapse / input shift: a sustained rise in mean entropy usually precedes any
visible accuracy drop. This detector reduces a batch of logits to a few device-resident
scalars — window mean, a tail quantile, and (optionally) a population-shift distance against
the baseline entropy histogram.

All math is device-agnostic ``torch`` so the identical code runs on CPU (for the reference
parity test) and on a CUDA monitor stream in production. ``compute`` is host-sync-free: it
never calls ``.item()``/``.cpu()``/``.numpy()``/``synchronize()`` and returns 0-dim tensors
that the engine drains asynchronously (see CLAUDE.md prime directive #2).
"""

from __future__ import annotations

import torch

from guard.baseline.schema import Baseline
from guard.detectors.base import DetectorResult

# Floor inside the log so we never evaluate log(0); small enough not to perturb parity.
_DEFAULT_EPS = 1e-12


def softmax_entropy(logits: torch.Tensor, eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """Per-sample Shannon entropy ``H_i = -Σ_c p[i,c]·log(p[i,c]+eps)`` over softmax probs.

    Args:
        logits: raw (pre-softmax) model outputs, shape ``[B, C]``.
        eps: additive floor inside the log for numerical safety.

    Returns:
        Tensor of shape ``[B]`` on the same device/dtype as ``logits``.
    """
    p = torch.softmax(logits, dim=1)
    return -(p * (p + eps).log()).sum(dim=1)


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

        if baseline is not None:
            self._ref_edges: torch.Tensor | None = baseline.entropy_bin_edges
            self._ref_hist: torch.Tensor | None = _normalize_hist(baseline.entropy_histogram, eps)
        else:
            self._ref_edges = None
            self._ref_hist = None

    def _population_shift(self, h: torch.Tensor) -> torch.Tensor:
        """Total-variation distance between the batch entropy histogram and the baseline.

        Bins ``h`` on the baseline's edges via ``bucketize`` + ``scatter_add`` (all on
        device, no host sync), normalizes, and returns ``0.5 * sum|p - q|`` (0-dim tensor).
        """
        assert self._ref_edges is not None and self._ref_hist is not None
        # Cache reference tensors on the input device (one small H2D copy, not a host sync).
        edges = self._ref_edges.to(h.device)
        q = self._ref_hist.to(h.device)
        n_bins = q.numel()

        # Interior boundaries map a value to a bin index in [0, n_bins-1].
        idx = torch.bucketize(h, edges[1:-1])
        counts = torch.zeros(n_bins, device=h.device, dtype=q.dtype)
        counts.scatter_add_(0, idx, torch.ones_like(h, dtype=q.dtype))
        p = counts / counts.sum().clamp_min(self.eps)

        return 0.5 * (p - q).abs().sum()

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Reduce one batch of logits to scalar entropy metrics.

        ``embeddings`` is unused by this detector but kept to satisfy the shared contract.
        """
        del embeddings  # entropy depends only on the output distribution
        h = softmax_entropy(logits, self.eps)

        q = torch.tensor(self.quantile, device=h.device, dtype=h.dtype)
        scores: dict[str, torch.Tensor] = {
            "entropy_mean": h.mean(),
            "entropy_quantile": torch.quantile(h, q),
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
