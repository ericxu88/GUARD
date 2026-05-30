"""KL / JS divergence detector (P1-04).

Tracks *prior / label shift*: when the mix of classes the model predicts drifts away from
the baseline class distribution ``Q``, the output distribution ``P`` diverges from ``Q``.

For one observed batch, ``P`` is the mean softmax over the batch — a soft "predicted-class
distribution" (the design doc allows either mean softmax or a class histogram; mean softmax
is smoother and stays host-sync-free). The detector reports both ``KL(P‖Q)`` and the
symmetric, bounded ``JS(P‖Q) ∈ [0, log 2]``; JS is the preferred alerting signal because its
bounded range makes thresholds stable.

All math is device-agnostic ``torch`` and numerically guarded (eps inside every log, clamped
probabilities) so zero-probability classes never produce NaN/inf. ``compute`` is
host-sync-free and returns 0-dim device-resident tensors (CLAUDE.md prime directive #2).
"""

from __future__ import annotations

import torch

from guard.baseline.schema import Baseline
from guard.detectors.base import DetectorResult

_DEFAULT_EPS = 1e-12


def _as_distribution(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp to non-negative, add ``eps`` smoothing, and renormalize into a distribution.

    Additive smoothing keeps this branch-free (no host sync): a degenerate all-zero input
    becomes uniform rather than a near-zero vector that fails to sum to 1.
    """
    x = x.clamp_min(0.0) + eps
    return x / x.sum()


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """``KL(P‖Q) = Σ_c P_c·log(P_c / Q_c)`` over distribution vectors ``[C]``.

    Numerically safe: inputs are renormalized and an ``eps`` floor inside the logs keeps the
    result finite even when ``Q`` (or ``P``) has zero-probability classes. Returns a 0-dim
    tensor; ``KL(P‖P) == 0`` exactly.
    """
    pc = _as_distribution(p, eps)
    qc = _as_distribution(q, eps)
    return (pc * ((pc + eps).log() - (qc + eps).log())).sum()


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """Jensen-Shannon divergence ``½KL(P‖M) + ½KL(Q‖M)``, ``M = ½(P+Q)``.

    Symmetric and bounded in ``[0, log 2]`` (nats). Returns a 0-dim tensor.
    """
    pc = _as_distribution(p, eps)
    qc = _as_distribution(q, eps)
    m = 0.5 * (pc + qc)
    return 0.5 * kl_divergence(pc, m, eps) + 0.5 * kl_divergence(qc, m, eps)


def mean_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Batch-mean softmax: the soft predicted-class distribution ``P``, shape ``[C]``."""
    return torch.softmax(logits, dim=1).mean(dim=0)


class DivergenceDetector:
    """Output-distribution drift via KL and JS divergence against baseline ``Q``.

    Args:
        baseline: reference whose ``class_distribution`` provides ``Q``.
        eps: numerical floor used in normalization and logs.
    """

    name = "divergence"

    def __init__(self, baseline: Baseline, eps: float = _DEFAULT_EPS) -> None:
        self.eps = eps
        self._baseline = baseline
        self._q = _as_distribution(baseline.class_distribution, eps)

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Reduce one batch of logits to KL and JS divergence against ``Q``."""
        del embeddings  # divergence depends only on the output distribution
        p = mean_softmax(logits)
        # Match Q to the live tensor's device/dtype (one small copy, not a host sync).
        q = self._q.to(device=p.device, dtype=p.dtype)
        return DetectorResult(
            scores={
                "kl_divergence": kl_divergence(p, q, self.eps),
                "js_divergence": js_divergence(p, q, self.eps),
            }
        )

    def reference(self) -> dict[str, object]:
        """Return the baseline artifact this detector compares against."""
        return {"class_distribution": self._baseline.class_distribution}
