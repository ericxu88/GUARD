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
from guard.detectors.base import DetectorResult, DeviceCache

_DEFAULT_EPS = 1e-12


def _promote(x: torch.Tensor) -> torch.Tensor:
    """Promote half precision to float32; leave float32/float64 alone.

    ``eps = 1e-12`` underflows to exactly 0 in float16, so the smoothing that is supposed
    to prevent ``log(0)`` silently stops working and a peaked ``P`` (an exact zero in the
    batch-mean softmax, which happens routinely under autocast) turns both KL and JS into
    NaN. A NaN never crosses a threshold, so the drift alert would be off with no error
    anywhere — the worst possible failure mode for a monitor.
    """
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _as_distribution(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp to non-negative, add ``eps`` smoothing, and renormalize into a distribution.

    Additive smoothing keeps this branch-free (no host sync): a degenerate all-zero input
    becomes uniform rather than a near-zero vector that fails to sum to 1.
    """
    x = _promote(x).clamp_min(0.0) + eps
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
    metric_names: tuple[str, ...] = ("kl_divergence", "js_divergence")

    def __init__(self, baseline: Baseline, eps: float = _DEFAULT_EPS) -> None:
        self.eps = eps
        self._baseline = baseline
        self._q = DeviceCache(_as_distribution(baseline.class_distribution, eps))

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Reduce one batch of logits to KL and JS divergence against ``Q``."""
        del embeddings  # divergence depends only on the output distribution
        # Promote before matching Q's dtype: casting Q *down* to half would defeat the
        # promotion inside _as_distribution and put the NaN back.
        p = mean_softmax(_promote(logits))
        # Q, already resident on the serving device — no per-batch host->device copy.
        q = self._q.like(p)
        return DetectorResult(
            scores={
                "kl_divergence": kl_divergence(p, q, self.eps),
                "js_divergence": js_divergence(p, q, self.eps),
            }
        )

    def reference(self) -> dict[str, object]:
        """Return the baseline artifact this detector compares against."""
        return {"class_distribution": self._baseline.class_distribution}
