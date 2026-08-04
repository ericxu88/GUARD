"""Baseline artifact schema (LOCKED for Phase 1).

A Baseline is the known-good reference, computed offline from representative data and
bound to a specific model version. Detectors compare live behavior against it.

Tolerances DEFAULT_RTOL / DEFAULT_ATOL are the contract for all CPU-reference parity
tests: a CUDA detector result must match its CPU reference within these.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-7


@dataclass
class Baseline:
    """Reference statistics for one model version.

    Tensors are stored on CPU and moved to the serving device at load time. Shapes are
    validated by `validate()`.
    """

    model_version: str
    created_at: str  # ISO-8601 UTC
    sample_count: int
    num_classes: int  # C
    embed_dim: int  # D

    class_distribution: torch.Tensor  # Q, shape [C], non-negative, sums to 1
    entropy_histogram: torch.Tensor  # shape [n_bins]
    entropy_bin_edges: torch.Tensor  # shape [n_bins + 1]
    embed_mean: torch.Tensor  # mu_ref, shape [D]
    embed_precision: torch.Tensor  # Sigma_ref^{-1}, shape [D, D], symmetric PSD

    checksum: str = ""  # filled by the store on save; verified on load

    def validate(self) -> None:
        """Cheap structural checks. Raises ValueError on any violation."""
        c, d = self.num_classes, self.embed_dim
        if self.class_distribution.shape != (c,):
            raise ValueError(
                f"class_distribution must be [{c}], got {tuple(self.class_distribution.shape)}"
            )
        if not torch.isclose(
            self.class_distribution.sum(),
            torch.ones((), dtype=self.class_distribution.dtype),
            atol=1e-4,
        ):
            raise ValueError("class_distribution must sum to 1")
        if (self.class_distribution < 0).any():
            raise ValueError("class_distribution must be non-negative")
        if self.entropy_bin_edges.numel() != self.entropy_histogram.numel() + 1:
            raise ValueError("entropy_bin_edges must have one more element than entropy_histogram")
        if (self.entropy_histogram < 0).any():
            raise ValueError("entropy_histogram must be non-negative")
        # An empty histogram is not merely useless, it is actively misleading: the entropy
        # detector normalizes it into an all-zero reference, against which every live batch
        # scores a total-variation distance of exactly 0.5 forever. That is a plausible
        # mid-range number, which is far worse than an obvious failure — so reject it here,
        # where a bad baseline can still be caught at build/save/load time.
        if float(self.entropy_histogram.sum()) <= 0.0:
            raise ValueError(
                "entropy_histogram is empty (sums to 0); it would make entropy_pop_shift a "
                "constant 0.5. Rebuild the baseline over data that produced entropy values "
                "inside [0, log C]."
            )
        if self.embed_mean.shape != (d,):
            raise ValueError(f"embed_mean must be [{d}], got {tuple(self.embed_mean.shape)}")
        if self.embed_precision.shape != (d, d):
            raise ValueError(
                f"embed_precision must be [{d}, {d}], got {tuple(self.embed_precision.shape)}"
            )
        if not torch.allclose(self.embed_precision, self.embed_precision.T, atol=1e-4):
            raise ValueError("embed_precision must be symmetric")
        # eigvalsh has no half-precision kernel; check in float32 so a legitimately
        # half-precision baseline is validated rather than dying on NotImplementedError.
        precision = self.embed_precision
        if precision.dtype in (torch.float16, torch.bfloat16):
            precision = precision.float()
        eigvals = torch.linalg.eigvalsh(precision)
        if not (eigvals > 0).all():
            raise ValueError(
                "embed_precision must be positive definite "
                f"(min eigenvalue: {eigvals.min().item():.4g})"
            )
