"""Detector contract (LOCKED for Phase 1).

A Detector turns one observed inference batch into one or more scalar metric tensors.
Implementations live next to this file (entropy.py, divergence.py, embedding.py).

Invariants every implementation MUST honor (see CLAUDE.md "Prime directives"):
  * `compute` is host-sync-free: no .item()/.cpu()/.numpy()/synchronize() inside it.
  * `compute` is allocation-light and writes device-resident 0-dim tensors.
  * The math uses device-agnostic torch ops so the same code runs on CPU (for the
    reference parity test) and CUDA (in production).

Phase-2 note: the overlap engine will add an optional zero-copy variant that writes
into a pre-allocated summary buffer. The `DetectorResult` return path defined here is the
stable v1 contract; do not change its shape without updating the engine and all callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class DetectorResult:
    """Per-batch scalar outputs of a detector.

    `scores` maps a metric name (stable, snake_case, e.g. "entropy_mean") to a 0-dim
    tensor on the same device as the inputs. Values are NOT moved to host here — the
    engine drains them asynchronously every K steps.
    """

    scores: dict[str, torch.Tensor]


@runtime_checkable
class Detector(Protocol):
    """Structural interface for all detectors."""

    name: str

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Compute metrics for one batch.

        Args:
            logits: model outputs, shape [B, C] (pre- or post-softmax per detector docs).
            embeddings: penultimate features, shape [B, D].

        Returns:
            DetectorResult with device-resident 0-dim score tensors.
        """
        ...

    def reference(self) -> dict[str, object]:
        """Return the baseline artifact this detector depends on (may be empty)."""
        ...
