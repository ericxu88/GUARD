"""Detector contract.

A Detector turns one observed inference batch into one or more scalar metric tensors.
Implementations live next to this file (entropy.py, divergence.py, embedding.py).

Invariants every implementation MUST honor (see CLAUDE.md "Prime directives"):
  * `compute` is host-sync-free: no .item()/.cpu()/.numpy()/synchronize() inside it, and
    no Python branch on a tensor *value* (which forces an implicit sync on CUDA).
  * `compute` is allocation-light and writes device-resident 0-dim tensors.
  * The math uses device-agnostic torch ops so the same code runs on CPU (for the
    reference parity test) and CUDA (in production).
  * `metric_names` is a fixed tuple, decided at construction, listing exactly the keys
    `compute` puts in `DetectorResult.scores`. The engine allocates a fixed-width summary
    buffer from it before the first batch, so the schema cannot change at runtime.

Baseline tensors and device residency
-------------------------------------
A detector holds reference tensors (a class distribution, a `[D, D]` precision matrix)
that live on CPU inside the `Baseline` and must be read on the serving device. Calling
`.to(device)` inside `compute` would issue a host→device copy of every one of them on
**every inference step** — at D=768 that is a 2.3 MB H2D transfer per request, on the
monitor stream, competing with the very inference traffic GUARD promises not to disturb.
:class:`DeviceCache` does the move once per (device, dtype) and hands back the same tensor
forever after.
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


class DeviceCache:
    """One immutable reference tensor, materialised once per ``(device, dtype)``.

    Args:
        tensor: the source tensor, detached on construction. Mutating the original
            afterwards does not affect already-cached copies — treat baselines as
            immutable, which the store guarantees by checksumming them.
        max_entries: cache bound. A process serves one or two device/dtype combinations;
            the bound exists only so a pathological caller cannot grow the cache without
            limit (prime directive #5).
    """

    __slots__ = ("_cache", "_max_entries", "_source")

    def __init__(self, tensor: torch.Tensor, *, max_entries: int = 4) -> None:
        self._source = tensor.detach()
        self._cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._max_entries = max_entries

    def get(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The source tensor on ``device`` with ``dtype``, copied at most once."""
        key = (device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        moved = self._source.to(device=device, dtype=dtype)
        if len(self._cache) >= self._max_entries:
            self._cache.clear()
        self._cache[key] = moved
        return moved

    def like(self, tensor: torch.Tensor) -> torch.Tensor:
        """The source tensor matched to ``tensor``'s device and dtype."""
        return self.get(tensor.device, tensor.dtype)

    @property
    def source(self) -> torch.Tensor:
        """The original (CPU-resident) tensor."""
        return self._source


@runtime_checkable
class Detector(Protocol):
    """Structural interface for all detectors."""

    name: str
    metric_names: tuple[str, ...]

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        """Compute metrics for one batch.

        Args:
            logits: model outputs, shape [B, C] (pre- or post-softmax per detector docs).
            embeddings: penultimate features, shape [B, D].

        Returns:
            DetectorResult whose `scores` contains exactly `metric_names`, each a 0-dim
            device-resident tensor.
        """
        ...

    def reference(self) -> dict[str, object]:
        """Return the baseline artifact this detector depends on (may be empty)."""
        ...
