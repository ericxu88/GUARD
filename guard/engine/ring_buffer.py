"""Pre-allocated on-device ring buffer for captured inference outputs (Phase 2).

Detector kernels must never read the tensors ``forward()`` returned. Those belong to the
inference stream's allocation pool and can be recycled the moment the next iteration needs
memory — while the monitor stream still has queued reads of them. GUARD therefore copies
every observed batch into memory **it owns for the process lifetime**, and the detectors
read only from there.

The buffer is a fixed ``[slots, max_batch, ...]`` block allocated once at construction:
constant memory, zero per-request allocation (prime directive #5). Slot reuse is safe
without any explicit fence because all monitor work is enqueued in order on a single
stream — the detector read of slot *s* is queued before the next write to slot *s*, so
CUDA's stream ordering guarantees the read finishes first. Extra slots therefore buy
pipelining depth, not correctness; the default of 8 is plenty and keeps the footprint
small.

Sizing::

    bytes = slots * max_batch * (num_classes + embed_dim) * itemsize

For ``slots=8, max_batch=64, C=1000, D=768, float32`` that is ~3.6 MB — negligible next to
a ViT-B/16's weights and activations. Multiply ``slots`` only if you have measured a need.
"""

from __future__ import annotations

import torch


class RingBuffer:
    """Fixed-size circular store of ``(logits, embeddings)`` batches on one device.

    Args:
        slots: number of batches held concurrently (>= 1).
        max_batch: largest batch size accepted; larger batches are rejected by the caller.
        num_classes: logits width ``C`` (>= 1).
        embed_dim: embedding width ``D``; ``0`` skips the embedding allocation entirely for
            deployments that only monitor the output distribution.
        device: device to allocate on.
        dtype: storage dtype. Defaults to ``float32`` regardless of the model's compute
            dtype: the ``copy_`` that fills a slot casts on the fly (still async, still no
            host sync), and running the detector reductions in float32 avoids the
            catastrophic precision loss float16 causes in entropy and covariance sums.
    """

    def __init__(
        self,
        *,
        slots: int,
        max_batch: int,
        num_classes: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots}")
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if embed_dim < 0:
            raise ValueError(f"embed_dim must be >= 0, got {embed_dim}")

        self.slots = slots
        self.max_batch = max_batch
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype

        self.logits = torch.empty((slots, max_batch, num_classes), device=device, dtype=dtype)
        self.embeddings = torch.empty((slots, max_batch, embed_dim), device=device, dtype=dtype)

    def write(
        self,
        slot: int,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy one batch into ``slot`` and return views sized to the actual batch.

        The returned views alias GUARD-owned memory, so detectors may read them for as
        long as the monitor stream's queued work takes. Batches smaller than ``max_batch``
        write a prefix of the slot; the stale tail is never read because the views are
        sliced to ``logits.shape[0]``.

        Both copies are issued on whatever stream is current — the caller is responsible
        for having activated the monitor stream first.
        """
        b = logits.shape[0]
        logits_view = self.logits[slot, :b]
        logits_view.copy_(logits, non_blocking=True)

        embed_view = self.embeddings[slot, :b]
        if self.embed_dim > 0:
            embed_view.copy_(embeddings, non_blocking=True)
        return logits_view, embed_view

    def nbytes(self) -> int:
        """Total device memory held by the ring, in bytes."""
        return (
            self.logits.element_size() * self.logits.numel()
            + self.embeddings.element_size() * self.embeddings.numel()
        )
