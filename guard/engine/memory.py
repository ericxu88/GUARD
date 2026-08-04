"""Bounded host-side staging buffers for the async scalar drain (Phase 2).

Scalar summaries leave the GPU in batches: every ``drain_every_k`` steps the engine
enqueues one asynchronous device→host copy of a ``[K, n_metrics]`` block into a **pinned**
host buffer and records a completion event. A background thread polls that event and reads
the buffer only once the copy has landed.

That handshake needs a buffer per copy in flight, and the number in flight must be
**bounded** (prime directive #5) — otherwise a stalled export thread turns into unbounded
pinned-memory growth, which is worse than dropping metrics because pinned memory is a
scarce, non-swappable system resource.

:class:`HostBufferPool` is that bound. It hands out a fixed number of buffers and returns
``None`` when they are all in flight; the caller drops the summary and increments
``guard_summaries_dropped_total``. Dropping monitoring data under backpressure is always
the right trade — the alternative is applying backpressure to inference.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from guard.engine.streams import MonitorEvent, pinned_empty


@dataclass(frozen=True)
class HostBuffer:
    """One pinned staging buffer plus the event that signals its copy has completed.

    Attributes:
        index: slot index in the owning pool; used to return the buffer.
        tensor: page-locked host tensor the device copies into.
        event: recorded on the monitor stream right after the copy is enqueued.
    """

    index: int
    tensor: torch.Tensor
    event: MonitorEvent


class HostBufferPool:
    """Fixed-size pool of pinned host buffers, safe for concurrent acquire/release.

    Args:
        count: number of buffers (== maximum drains in flight). Must be >= 1.
        shape: shape of each buffer, normally ``(drain_every_k, n_metrics)``.
        dtype: buffer dtype, normally ``torch.float32``.
        device: the *device* the copies come from; decides whether the host allocation is
            page-locked.
    """

    def __init__(
        self,
        *,
        count: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if count < 1:
            raise ValueError(f"pool count must be >= 1, got {count}")
        self._lock = threading.Lock()
        self._buffers = [
            HostBuffer(
                index=i, tensor=pinned_empty(shape, dtype, device), event=MonitorEvent(device)
            )
            for i in range(count)
        ]
        self._free: list[int] = list(range(count))
        self.capacity = count

    def acquire(self) -> HostBuffer | None:
        """Take a free buffer, or ``None`` when every buffer is still in flight."""
        with self._lock:
            if not self._free:
                return None
            return self._buffers[self._free.pop()]

    def release(self, buf: HostBuffer) -> None:
        """Return a buffer to the pool once its contents have been consumed.

        Idempotent: releasing a buffer that is already free is ignored rather than
        corrupting the free list, because the release path runs in the drain thread where
        an exception would kill metric export entirely.
        """
        with self._lock:
            if buf.index not in self._free:
                self._free.append(buf.index)

    @property
    def free(self) -> int:
        """Number of buffers currently available."""
        with self._lock:
            return len(self._free)

    @property
    def in_flight(self) -> int:
        """Number of buffers currently holding an un-consumed copy."""
        return self.capacity - self.free

    def nbytes(self) -> int:
        """Total pinned host memory held by the pool, in bytes."""
        return sum(b.tensor.element_size() * b.tensor.numel() for b in self._buffers)
