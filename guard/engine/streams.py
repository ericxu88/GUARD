"""CUDA stream / event facade with a synchronous fallback (Phase 2).

Every CUDA-specific primitive the overlap engine needs lives behind one of two tiny
classes here, and every one of them has a no-op counterpart for non-CUDA devices. That
buys two things:

* :mod:`guard.engine.monitor` contains **zero** ``if device.type == "cuda"`` branches, so
  the engine's logic (ring indexing, drain batching, backpressure, failure isolation) is
  one code path that is fully exercised by the CPU test suite in CI.
* On a CPU or MPS device GUARD still works — detectors simply run inline on the calling
  thread instead of overlapping on a monitor stream. Correct, just not free.

Stream priority
---------------
CUDA priorities are *inverted*: **more negative = higher priority**, and
``cudaDeviceGetStreamPriorityRange`` returns ``(least, greatest)`` where ``least`` is
``0``. The monitor stream is therefore created at priority ``0`` — the lowest the device
offers — so the scheduler always prefers inference kernels and monitoring only fills idle
SM capacity. If you own the serving loop, elevate *inference* with
``torch.cuda.Stream(priority=-1)`` to widen the gap further; GUARD cannot do that for you
because it does not own the stream inference runs on.

Nothing in this module synchronises. ``record``/``wait_event``/``query`` are all enqueue-
or poll-side operations; the only synchronising call is :meth:`MonitorStream.synchronize`,
which exists exclusively for shutdown and for tests.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

# CUDA's least (lowest) stream priority. See the module docstring: negative is higher.
LOW_PRIORITY = 0


def resolve_device(device: torch.device | str | None) -> torch.device:
    """Normalise a device argument, defaulting to CUDA when one is present.

    An unindexed accelerator name is resolved to a concrete index, because a tensor's
    ``.device`` always carries one. Leaving ``torch.device("mps")`` unresolved while every
    tensor reports ``mps:0`` makes an equality check fail for every batch — the monitor
    then rejects 100% of observations and reports nothing but a drop counter.

    ``"cuda"`` resolves to the *current* device index specifically, so ring buffers,
    streams, and events all land on the same physical GPU; a mismatch there is a
    notoriously hard-to-diagnose multi-GPU bug.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.index is not None or dev.type in ("cpu", "meta"):
        return dev
    if dev.type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    # Every other backend (mps, xpu, …) resolves an unindexed name to index 0.
    return torch.device(dev.type, 0)


def same_device(a: torch.device, b: torch.device) -> bool:
    """True when two devices refer to the same physical device.

    Tolerant of a missing index on either side: PyTorch resolves an unindexed accelerator
    name to the current device of that type, so ``cuda`` and ``cuda:0`` are the same place
    in a single-GPU process. Strict ``==`` on ``torch.device`` is not, which is exactly the
    trap that made the monitor drop every batch on MPS.
    """
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index


def is_cuda(device: torch.device) -> bool:
    """True when ``device`` is a CUDA device *and* CUDA is actually usable."""
    return device.type == "cuda" and torch.cuda.is_available()


def current_stream(device: torch.device) -> Any:
    """The stream work is currently being submitted to **on ``device``**, or ``None``.

    ``torch.cuda.current_stream()`` with no argument answers for
    ``torch.cuda.current_device()``, which is not the same question when a process drives
    more than one GPU. Always pass the device you mean.
    """
    return torch.cuda.current_stream(device) if is_cuda(device) else None


def capturing_graph(device: torch.device) -> bool:
    """True when ``device``'s current stream is capturing a CUDA graph.

    Recording an event or switching streams inside a capture corrupts the graph (design doc
    §6.4), so the engine drops observations while this is true. Note that it detects
    *capture*, not *replay*: a replayed graph writes its outputs into a static pool, which
    is a different hazard the engine handles by copying on the inference stream.
    """
    if not is_cuda(device):
        return False
    try:
        with torch.cuda.device(device):
            return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # pragma: no cover - older/newer torch without the helper
        return False


class MonitorEvent:
    """A CUDA event, or a no-op stand-in on non-CUDA devices.

    On CPU every operation has already completed by the time it is enqueued, so
    :meth:`query` is unconditionally ``True`` and :meth:`record` does nothing.
    """

    __slots__ = ("_event", "_recorded", "_timing")

    def __init__(self, device: torch.device, *, enable_timing: bool = False) -> None:
        self._timing = enable_timing
        self._recorded = False
        self._event: Any = (
            torch.cuda.Event(enable_timing=enable_timing)  # type: ignore[no-untyped-call]
            if is_cuda(device)
            else None
        )

    def record(self, stream: MonitorStream | Any | None = None) -> None:
        """Record the event on ``stream`` (a :class:`MonitorStream` or a raw CUDA stream).

        Passing ``None`` falls back to ``torch.cuda.current_stream()``, which resolves to
        the **current device's** current stream — not necessarily the device this event
        belongs to. In a multi-GPU process serving a model on ``cuda:1`` while the ambient
        current device is still ``cuda:0``, that silently records against an idle stream on
        the wrong card, so the dependency the monitor then waits on has nothing to do with
        the forward pass it is supposed to follow. Callers that care should pass the stream
        explicitly; :func:`current_stream` gets the right one for a known device.
        """
        self._recorded = True
        if self._event is None:
            return
        raw = stream.raw if isinstance(stream, MonitorStream) else stream
        if raw is None:
            self._event.record()
        else:
            self._event.record(raw)

    def wait(self, stream: MonitorStream | Any | None = None) -> None:
        """Make future work on ``stream`` wait for this event. Does not block the host.

        A never-recorded event is a no-op (PyTorch only creates the underlying
        ``cudaEvent_t`` on first record), which is what makes it safe to wait on a ring
        slot's completion event during the first pass through the ring.
        """
        if self._event is None or not self._recorded:
            return
        raw = stream.raw if isinstance(stream, MonitorStream) else stream
        if raw is None:
            self._event.wait()
        else:
            self._event.wait(raw)

    def query(self) -> bool:
        """True when all work captured by the event has completed. Never blocks."""
        if self._event is None:
            return True
        if not self._recorded:
            # An un-recorded CUDA event reports "complete"; that would let the drain
            # thread read a buffer whose copy was never enqueued.
            return False
        return bool(self._event.query())

    def synchronize(self) -> None:
        """Block the calling thread until the event completes. Shutdown/tests only."""
        if self._event is not None and self._recorded:
            self._event.synchronize()

    def elapsed_time(self, end: MonitorEvent) -> float:
        """Milliseconds between this event and ``end``; ``0.0`` if timing is unavailable.

        Returns ``0.0`` rather than raising when either event is still pending, so a
        caller polling for timing never blocks the thread it runs on.
        """
        if self._event is None or end._event is None or not (self._timing and end._timing):
            return 0.0
        if not (self.query() and end.query()):
            return 0.0
        return float(self._event.elapsed_time(end._event))


class MonitorStream:
    """A low-priority CUDA stream, or a transparent pass-through on non-CUDA devices.

    Args:
        device: device the stream belongs to.
        priority: CUDA stream priority; ``0`` (the default) is the *lowest* priority the
            device offers, which is what the monitor wants. Negative values raise the
            monitor above inference and will hurt serving latency — don't.
    """

    __slots__ = ("_priority", "_stream", "device")

    def __init__(self, device: torch.device, *, priority: int = LOW_PRIORITY) -> None:
        if priority < 0:
            raise ValueError(
                f"monitor stream priority must be >= 0 (0 is the lowest CUDA priority); "
                f"got {priority}. A negative priority would let monitoring preempt "
                f"inference, which violates prime directive #1."
            )
        self.device = device
        self._priority = priority
        self._stream: Any = (
            torch.cuda.Stream(device=device, priority=priority)  # type: ignore[no-untyped-call]
            if is_cuda(device)
            else None
        )

    @property
    def raw(self) -> Any:
        """The underlying ``torch.cuda.Stream``, or ``None`` on non-CUDA devices."""
        return self._stream

    @property
    def priority(self) -> int:
        """The priority the stream was created with."""
        return self._priority

    @property
    def is_cuda(self) -> bool:
        """True when this wraps a real CUDA stream."""
        return self._stream is not None

    @contextlib.contextmanager
    def activate(self) -> Iterator[None]:
        """Make this the current stream for the calling thread inside the block.

        A no-op on non-CUDA devices, where work runs inline. PyTorch's current-stream
        state is thread-local, so concurrent callers each get their own context.
        """
        if self._stream is None:
            yield
            return
        with torch.cuda.stream(self._stream):
            yield

    def wait_event(self, event: MonitorEvent) -> None:
        """Make future work on this stream wait for ``event`` — a GPU-side dependency.

        The calling thread does **not** block: this only inserts a wait node into the
        stream's queue.
        """
        if self._stream is None or event._event is None:
            return
        self._stream.wait_event(event._event)

    def protect(self, tensor: torch.Tensor) -> None:
        """Tell the caching allocator this stream will read ``tensor`` (prime directive #3).

        PyTorch's caching allocator frees a block back to the pool once the stream that
        *allocated* it is done. A tensor produced by ``forward()`` on the inference stream
        can therefore be handed to the next iteration while the monitor stream still has a
        queued read of it — a use-after-free that shows up as silently corrupted metrics,
        not as a crash. ``record_stream`` defers the reuse until this stream's work
        completes.

        Only meaningful for CUDA tensors allocated by the caching allocator; a no-op
        otherwise.
        """
        if self._stream is None or not tensor.is_cuda:
            return
        tensor.record_stream(self._stream)

    def synchronize(self) -> None:
        """Block until all work on this stream completes. Shutdown and tests only —
        calling this from ``observe()`` would serialise inference (prime directive #2)."""
        if self._stream is not None:
            self._stream.synchronize()


def pinned_empty(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Allocate a host buffer, page-locked when the target device is CUDA.

    Page-locked (pinned) memory is what makes ``copy_(..., non_blocking=True)`` genuinely
    asynchronous; copying to pageable memory silently degrades to a synchronising copy,
    which would stall the inference stream every ``drain_every_k`` steps.
    """
    return torch.empty(shape, dtype=dtype, pin_memory=is_cuda(device))
