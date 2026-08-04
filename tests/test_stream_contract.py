"""The CUDA ordering contract, verified without a CUDA device.

GUARD's memory-safety argument is not about *what* operations it issues — it is about the
**order** it issues them in. CUDA guarantees that work submitted to one stream executes in
submission order, and that a stream waiting on an event resumes only after the work
recorded before that event completes. Given those guarantees, correctness follows entirely
from program order:

    wait(slot_done[s], inference)   ─ do not overwrite a slot the monitor is still reading
    ring.write(s)                   ─ capture, in the caller's stream
    record(slot_ready[s], inference)─ signal only after the capture is enqueued
    ── switch to the monitor stream ──
    wait_event(slot_ready[s])       ─ detectors do not start before the capture
    detector.compute() × N
    record(slot_done[s], monitor)   ─ release the slot for reuse

A CPU run cannot check any of that: the stream and event facade degrades to no-ops, so
every ordering bug is invisible. What *is* checkable anywhere is the program order itself.
This module swaps in a recording fake for the whole CUDA layer, drives the real engine
through it, and asserts each link in the chain above.

That makes these tests the CPU-side proof of the design's central claim, and the regression
guard for the bug that motivated the redesign: capturing on the monitor stream instead of
the inference stream, where a static output buffer (CUDA-graph replay,
``torch.compile(mode="reduce-overhead")``, a TensorRT binding) is rewritten by the next
iteration while the capture is still queued. They do not replace ``tests/test_gpu_engine.py``
— only hardware can confirm the CUDA runtime honours the contract — but they confirm GUARD
asks for the right thing.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from guard.baseline.schema import Baseline
from guard.detectors.base import DetectorResult
from guard.engine.monitor import GuardMonitor

C, D, MAXB = 6, 4, 8


# --------------------------------------------------------------------------------------
# A recording stand-in for the CUDA layer
# --------------------------------------------------------------------------------------
class Trace(list[tuple[Any, ...]]):
    """The ordered log of every CUDA-layer operation the engine issued."""

    def names(self) -> list[str]:
        return [entry[0] for entry in self]

    def index_of(self, entry: tuple[Any, ...]) -> int:
        assert entry in self, f"{entry} never happened; trace was:\n{self.pretty()}"
        return self.index(entry)

    def before(self, first: tuple[Any, ...], second: tuple[Any, ...]) -> bool:
        return self.index_of(first) < self.index_of(second)

    def pretty(self) -> str:
        return "\n".join(f"  {i:3d} {e}" for i, e in enumerate(self))


class FakeStream:
    """Stands in for ``torch.cuda.Stream``; records instead of enqueuing."""

    def __init__(self, label: str, trace: Trace, priority: int = 0) -> None:
        self.label = label
        self.priority = priority
        self._trace = trace

    def wait_event(self, event: FakeEvent) -> None:
        self._trace.append(("stream.wait_event", self.label, event.label))

    def synchronize(self) -> None:
        # Recorded so a test can assert it never happens inside observe(): a synchronize on
        # the hot path is a prime-directive #2 violation.
        self._trace.append(("stream.synchronize", self.label))

    def __repr__(self) -> str:
        return f"FakeStream({self.label})"


class FakeEvent:
    """Stands in for ``torch.cuda.Event``. ``query()`` reports complete so drains land."""

    def __init__(self, label: str, trace: Trace, enable_timing: bool = False) -> None:
        self.label = label
        self.enable_timing = enable_timing
        self._trace = trace
        self.recorded_on: str | None = None

    def record(self, stream: FakeStream | None = None) -> None:
        label = stream.label if stream is not None else "<ambient-current>"
        self.recorded_on = label
        self._trace.append(("event.record", self.label, label))

    def wait(self, stream: FakeStream | None = None) -> None:
        label = stream.label if stream is not None else "<ambient-current>"
        self._trace.append(("event.wait", self.label, label))

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        self._trace.append(("event.synchronize", self.label))

    def elapsed_time(self, end: FakeEvent) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"FakeEvent({self.label})"


class FakeCuda:
    """Installs the fake CUDA layer and hands back the trace."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.trace = Trace()
        self._counter = 0
        self.inference = FakeStream("inference", self.trace)
        self._labels: dict[int, str] = {}
        self._install(monkeypatch)

    def _next_label(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import guard.engine.memory as memory_mod
        import guard.engine.monitor as monitor_mod
        import guard.engine.streams as streams_mod

        trace = self.trace

        def fake_stream_ctor(device: Any = None, priority: int = 0) -> FakeStream:
            return FakeStream(self._next_label("stream"), trace, priority)

        def fake_event_ctor(enable_timing: bool = False, **_: Any) -> FakeEvent:
            return FakeEvent(self._next_label("event"), trace, enable_timing)

        @contextlib.contextmanager
        def fake_stream_ctx(stream: FakeStream) -> Iterator[None]:
            trace.append(("stream.enter", stream.label))
            try:
                yield
            finally:
                trace.append(("stream.exit", stream.label))

        # `is_cuda` is what flips the facade from no-op to real; forcing it True is what
        # makes the engine take the CUDA code path on a machine that has no CUDA.
        monkeypatch.setattr(streams_mod, "is_cuda", lambda device: device.type != "cpu_only")
        monkeypatch.setattr(torch.cuda, "Stream", fake_stream_ctor)
        monkeypatch.setattr(torch.cuda, "Event", fake_event_ctor)
        monkeypatch.setattr(torch.cuda, "stream", fake_stream_ctx)
        # The monitor asks for the current stream *of its own device*; hand back the one
        # standing in for inference so the trace can tell the two apart.
        monkeypatch.setattr(monitor_mod, "current_stream", lambda device: self.inference)
        monkeypatch.setattr(monitor_mod, "capturing_graph", lambda device: False)
        # Pinned allocation would need a real CUDA context.
        monkeypatch.setattr(
            memory_mod,
            "pinned_empty",
            lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
        )

    def label_events(self, monitor: GuardMonitor) -> None:
        """Give the engine's events meaningful names once it has been constructed."""
        for i, event in enumerate(monitor._slot_ready):
            event._event.label = f"ready[{i}]"
        for i, event in enumerate(monitor._slot_done):
            event._event.label = f"done[{i}]"
        for buf in monitor._pool._buffers:
            buf.event._event.label = f"drain[{buf.index}]"
        monitor._stream.raw.label = "monitor"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
class TracingDetector:
    """Logs each compute into the same trace, so it can be ordered against stream ops."""

    name = "tracing"
    metric_names: tuple[str, ...] = ("traced_value",)

    def __init__(self, trace: Trace) -> None:
        self._trace = trace

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        del embeddings
        self._trace.append(("detector.compute", self.name))
        return DetectorResult(scores={"traced_value": logits.mean()})

    def reference(self) -> dict[str, object]:
        return {}


def _baseline() -> Baseline:
    return Baseline(
        model_version="contract@1",
        created_at="2026-01-01T00:00:00Z",
        sample_count=128,
        num_classes=C,
        embed_dim=D,
        class_distribution=torch.full((C,), 1.0 / C),
        entropy_histogram=torch.full((8,), 16.0),
        entropy_bin_edges=torch.linspace(0.0, math.log(C), 9),
        embed_mean=torch.zeros(D),
        embed_precision=torch.eye(D),
    )


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    return FakeCuda(monkeypatch)


def _build(fake: FakeCuda, **kwargs: Any) -> GuardMonitor:
    params: dict[str, Any] = {
        "num_classes": C,
        "embed_dim": D,
        "max_batch": MAXB,
        "ring_slots": 2,
        "drain_every_k": 2,
        "start_drain_thread": False,
    }
    params.update(kwargs)
    monitor = GuardMonitor([TracingDetector(fake.trace)], **params)
    fake.label_events(monitor)
    fake.trace.clear()  # discard construction noise; only observe() is under test
    return monitor


def _batch(b: int = MAXB) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(b, C), torch.randn(b, D)


def _wrap_ring_write(monitor: GuardMonitor, trace: Trace) -> None:
    """Log ring writes so they can be ordered against the stream operations."""
    original = monitor._ring.write

    def traced(slot: int, logits: torch.Tensor, embeddings: torch.Tensor) -> Any:
        trace.append(("ring.write", slot))
        return original(slot, logits, embeddings)

    monitor._ring.write = traced  # type: ignore[method-assign]


# --------------------------------------------------------------------------------------
# The contract, link by link
# --------------------------------------------------------------------------------------
def test_capture_is_issued_on_the_inference_stream(fake_cuda: FakeCuda) -> None:
    """The whole memory-safety argument: the ring copy is ordered against the caller's own
    stream, so it cannot race an in-place rewrite of a static output buffer."""
    monitor = _build(fake_cuda)
    _wrap_ring_write(monitor, fake_cuda.trace)
    monitor.observe(*_batch())
    trace = fake_cuda.trace

    write = trace.index_of(("ring.write", 0))
    enter = trace.index_of(("stream.enter", "monitor"))
    assert write < enter, (
        "the ring copy was issued inside the monitor-stream context. It must be issued in "
        f"the caller's stream, before the switch.\n{trace.pretty()}"
    )
    monitor.close()


def test_readiness_event_is_recorded_on_the_inference_stream(fake_cuda: FakeCuda) -> None:
    """Recorded on the monitor stream, the event would signal the wrong work entirely; via
    a bare ``record()`` it would land on whatever device happens to be current."""
    monitor = _build(fake_cuda)
    monitor.observe(*_batch())
    trace = fake_cuda.trace

    assert ("event.record", "ready[0]", "inference") in trace, trace.pretty()
    assert ("event.record", "ready[0]", "monitor") not in trace
    assert ("event.record", "ready[0]", "<ambient-current>") not in trace, (
        "the readiness event was recorded without naming a stream, so it lands on the "
        "current device's current stream — the wrong GPU in a multi-GPU process"
    )
    monitor.close()


def test_capture_precedes_the_readiness_signal(fake_cuda: FakeCuda) -> None:
    """Signalling before copying would let the detectors read an unwritten slot."""
    monitor = _build(fake_cuda)
    _wrap_ring_write(monitor, fake_cuda.trace)
    monitor.observe(*_batch())
    trace = fake_cuda.trace

    assert trace.before(("ring.write", 0), ("event.record", "ready[0]", "inference")), (
        f"the slot was signalled ready before it was written\n{trace.pretty()}"
    )
    monitor.close()


def test_detectors_wait_for_the_capture_before_running(fake_cuda: FakeCuda) -> None:
    monitor = _build(fake_cuda)
    monitor.observe(*_batch())
    trace = fake_cuda.trace

    assert trace.before(
        ("stream.wait_event", "monitor", "ready[0]"), ("detector.compute", "tracing")
    ), f"detectors were queued without waiting for the capture\n{trace.pretty()}"
    # ...and the detector work belongs to the monitor stream, not the caller's.
    enter = trace.index_of(("stream.enter", "monitor"))
    exit_ = trace.index_of(("stream.exit", "monitor"))
    compute = trace.index_of(("detector.compute", "tracing"))
    assert enter < compute < exit_, f"detector ran outside the monitor stream\n{trace.pretty()}"
    monitor.close()


def test_slot_reuse_waits_for_the_previous_detector_read(fake_cuda: FakeCuda) -> None:
    """The mirror-image hazard of capturing on the inference stream.

    With the copy in the caller's stream, inference could overwrite ring slot *s* while the
    monitor is still reading it from ``ring_slots`` steps ago. The engine must make the
    inference stream wait on that slot's completion event *before* the write.
    """
    monitor = _build(fake_cuda, ring_slots=2)
    _wrap_ring_write(monitor, fake_cuda.trace)
    # Prime the ring: on the very first pass a slot has never been read, so there is
    # genuinely nothing to wait for and the engine correctly issues no wait. The contract
    # under test is the steady state, once slots start being reused.
    for _ in range(2):
        monitor.observe(*_batch())
    fake_cuda.trace.clear()
    for _ in range(4):  # two more full wraps of a two-slot ring
        monitor.observe(*_batch())
    trace = fake_cuda.trace

    # Every write is preceded by a wait on that slot's completion event, on the *inference*
    # stream — a wait issued on the monitor stream would order nothing.
    waits = [i for i, e in enumerate(trace) if e == ("event.wait", "done[0]", "inference")]
    writes = [i for i, e in enumerate(trace) if e == ("ring.write", 0)]
    assert len(writes) == 2, f"expected slot 0 to be written twice\n{trace.pretty()}"
    assert len(waits) == 2, f"expected a wait before each write of slot 0\n{trace.pretty()}"
    for wait, write in zip(waits, writes, strict=True):
        assert wait < write, f"slot 0 was overwritten before the wait\n{trace.pretty()}"

    # And the slot is only released after its detectors have run.
    for i, entry in enumerate(trace):
        if entry == ("event.record", "done[0]", "monitor"):
            preceding = trace[:i]
            assert ("detector.compute", "tracing") in preceding, (
                f"slot 0 was released before any detector read it\n{trace.pretty()}"
            )
            break
    else:
        pytest.fail(f"slot 0 was never released for reuse\n{trace.pretty()}")
    monitor.close()


def test_slot_release_follows_the_detectors(fake_cuda: FakeCuda) -> None:
    monitor = _build(fake_cuda)
    monitor.observe(*_batch())
    trace = fake_cuda.trace
    assert trace.before(("detector.compute", "tracing"), ("event.record", "done[0]", "monitor")), (
        f"the slot was released before the detectors ran\n{trace.pretty()}"
    )
    monitor.close()


def test_drain_copy_is_recorded_on_the_monitor_stream(fake_cuda: FakeCuda) -> None:
    """The device→host copy must be queued behind the detectors on the monitor stream, so
    the block it copies is complete, and off the inference stream so it costs nothing."""
    monitor = _build(fake_cuda, drain_every_k=2)
    monitor.observe(*_batch())
    monitor.observe(*_batch())
    trace = fake_cuda.trace

    drains = [e for e in trace if e[0] == "event.record" and e[1].startswith("drain")]
    assert len(drains) == 1, f"expected exactly one drain per K steps\n{trace.pretty()}"
    drain = drains[0]
    assert drain[2] == "monitor", (
        f"the D2H copy was queued on {drain[2]}, not the monitor stream — it would then "
        f"cost the inference stream a transfer every K steps\n{trace.pretty()}"
    )
    assert trace.before(("detector.compute", "tracing"), drain)
    # It belongs to the monitor stream's context, not the caller's.
    enters = [i for i, e in enumerate(trace) if e == ("stream.enter", "monitor")]
    exits = [i for i, e in enumerate(trace) if e == ("stream.exit", "monitor")]
    idx = trace.index_of(drain)
    assert any(a < idx < b for a, b in zip(enters, exits, strict=True)), trace.pretty()
    monitor.close()


def test_observe_never_synchronises(fake_cuda: FakeCuda) -> None:
    """Prime directive #2, at the level of program order.

    ``tests/test_gpu_engine.py`` asserts the same property on hardware with
    ``set_sync_debug_mode("error")``, which also catches implicit syncs inside torch ops.
    This catches an explicit one anywhere in the enqueue path, on any machine.
    """
    monitor = _build(fake_cuda, drain_every_k=2)
    for _ in range(8):
        monitor.observe(*_batch())
    names = fake_cuda.trace.names()
    assert "stream.synchronize" not in names, fake_cuda.trace.pretty()
    assert "event.synchronize" not in names, fake_cuda.trace.pretty()
    monitor.close()


def test_full_step_order_is_exactly_the_contract(fake_cuda: FakeCuda) -> None:
    """One step, whole sequence, pinned. Any reordering of the engine breaks this."""
    monitor = _build(fake_cuda, ring_slots=1, drain_every_k=4)
    _wrap_ring_write(monitor, fake_cuda.trace)
    monitor.observe(*_batch())  # prime: nothing has read slot 0 yet, so no wait is due
    fake_cuda.trace.clear()
    monitor.observe(*_batch())  # steady state: the slot is being reused

    assert list(fake_cuda.trace) == [
        ("event.wait", "done[0]", "inference"),  # do not clobber a slot still being read
        ("ring.write", 0),  # capture, in the caller's stream
        ("event.record", "ready[0]", "inference"),  # signal only after the copy is queued
        ("stream.enter", "monitor"),
        ("stream.wait_event", "monitor", "ready[0]"),  # detectors wait for the capture
        ("detector.compute", "tracing"),
        ("event.record", "done[0]", "monitor"),  # release the slot for reuse
        ("stream.exit", "monitor"),
    ], fake_cuda.trace.pretty()
    monitor.close()


def test_a_failing_detector_still_releases_its_slot(fake_cuda: FakeCuda) -> None:
    """If a detector exception skipped the release, the inference stream would wait on an
    event that is never recorded again and the ring would seize after one wrap."""

    class Boom:
        name = "boom"
        metric_names: tuple[str, ...] = ("boom_metric",)

        def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
            raise RuntimeError("detector exploded")

        def reference(self) -> dict[str, object]:
            return {}

    monitor = GuardMonitor(
        [Boom()],
        num_classes=C,
        embed_dim=D,
        max_batch=MAXB,
        ring_slots=2,
        drain_every_k=2,
        start_drain_thread=False,
        max_detector_failures=100,
    )
    fake_cuda.label_events(monitor)
    fake_cuda.trace.clear()

    for _ in range(4):
        monitor.observe(*_batch())

    releases = [e for e in fake_cuda.trace if e[0] == "event.record" and e[1].startswith("done")]
    assert len(releases) == 4, (
        f"a raising detector suppressed the slot release; the ring would seize\n"
        f"{fake_cuda.trace.pretty()}"
    )
    monitor.close()


def test_real_detectors_run_unchanged_through_the_cuda_path(fake_cuda: FakeCuda) -> None:
    """The fake layer must not be exercising a different code path than production."""
    from guard.detectors import DivergenceDetector, EmbeddingDriftDetector, EntropyDetector

    baseline = _baseline()
    monitor = GuardMonitor(
        [
            EntropyDetector(baseline=baseline),
            DivergenceDetector(baseline),
            EmbeddingDriftDetector(baseline, cov_every=1),
        ],
        num_classes=C,
        embed_dim=D,
        max_batch=MAXB,
        ring_slots=2,
        drain_every_k=2,
        start_drain_thread=False,
    )
    fake_cuda.label_events(monitor)
    torch.manual_seed(0)
    for _ in range(4):
        monitor.observe(*_batch())
        monitor.drain_pending()

    snapshot = monitor.aggregator.snapshot()
    assert set(snapshot) == set(monitor.metric_names)
    assert all(math.isfinite(v) for v in snapshot.values()), snapshot
    assert monitor.stats().disabled_detectors == ()
    monitor.close()


# --------------------------------------------------------------------------------------
# The CUDA API surface the engine depends on
# --------------------------------------------------------------------------------------
# The fake layer above proves the engine issues operations in the right ORDER. It cannot
# prove those operations exist with the signatures assumed — the fakes would happily accept
# a call that real torch rejects. These checks close that gap against the installed torch,
# so a version bump that moves an API breaks CI here rather than on a GPU box at 3am.


def test_cuda_stream_accepts_the_arguments_the_engine_passes() -> None:
    # Stream is C-backed, so introspect the documented signature rather than the ctor.
    doc = torch.cuda.Stream.__doc__ or ""
    assert "priority" in doc, "torch.cuda.Stream no longer documents a priority argument"
    for method in ("wait_event", "synchronize", "record_event", "query"):
        assert hasattr(torch.cuda.Stream, method), f"torch.cuda.Stream lost {method}()"
    assert hasattr(torch.cuda.Stream, "priority")


def test_cuda_event_exposes_record_wait_query_and_timing() -> None:
    import inspect

    for method in ("record", "wait", "query", "synchronize", "elapsed_time"):
        assert hasattr(torch.cuda.Event, method), f"torch.cuda.Event lost {method}()"
    # The engine relies on record()/wait() taking an explicit stream — passing None would
    # silently target the *current device's* current stream, which is the wrong GPU in a
    # multi-GPU process.
    for method in ("record", "wait"):
        params = inspect.signature(getattr(torch.cuda.Event, method)).parameters
        assert "stream" in params, f"Event.{method}() no longer accepts a stream"
    assert "enable_timing" in inspect.signature(torch.cuda.Event.__new__).parameters


def test_event_record_without_a_stream_still_defaults_to_the_current_device() -> None:
    """Pins the behaviour the engine works around.

    If a future torch made a bare ``record()`` resolve per-event-device instead, the
    explicit-stream plumbing would become unnecessary — but until then it is load-bearing,
    and this test documents why by reading the source torch actually ships.
    """
    import inspect

    source = inspect.getsource(torch.cuda.Event.record)
    assert "torch.cuda.current_stream()" in source, (
        "torch.cuda.Event.record() no longer falls back to the current device's stream; "
        "re-check whether GuardMonitor still needs to pass the stream explicitly"
    )


def test_stream_context_manager_and_current_stream_exist() -> None:
    import inspect

    assert callable(torch.cuda.stream)
    assert "device" in inspect.signature(torch.cuda.current_stream).parameters, (
        "current_stream() must accept a device — asking without one answers for whatever "
        "device happens to be current"
    )


def test_graph_capture_detection_api_exists() -> None:
    # The engine drops observations while a graph is being captured (design doc §6.4).
    assert callable(torch.cuda.is_current_stream_capturing)
    assert hasattr(torch.cuda, "CUDAGraph")
    assert callable(torch.cuda.graph)


def test_pinned_memory_and_record_stream_api_exist() -> None:
    assert torch.empty(2, pin_memory=False).is_pinned() is False  # kwarg still accepted
    assert hasattr(torch.Tensor, "is_pinned")
    assert hasattr(torch.Tensor, "record_stream")
    # The GPU suite's no-sync assertion depends on this.
    assert callable(torch.cuda.set_sync_debug_mode)


def test_non_blocking_copy_requires_pinned_memory_to_be_async() -> None:
    """The invariant behind `_enqueue_drain`'s `non_blocking=buf.tensor.is_pinned()`.

    An async copy into pageable memory is not merely slower — where the completion event is
    a stub (any non-CUDA device), the drain reads the buffer before the copy lands and
    publishes zeros. The engine therefore ties `non_blocking` to `is_pinned()`, and this
    records that a plain host tensor is not pinned so the tie has effect.
    """
    assert torch.empty(4, 4).is_pinned() is False
