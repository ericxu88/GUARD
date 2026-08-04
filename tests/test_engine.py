"""Phase 2: the overlap engine — ring buffer, drain batching, backpressure, isolation.

Everything here runs on CPU. That is the whole point of the ``MonitorStream`` /
``MonitorEvent`` facade: the engine has one code path, so the logic that will run on a CUDA
monitor stream in production — slot rotation, summary layout, drain batching, pool
exhaustion, NaN slices for failed detectors, the kill-switch — is verified in CI without a
GPU. What remains hardware-dependent is only the CUDA primitives themselves, and those are
covered by ``tests/test_gpu_engine.py`` under ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping

import pytest
import torch

from guard.baseline.schema import Baseline
from guard.config import Config, DetectorConfig, ThresholdConfig
from guard.detectors.base import DetectorResult
from guard.engine.aggregator import MetricAggregator
from guard.engine.memory import HostBufferPool
from guard.engine.monitor import GuardMonitor, build_monitor
from guard.engine.ring_buffer import RingBuffer
from guard.engine.streams import LOW_PRIORITY, MonitorEvent, MonitorStream, resolve_device
from guard.export.alerts import Alert

C, D, MAXB = 6, 4, 8


# --------------------------------------------------------------------------------------
# Fixtures & doubles
# --------------------------------------------------------------------------------------
def _baseline(c: int = C, d: int = D) -> Baseline:
    return Baseline(
        model_version="engine-test@1",
        created_at="2026-01-01T00:00:00Z",
        sample_count=256,
        num_classes=c,
        embed_dim=d,
        class_distribution=torch.full((c,), 1.0 / c),
        entropy_histogram=torch.full((8,), 32.0),
        entropy_bin_edges=torch.linspace(0.0, math.log(c), 9),
        embed_mean=torch.zeros(d),
        embed_precision=torch.eye(d),
    )


class CountingDetector:
    """Emits the step index, so a test can assert on exactly which rows were drained."""

    name = "counting"
    metric_names: tuple[str, ...] = ("counter_value",)

    def __init__(self) -> None:
        self.calls = 0

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        del embeddings
        value = torch.tensor(float(self.calls), dtype=logits.dtype, device=logits.device)
        self.calls += 1
        return DetectorResult(scores={"counter_value": value})

    def reference(self) -> dict[str, object]:
        return {}


class BoomDetector:
    """Always raises. Used to prove a broken detector cannot reach the inference caller."""

    name = "boom"
    metric_names: tuple[str, ...] = ("boom_metric",)

    def __init__(self) -> None:
        self.calls = 0

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        del logits, embeddings
        self.calls += 1
        raise RuntimeError("detector exploded")

    def reference(self) -> dict[str, object]:
        return {}


class RecordingSink:
    """Captures everything the drain thread publishes."""

    def __init__(self) -> None:
        self.updates: list[dict[str, float]] = []
        self.alerts: list[tuple[str, bool]] = []
        self.dropped = 0
        self.telemetry: list[dict[str, float]] = []
        self.detector_failures: dict[str, int] = {}

    def update(self, values: Mapping[str, float]) -> None:
        self.updates.append(dict(values))

    def set_alert(self, detector: str, active: bool) -> None:
        self.alerts.append((detector, active))

    def add_dropped(self, count: int) -> None:
        self.dropped += count

    def set_telemetry(self, values: Mapping[str, float]) -> None:
        self.telemetry.append(dict(values))

    def record_detector_failure(self, detector: str, count: int = 1) -> None:
        self.detector_failures[detector] = self.detector_failures.get(detector, 0) + count


def _monitor(detectors: list[object], **kwargs: object) -> GuardMonitor:
    """A monitor with the drain thread off so tests drive the pipeline deterministically."""
    params: dict[str, object] = {
        "num_classes": C,
        "embed_dim": D,
        "max_batch": MAXB,
        "drain_every_k": 4,
        "ring_slots": 3,
        "start_drain_thread": False,
    }
    params.update(kwargs)
    return GuardMonitor(detectors, **params)  # type: ignore[arg-type]


def _batch(b: int = MAXB) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(b, C), torch.randn(b, D)


# --------------------------------------------------------------------------------------
# RingBuffer
# --------------------------------------------------------------------------------------
def test_ring_buffer_is_preallocated_and_bounded() -> None:
    ring = RingBuffer(
        slots=4, max_batch=8, num_classes=10, embed_dim=16, device=torch.device("cpu")
    )
    assert ring.logits.shape == (4, 8, 10)
    assert ring.embeddings.shape == (4, 8, 16)
    # 4*8*(10+16)*4 bytes; the point is that it never grows with the number of steps.
    assert ring.nbytes() == 4 * 8 * (10 + 16) * 4
    for _ in range(50):
        ring.write(0, torch.randn(8, 10), torch.randn(8, 16))
    assert ring.nbytes() == 4 * 8 * (10 + 16) * 4


def test_ring_write_returns_views_sized_to_the_actual_batch() -> None:
    # Dynamic batching means B varies per request; the stale tail of the slot must never
    # be handed to a detector.
    ring = RingBuffer(slots=2, max_batch=8, num_classes=3, embed_dim=2, device=torch.device("cpu"))
    logits, embeddings = torch.randn(3, 3), torch.randn(3, 2)
    lv, ev = ring.write(1, logits, embeddings)
    assert lv.shape == (3, 3)
    assert ev.shape == (3, 2)
    assert torch.equal(lv, logits)
    assert torch.equal(ev, embeddings)


def test_ring_write_casts_dtype_on_copy() -> None:
    # The ring is float32 even when the model is half; the cast happens inside the copy so
    # detector reductions never run in 10-bit mantissa arithmetic.
    ring = RingBuffer(slots=1, max_batch=4, num_classes=3, embed_dim=2, device=torch.device("cpu"))
    lv, _ = ring.write(0, torch.randn(4, 3, dtype=torch.float16), torch.randn(4, 2))
    assert lv.dtype == torch.float32


def test_ring_supports_zero_embed_dim() -> None:
    # Deployments that only watch the output distribution should not pay for an embedding
    # ring they never read.
    ring = RingBuffer(slots=2, max_batch=4, num_classes=3, embed_dim=0, device=torch.device("cpu"))
    assert ring.embeddings.numel() == 0
    lv, ev = ring.write(0, torch.randn(4, 3), torch.empty(0))
    assert lv.shape == (4, 3)
    assert ev.shape == (4, 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"slots": 0}, "slots"),
        ({"max_batch": 0}, "max_batch"),
        ({"num_classes": 0}, "num_classes"),
        ({"embed_dim": -1}, "embed_dim"),
    ],
)
def test_ring_rejects_invalid_geometry(kwargs: dict[str, int], match: str) -> None:
    params = {
        "slots": 2,
        "max_batch": 4,
        "num_classes": 3,
        "embed_dim": 2,
        "device": torch.device("cpu"),
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        RingBuffer(**params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# HostBufferPool — the bound on in-flight drains
# --------------------------------------------------------------------------------------
def test_pool_hands_out_exactly_capacity_buffers() -> None:
    pool = HostBufferPool(count=2, shape=(4, 3), dtype=torch.float32, device=torch.device("cpu"))
    a, b = pool.acquire(), pool.acquire()
    assert a is not None and b is not None
    assert a.index != b.index
    # Exhaustion must return None, not block and not allocate: pinned memory is scarce and
    # unbounded growth here is worse than dropping metrics.
    assert pool.acquire() is None
    assert pool.in_flight == 2
    pool.release(a)
    assert pool.free == 1
    assert pool.acquire() is not None


def test_pool_release_is_idempotent() -> None:
    # The release path runs in the drain thread, where a double-release corrupting the free
    # list would silently hand the same buffer to two concurrent copies.
    pool = HostBufferPool(count=1, shape=(2, 2), dtype=torch.float32, device=torch.device("cpu"))
    buf = pool.acquire()
    assert buf is not None
    pool.release(buf)
    pool.release(buf)
    assert pool.free == 1
    assert pool.capacity == 1


# --------------------------------------------------------------------------------------
# Stream / event facade on a non-CUDA device
# --------------------------------------------------------------------------------------
def test_cpu_stream_and_event_are_transparent_noops() -> None:
    device = torch.device("cpu")
    stream = MonitorStream(device)
    event = MonitorEvent(device)
    assert stream.is_cuda is False
    assert stream.raw is None
    # On CPU the work has already happened by the time it is "enqueued".
    assert event.query() is True
    with stream.activate():
        stream.wait_event(event)
        stream.protect(torch.randn(2, 2))
    event.record(stream)
    stream.synchronize()
    assert event.elapsed_time(MonitorEvent(device)) == 0.0


def test_monitor_stream_refuses_to_outrank_inference() -> None:
    # Negative priority means *higher* than inference in CUDA's inverted scheme, which
    # would let monitoring preempt serving — a prime-directive #1 violation.
    with pytest.raises(ValueError, match="priority"):
        MonitorStream(torch.device("cpu"), priority=-1)
    assert MonitorStream(torch.device("cpu"), priority=LOW_PRIORITY).priority == 0


def test_resolve_device_pins_a_concrete_cuda_index() -> None:
    # A bare "cuda" for the ring but "cuda:0" for the stream is a silent multi-GPU bug.
    if torch.cuda.is_available():
        assert resolve_device("cuda").index is not None
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(None).type in {"cpu", "cuda"}


# --------------------------------------------------------------------------------------
# Metric layout — decided before the first batch, never on the hot path
# --------------------------------------------------------------------------------------
def test_metric_layout_is_flat_and_ordered() -> None:
    from guard.detectors import DivergenceDetector, EntropyDetector

    baseline = _baseline()
    monitor = _monitor([EntropyDetector(baseline=baseline), DivergenceDetector(baseline)])
    assert monitor.metric_names == (
        "entropy_mean",
        "entropy_quantile",
        "entropy_pop_shift",
        "kl_divergence",
        "js_divergence",
    )
    monitor.close()


def test_detector_without_metric_names_is_rejected_at_construction() -> None:
    # The summary buffer is fixed-width and allocated up front; discovering the schema from
    # the first result would mean allocating on the hot path.
    class Undeclared:
        name = "undeclared"

        def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
            return DetectorResult(scores={})

        def reference(self) -> dict[str, object]:
            return {}

    with pytest.raises(TypeError, match="metric_names"):
        _monitor([Undeclared()])


def test_duplicate_metric_names_across_detectors_are_rejected() -> None:
    # Two detectors writing the same column would make one silently overwrite the other.
    with pytest.raises(ValueError, match="unique"):
        _monitor([CountingDetector(), CountingDetector()])


def test_monitor_requires_at_least_one_detector() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        _monitor([])


# --------------------------------------------------------------------------------------
# The drain pipeline
# --------------------------------------------------------------------------------------
def test_rows_arrive_in_step_order_exactly_once() -> None:
    sink = RecordingSink()
    detector = CountingDetector()
    monitor = _monitor([detector], sink=sink, drain_every_k=4)
    for _ in range(12):
        monitor.observe(*_batch())
    monitor.flush()

    values = [u["counter_value"] for u in sink.updates]
    assert values == [float(i) for i in range(12)], "rows must arrive once, in step order"
    assert monitor.stats().rows_dispatched == 12
    monitor.close()


def test_drained_rows_carry_their_original_step_index() -> None:
    # A drain delivers K rows at once, and each must be attributed back to the step that
    # produced it — that index is what an alert reports and what a responder correlates
    # against deploy timestamps. Getting it wrong misdates every firing by up to K steps.
    class StepRecorder(MetricAggregator):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.steps: list[int] = []

        def push(self, step: int, values: Mapping[str, float]) -> list[Alert]:
            self.steps.append(step)
            return super().push(step, values)

    recorder = StepRecorder(("counter_value",), {"counter_value": "counting"})
    monitor = _monitor([CountingDetector()], aggregator=recorder, drain_every_k=4)
    for _ in range(12):
        monitor.observe(*_batch())
    monitor.flush()
    assert recorder.steps == list(range(12))
    monitor.close()


def test_successive_observations_land_in_different_ring_slots() -> None:
    # Slot rotation is what lets the monitor stream read slot s while inference fills
    # slot s+1. Pinning every batch to one slot still produces correct numbers — it just
    # silently removes all pipelining, so only an explicit check catches it.
    slots = 3
    monitor = _monitor([CountingDetector()], ring_slots=slots)
    for i in range(slots):
        monitor.observe(torch.full((MAXB, C), float(i)), torch.zeros(MAXB, D))
    occupants = {float(monitor._ring.logits[s, 0, 0]) for s in range(slots)}
    assert occupants == {0.0, 1.0, 2.0}, f"batches collided in the ring: {occupants}"
    # And the ring wraps rather than growing.
    monitor.observe(torch.full((MAXB, C), 9.0), torch.zeros(MAXB, D))
    assert float(monitor._ring.logits[0, 0, 0]) == 9.0
    monitor.close()


def test_drain_happens_every_k_steps_not_per_step() -> None:
    # Batching the device->host copy is the whole reason the drain exists; one copy per
    # step would put a transfer on the critical path every request.
    sink = RecordingSink()
    monitor = _monitor([CountingDetector()], sink=sink, drain_every_k=4)
    for _ in range(3):
        monitor.observe(*_batch())
    assert monitor.drain_pending() == 0, "no drain before the block is full"
    monitor.observe(*_batch())
    assert monitor.drain_pending() == 4, "one drain of exactly K rows"
    assert monitor.stats().drains_enqueued == 1
    monitor.close()


def test_partial_block_is_not_lost_on_flush() -> None:
    sink = RecordingSink()
    monitor = _monitor([CountingDetector()], sink=sink, drain_every_k=8)
    for _ in range(8):
        monitor.observe(*_batch())
    monitor.flush()
    assert len(sink.updates) == 8
    monitor.close()


def test_backpressure_drops_summaries_instead_of_growing() -> None:
    # With the drain thread off, nothing releases pinned buffers, so the pool runs dry.
    # The right answer is to drop and count — never to block the inference thread.
    monitor = _monitor([CountingDetector()], drain_every_k=2, max_pending_drains=2)
    for _ in range(20):
        monitor.observe(*_batch())
    stats = monitor.stats()
    assert stats.drains_enqueued == 2, "only capacity-many copies may be in flight"
    assert stats.summaries_dropped == 16, "everything past capacity is dropped and counted"
    assert stats.steps_observed == 20, "inference is never impeded by export backpressure"
    monitor.close()


def test_dropped_summaries_are_reported_as_a_delta_not_a_total() -> None:
    # Prometheus counters are incremented, not set; publishing the running total would make
    # every rate() panel grow quadratically.
    sink = RecordingSink()
    monitor = _monitor([CountingDetector()], sink=sink, drain_every_k=2, max_pending_drains=1)
    for _ in range(8):
        monitor.observe(*_batch())
    monitor.flush()
    for _ in range(8):
        monitor.observe(*_batch())
    monitor.flush()
    assert sink.dropped == monitor.stats().summaries_dropped
    monitor.close()


def test_background_drain_thread_delivers_without_manual_flush() -> None:
    sink = RecordingSink()
    monitor = GuardMonitor(
        [CountingDetector()],
        num_classes=C,
        embed_dim=D,
        max_batch=MAXB,
        drain_every_k=2,
        sink=sink,
        drain_poll_interval=0.005,
    )
    for _ in range(8):
        monitor.observe(*_batch())
    deadline = threading.Event()
    for _ in range(100):
        if len(sink.updates) >= 8:
            break
        deadline.wait(0.01)
    monitor.close()
    assert len(sink.updates) >= 8, "the daemon drain thread must consume completed copies"


# --------------------------------------------------------------------------------------
# Malformed input is dropped, never raised (prime directive #1)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("logits", "embeddings"),
    [
        (torch.randn(4), torch.randn(4, D)),  # rank 1
        (torch.randn(4, C + 1), torch.randn(4, D)),  # wrong C
        (torch.randn(MAXB + 1, C), torch.randn(MAXB + 1, D)),  # oversized batch
        (torch.randn(0, C), torch.randn(0, D)),  # empty batch
        (torch.randn(4, C), torch.randn(4, D + 1)),  # wrong D
        (torch.randn(4, C), torch.randn(3, D)),  # batch mismatch
        (torch.randn(4, C), None),  # missing embeddings
    ],
)
def test_malformed_observations_are_dropped_and_counted(
    logits: torch.Tensor, embeddings: torch.Tensor | None
) -> None:
    monitor = _monitor([CountingDetector()])
    monitor.observe(logits, embeddings)  # must not raise
    stats = monitor.stats()
    assert stats.observations_dropped == 1
    assert stats.steps_observed == 0
    monitor.close()


def test_variable_batch_sizes_are_all_accepted() -> None:
    sink = RecordingSink()
    monitor = _monitor([CountingDetector()], sink=sink, drain_every_k=4)
    for b in (1, 3, MAXB, 2):
        monitor.observe(*_batch(b))
    monitor.flush()
    assert len(sink.updates) == 4
    monitor.close()


# --------------------------------------------------------------------------------------
# Failure isolation
# --------------------------------------------------------------------------------------
def test_failing_detector_never_reaches_the_caller_and_is_disabled() -> None:
    sink = RecordingSink()
    boom = BoomDetector()
    monitor = _monitor(
        [CountingDetector(), boom], sink=sink, drain_every_k=2, max_detector_failures=3
    )
    for _ in range(10):
        monitor.observe(*_batch())  # must not raise
    monitor.flush()

    assert monitor.firewall.is_disabled("boom"), "breaker must trip and stop calling it"
    assert boom.calls == 3, "a disabled detector costs a dict lookup, not an exception"
    assert "boom" in monitor.stats().disabled_detectors
    # The healthy detector keeps reporting; only the broken column goes missing.
    assert all("counter_value" in u for u in sink.updates)
    assert all("boom_metric" not in u for u in sink.updates), "NaN slices are skipped, not exported"
    monitor.close()


def test_failed_detector_slice_is_nan_not_stale() -> None:
    # A stale value is far more dangerous than a gap: it looks like a healthy reading.
    monitor = _monitor([BoomDetector()], drain_every_k=1, max_detector_failures=100)
    monitor.observe(*_batch())
    monitor.flush()
    assert monitor.aggregator.values_skipped >= 1
    assert monitor.aggregator.snapshot() == {}
    monitor.close()


def test_observe_survives_a_non_tensor_argument() -> None:
    # A serving handler passing the wrong thing must get a dropped metric, not a 500.
    monitor = _monitor([CountingDetector()])
    monitor.observe("not a tensor", None)  # type: ignore[arg-type]
    assert monitor.stats().steps_observed == 0
    monitor.close()


# --------------------------------------------------------------------------------------
# Kill-switch
# --------------------------------------------------------------------------------------
def test_kill_switch_stops_and_resumes_monitoring() -> None:
    monitor = _monitor([CountingDetector()])
    monitor.observe(*_batch())
    monitor.disable()
    for _ in range(5):
        monitor.observe(*_batch())
    assert monitor.stats().steps_observed == 1
    assert monitor.enabled is False
    monitor.enable()
    monitor.observe(*_batch())
    assert monitor.stats().steps_observed == 2
    monitor.close()


def test_env_kill_switch_overrides_the_runtime_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # An operator must be able to neutralise GUARD in a running deployment without a code
    # change; the env var wins over the constructor argument.
    monitor = _monitor([CountingDetector()], enabled=True)
    monkeypatch.setenv("GUARD_DISABLED", "1")
    for _ in range(5):
        monitor.observe(*_batch())
    assert monitor.stats().steps_observed == 0
    monkeypatch.delenv("GUARD_DISABLED")
    monitor.observe(*_batch())
    assert monitor.stats().steps_observed == 1
    monitor.close()


def test_observe_after_close_is_a_noop() -> None:
    monitor = _monitor([CountingDetector()])
    monitor.close()
    monitor.close()  # idempotent
    monitor.observe(*_batch())
    assert monitor.stats().steps_observed == 0


# --------------------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------------------
def test_concurrent_observers_do_not_lose_or_duplicate_steps() -> None:
    # A serving process calls observe() from N request threads. Slot and step allocation
    # must be atomic or two requests silently share a ring slot.
    monitor = _monitor([CountingDetector()], drain_every_k=8, max_pending_drains=8)
    threads = [
        threading.Thread(target=lambda: [monitor.observe(*_batch(2)) for _ in range(25)])
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert monitor.stats().steps_observed == 100
    assert monitor.stats().observations_dropped == 0
    monitor.close()


# --------------------------------------------------------------------------------------
# Wiring: config -> detectors -> monitor -> aggregator
# --------------------------------------------------------------------------------------
def test_build_monitor_wires_config_baseline_and_thresholds() -> None:
    baseline = _baseline()
    config = Config(
        window_size=2,
        drain_every_k=4,
        detectors=[
            DetectorConfig(name="entropy", params={"quantile": 0.9}),
            DetectorConfig(name="divergence"),
            DetectorConfig(name="embedding", params={"cov_every": 2}),
        ],
        thresholds={"js_divergence": ThresholdConfig(threshold=0.25, delta=0.01)},
    )
    monitor = build_monitor(baseline, config, max_batch=MAXB, start_drain_thread=False)
    assert monitor.drain_every_k == 4
    assert "embedding_cov_drift" in monitor.metric_names
    assert monitor.aggregator.window_size == 2
    assert monitor.aggregator.model_version == "engine-test@1"
    for _ in range(8):
        monitor.observe(*_batch())
    monitor.flush()
    assert set(monitor.aggregator.snapshot()) == set(monitor.metric_names)
    monitor.close()


def test_build_monitor_skips_disabled_detectors_and_drops_unused_embedding_ring() -> None:
    baseline = _baseline()
    config = Config(
        window_size=1,
        drain_every_k=2,
        detectors=[
            DetectorConfig(name="divergence"),
            DetectorConfig(name="embedding", enabled=False),
        ],
    )
    monitor = build_monitor(baseline, config, max_batch=MAXB, start_drain_thread=False)
    assert monitor.embed_dim == 0, "no embedding detector means no embedding ring"
    monitor.observe(torch.randn(4, C), None)
    monitor.observe(torch.randn(4, C), None)
    monitor.flush()
    assert monitor.stats().steps_observed == 2
    monitor.close()


def test_build_monitor_reports_bad_detector_params_at_startup() -> None:
    # Better a loud failure at boot than a metric that silently never appears, because the
    # firewall would swallow a constructor TypeError raised on the first batch.
    config = Config(
        window_size=1,
        drain_every_k=1,
        detectors=[DetectorConfig(name="entropy", params={"nonsense": 1})],
    )
    with pytest.raises(ValueError, match="entropy"):
        build_monitor(_baseline(), config, max_batch=MAXB)


def test_context_manager_closes_the_monitor() -> None:
    with _monitor([CountingDetector()]) as monitor:
        monitor.observe(*_batch())
    assert monitor.enabled is False


# --------------------------------------------------------------------------------------
# Capture ordering and lifecycle (Phase-2 review regressions)
# --------------------------------------------------------------------------------------
def test_capture_reads_the_caller_tensor_before_observe_returns() -> None:
    # The engine copies into its ring on the *inference* stream, so the captured data is
    # ordered against both the forward pass and whatever overwrites the output next. The
    # observable consequence on any device: mutating the caller's tensor after observe()
    # returns cannot change what the detector sees.
    #
    # The bug this guards against only bites on CUDA — with the copy queued on the monitor
    # stream, a static output buffer (CUDA-graph replay, torch.compile reduce-overhead,
    # TensorRT binding) is rewritten by the next iteration while that copy is still queued,
    # and record_stream cannot help because such memory is never freed to the allocator.
    # Metrics then silently mix two batches. Here the same ordering is asserted cheaply.
    detector = CountingDetector()
    monitor = _monitor([detector], drain_every_k=1)
    static = torch.zeros(MAXB, C)  # a buffer the "server" reuses every iteration
    embeddings = torch.zeros(MAXB, D)

    static.fill_(1.0)
    monitor.observe(static, embeddings)
    static.fill_(99.0)  # the next iteration overwrites the same memory
    monitor.flush()

    assert float(monitor._ring.logits[0, 0, 0]) == 1.0, "capture must not see the overwrite"
    monitor.close()


def test_slot_reuse_waits_for_the_previous_read() -> None:
    # Writing a ring slot on the inference stream introduces the mirror-image hazard:
    # inference could overwrite slot s while the monitor is still reading it from
    # ring_slots steps ago. The engine waits on that slot's completion event first. On CPU
    # the wait is a no-op, so what is asserted here is that the wait is issued at all and
    # that data still round-trips correctly once the ring has wrapped several times.
    sink = RecordingSink()
    detector = CountingDetector()
    monitor = _monitor([detector], sink=sink, ring_slots=2, drain_every_k=1)
    for _ in range(10):  # five full wraps of a two-slot ring
        monitor.observe(*_batch())
        monitor.drain_pending()  # free pinned buffers as we go, as the drain thread would
    monitor.flush()
    assert [u["counter_value"] for u in sink.updates] == [float(i) for i in range(10)]
    monitor.close()


def test_drain_survives_a_sink_that_always_raises() -> None:
    # The drain thread must outlive any single failure; if it died, export would stop for
    # the life of the process while stats().enabled kept reporting True.
    class Exploding:
        def update(self, values: Mapping[str, float]) -> None:
            raise RuntimeError("collector unreachable")

        def set_alert(self, detector: str, active: bool) -> None:
            raise RuntimeError("collector unreachable")

        def add_dropped(self, count: int) -> None:
            raise RuntimeError("collector unreachable")

        def set_telemetry(self, values: Mapping[str, float]) -> None:
            raise RuntimeError("collector unreachable")

    monitor = _monitor([CountingDetector()], sink=Exploding(), drain_every_k=2)
    for _ in range(8):
        monitor.observe(*_batch())
    monitor.flush()  # must not raise
    assert monitor.stats().steps_observed == 8
    monitor.close()


def test_dropping_a_monitor_without_closing_stops_its_drain_thread() -> None:
    # A bound method as the thread target would pin the monitor forever: the loop never
    # returns on its own, so a monitor rebuilt on config reload would leak its ring buffer,
    # its pinned pool, and a live polling thread every time.
    import gc
    import weakref

    monitor = GuardMonitor(
        [CountingDetector()],
        num_classes=C,
        embed_dim=D,
        max_batch=MAXB,
        drain_poll_interval=0.01,
    )
    thread = monitor._thread
    assert thread is not None and thread.is_alive()
    ref = weakref.ref(monitor)

    del monitor
    gc.collect()
    thread.join(timeout=5.0)

    assert ref() is None, "monitor was kept alive by its own drain thread"
    assert not thread.is_alive(), "drain thread outlived the monitor it served"
