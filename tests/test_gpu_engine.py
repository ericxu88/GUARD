"""Phase 2 hardware tests: the things only a real CUDA device can prove.

Everything in this file is ``@pytest.mark.gpu`` and auto-skips without CUDA (conftest.py).
The CPU suite already covers the engine's *logic*; what is left — and what genuinely cannot
be faked — is whether the CUDA primitives behave the way the design assumes:

* **No host synchronisation in the hot path** (prime directive #2). Asserted with
  ``torch.cuda.set_sync_debug_mode("error")``, which turns any implicit device→host sync
  into an exception. This is the single most important test in the repository: one stray
  ``.item()`` silently serialises inference and the overhead claim evaporates.
* **Memory safety** (prime directive #3). ``record_stream`` misuse does not crash — it
  silently corrupts, because the allocator hands a still-queued tensor's memory to the next
  iteration. The stress test below makes that corruption *observable* by writing a known
  value per step and checking every drained value.
* **CPU-reference parity** (prime directive #4). Detector math must agree across devices.
* **Bounded memory** (prime directive #5).

Run with ``pytest -m gpu`` on a machine with compute capability >= 8.0. The p99-overhead
SLO is *not* asserted here — a shared CI GPU cannot measure it honestly. That belongs to
``benchmarks/ab_latency.py``; see docs/GPU_VERIFICATION.md.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Iterator

import pytest
import torch

from guard.baseline.compute import build_baseline
from guard.baseline.schema import DEFAULT_ATOL, DEFAULT_RTOL, Baseline
from guard.config import Config, DetectorConfig, ThresholdConfig
from guard.detectors import DivergenceDetector, EmbeddingDriftDetector, EntropyDetector
from guard.detectors.base import DetectorResult
from guard.engine.monitor import GuardMonitor, build_monitor

pytestmark = pytest.mark.gpu

C, D, MAXB = 32, 16, 8


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _baseline(c: int = C, d: int = D) -> Baseline:
    return Baseline(
        model_version="gpu-test@1",
        created_at="2026-01-01T00:00:00Z",
        sample_count=1024,
        num_classes=c,
        embed_dim=d,
        class_distribution=torch.full((c,), 1.0 / c),
        entropy_histogram=torch.full((16,), 64.0),
        entropy_bin_edges=torch.linspace(0.0, math.log(c), 17),
        embed_mean=torch.zeros(d),
        embed_precision=torch.eye(d),
    )


class MeanDetector:
    """Reports ``logits.mean()`` — a value the test controls exactly.

    Entropy and divergence are poor corruption probes: they are invariant to the constants
    a stress test can write. A plain mean makes every drained number predictable, so a
    single byte of allocator reuse shows up as a wrong value rather than as noise.
    """

    name = "mean"
    metric_names: tuple[str, ...] = ("mean_value",)

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        del embeddings
        return DetectorResult(scores={"mean_value": logits.mean()})

    def reference(self) -> dict[str, object]:
        return {}


@contextlib.contextmanager
def strict_no_sync() -> Iterator[None]:
    """Make any implicit device→host synchronisation raise instead of silently stalling."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _monitor(**kwargs: object) -> GuardMonitor:
    params: dict[str, object] = {
        "num_classes": C,
        "embed_dim": D,
        "max_batch": MAXB,
        "device": "cuda",
        "drain_every_k": 4,
        "ring_slots": 4,
        "start_drain_thread": False,
    }
    params.update(kwargs)
    detectors = params.pop("detectors", None) or [MeanDetector()]
    return GuardMonitor(detectors, **params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Prime directive #2 — no host synchronisation on the hot path
# --------------------------------------------------------------------------------------
def test_observe_performs_no_host_synchronisation() -> None:
    baseline = _baseline()
    monitor = _monitor(
        detectors=[
            EntropyDetector(baseline=baseline),
            DivergenceDetector(baseline),
            EmbeddingDriftDetector(baseline, cov_every=2),
        ]
    )
    logits = torch.randn(MAXB, C, device="cuda")
    embeddings = torch.randn(MAXB, D, device="cuda")

    monitor.observe(logits, embeddings)  # warm-up: first-call lazy allocations
    torch.cuda.synchronize()

    with strict_no_sync():
        for _ in range(monitor.drain_every_k * 3):
            monitor.observe(logits, embeddings)

    # A sync inside observe() would have raised above; a *swallowed* one would show up here
    # as the engine breaker having tripped.
    assert monitor.firewall.disabled == frozenset()
    assert monitor.stats().steps_observed >= monitor.drain_every_k * 3
    monitor.close()


def test_observe_returns_while_the_gpu_is_still_busy() -> None:
    # The direct behavioural statement of "fire-and-forget": queue a long chain of work on
    # the inference stream, then observe. If observe waited on the device, the event
    # recorded before it would necessarily have completed by the time it returned.
    monitor = _monitor()
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    out = torch.empty_like(a)
    logits = torch.randn(MAXB, C, device="cuda")
    embeddings = torch.randn(MAXB, D, device="cuda")
    monitor.observe(logits, embeddings)
    torch.cuda.synchronize()

    for _ in range(200):
        torch.mm(a, b, out=out)
    busy = torch.cuda.Event()
    busy.record()

    start = time.perf_counter()
    monitor.observe(logits, embeddings)
    elapsed = time.perf_counter() - start

    assert not busy.query(), "observe() waited for the GPU — the hot path is synchronising"
    assert elapsed < 0.01, f"observe() took {elapsed * 1e3:.2f} ms; it should be a few enqueues"
    torch.cuda.synchronize()
    monitor.close()


# --------------------------------------------------------------------------------------
# Prime directive #3 — memory safety under allocator reuse
# --------------------------------------------------------------------------------------
def test_capture_survives_allocator_reuse() -> None:
    """Hostile allocator churn: free the captured tensor immediately and force the block to
    be handed straight back out for a same-sized allocation.

    If the capture were queued behind the monitor stream's backlog, the copy would read
    whatever the next allocation wrote and the drained means would come back wrong.
    """
    steps = 256
    monitor = _monitor(drain_every_k=8, max_pending_drains=8, ring_slots=4)
    seen: list[float] = []
    monitor.aggregator.sink = _CollectingSink(seen)
    embeddings = torch.randn(MAXB, D, device="cuda")

    for i in range(steps):
        logits = torch.full((MAXB, C), float(i), device="cuda")
        monitor.observe(logits, embeddings)
        del logits
        # Same shape and stream => the allocator's best candidate is the block just freed.
        churn = torch.empty((MAXB, C), device="cuda")
        churn.fill_(-999.0)
        del churn
        monitor.drain_pending()

    monitor.flush()
    expected = [float(i) for i in range(steps)]
    assert len(seen) == steps, f"drained {len(seen)} of {steps} rows"
    assert seen == pytest.approx(expected, abs=1e-4), "captured logits were corrupted"
    monitor.close()


def test_capture_survives_a_static_output_buffer() -> None:
    """The case ``record_stream`` cannot cover, and the reason capture moved onto the
    inference stream.

    ``record_stream`` tells the *caching allocator* to defer recycling a block that gets
    freed. A growing share of serving paths never free their outputs at all: CUDA-graph
    replay, ``torch.compile(mode="reduce-overhead")``, TensorRT bindings and any
    pre-allocated ``out=`` buffer write results into the same device addresses every
    iteration. The allocator is never involved, so nothing would stop iteration N+1 from
    overwriting the buffer while a monitor-stream copy of iteration N was still queued.

    This simulates exactly that: one buffer, rewritten in place, immediately after each
    ``observe()``. Every drained mean must be the value that was in the buffer at the time
    of the call. A failure here is not a crash — it is metrics computed on a torn mix of
    two batches and exported as if they were real.
    """
    steps = 256
    monitor = _monitor(drain_every_k=8, max_pending_drains=8, ring_slots=4)
    seen: list[float] = []
    monitor.aggregator.sink = _CollectingSink(seen)
    embeddings = torch.randn(MAXB, D, device="cuda")
    static_logits = torch.empty((MAXB, C), device="cuda")  # never freed, always reused

    for i in range(steps):
        static_logits.fill_(float(i))
        monitor.observe(static_logits, embeddings)
        # Immediately overwrite, as the next graph replay / compiled step would.
        static_logits.fill_(-1.0)
        monitor.drain_pending()

    monitor.flush()
    expected = [float(i) for i in range(steps)]
    assert len(seen) == steps, f"drained {len(seen)} of {steps} rows"
    assert seen == pytest.approx(expected, abs=1e-4), (
        "a static output buffer was overwritten before GUARD captured it — the capture is "
        "not ordered against the inference stream"
    )
    monitor.close()


class _CollectingSink:
    """Appends one metric per drained row so the stress test can compare exact values."""

    def __init__(self, into: list[float]) -> None:
        self._into = into

    def update(self, values: dict[str, float]) -> None:  # type: ignore[override]
        if "mean_value" in values:
            self._into.append(values["mean_value"])

    def set_alert(self, detector: str, active: bool) -> None: ...

    def add_dropped(self, count: int) -> None: ...

    def set_telemetry(self, values: dict[str, float]) -> None: ...


# --------------------------------------------------------------------------------------
# Prime directive #5 — bounded resources
# --------------------------------------------------------------------------------------
def test_device_memory_does_not_grow_with_steps() -> None:
    baseline = _baseline()
    monitor = _monitor(
        detectors=[
            EntropyDetector(baseline=baseline),
            DivergenceDetector(baseline),
            EmbeddingDriftDetector(baseline, cov_every=8),
        ],
        drain_every_k=8,
        max_pending_drains=4,
    )
    logits = torch.randn(MAXB, C, device="cuda")
    embeddings = torch.randn(MAXB, D, device="cuda")

    for _ in range(64):  # warm-up: lazy Welford/scratch allocations happen here
        monitor.observe(logits, embeddings)
        monitor.drain_pending()
    monitor.flush()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    for _ in range(2000):
        monitor.observe(logits, embeddings)
        monitor.drain_pending()
    monitor.flush()
    torch.cuda.synchronize()
    growth = torch.cuda.memory_allocated() - before

    assert growth < 1 << 20, f"device memory grew {growth} bytes over 2000 steps"
    monitor.close()


def test_pinned_host_buffers_are_page_locked() -> None:
    # Pageable destinations silently downgrade `non_blocking=True` to a synchronising copy,
    # which would stall the inference stream once every drain_every_k steps.
    monitor = _monitor()
    buf = monitor._pool.acquire()
    assert buf is not None
    assert buf.tensor.is_pinned(), "drain buffers must be page-locked for async D2H"
    monitor._pool.release(buf)
    monitor.close()


# --------------------------------------------------------------------------------------
# Prime directive #4 — CUDA results match the CPU reference
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_detectors_match_cpu_reference_on_cuda(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    baseline = _baseline()
    logits = torch.randn(64, C, dtype=dtype)
    embeddings = torch.randn(64, D, dtype=dtype)

    for make in (
        lambda: EntropyDetector(baseline=baseline),
        lambda: DivergenceDetector(baseline),
        lambda: EmbeddingDriftDetector(baseline, cov_every=1),
    ):
        cpu_scores = make().compute(logits, embeddings).scores
        cuda_scores = make().compute(logits.cuda(), embeddings.cuda()).scores
        assert set(cpu_scores) == set(cuda_scores)
        for name, cpu_value in cpu_scores.items():
            assert torch.allclose(
                cuda_scores[name].cpu().to(cpu_value.dtype),
                cpu_value,
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
            ), f"{name} diverged between CPU and CUDA"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_serving_path_produces_finite_metrics(dtype: torch.dtype) -> None:
    # Autocast serving is the norm for ViT-class models; half-precision logits must not
    # produce NaN metrics (which would never cross a threshold, silently disabling alerts).
    baseline = _baseline()
    monitor = _monitor(
        detectors=[
            EntropyDetector(baseline=baseline),
            DivergenceDetector(baseline),
            EmbeddingDriftDetector(baseline, cov_every=1),
        ],
        drain_every_k=2,
    )
    for _ in range(4):
        monitor.observe(
            (torch.randn(MAXB, C, device="cuda") * 20).to(dtype),
            torch.randn(MAXB, D, device="cuda").to(dtype),
        )
    monitor.flush()
    snapshot = monitor.aggregator.snapshot()
    assert set(snapshot) == set(monitor.metric_names), f"missing metrics: {snapshot}"
    values = list(snapshot.values())
    assert all(math.isfinite(v) for v in values), snapshot
    monitor.close()


# --------------------------------------------------------------------------------------
# Stream configuration
# --------------------------------------------------------------------------------------
def test_monitor_stream_is_real_separate_and_lowest_priority() -> None:
    monitor = _monitor()
    stream = monitor._stream
    assert stream.is_cuda
    assert stream.raw is not None
    assert stream.raw != torch.cuda.current_stream(), "must not share the inference stream"
    # 0 is CUDA's *least* priority: the scheduler always prefers inference kernels.
    assert stream.priority == 0
    monitor.close()


def test_observation_is_dropped_during_cuda_graph_capture() -> None:
    # Recording events or switching streams inside a capture corrupts the graph
    # (design doc §6.4). Dropping is the only safe answer, and it must be counted.
    monitor = _monitor()
    logits = torch.randn(MAXB, C, device="cuda")
    embeddings = torch.randn(MAXB, D, device="cuda")
    monitor.observe(logits, embeddings)
    torch.cuda.synchronize()
    before = monitor.stats()

    static = torch.randn(MAXB, C, device="cuda")
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.cuda.graph(graph):
        static.mul_(1.0001)
        monitor.observe(logits, embeddings)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    after = monitor.stats()
    assert after.steps_observed == before.steps_observed, "observation must be skipped"
    assert after.observations_dropped == before.observations_dropped + 1
    monitor.close()


# --------------------------------------------------------------------------------------
# End-to-end on device
# --------------------------------------------------------------------------------------
def test_build_baseline_accepts_cuda_batches() -> None:
    torch.manual_seed(0)
    batches = [(torch.randn(64, C, device="cuda"), torch.randn(64, D, device="cuda"))]
    baseline = build_baseline(batches, model_version="cuda@1", num_classes=C, embed_dim=D)
    baseline.validate()
    assert float(baseline.entropy_histogram.sum()) == 64.0
    assert baseline.class_distribution.device.type == "cpu", "baselines are stored on host"


def test_full_pipeline_detects_injected_drift_on_device() -> None:
    torch.manual_seed(0)
    clean = [(torch.randn(64, C) * 2, torch.randn(64, D)) for _ in range(8)]
    baseline = build_baseline(clean, model_version="cuda@1", num_classes=C, embed_dim=D)
    config = Config(
        window_size=1,
        drain_every_k=4,
        detectors=[
            DetectorConfig(name="entropy"),
            DetectorConfig(name="divergence"),
            DetectorConfig(name="embedding", params={"cov_every": 8}),
        ],
        thresholds={
            "embedding_mahalanobis_mean": ThresholdConfig(threshold=3.0, delta=0.5, debounce=2)
        },
    )
    monitor = build_monitor(baseline, config, max_batch=64, device="cuda", start_drain_thread=False)
    for _ in range(40):  # in-distribution
        monitor.observe((torch.randn(64, C) * 2).cuda(), torch.randn(64, D).cuda())
    monitor.flush()
    assert monitor.aggregator.alerts_fired == 0, "clean traffic must not alert"

    for _ in range(40):  # covariate shift
        monitor.observe((torch.randn(64, C) * 2).cuda(), (torch.randn(64, D) + 8.0).cuda())
    monitor.flush()
    assert monitor.aggregator.alerts_fired > 0, "injected embedding shift must alert"
    assert monitor.aggregator.alert_states()["embedding"] is True
    monitor.close()


def test_concurrent_observers_on_device() -> None:
    monitor = _monitor(drain_every_k=16, max_pending_drains=8)
    logits = torch.randn(MAXB, C, device="cuda")
    embeddings = torch.randn(MAXB, D, device="cuda")

    def worker() -> None:
        for _ in range(50):
            monitor.observe(logits, embeddings)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    monitor.flush()
    assert monitor.stats().steps_observed == 200
    assert monitor.stats().observations_dropped == 0
    monitor.close()


def test_cross_device_tensors_are_rejected_not_crashed() -> None:
    monitor = _monitor()
    monitor.observe(torch.randn(MAXB, C), torch.randn(MAXB, D))  # CPU tensors, CUDA monitor
    assert monitor.stats().observations_dropped == 1
    assert monitor.stats().steps_observed == 0
    monitor.close()
