"""The overlap engine: streams, events, ring buffer, async drain (Phase 2).

This is where "near-zero overhead" is won or lost. :meth:`GuardMonitor.observe` is called
on the inference thread immediately after ``forward()`` and must return in microseconds
having done nothing but *enqueue* work:

1. On the **inference** stream, copy the outputs into GUARD's own pre-allocated ring
   buffer and record a CUDA event — "the ring slot holds this step".
2. Switch to the low-priority **monitor** stream and ``wait_event`` on it. That is a
   GPU-side dependency: the monitor's kernels are queued instantly, and the CPU never
   blocks. They execute only once the capture has landed, and only in whatever SM capacity
   inference leaves free.
3. Run the detectors against the ring slot and write their 0-dim outputs into a
   device-resident summary block.
4. Every ``drain_every_k`` steps, enqueue one async device→host copy of that block into a
   pinned buffer and record a completion event. A background thread polls the event and
   only then reads the numbers, feeds the change-point tests, and updates the exporter.

Step 1 is the part that took a redesign to get right. The obvious arrangement — queue the
capture on the *monitor* stream and rely on ``record_stream`` to keep the allocator from
recycling the source — is what the design sketch called for, and it is not sufficient.
``record_stream`` only defers reuse of a block that gets **freed**, and an increasing share
of serving paths never free their outputs: CUDA-graph replay,
``torch.compile(mode="reduce-overhead")``, TensorRT bindings and any pre-allocated ``out=``
buffer rewrite the same device addresses on every iteration. Nothing would order the next
iteration's write against a capture still queued behind the monitor's backlog, and the
result would not be a crash — it would be entropy and divergence computed on a torn mix of
two batches and exported as if they were real. Copying in the caller's stream makes the
capture ordered against both the forward pass and whatever overwrites it next, for every
kind of memory. Inference pays two small device-to-device copies for that; see ``_enqueue``.

Nothing on that path reads a tensor value on the host. No ``.item()``, no ``.cpu()``, no
``synchronize()`` — a single one of those would serialise the inference stream and turn a
free monitor into a 10-40% tax (prime directive #2).

Everything else here is failure containment. Every detector call goes through
:class:`~guard.safety.Firewall`; a detector that raises is disabled and its metric slice
stays NaN, which the aggregator skips. The engine itself has a breaker: if the enqueue path
fails repeatedly, monitoring turns itself off and inference carries on untouched. Under
export backpressure summaries are dropped and counted, never buffered without bound.

On a non-CUDA device every stream and event degrades to a no-op (see
:mod:`guard.engine.streams`) and the detectors run inline. The logic below is one code
path, which is why the CPU test suite can verify ring indexing, drain batching,
backpressure, latching, and failure isolation without a GPU — leaving only the CUDA
primitives themselves for the ``@pytest.mark.gpu`` suite to confirm on hardware.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch

from guard.baseline.schema import Baseline
from guard.config import Config, ThresholdConfig
from guard.detectors.base import Detector, DetectorResult
from guard.engine.aggregator import MetricAggregator
from guard.engine.memory import HostBuffer, HostBufferPool
from guard.engine.ring_buffer import RingBuffer
from guard.engine.streams import (
    LOW_PRIORITY,
    MonitorEvent,
    MonitorStream,
    capturing_graph,
    current_stream,
    is_cuda,
    resolve_device,
    same_device,
)
from guard.export.alerts import AlertSink
from guard.export.base import MetricSink, NullSink
from guard.safety.firewall import Firewall, KillSwitch

logger = logging.getLogger("guard.engine")

# Firewall component name for the enqueue path itself. If this trips, GUARD is off.
_ENGINE = "engine"


@dataclass(frozen=True)
class MonitorStats:
    """A snapshot of the engine's own health. Cheap to take; safe from any thread."""

    enabled: bool
    device: str
    steps_observed: int
    observations_dropped: int
    summaries_dropped: int
    drains_enqueued: int
    drains_completed: int
    rows_dispatched: int
    ring_utilization: float
    last_enqueue_us: float
    mean_enqueue_us: float
    gpu_time_us: float
    disabled_detectors: tuple[str, ...]
    ring_bytes: int
    pinned_bytes: int


@dataclass(frozen=True)
class _PendingDrain:
    """One in-flight device→host summary copy awaiting its completion event."""

    buf: HostBuffer
    first_step: int


class GuardMonitor:
    """Fire-and-forget GPU monitoring attached to a live inference loop.

    Args:
        detectors: detector instances to run each step. Each must expose ``name`` and
            ``metric_names`` (see :mod:`guard.detectors.base`).
        num_classes: logits width ``C``.
        embed_dim: embedding width ``D``; ``0`` if no detector consumes embeddings.
        max_batch: largest batch ``observe`` will accept. Larger batches are dropped and
            counted rather than silently truncated.
        device: device to monitor on. Defaults to CUDA when available, else CPU.
        dtype: ring-buffer and detector compute dtype. Float32 regardless of the model's
            dtype is the right default — see :class:`~guard.engine.ring_buffer.RingBuffer`.
        ring_slots: ring depth. Correctness does not depend on it (stream ordering does);
            more slots only buy pipelining.
        drain_every_k: steps per device→host summary copy. Larger = fewer, bigger copies
            and staler metrics.
        max_pending_drains: pinned buffers, i.e. copies allowed in flight. When they are
            all busy, summaries are dropped and counted.
        window_size: steps per aggregation window (see
            :class:`~guard.engine.aggregator.MetricAggregator`).
        thresholds: metric name → change-point config.
        sink: metric destination. Defaults to :class:`~guard.export.base.NullSink`.
        alert_sink: alert destination. ``None`` keeps alerts to the latched gauge.
        aggregator: supply a pre-built aggregator to override windowing/thresholds wholesale.
        max_detector_failures: failures before a detector is disabled by its breaker.
        enable_gpu_timing: record CUDA timing events around the monitor work on drain
            steps. Adds two event records per ``drain_every_k`` steps; off by default
            because the honest cost metric is ``last_enqueue_us``, which is what inference
            actually pays.
        enabled: initial kill-switch state. ``GUARD_DISABLED=1`` in the environment
            overrides this to off.
        stream_priority: monitor stream priority; ``0`` (lowest) unless you know better.
        start_drain_thread: run the host-side drain in a daemon thread. Set ``False`` to
            drive :meth:`drain_pending` yourself (tests, or a server with its own loop).
        drain_poll_interval: seconds the drain thread sleeps when nothing is ready.
        model_version / baseline_version: stamped onto alerts for triage.
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        *,
        num_classes: int,
        embed_dim: int,
        max_batch: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        ring_slots: int = 8,
        drain_every_k: int = 32,
        max_pending_drains: int = 4,
        window_size: int = 1,
        thresholds: Mapping[str, ThresholdConfig] | None = None,
        sink: MetricSink | None = None,
        alert_sink: AlertSink | None = None,
        aggregator: MetricAggregator | None = None,
        max_detector_failures: int = 3,
        enable_gpu_timing: bool = False,
        enabled: bool = True,
        stream_priority: int = LOW_PRIORITY,
        start_drain_thread: bool = True,
        drain_poll_interval: float = 0.05,
        model_version: str = "",
        baseline_version: str = "",
    ) -> None:
        if not detectors:
            raise ValueError("GuardMonitor needs at least one detector")
        if drain_every_k < 1:
            raise ValueError(f"drain_every_k must be >= 1, got {drain_every_k}")
        if max_pending_drains < 1:
            raise ValueError(f"max_pending_drains must be >= 1, got {max_pending_drains}")

        self.device = resolve_device(device)
        self.dtype = dtype
        self.drain_every_k = drain_every_k
        self.max_batch = max_batch
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.enable_gpu_timing = enable_gpu_timing

        self._detectors = tuple(detectors)
        self._layout = _metric_layout(self._detectors)
        self.metric_names = tuple(name for name, _ in self._layout)
        metric_detectors = dict(self._layout)
        # Each detector owns a contiguous span of the summary row, so writing its results
        # is one stack kernel into a view rather than a copy per metric.
        self._detector_slices = _detector_slices(self._detectors)

        self._kill = KillSwitch(enabled)
        self._firewall = Firewall(max_failures=max_detector_failures, log=logger)

        self._stream = MonitorStream(self.device, priority=stream_priority)
        self._ring = RingBuffer(
            slots=ring_slots,
            max_batch=max_batch,
            num_classes=num_classes,
            embed_dim=embed_dim,
            device=self.device,
            dtype=dtype,
        )
        # "Inference outputs ready" and "monitor work for this slot done", one pair per slot.
        self._slot_ready = [MonitorEvent(self.device) for _ in range(ring_slots)]
        self._slot_done = [MonitorEvent(self.device) for _ in range(ring_slots)]

        n_metrics = len(self.metric_names)
        self._summary = torch.zeros((drain_every_k, n_metrics), device=self.device, dtype=dtype)
        self._pool = HostBufferPool(
            count=max_pending_drains,
            shape=(drain_every_k, n_metrics),
            dtype=dtype,
            device=self.device,
        )
        self._timing: list[tuple[MonitorEvent, MonitorEvent]] = [
            (
                MonitorEvent(self.device, enable_timing=True),
                MonitorEvent(self.device, enable_timing=True),
            )
            for _ in range(max_pending_drains)
        ]

        self.sink: MetricSink = sink if sink is not None else NullSink()
        self.aggregator = (
            aggregator
            if aggregator is not None
            else MetricAggregator(
                self.metric_names,
                metric_detectors,
                window_size=window_size,
                thresholds=thresholds,
                sink=self.sink,
                alert_sink=alert_sink,
                model_version=model_version,
                baseline_version=baseline_version,
            )
        )
        self._alert_sink = alert_sink

        # Hot-path state, all guarded by _lock.
        self._lock = threading.Lock()
        self._slot = 0
        self._step = 0
        self._observations_dropped = 0
        self._summaries_dropped = 0
        self._drains_enqueued = 0
        self._last_enqueue_us = 0.0
        self._enqueue_us_ema = 0.0
        self._warned_device_mismatch = False
        # High-water marks so counter metrics are published as deltas, not totals.
        self._summaries_published = 0
        self._steps_published = 0
        self._rejects_published = 0
        self._failures_published: dict[str, int] = {}

        # Drain-side state, guarded by _pending_lock (never taken while holding _lock for
        # anything slow — see _enqueue_drain, the one place both are held).
        self._pending_lock = threading.Lock()
        # Serialises the whole pop-and-dispatch sequence: the aggregator's windows and
        # change-point state are not thread-safe, so only one drainer may be inside it.
        self._drain_lock = threading.Lock()
        self._pending: deque[_PendingDrain] = deque()
        self._drains_completed = 0
        self._rows_dispatched = 0
        self._gpu_time_us = 0.0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        if start_drain_thread:
            # A bound method as the thread target would keep the monitor alive forever: the
            # loop never returns on its own, so a monitor dropped without close() (a config
            # reload that rebuilds it, a test that forgets) would be uncollectable and its
            # ring buffer, pinned pool, and polling thread would leak. A weak reference plus
            # a finalizer means dropping the last reference stops the thread and frees the
            # buffers, whether or not anyone called close().
            self._thread = threading.Thread(
                target=_drain_loop,
                args=(weakref.ref(self), drain_poll_interval, self._stop),
                name="guard-drain",
                daemon=True,
            )
            self._finalizer = weakref.finalize(self, self._stop.set)
            self._thread.start()

        logger.info(
            "GUARD monitor ready on %s: %d detector(s), %d metric(s), ring %.1f MiB, "
            "pinned %.1f KiB, drain every %d steps%s",
            self.device,
            len(self._detectors),
            n_metrics,
            self._ring.nbytes() / 2**20,
            self._pool.nbytes() / 2**10,
            drain_every_k,
            "" if self._stream.is_cuda else " (no CUDA: detectors run inline, no overlap)",
        )

    # ── construction helpers ─────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: Config,
        detectors: Sequence[Detector],
        *,
        num_classes: int,
        embed_dim: int,
        max_batch: int,
        **kwargs: Any,
    ) -> GuardMonitor:
        """Build a monitor from a validated :class:`~guard.config.Config`.

        Config supplies ``enabled``, ``window_size``, ``drain_every_k``, and ``thresholds``;
        anything else can be overridden through ``kwargs``.
        """
        params: dict[str, Any] = {
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "max_batch": max_batch,
            "enabled": config.enabled,
            "window_size": config.window_size,
            "drain_every_k": config.drain_every_k,
            "thresholds": config.thresholds,
        }
        params.update(kwargs)
        return cls(detectors, **params)

    # ── the hot path ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def observe(self, logits: torch.Tensor, embeddings: torch.Tensor | None = None) -> None:
        """Enqueue monitoring for one inference batch. Fire-and-forget; never raises.

        Call this on the inference thread immediately after ``forward()``, while the
        current stream is still the one the model ran on — that is how the "outputs ready"
        event lands on the right stream.

        Args:
            logits: model outputs ``[B, C]``, on the monitor's device.
            embeddings: penultimate features ``[B, D]``. Optional when ``embed_dim == 0``.

        Anything malformed (wrong rank, wrong width, wrong device, oversized batch) is
        dropped and counted in ``observations_dropped`` rather than raised: a monitoring
        hook that throws into a serving handler is a production incident, a dropped metric
        is a dashboard gap.
        """
        if not self._kill.enabled or self._closed:
            return
        start = time.perf_counter()
        self._firewall.call(_ENGINE, self._enqueue, logits, embeddings)
        elapsed_us = (time.perf_counter() - start) * 1e6
        self._last_enqueue_us = elapsed_us
        # EMA over ~100 steps; a plain mean would hide a regression behind warm-up.
        self._enqueue_us_ema += (elapsed_us - self._enqueue_us_ema) * 0.01

    def _enqueue(self, logits: torch.Tensor, embeddings: torch.Tensor | None) -> None:
        """The whole enqueue path. Runs behind the engine's firewall entry."""
        if embeddings is None:
            embeddings = self._ring.embeddings[0, :0]

        with self._lock:
            if not self._validate(logits, embeddings):
                self._observations_dropped += 1
                return

            step = self._step
            slot = self._slot
            self._step = step + 1
            self._slot = (slot + 1) % self._ring.slots

            is_drain_step = (step + 1) % self.drain_every_k == 0
            buf = self._pool.acquire() if is_drain_step else None
            if is_drain_step and buf is None:
                self._summaries_dropped += self.drain_every_k
            timing = (
                self._timing[buf.index] if (buf is not None and self.enable_gpu_timing) else None
            )

            # The stream inference is submitting to *on the monitored device*. Asking for
            # the current stream without naming the device answers for whatever device
            # happens to be current, which is the wrong card in a multi-GPU process.
            inference = current_stream(self.device)

            # ── capture, on the inference stream ──────────────────────────────
            #
            # The copy into GUARD's ring runs here, in the caller's stream, rather than on
            # the monitor stream. That ordering is the whole ballgame for memory safety.
            #
            # `record_stream` alone is not enough. It tells the *caching allocator* to
            # defer recycling a block that gets freed — but a growing share of serving
            # paths never free their output tensors at all: CUDA-graph replay,
            # `torch.compile(mode="reduce-overhead")`, TensorRT bindings and any
            # pre-allocated `out=` buffer write their results into the same device
            # addresses on every iteration. The allocator is not involved, so nothing
            # stops iteration N+1 from overwriting the buffer while a monitor-stream copy
            # of iteration N is still queued behind a backlog of detector kernels. The
            # result is not a crash: it is entropy and divergence computed on a torn mix
            # of two batches, exported as if they were real.
            #
            # Copying in the caller's stream makes the capture ordered against both the
            # forward pass that produced the tensors and whatever overwrites them next,
            # for every kind of memory. Inference pays two small device-to-device copies
            # (~450 KB at B=64, C=1000, D=768 — single-digit microseconds) instead of a
            # class of silent corruption.
            #
            # Slot reuse is the mirror-image hazard, and the wait below closes it: the
            # monitor may still be reading this slot from `ring_slots` steps ago. By
            # default that event completed long before we get here, so the wait is free;
            # when it is not free, `ring_utilization` is the metric that says so and
            # `ring_slots` is the knob. A never-recorded event (the first pass through the
            # ring) is a no-op.
            self._slot_done[slot].wait(inference)
            logits_view, embed_view = self._ring.write(slot, logits, embeddings)

            # Marks "the ring slot holds this step's outputs".
            ready = self._slot_ready[slot]
            ready.record(inference)

            with self._stream.activate():
                # GPU-side dependency only: this thread does not wait.
                self._stream.wait_event(ready)

                if timing is not None:
                    timing[0].record(self._stream)

                row = self._summary[step % self.drain_every_k]
                # NaN means "no reading this step". A disabled detector leaves its slice
                # untouched, and the aggregator skips non-finite values.
                row.fill_(float("nan"))
                for detector, metric_slice in self._detector_slices:
                    self._firewall.call(
                        detector.name,
                        self._run_detector,
                        detector,
                        metric_slice,
                        row,
                        logits_view,
                        embed_view,
                    )

                self._slot_done[slot].record(self._stream)
                if timing is not None:
                    timing[1].record(self._stream)
                if buf is not None:
                    self._enqueue_drain(buf, step)

    def _validate(self, logits: torch.Tensor, embeddings: torch.Tensor) -> bool:
        """Cheap structural checks on shapes and devices. Never reads tensor *values*.

        Every check here is on ``.shape``/``.device``/``.ndim`` — Python-side metadata that
        costs nothing and, crucially, does not synchronise with the device.
        """
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            return self._reject(
                f"expected logits [B, {self.num_classes}], got {tuple(logits.shape)}"
            )
        batch = logits.shape[0]
        if batch < 1 or batch > self.max_batch:
            return self._reject(f"batch size {batch} outside [1, {self.max_batch}]")
        if not same_device(logits.device, self.device):
            return self._reject_device(logits.device)
        if self.embed_dim > 0:
            if embeddings.ndim != 2 or embeddings.shape[1] != self.embed_dim:
                return self._reject(
                    f"expected embeddings [B, {self.embed_dim}], got {tuple(embeddings.shape)}"
                )
            if embeddings.shape[0] != batch:
                return self._reject(
                    f"batch mismatch: logits {batch} vs embeddings {embeddings.shape[0]}"
                )
            if not same_device(embeddings.device, self.device):
                return self._reject_device(embeddings.device)
        if capturing_graph(self.device):
            # Recording events or switching streams inside a graph capture corrupts the
            # graph (design doc §6.4). Dropping is the only safe answer.
            return self._reject("CUDA graph capture is active; observation dropped")
        return True

    def _reject_device(self, actual: torch.device) -> bool:
        """Reject a cross-device observation, loudly.

        This one is not a transient drop, it is a misconfiguration: every subsequent batch
        will be rejected too, so GUARD reports nothing at all while looking installed and
        healthy. It gets its own ERROR with the fix in it, once.
        """
        if not self._warned_device_mismatch:
            self._warned_device_mismatch = True
            logger.error(
                "GUARD is monitoring %s but is being handed tensors on %s, so EVERY "
                "observation is being dropped. Build the monitor on the model's device — "
                "e.g. build_monitor(..., device=%r). One monitor per device.",
                self.device,
                actual,
                str(actual),
            )
        return False

    def _reject(self, reason: str) -> bool:
        """Log the first rejection only — a mis-shaped call repeats every single step."""
        if not self._warned_device_mismatch:
            self._warned_device_mismatch = True
            logger.warning(
                "GUARD dropped an observation (%s). Further drops are counted in "
                "observations_dropped but not logged.",
                reason,
            )
        return False

    def _run_detector(
        self,
        detector: Detector,
        metric_slice: slice,
        row: torch.Tensor,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        """Run one detector and write its scores into the summary row. Host-sync-free."""
        result: DetectorResult = detector.compute(logits, embeddings)
        names = detector.metric_names
        values = [result.scores[name].reshape(()) for name in names]
        if row.dtype != values[0].dtype:
            values = [v.to(row.dtype) for v in values]
        # One stack kernel writing straight into the summary row: no intermediate
        # allocation, no per-metric copy, no host round-trip.
        torch.stack(values, out=row[metric_slice])

    def _enqueue_drain(self, buf: HostBuffer, step: int) -> None:
        """Enqueue the async D2H copy of the summary block. Called on the monitor stream.

        A buffer that is acquired but never reaches ``_pending`` can never be released, and
        the pool has no way to reclaim it — a handful of failures would starve the drain
        permanently. Every path out of here therefore either queues the buffer or returns
        it.
        """
        try:
            # `non_blocking` is only safe into **pinned** memory, and only where the
            # completion event is real. On CUDA both hold: the pool allocates page-locked
            # buffers and `buf.event` is a genuine cudaEvent the drain thread polls before
            # reading. Anywhere else — MPS in particular, which is asynchronous but has no
            # stream/event facade here — an async copy would be read by a stub event that
            # reports "complete" immediately, and the drain would publish whatever the
            # buffer happened to contain. That failure is silent: plausible-looking zeros,
            # exported as real metrics.
            buf.tensor.copy_(self._summary, non_blocking=buf.tensor.is_pinned())
            buf.event.record(self._stream)
        except Exception:
            self._pool.release(buf)
            raise
        self._drains_enqueued += 1
        with self._pending_lock:
            self._pending.append(_PendingDrain(buf=buf, first_step=step - self.drain_every_k + 1))

    # ── the host side ────────────────────────────────────────────────────────

    def drain_pending(self) -> int:
        """Consume every completed device→host copy. Returns the number of rows dispatched.

        Non-blocking: a copy whose event has not completed is left in the queue.

        Serialised by ``_drain_lock``. The aggregator holds unsynchronised windowing and
        change-point state, so exactly one thread may be inside it at a time — otherwise a
        ``close()`` whose thread join timed out would run concurrently with the still-live
        drain thread, interleaving pushes and feeding the change-point tests out of order.
        """
        with self._drain_lock:
            return self._drain_once()

    def _drain_once(self) -> int:
        """Pop every landed buffer and dispatch it. Caller holds ``_drain_lock``."""
        ready: list[_PendingDrain] = []
        try:
            with self._pending_lock:
                # query() is a CUDA call and can raise; anything already popped must not be
                # stranded outside the queue, or its pinned buffer leaks and the pool
                # eventually starves.
                while self._pending and self._pending[0].buf.event.query():
                    ready.append(self._pending.popleft())
        except Exception:
            logger.exception("GUARD failed while polling drain completion")

        rows = 0
        for item in ready:
            try:
                ok, count = self._firewall.call("aggregator", self._dispatch_buffer, item)
                if ok and count is not None:
                    rows += count
            finally:
                self._pool.release(item.buf)
                self._drains_completed += 1
        if ready:
            # Behind the firewall like everything else: the sink is caller-supplied, and a
            # broken exporter must degrade metric export, not kill the drain thread.
            self._firewall.call("export", self._publish_telemetry)
        return rows

    def _dispatch_buffer(self, item: _PendingDrain) -> int:
        """Read one landed pinned buffer and feed every row to the aggregator.

        ``.tolist()`` here reads **host** memory the device already wrote into. It is not a
        device synchronisation — the event we polled is what guarantees the data is there.
        """
        values = item.buf.tensor.tolist()
        for offset, row in enumerate(values):
            self.aggregator.push(
                item.first_step + offset, dict(zip(self.metric_names, row, strict=True))
            )
        self._rows_dispatched += len(values)

        if self.enable_gpu_timing:
            start, stop = self._timing[item.buf.index]
            self._gpu_time_us = start.elapsed_time(stop) * 1000.0
        return len(values)

    def _publish_telemetry(self) -> None:
        """Push GUARD's own health metrics. Drain thread only — never the hot path.

        Counters are published as **deltas** since the last publish, because a Prometheus
        counter is incremented, not set; sending the running total each time would make
        every rate panel grow quadratically.
        """
        with self._lock:
            summaries_now = self._summaries_dropped
            steps_now = self._step
            rejects_now = self._observations_dropped
        dropped_delta = summaries_now - self._summaries_published
        observed_delta = steps_now - self._steps_published
        rejected_delta = rejects_now - self._rejects_published

        try:
            utilization = self.ring_utilization()
        except Exception:
            logger.exception("GUARD could not read ring utilization")
            utilization = float("nan")

        telemetry = {
            "monitor_overhead_us": self._enqueue_us_ema,
            "ring_utilization": utilization,
            "detectors_disabled": float(len(self._firewall.disabled)),
            "steps_observed": float(max(0, observed_delta)),
            "observations_dropped": float(max(0, rejected_delta)),
        }
        if self.enable_gpu_timing:
            telemetry["monitor_gpu_time_us"] = self._gpu_time_us
        self.sink.set_telemetry(telemetry)
        if dropped_delta > 0:
            self.sink.add_dropped(dropped_delta)
        # Advance the high-water marks only once the sink has accepted the deltas. Moving
        # them first means a single failed publish silently erases those counts forever,
        # and a counter that under-reports is worse than one that is briefly stale.
        self._summaries_published = summaries_now
        self._steps_published = steps_now
        self._rejects_published = rejects_now

        # Per-detector failure counts, as deltas. Without this the only signal that a
        # detector is misbehaving is `detectors_disabled` flipping to 1 — which tells you
        # something broke but not what, and only after the breaker has already tripped.
        #
        # Looked up rather than called directly: `record_detector_failure` was added to the
        # MetricSink protocol after the first sinks existed, and a third-party sink written
        # against the older contract must keep working rather than start raising here.
        record = getattr(self.sink, "record_detector_failure", None)
        if record is None:
            return
        for detector in self._detectors:
            total = self._firewall.failure_count(detector.name)
            delta = total - self._failures_published.get(detector.name, 0)
            if delta > 0:
                self._failures_published[detector.name] = total
                record(detector.name, delta)

    # ── introspection & lifecycle ────────────────────────────────────────────

    def ring_utilization(self) -> float:
        """Fraction of used ring slots whose monitor work has not finished on the GPU.

        Polled with ``cudaEventQuery``, which never blocks. A number persistently near 1.0
        means the monitor stream is falling behind inference — the honest signal that
        overlap has stopped being free on this workload.
        """
        slots = self._ring.slots
        used = min(self._step, slots)
        if used == 0:
            return 0.0
        in_flight = sum(1 for i in range(used) if not self._slot_done[i].query())
        return in_flight / slots

    def stats(self) -> MonitorStats:
        """Snapshot of engine health. Cheap; safe from any thread."""
        return MonitorStats(
            enabled=self._kill.enabled and not self._closed,
            device=str(self.device),
            steps_observed=self._step,
            observations_dropped=self._observations_dropped,
            summaries_dropped=self._summaries_dropped,
            drains_enqueued=self._drains_enqueued,
            drains_completed=self._drains_completed,
            rows_dispatched=self._rows_dispatched,
            ring_utilization=self.ring_utilization(),
            last_enqueue_us=self._last_enqueue_us,
            mean_enqueue_us=self._enqueue_us_ema,
            gpu_time_us=self._gpu_time_us,
            disabled_detectors=tuple(sorted(self._firewall.disabled)),
            ring_bytes=self._ring.nbytes(),
            pinned_bytes=self._pool.nbytes(),
        )

    @property
    def firewall(self) -> Firewall:
        """The failure-isolation firewall; inspect it to see what has been disabled."""
        return self._firewall

    @property
    def enabled(self) -> bool:
        """Whether monitoring is currently running (kill-switch and env var respected)."""
        return self._kill.enabled and not self._closed

    def enable(self) -> None:
        """Re-enable monitoring."""
        self._kill.enable()

    def disable(self) -> None:
        """Global kill-switch: stop monitoring without touching the serving code."""
        self._kill.disable()

    def flush(self, timeout: float = 5.0) -> int:
        """Wait for all enqueued monitor work, then drain. **Synchronises** — tests only.

        Never call this from a serving path: it blocks the calling thread on the GPU, which
        is precisely what ``observe`` exists to avoid.
        """
        del timeout  # stream.synchronize() has no timeout; kept for call-site clarity
        self._stream.synchronize()
        return self.drain_pending()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the drain thread, flush what is in flight, and release sinks. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        stalled = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            stalled = self._thread.is_alive()
        try:
            self._stream.synchronize()
            if stalled:
                # The drain thread is wedged (a hung sink, a slow webhook). `drain_pending`
                # is mutually exclusive, so a final drain here would be correct but would
                # block on that same wedge. Report it and leave the last block undelivered
                # rather than hang the caller's shutdown.
                logger.error(
                    "GUARD drain thread did not stop within %.1fs; skipping the final "
                    "flush, %d summary block(s) undelivered.",
                    timeout,
                    len(self._pending),
                )
            else:
                self.drain_pending()
        except Exception:
            logger.exception("GUARD failed to flush during close()")
        if self._alert_sink is not None:
            try:
                self._alert_sink.close()
            except Exception:
                logger.exception("GUARD alert sink failed to close")

    def __enter__(self) -> GuardMonitor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GuardMonitor(device={self.device}, detectors="
            f"{[d.name for d in self._detectors]}, metrics={len(self.metric_names)}, "
            f"steps={self._step}, enabled={self.enabled})"
        )


def _metric_layout(detectors: Sequence[Detector]) -> tuple[tuple[str, str], ...]:
    """Flatten detectors into ``(metric_name, detector_name)`` pairs in a stable order.

    Raises:
        TypeError: if a detector does not declare ``metric_names``. The engine allocates a
            fixed-width summary buffer at construction, so the metric schema has to be
            known before the first batch — discovering it from the first result would mean
            an allocation on the hot path and a schema that can change under the exporter.
        ValueError: on duplicate metric names across detectors, which would silently make
            one detector overwrite another's column.
    """
    layout: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for detector in detectors:
        names = getattr(detector, "metric_names", None)
        if not names:
            raise TypeError(
                f"detector {getattr(detector, 'name', detector)!r} does not declare "
                f"'metric_names'. Add a class or instance attribute listing the keys its "
                f"compute() returns, e.g. metric_names = ('my_metric',)."
            )
        for name in names:
            if name in seen:
                raise ValueError(
                    f"metric name {name!r} is emitted by both {seen[name]!r} and "
                    f"{detector.name!r}; metric names must be unique across detectors"
                )
            seen[name] = detector.name
            layout.append((name, detector.name))
    return tuple(layout)


def _drain_loop(
    monitor_ref: weakref.ReferenceType[GuardMonitor],
    poll_interval: float,
    stop: threading.Event,
) -> None:
    """Background drain body, holding only a **weak** reference to the monitor.

    Module-level and weak on purpose: a bound method would keep the monitor alive for the
    process lifetime, since this loop only exits when told to. Taking a strong reference
    per iteration and dropping it before sleeping means a monitor nobody else holds is
    collected, its finalizer sets ``stop``, and the thread ends.

    The body is contained. If an exception escaped, the thread would die silently and
    metric export would stop for the rest of the process while ``stats().enabled`` kept
    reporting ``True`` — a monitor that has stopped monitoring and does not say so.
    """
    while not stop.is_set():
        monitor = monitor_ref()
        if monitor is None:
            return
        try:
            idle = monitor.drain_pending() == 0
        except Exception:
            logger.exception("GUARD drain iteration failed; continuing")
            idle = True
        del monitor  # do not hold the monitor alive across the sleep
        if idle:
            stop.wait(poll_interval)


def _detector_slices(detectors: Sequence[Detector]) -> tuple[tuple[Detector, slice], ...]:
    """Map each detector to its contiguous span of columns in the summary row."""
    spans: list[tuple[Detector, slice]] = []
    offset = 0
    for detector in detectors:
        width = len(detector.metric_names)
        spans.append((detector, slice(offset, offset + width)))
        offset += width
    return tuple(spans)


def build_monitor(
    baseline: Baseline,
    config: Config,
    *,
    max_batch: int,
    device: torch.device | str | None = None,
    sink: MetricSink | None = None,
    alert_sink: AlertSink | None = None,
    **kwargs: Any,
) -> GuardMonitor:
    """Build a fully wired monitor from a baseline and a config. The one-liner entry point.

    Constructs exactly the detectors ``config.detectors`` enables, sizes the ring from the
    baseline's ``num_classes``/``embed_dim``, and stamps the baseline's model version and
    checksum onto every alert so a firing is traceable to the reference it was judged
    against.

    When ``sink`` is omitted, one is built from ``config.export``: a
    :class:`~guard.export.prometheus.PrometheusExporter` in the configured namespace, or a
    :class:`~guard.export.base.NullSink` when export is disabled. It is **not** served —
    call ``monitor.sink.start_http_server(config.export.prometheus_port)`` when you want the
    endpoint. Binding a port is not something a library should do behind your back.
    """
    from guard.detectors import build_detectors

    detectors = build_detectors(config, baseline)
    if not detectors:
        raise ValueError("config enables no detectors; nothing to monitor")
    needs_embeddings = any(d.name == "embedding" for d in detectors)
    if sink is None:
        sink = _sink_from_config(config)
    _configure_exporter(sink, detectors)
    return GuardMonitor.from_config(
        config,
        detectors,
        num_classes=baseline.num_classes,
        embed_dim=baseline.embed_dim if needs_embeddings else 0,
        max_batch=max_batch,
        device=device,
        sink=sink,
        alert_sink=alert_sink,
        model_version=baseline.model_version,
        baseline_version=baseline.checksum[:12],
        **kwargs,
    )


def _sink_from_config(config: Config) -> MetricSink:
    """Build the metric sink ``config.export`` describes."""
    from guard.export.prometheus import PrometheusExporter

    if not config.export.enabled:
        return NullSink()
    return PrometheusExporter(namespace=config.export.namespace)


def _configure_exporter(sink: MetricSink | None, detectors: Sequence[Detector]) -> None:
    """Teach a Prometheus sink about the metric schema it is about to receive.

    Also decides whether ``guard_entropy_p99`` gets fed: the design doc locks that name,
    but the entropy detector's quantile is configurable, and publishing a p75 under a gauge
    whose HELP text says "99th percentile" is the kind of quiet lie that makes an on-call
    engineer stop trusting the dashboard. ``guard_entropy_quantile`` always carries the
    true value; the p99 alias is fed only when the configured quantile really is 0.99.
    """
    from guard.export.prometheus import PrometheusExporter

    if not isinstance(sink, PrometheusExporter):
        return
    for detector in detectors:
        sink.ensure_metrics(detector.metric_names)
    quantiles = [getattr(d, "quantile", None) for d in detectors if d.name == "entropy"]
    sink.emit_p99_alias = all(q is None or abs(q - 0.99) < 1e-9 for q in quantiles)


__all__ = [
    "GuardMonitor",
    "MonitorStats",
    "build_monitor",
    "is_cuda",
]
