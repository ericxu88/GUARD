"""End-to-end verification on a real non-CPU device, whichever one is present.

Runs on CUDA when there is a GPU and on MPS on Apple silicon; skips cleanly when neither
exists. These are deliberately *not* marked ``gpu`` — they are not about stream overlap,
they are about the thing every accelerator shares and CPU hides: a **separate memory
space**. Device-resident ring buffers, real device→host drain copies, indexed device
identity, and detector kernels that have to exist on a non-CPU backend at all.

That last point is not hypothetical. ``torch.histogram`` is CPU-only even in a CUDA build,
and several detector ops (``bucketize``, ``scatter_add_``, ``quantile``, ``mm(out=)``,
``outer(out=)``, ``linalg.norm``) have uneven backend coverage. A CPU-only suite cannot
tell you whether the detector math is genuinely device-agnostic; running it on any real
accelerator can.

MPS was also where a real bug surfaced: ``torch.device("mps")`` carries no index while
every tensor reports ``mps:0``, so a strict ``==`` rejected 100% of observations while the
monitor looked healthy. See :func:`test_unindexed_device_still_matches_its_tensors`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch

from guard.baseline.schema import Baseline
from guard.config import Config, DetectorConfig, ThresholdConfig
from guard.detectors import DivergenceDetector, EmbeddingDriftDetector, EntropyDetector
from guard.engine.monitor import GuardMonitor, build_monitor
from guard.engine.streams import resolve_device, same_device

C, D, B = 16, 8, 16


def _accelerator() -> torch.device | None:
    """The best available non-CPU device, or None."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


ACCEL = _accelerator()
requires_accelerator = pytest.mark.skipif(
    ACCEL is None, reason="no non-CPU device (CUDA or MPS) available"
)


def _baseline() -> Baseline:
    return Baseline(
        model_version="accel@1",
        created_at="2026-01-01T00:00:00Z",
        sample_count=512,
        num_classes=C,
        embed_dim=D,
        class_distribution=torch.full((C,), 1.0 / C),
        entropy_histogram=torch.full((16,), 32.0),
        entropy_bin_edges=torch.linspace(0.0, math.log(C), 17),
        embed_mean=torch.zeros(D),
        embed_precision=torch.eye(D),
    )


@pytest.fixture
def device() -> Iterator[torch.device]:
    assert ACCEL is not None
    yield ACCEL


# --------------------------------------------------------------------------------------
# Device identity — the bug MPS exposed
# --------------------------------------------------------------------------------------
def test_resolve_device_gives_every_accelerator_a_concrete_index() -> None:
    # A tensor's `.device` always carries an index. Leaving the monitor's device unindexed
    # made `logits.device != self.device` true for every batch, so the engine dropped 100%
    # of observations and reported nothing but a counter.
    assert resolve_device("cpu") == torch.device("cpu")
    for name in ("mps", "xpu"):
        resolved = resolve_device(name)
        assert resolved.type == name
        assert resolved.index == 0, f"{name} was left unindexed: {resolved}"
    if torch.cuda.is_available():
        assert resolve_device("cuda").index is not None


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("cpu", "cpu", True),
        ("mps", "mps:0", True),  # the exact pairing that used to reject every batch
        ("mps:0", "mps", True),
        ("cuda", "cuda:0", True),
        ("cuda:0", "cuda:1", False),
        ("cpu", "mps", False),
    ],
)
def test_same_device_tolerates_a_missing_index(a: str, b: str, expected: bool) -> None:
    assert same_device(torch.device(a), torch.device(b)) is expected


@requires_accelerator
def test_unindexed_device_still_matches_its_tensors(device: torch.device) -> None:
    """The regression itself: build the monitor from the bare device name, hand it tensors,
    and require that every one is accepted."""
    monitor = GuardMonitor(
        [EntropyDetector(baseline=_baseline())],
        num_classes=C,
        embed_dim=D,
        max_batch=B,
        device=device.type,  # bare name, no index — as a user would write it
        drain_every_k=2,
        start_drain_thread=False,
    )
    for _ in range(6):
        monitor.observe(torch.randn(B, C, device=device), torch.randn(B, D, device=device))
    monitor.flush()

    stats = monitor.stats()
    assert stats.observations_dropped == 0, (
        f"{stats.observations_dropped} observations rejected as cross-device on "
        f"{monitor.device}; the monitor is monitoring nothing"
    )
    assert stats.steps_observed == 6
    monitor.close()


@requires_accelerator
def test_tensors_from_another_device_are_rejected_and_counted(device: torch.device) -> None:
    # The check must still be strict about genuinely different devices — one monitor per
    # device is the documented topology.
    monitor = GuardMonitor(
        [EntropyDetector(baseline=_baseline())],
        num_classes=C,
        embed_dim=D,
        max_batch=B,
        device=device,
        start_drain_thread=False,
    )
    monitor.observe(torch.randn(B, C), torch.randn(B, D))  # CPU tensors
    assert monitor.stats().observations_dropped == 1
    assert monitor.stats().steps_observed == 0
    monitor.close()


# --------------------------------------------------------------------------------------
# Detector math is genuinely device-agnostic
# --------------------------------------------------------------------------------------
@requires_accelerator
@pytest.mark.parametrize("name", ["entropy", "divergence", "embedding"])
def test_detector_matches_cpu_reference_on_device(name: str, device: torch.device) -> None:
    """Prime directive #4, against whatever accelerator this machine has.

    Tolerance is loose relative to DEFAULT_RTOL because MPS computes in a different order
    than CPU and is float32-only; the point is that the kernels exist and agree, not that
    they are bit-identical.
    """
    baseline = _baseline()
    builders = {
        "entropy": lambda: EntropyDetector(baseline=baseline),
        "divergence": lambda: DivergenceDetector(baseline),
        "embedding": lambda: EmbeddingDriftDetector(baseline, cov_every=1),
    }
    torch.manual_seed(0)
    logits, embeddings = torch.randn(64, C) * 2, torch.randn(64, D)

    cpu = builders[name]().compute(logits, embeddings).scores
    acc = builders[name]().compute(logits.to(device), embeddings.to(device)).scores

    assert set(cpu) == set(acc)
    for key, expected in cpu.items():
        actual = acc[key].to("cpu")
        assert torch.isfinite(actual), f"{key} is not finite on {device}"
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (
            f"{key}: {device} gave {float(actual)!r}, CPU gave {float(expected)!r}"
        )


# --------------------------------------------------------------------------------------
# The whole pipeline, on device
# --------------------------------------------------------------------------------------
@requires_accelerator
def test_engine_state_lives_on_the_device(device: torch.device) -> None:
    monitor = GuardMonitor(
        [EntropyDetector(baseline=_baseline())],
        num_classes=C,
        embed_dim=D,
        max_batch=B,
        device=device,
        start_drain_thread=False,
    )
    assert monitor._ring.logits.device.type == device.type
    assert monitor._summary.device.type == device.type
    # The drain destination is host memory — that is the whole point of the drain.
    assert monitor._pool._buffers[0].tensor.device.type == "cpu"
    monitor.close()


@requires_accelerator
def test_full_pipeline_detects_drift_on_device(device: torch.device) -> None:
    """Clean traffic is quiet, injected covariate shift alerts — with every tensor living
    in the accelerator's memory space and every metric crossing a real D2H boundary."""
    baseline = _baseline()
    config = Config(
        window_size=2,
        drain_every_k=4,
        detectors=[
            DetectorConfig(name="entropy"),
            DetectorConfig(name="divergence"),
            DetectorConfig(name="embedding", params={"cov_every": 4}),
        ],
        thresholds={
            "embedding_mahalanobis_mean": ThresholdConfig(threshold=3.0, delta=0.5, debounce=2)
        },
    )
    monitor = build_monitor(baseline, config, max_batch=B, device=device, start_drain_thread=False)
    torch.manual_seed(0)

    for _ in range(40):
        monitor.observe((torch.randn(B, C) * 2).to(device), torch.randn(B, D).to(device))
        monitor.drain_pending()
    monitor.flush()
    clean = dict(monitor.aggregator.snapshot())
    assert set(clean) == set(monitor.metric_names)
    assert all(math.isfinite(v) for v in clean.values()), clean
    assert monitor.aggregator.alerts_fired == 0, "clean traffic must not alert"

    for _ in range(40):
        monitor.observe((torch.randn(B, C) * 2).to(device), (torch.randn(B, D) + 7.0).to(device))
        monitor.drain_pending()
    monitor.flush()
    drifted = monitor.aggregator.snapshot()

    assert monitor.aggregator.alerts_fired > 0, "injected embedding shift must alert"
    assert drifted["embedding_mahalanobis_mean"] > 3 * clean["embedding_mahalanobis_mean"]
    assert monitor.stats().observations_dropped == 0
    assert monitor.stats().disabled_detectors == ()
    monitor.close()


@requires_accelerator
def test_drained_values_survive_the_device_to_host_copy(device: torch.device) -> None:
    """The drain is the only place data leaves the device; a wrong dtype, stride, or
    non-blocking copy from unpinned memory would show up here as garbage."""

    class MeanDetector:
        name = "mean"
        metric_names: tuple[str, ...] = ("mean_value",)

        def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> object:
            from guard.detectors.base import DetectorResult

            del embeddings
            return DetectorResult(scores={"mean_value": logits.mean()})

        def reference(self) -> dict[str, object]:
            return {}

    seen: list[float] = []

    class Collect:
        def update(self, values: dict[str, float]) -> None:
            if "mean_value" in values:
                seen.append(values["mean_value"])

        def set_alert(self, detector: str, active: bool) -> None: ...

        def add_dropped(self, count: int) -> None: ...

        def set_telemetry(self, values: dict[str, float]) -> None: ...

    monitor = GuardMonitor(
        [MeanDetector()],  # type: ignore[list-item]
        num_classes=C,
        embed_dim=D,
        max_batch=B,
        device=device,
        drain_every_k=4,
        max_pending_drains=8,
        start_drain_thread=False,
    )
    monitor.aggregator.sink = Collect()  # type: ignore[assignment]
    embeddings = torch.randn(B, D, device=device)

    for i in range(32):
        monitor.observe(torch.full((B, C), float(i), device=device), embeddings)
        monitor.drain_pending()
    monitor.flush()

    assert seen == pytest.approx([float(i) for i in range(32)], abs=1e-4), (
        "values were corrupted crossing the device→host boundary"
    )
    monitor.close()
