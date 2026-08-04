"""Integration layer: ``GuardModule`` — hook capture, output transparency, containment.

The wrapper is the only piece of GUARD that sits *inside* the serving call stack, so its
contract is narrower and stricter than any detector's: it must hand the engine the two
tensors it needs, and it must return the model's output identically, on the same code
path, no matter what monitoring does behind it (prime directive #1).

Everything here runs on small hand-built ``nn.Module``s — no torchvision, no downloads —
because the properties under test are about hooks, thread-local state, and exception
containment, none of which depend on the model being real.
"""

from __future__ import annotations

import copy
import pickle
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

from guard.detectors.base import DetectorResult
from guard.engine.monitor import GuardMonitor
from guard.integration.torch_module import (
    EmbedReduce,
    GuardModule,
    candidate_embed_layers,
    reduce_embedding,
)

IN_DIM = 6
EMBED_DIM = 4
NUM_CLASSES = 3


# --------------------------------------------------------------------------------------
# Detectors: the engine's view of what the wrapper delivered
# --------------------------------------------------------------------------------------
class _RecordingDetector:
    """Keeps a copy of every ``(logits, embeddings)`` pair the engine hands it.

    Copies, not references: the tensors a detector receives are views into GUARD's ring
    buffer, whose slots are overwritten a few steps later.
    """

    name = "recording"
    metric_names: tuple[str, ...] = ("recording_mean",)

    def __init__(self) -> None:
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        self.seen.append((logits.clone(), embeddings.clone()))
        return DetectorResult({"recording_mean": logits.mean()})

    def reference(self) -> dict[str, object]:
        return {}


class _ExplodingDetector:
    """A detector that always raises — the thing prime directive #1 must contain."""

    name = "exploding"
    metric_names: tuple[str, ...] = ("exploding_score",)

    def __init__(self) -> None:
        self.calls = 0

    def compute(self, logits: torch.Tensor, embeddings: torch.Tensor) -> DetectorResult:
        self.calls += 1
        raise RuntimeError("detector blew up")

    def reference(self) -> dict[str, object]:
        return {}


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class _TinyNet(nn.Module):
    """``features`` (the tap point, ``[B, D]``) feeding a linear ``head`` (``[B, C]``)."""

    def __init__(self, in_dim: int = IN_DIM, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.features = nn.Linear(in_dim, embed_dim)
        self.head = nn.Linear(embed_dim, NUM_CLASSES)
        self.model_tag = "tiny-net"

    def describe(self, prefix: str) -> str:
        """A plain method, to check the wrapper delegates callables and not just data."""
        return f"{prefix}:{self.model_tag}"

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.head(self.features(x))
        return logits

    def forward(self, x: torch.Tensor) -> Any:
        self.last_output: Any = self._logits(x)
        return self.last_output


class _TupleNet(_TinyNet):
    """Returns ``(logits, aux)`` — the shape a torchvision auxiliary-head model returns."""

    def forward(self, x: torch.Tensor) -> Any:
        self.last_output = (self._logits(x), torch.zeros(1))
        return self.last_output


class _DictNet(_TinyNet):
    """Returns a dict keyed ``"logits"`` — the HuggingFace/detection dict convention."""

    def forward(self, x: torch.Tensor) -> Any:
        self.last_output = {"logits": self._logits(x), "loss": None}
        return self.last_output


class _OutKeyNet(_TinyNet):
    """Returns a dict keyed ``"out"`` — torchvision's segmentation convention."""

    def forward(self, x: torch.Tensor) -> Any:
        self.last_output = {"out": self._logits(x)}
        return self.last_output


@dataclass
class _Output:
    """Stand-in for a HuggingFace ``ModelOutput``: logits live on an attribute."""

    logits: torch.Tensor
    hidden: torch.Tensor


class _AttrNet(_TinyNet):
    """Returns an object exposing ``.logits`` — the HuggingFace convention."""

    def forward(self, x: torch.Tensor) -> Any:
        self.last_output = _Output(logits=self._logits(x), hidden=torch.zeros(1))
        return self.last_output


class _OpaqueNet(_TinyNet):
    """Returns something with no recoverable logits, to prove the wrapper just skips."""

    def forward(self, x: torch.Tensor) -> Any:
        self.last_output = "not a tensor"
        return self.last_output


class _SeqTap(nn.Module):
    """Emits a ``[B, T, D]`` sequence activation, like a transformer encoder's final norm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token 0 carries the input and the rest are zeros, so embed_reduce='cls' must
        # hand the engine exactly the input back — a mean would not.
        out = torch.zeros(x.shape[0], 3, x.shape[1], dtype=x.dtype)
        out[:, 0] = x
        return out


class _SeqNet(nn.Module):
    """Sequence model whose tap point is 3-D, exercising the reduce path end to end."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _SeqTap()
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.head(self.encoder(x)[:, 0])
        return logits


class _TupleTap(nn.Module):
    """Submodule returning ``(tensor, state)`` — an LSTM-shaped hook payload."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, str]:
        return x * 2.0, "state"


class _TupleTapNet(nn.Module):
    """The hooked submodule returns a tuple; ``_capture`` must take element 0."""

    def __init__(self) -> None:
        super().__init__()
        self.features = _TupleTap()
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.head(self.features(x)[0])
        return logits


class _TagNet(nn.Module):
    """Embedding is the input; logits are its first ``C`` columns.

    A per-request tag written into every element therefore appears in *both* tensors, so a
    thread that observes another thread's captured activation shows up as a tag mismatch
    rather than as a plausible-looking number.
    """

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Identity()
        self.delay = delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.features(x)
        if self.delay:
            # Widen the window between the hook stashing the activation and forward()
            # handing it to the monitor. That gap is exactly where a shared (non
            # thread-local) stash would be clobbered by a concurrent request.
            time.sleep(self.delay)
        logits: torch.Tensor = embedding[:, :NUM_CLASSES].contiguous()
        return logits


# --------------------------------------------------------------------------------------
# Fixtures & helpers
# --------------------------------------------------------------------------------------
MonitorFactory = Callable[..., GuardMonitor]


@pytest.fixture
def make_monitor() -> Iterator[MonitorFactory]:
    """Build monitors that are always closed, so no drain thread leaks out of a test."""
    built: list[GuardMonitor] = []

    def factory(
        detectors: list[Any] | None = None,
        *,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        max_batch: int = 8,
        **kwargs: Any,
    ) -> GuardMonitor:
        monitor = GuardMonitor(
            detectors if detectors is not None else [_RecordingDetector()],
            num_classes=num_classes,
            embed_dim=embed_dim,
            max_batch=max_batch,
            start_drain_thread=False,
            drain_every_k=1,
            **kwargs,
        )
        built.append(monitor)
        return monitor

    yield factory
    for monitor in built:
        monitor.close()


class _StashSpy(GuardModule):
    """Exposes the thread-local stash, which ``forward`` otherwise consumes and clears."""

    stashed: torch.Tensor | None = None

    def _take_embedding(self) -> torch.Tensor | None:
        self.stashed = getattr(self._local, "embedding", None)
        return super()._take_embedding()


def _wrap(model: nn.Module, monitor: GuardMonitor, **kwargs: Any) -> GuardModule:
    """Wrap ``model`` and put it in eval mode, which is what ``observe_in_eval_only`` gates on."""
    wrapped = GuardModule(model, monitor, **kwargs)
    wrapped.eval()
    return wrapped


def _inputs(batch: int = 2, seed: int = 0, in_dim: int = IN_DIM) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, in_dim)


# --------------------------------------------------------------------------------------
# Hook capture: does the engine see the right tensors?
# --------------------------------------------------------------------------------------
def test_hook_delivers_named_submodule_output(make_monitor: MonitorFactory) -> None:
    torch.manual_seed(0)
    model = _TinyNet().eval()
    x = _inputs(batch=5)
    # Compute the expectations BEFORE wrapping: calling model.features(x) once the hook is
    # installed would stash a phantom activation for the next forward to consume.
    expected_embed = model.features(x)
    expected_logits = model(x)

    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    wrapped = _wrap(model, monitor, embed_layer="features")

    wrapped(x)
    monitor.flush()

    assert len(detector.seen) == 1
    logits, embeddings = detector.seen[0]
    assert logits.shape == (5, NUM_CLASSES)
    assert embeddings.shape == (5, EMBED_DIM)
    # The ring holds float32 copies of exactly these tensors, so equality is exact.
    assert torch.equal(logits, expected_logits)
    assert torch.equal(embeddings, expected_embed)


def test_steps_observed_increments_once_per_forward(make_monitor: MonitorFactory) -> None:
    torch.manual_seed(1)
    monitor = make_monitor()
    wrapped = _wrap(_TinyNet().eval(), monitor, embed_layer="features")

    for i in range(4):
        wrapped(_inputs(batch=2, seed=i))
        monitor.flush()
        assert monitor.stats().steps_observed == i + 1

    # A dropped observation would surface here rather than as a silent gap in the stream.
    assert monitor.stats().observations_dropped == 0


def test_tuple_returning_submodule_is_captured(make_monitor: MonitorFactory) -> None:
    """A hooked module returning ``(tensor, state)`` must yield element 0, not the tuple."""
    torch.manual_seed(2)
    x = _inputs(batch=3, in_dim=EMBED_DIM)

    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    _wrap(_TupleTapNet().eval(), monitor, embed_layer="features")(x)
    monitor.flush()

    assert torch.equal(detector.seen[0][1], x * 2.0)


def test_three_dim_activation_is_reduced_before_the_engine(make_monitor: MonitorFactory) -> None:
    """Without a reduce the engine would reject a [B, T, D] activation as mis-shaped."""
    torch.manual_seed(3)
    x = _inputs(batch=2, in_dim=EMBED_DIM)

    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    _wrap(_SeqNet().eval(), monitor, embed_layer="encoder", embed_reduce="cls")(x)
    monitor.flush()

    assert monitor.stats().observations_dropped == 0
    assert torch.equal(detector.seen[0][1], x)


# --------------------------------------------------------------------------------------
# Output transparency: forward() returns the model's object, untouched
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory",
    [_TinyNet, _TupleNet, _DictNet, _OutKeyNet, _AttrNet, _OpaqueNet],
    ids=["tensor", "tuple", "dict", "dict_out_key", "logits_attr", "opaque"],
)
def test_forward_returns_the_models_object_identically(
    factory: Callable[[], _TinyNet], make_monitor: MonitorFactory
) -> None:
    torch.manual_seed(4)
    model = factory().eval()
    monitor = make_monitor()
    wrapped = _wrap(model, monitor, embed_layer="features")

    result = wrapped(_inputs(batch=2))
    monitor.flush()

    # Identity, not equality: a wrapper that rebuilt the output (even correctly) would
    # break callers that mutate it or compare it by identity.
    assert result is model.last_output


@pytest.mark.parametrize(
    ("factory", "extract"),
    [
        (_TinyNet, lambda out: out),
        (_TupleNet, lambda out: out[0]),
        (_DictNet, lambda out: out["logits"]),
        (_OutKeyNet, lambda out: out["out"]),
        (_AttrNet, lambda out: out.logits),
    ],
    ids=["tensor", "tuple", "dict", "dict_out_key", "logits_attr"],
)
def test_logits_are_found_in_every_supported_output_shape(
    factory: Callable[[], _TinyNet],
    extract: Callable[[Any], torch.Tensor],
    make_monitor: MonitorFactory,
) -> None:
    torch.manual_seed(5)
    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    wrapped = _wrap(factory().eval(), monitor, embed_layer="features")

    result = wrapped(_inputs(batch=2))
    monitor.flush()

    assert len(detector.seen) == 1
    assert torch.equal(detector.seen[0][0], extract(result))


def test_unrecognized_output_skips_the_step_without_raising(make_monitor: MonitorFactory) -> None:
    """No logits means no observation — but never an exception into the serving path."""
    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    wrapped = _wrap(_OpaqueNet().eval(), monitor, embed_layer="features")

    assert wrapped(_inputs(batch=2)) == "not a tensor"
    monitor.flush()
    assert detector.seen == []
    assert monitor.stats().steps_observed == 0


def test_custom_logits_from_overrides_the_heuristic(make_monitor: MonitorFactory) -> None:
    """The escape hatch for output types the default heuristic cannot know about."""
    torch.manual_seed(6)
    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    wrapped = _wrap(
        _TupleNet().eval(),
        monitor,
        embed_layer="features",
        # Deliberately the *aux* slot, broadcast to [B, C]: proves the callable is what is
        # consulted, since the tuple heuristic would have taken element 0 instead.
        logits_from=lambda out: out[1].reshape(1, 1).expand(2, NUM_CLASSES).contiguous(),
    )

    wrapped(_inputs(batch=2))
    monitor.flush()

    assert torch.equal(detector.seen[0][0], torch.zeros(2, NUM_CLASSES))


# --------------------------------------------------------------------------------------
# reduce_embedding
# --------------------------------------------------------------------------------------
def test_reduce_two_dim_is_passthrough() -> None:
    tensor = torch.randn(4, EMBED_DIM)
    modes: list[EmbedReduce] = ["auto", "none", "cls", "mean", "flatten"]
    for mode in modes:
        # Same object, not a copy: an already-[B, D] activation must cost nothing.
        assert reduce_embedding(tensor, mode) is tensor


def test_reduce_cls_takes_token_zero() -> None:
    torch.manual_seed(7)
    tensor = torch.randn(2, 5, EMBED_DIM)
    reduced = reduce_embedding(tensor, "cls")
    assert torch.equal(reduced, tensor[:, 0])
    # Token 0 and the token-mean must be distinguishable, or the assertion above is empty.
    assert not torch.allclose(reduced, tensor.mean(dim=1))


def test_reduce_mean_averages_over_tokens() -> None:
    torch.manual_seed(8)
    tensor = torch.randn(2, 5, EMBED_DIM)
    reduced = reduce_embedding(tensor, "mean")
    assert torch.allclose(reduced, tensor.mean(dim=1))
    assert not torch.allclose(reduced, tensor[:, 0])


def test_reduce_auto_global_average_pools_a_feature_map() -> None:
    """[B, C, H, W] -> [B, C]: 'auto' must pool spatially, not flatten to C*H*W."""
    torch.manual_seed(9)
    tensor = torch.randn(2, EMBED_DIM, 3, 3)
    reduced = reduce_embedding(tensor, "auto")
    assert reduced.shape == (2, EMBED_DIM)
    assert torch.allclose(reduced, tensor.mean(dim=(2, 3)))


def test_reduce_auto_averages_tokens_for_a_sequence() -> None:
    torch.manual_seed(10)
    tensor = torch.randn(2, 5, EMBED_DIM)
    reduced = reduce_embedding(tensor, "auto")
    # Averaging, not flattening: flattening a [B, 197, 768] ViT activation would give a
    # 151k-wide "embedding" whose reference covariance is a 23 GB matrix.
    assert reduced.shape == (2, EMBED_DIM)
    assert torch.allclose(reduced, tensor.mean(dim=1))


def test_reduce_flatten_keeps_every_element() -> None:
    torch.manual_seed(11)
    tensor = torch.randn(2, 5, EMBED_DIM)
    reduced = reduce_embedding(tensor, "flatten")
    assert reduced.shape == (2, 5 * EMBED_DIM)
    assert torch.equal(reduced, tensor.reshape(2, -1))


def test_reduce_none_rejects_a_three_dim_activation() -> None:
    with pytest.raises(ValueError, match="embed_reduce='none'"):
        reduce_embedding(torch.randn(2, 5, EMBED_DIM), "none")


def test_reduce_cls_rejects_a_four_dim_activation() -> None:
    with pytest.raises(ValueError, match="embed_reduce='cls'"):
        reduce_embedding(torch.randn(2, EMBED_DIM, 3, 3), "cls")


# --------------------------------------------------------------------------------------
# Containment: a broken detector must not reach the caller
# --------------------------------------------------------------------------------------
def test_raising_detector_leaves_inference_untouched(make_monitor: MonitorFactory) -> None:
    torch.manual_seed(12)
    model = _TinyNet().eval()
    x = _inputs(batch=2)
    expected = model(x)

    boom = _ExplodingDetector()
    recorder = _RecordingDetector()
    monitor = make_monitor([boom, recorder])
    wrapped = _wrap(model, monitor, embed_layer="features")

    result = wrapped(x)
    monitor.flush()

    assert torch.equal(result, expected)
    assert monitor.firewall.failure_count("exploding") > 0
    # Isolation, not merely containment: the healthy detector still ran this step.
    assert len(recorder.seen) == 1
    assert monitor.stats().steps_observed == 1


def test_repeatedly_failing_detector_is_disabled_by_its_breaker(
    make_monitor: MonitorFactory,
) -> None:
    torch.manual_seed(13)
    boom = _ExplodingDetector()
    monitor = make_monitor([boom, _RecordingDetector()], max_detector_failures=3)
    wrapped = _wrap(_TinyNet().eval(), monitor, embed_layer="features")

    for i in range(6):
        wrapped(_inputs(batch=2, seed=i))
    monitor.flush()

    assert "exploding" in monitor.stats().disabled_detectors
    # Once tripped the firewall stops calling it: 3 failures, then no more invocations.
    assert boom.calls == 3
    assert monitor.stats().steps_observed == 6


def test_misconfigured_reduce_cannot_break_inference(make_monitor: MonitorFactory) -> None:
    """``embed_reduce='none'`` on a 3-D activation raises *inside* the wrapper.

    ``monitor.observe`` has its own firewall, but this failure happens before it — in
    ``_take_embedding`` — so only ``forward``'s own try/except stands between a wiring
    mistake and a 500 from the serving handler.
    """
    torch.manual_seed(14)
    model = _SeqNet().eval()
    x = _inputs(batch=2, in_dim=EMBED_DIM)
    expected = model(x)

    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    wrapped = _wrap(model, monitor, embed_layer="encoder", embed_reduce="none")

    result = wrapped(x)
    monitor.flush()

    assert torch.equal(result, expected)
    assert detector.seen == []
    assert monitor.stats().steps_observed == 0


def test_captured_activation_is_detached_from_the_autograd_graph(
    make_monitor: MonitorFactory,
) -> None:
    """A stashed activation that still carries ``grad_fn`` pins the whole graph alive.

    In a server that forgot ``torch.no_grad()`` that is a per-request memory leak of the
    entire backward graph — caused by monitoring, which is exactly what must not happen.
    """
    torch.manual_seed(15)
    monitor = make_monitor()
    wrapped = _StashSpy(_TinyNet(), monitor, embed_layer="features")
    wrapped.eval()

    with torch.enable_grad():
        wrapped(_inputs(batch=2).requires_grad_(True))
    monitor.flush()

    assert wrapped.stashed is not None
    assert wrapped.stashed.requires_grad is False
    assert wrapped.stashed.grad_fn is None


# --------------------------------------------------------------------------------------
# Training-mode gating
# --------------------------------------------------------------------------------------
def test_train_mode_suppresses_observation_and_eval_resumes(make_monitor: MonitorFactory) -> None:
    torch.manual_seed(14)
    monitor = make_monitor()
    wrapped = GuardModule(_TinyNet(), monitor, embed_layer="features")

    wrapped.train()
    wrapped(_inputs(batch=2, seed=0))
    monitor.flush()
    # Training activations are not what the baseline describes; observing them would raise
    # drift alerts on your own fine-tuning job.
    assert monitor.stats().steps_observed == 0
    assert monitor.stats().observations_dropped == 0

    wrapped.eval()
    wrapped(_inputs(batch=2, seed=1))
    monitor.flush()
    assert monitor.stats().steps_observed == 1


def test_observe_in_eval_only_false_monitors_training_too(make_monitor: MonitorFactory) -> None:
    torch.manual_seed(15)
    monitor = make_monitor()
    wrapped = GuardModule(_TinyNet(), monitor, embed_layer="features", observe_in_eval_only=False)

    wrapped.train()
    wrapped(_inputs(batch=2))
    monitor.flush()

    assert monitor.stats().steps_observed == 1


def test_wrapping_an_already_eval_model_observes(make_monitor: MonitorFactory) -> None:
    """The docstring's own example — no .eval() on the wrapper — must actually monitor.

    ``nn.Module.__init__`` defaults ``training`` to True, so a wrapper that gated on its
    own freshly-initialised flag would install GUARD, leave the dashboards empty, and
    report no error anywhere.
    """
    torch.manual_seed(16)
    monitor = make_monitor()
    wrapped = GuardModule(_TinyNet().eval(), monitor, embed_layer="features")

    wrapped(_inputs(batch=2))
    monitor.flush()

    assert monitor.stats().steps_observed == 1


def test_flipping_the_wrapped_model_to_train_suppresses_observation(
    make_monitor: MonitorFactory,
) -> None:
    """The gate reads the *wrapped* model's mode, which is the one training code flips."""
    torch.manual_seed(16)
    monitor = make_monitor()
    wrapped = _wrap(_TinyNet(), monitor, embed_layer="features")

    wrapped.model.train()  # a reference to the inner model, as a training loop would hold
    wrapped(_inputs(batch=2))
    monitor.flush()

    assert monitor.stats().steps_observed == 0


def test_suppressed_step_does_not_leak_a_stale_activation(make_monitor: MonitorFactory) -> None:
    """A skipped step must drop its capture, or a later step is scored on the wrong batch.

    Detaching the hook in between is what makes a leak visible: the stale activation has
    the same batch size as the live logits, so the engine's shape validation would wave the
    mismatched pair straight through to the detectors.
    """
    torch.manual_seed(17)
    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    wrapped = _wrap(_TagNet(), monitor, embed_layer="features", embed_reduce="none")

    wrapped.train()
    wrapped(torch.full((2, EMBED_DIM), 7.0))  # captured by the hook, must be discarded
    wrapped.eval()
    wrapped.detach_hook()
    wrapped(torch.full((2, EMBED_DIM), 9.0))  # nothing is captured this step
    monitor.flush()

    assert detector.seen == []
    assert monitor.stats().observations_dropped == 1


def test_disabled_monitor_drops_the_capture_too(make_monitor: MonitorFactory) -> None:
    """Same leak, reached through the kill-switch instead of through training mode."""
    torch.manual_seed(18)
    detector = _RecordingDetector()
    monitor = make_monitor([detector])
    wrapped = _wrap(_TagNet(), monitor, embed_layer="features", embed_reduce="none")

    monitor.disable()
    wrapped(torch.full((2, EMBED_DIM), 7.0))
    monitor.enable()
    wrapped.detach_hook()
    wrapped(torch.full((2, EMBED_DIM), 9.0))
    monitor.flush()

    assert detector.seen == []
    assert monitor.stats().observations_dropped == 1


def test_wrapper_adopts_the_wrapped_models_mode(make_monitor: MonitorFactory) -> None:
    """``guarded.training`` must agree with the model it wraps.

    Serving code branches on it, and a ``.train()``/``.eval()`` round-trip through the
    wrapper would otherwise silently flip the inner model's mode.
    """
    assert GuardModule(_TinyNet().eval(), make_monitor(), embed_layer="features").training is False
    assert GuardModule(_TinyNet(), make_monitor(), embed_layer="features").training is True


# --------------------------------------------------------------------------------------
# Construction contract
# --------------------------------------------------------------------------------------
def test_missing_embed_layer_is_rejected_at_construction(make_monitor: MonitorFactory) -> None:
    """Fail loudly at wire-up rather than dropping every observation at runtime."""
    monitor = make_monitor(embed_dim=EMBED_DIM)
    with pytest.raises(ValueError, match="embed_layer"):
        GuardModule(_TinyNet(), monitor)


def test_unknown_embed_layer_lists_candidates(make_monitor: MonitorFactory) -> None:
    monitor = make_monitor()
    with pytest.raises(ValueError, match="no submodule named 'featurez'") as excinfo:
        GuardModule(_TinyNet(), monitor, embed_layer="featurez")
    # The message has to be actionable, so the real names must be in it.
    assert "features" in str(excinfo.value)
    assert "head" in str(excinfo.value)


def test_candidate_embed_layers_lists_submodules_in_forward_order() -> None:
    assert candidate_embed_layers(_TinyNet()) == ["features", "head"]
    # The cap keeps the error message readable on a 300-module network, and keeps the
    # *last* names — the ones near the classifier head, which is where the tap goes.
    assert candidate_embed_layers(_SeqNet(), max_results=1) == ["head"]


def test_logits_only_monitoring_needs_no_embed_layer(make_monitor: MonitorFactory) -> None:
    """embed_dim=0 is the output-distribution-only deployment: no hook, no embeddings."""
    torch.manual_seed(19)
    detector = _RecordingDetector()
    monitor = make_monitor([detector], embed_dim=0)
    model = _TinyNet().eval()
    wrapped = _wrap(model, monitor, embed_layer=None)

    wrapped(_inputs(batch=2))
    monitor.flush()

    assert wrapped._handle is None
    assert model.features._forward_hooks == {}
    assert monitor.stats().steps_observed == 1
    assert detector.seen[0][1].shape == (2, 0)


# --------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------
def test_detach_hook_removes_the_hook(make_monitor: MonitorFactory) -> None:
    torch.manual_seed(20)
    model = _TinyNet().eval()
    monitor = make_monitor()
    wrapped = _wrap(model, monitor, embed_layer="features")

    wrapped(_inputs(batch=2, seed=0))
    monitor.flush()
    assert len(model.features._forward_hooks) == 1
    assert monitor.stats().steps_observed == 1

    wrapped.detach_hook()
    assert model.features._forward_hooks == {}

    wrapped(_inputs(batch=2, seed=1))
    monitor.flush()
    # No capture means no embedding, and the engine rejects rather than inventing one.
    assert monitor.stats().steps_observed == 1
    assert monitor.stats().observations_dropped == 1


def test_detach_hook_is_idempotent(make_monitor: MonitorFactory) -> None:
    model = _TinyNet().eval()
    wrapped = _wrap(model, make_monitor(), embed_layer="features")
    wrapped.detach_hook()
    wrapped.detach_hook()  # a second remove() on a stale handle must not raise
    assert model.features._forward_hooks == {}


def test_close_detaches_the_hook_and_stops_the_monitor(make_monitor: MonitorFactory) -> None:
    model = _TinyNet().eval()
    monitor = make_monitor()
    wrapped = _wrap(model, monitor, embed_layer="features")

    wrapped.close()

    assert model.features._forward_hooks == {}
    assert monitor.enabled is False
    # Still serving after shutdown: close() removes monitoring, not inference.
    assert wrapped(_inputs(batch=2)).shape == (2, NUM_CLASSES)


# --------------------------------------------------------------------------------------
# Transparency of the wrapper itself
# --------------------------------------------------------------------------------------
def test_attributes_delegate_to_the_wrapped_model(make_monitor: MonitorFactory) -> None:
    """Serving code does ``model.config`` / ``model.generate``; wrapping must not break it."""
    model = _TinyNet().eval()
    wrapped = _wrap(model, make_monitor(), embed_layer="features")

    assert wrapped.model_tag == "tiny-net"
    assert wrapped.describe("hi") == "hi:tiny-net"
    assert wrapped.head is model.head  # submodules stay reachable by name
    # The wrapper's own attributes still win over the delegate.
    assert wrapped.embed_layer == "features"
    assert wrapped.model is model


def test_missing_attribute_still_raises_attribute_error(make_monitor: MonitorFactory) -> None:
    wrapped = _wrap(_TinyNet(), make_monitor(), embed_layer="features")
    with pytest.raises(AttributeError):
        _ = wrapped.definitely_not_here


# --------------------------------------------------------------------------------------
# Thread safety: N request threads must never cross embeddings
# --------------------------------------------------------------------------------------
def test_concurrent_forwards_never_cross_embeddings(make_monitor: MonitorFactory) -> None:
    """Four "request threads", each with its own tag and batch size, sharing one wrapper.

    A wrapper that stashed the captured activation on ``self`` instead of in thread-local
    storage would pair thread A's logits with thread B's embedding — visible here either as
    a tag mismatch or, because the batch sizes differ, as a dropped observation.
    """
    n_threads = 4
    iterations = 25
    batch_sizes = [2, 3, 4, 5]

    detector = _RecordingDetector()
    monitor = make_monitor([detector], embed_dim=EMBED_DIM, max_batch=max(batch_sizes))
    wrapped = _wrap(
        _TagNet(delay=0.001).eval(), monitor, embed_layer="features", embed_reduce="none"
    )
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        tag = float(index + 1)
        x = torch.full((batch_sizes[index], EMBED_DIM), tag)
        try:
            for _ in range(iterations):
                out = wrapped(x)
                assert torch.equal(out, torch.full((batch_sizes[index], NUM_CLASSES), tag))
        except BaseException as exc:  # re-raised on the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    monitor.flush()

    assert not errors, errors
    assert all(not t.is_alive() for t in threads)

    total = n_threads * iterations
    assert monitor.stats().steps_observed == total
    assert monitor.stats().observations_dropped == 0
    assert len(detector.seen) == total

    counts = dict.fromkeys(range(1, n_threads + 1), 0)
    for logits, embeddings in detector.seen:
        tag = int(logits[0, 0].item())
        # The same tag in both tensors, and both batch dims from the same request.
        assert torch.equal(embeddings, torch.full_like(embeddings, float(tag)))
        assert embeddings.shape[0] == logits.shape[0] == batch_sizes[tag - 1]
        counts[tag] += 1
    assert counts == dict.fromkeys(range(1, n_threads + 1), iterations)


# --------------------------------------------------------------------------------------
# Checkpoint and copy transparency
# --------------------------------------------------------------------------------------
def test_state_dict_has_no_wrapper_prefix(make_monitor: MonitorFactory) -> None:
    # Wrapping must not change how a model checkpoints. nn.Module's default would rename
    # every key to "model.<orig>", so a checkpoint saved through the wrapper would no
    # longer load into the bare model — a break that surfaces at deploy time, not here.
    model = _TinyNet()
    guarded = _wrap(model, make_monitor(), embed_layer="features")

    state = guarded.state_dict()
    assert all(not k.startswith("model.") for k in state), list(state)
    assert set(state) == set(model.state_dict())
    # ...and it round-trips into a fresh, unwrapped instance.
    assert _TinyNet().load_state_dict(state)
    guarded.load_state_dict(state)


def test_unwrap_returns_the_original_module(make_monitor: MonitorFactory) -> None:
    model = _TinyNet()
    guarded = _wrap(model, make_monitor(), embed_layer="features")
    assert guarded.unwrap() is model


@pytest.mark.parametrize("operation", [copy.deepcopy, pickle.dumps])
def test_copying_a_wrapper_fails_with_an_actionable_message(
    operation: Any, make_monitor: MonitorFactory
) -> None:
    # copy.deepcopy(model) is routine in EMA and trainer code. Left to nn.Module's default
    # it dies with "cannot pickle '_thread.lock' object", which says nothing about what to
    # do. The monitor owns a drain thread and CUDA streams; copying it is meaningless.
    guarded = _wrap(_TinyNet(), make_monitor(), embed_layer="features")
    with pytest.raises(TypeError, match="unwrap"):
        operation(guarded)
