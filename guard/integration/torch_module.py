"""``GuardModule``: wrap any ``nn.Module`` and monitor it (design doc §12, "Embedded wrapper").

The integration contract is deliberately minimal — GUARD needs two tensors per forward
pass, logits ``[B, C]`` and a penultimate embedding ``[B, D]`` — and the wrapper gets them
without changing the model:

* **logits** come from the model's return value. Most classifiers return the tensor
  directly; HuggingFace-style models return an object with a ``.logits`` attribute, and
  some return a tuple. All three are handled, and ``logits_from`` covers the rest.
* **embeddings** come from a ``register_forward_hook`` on a named submodule. No weights are
  touched, no ``forward`` is rewritten, and removing the wrapper removes the hook.

Everything downstream is fire-and-forget: ``observe()`` cannot raise, so the wrapper's
``forward`` returns exactly what the wrapped model returned, on exactly the same path, even
if every detector is broken.

Picking ``embed_layer``
-----------------------
It is the module whose *output* is the feature vector the classifier head consumes. Use
:func:`candidate_embed_layers` to list the plausible ones::

    >>> from torchvision.models import vit_b_16
    >>> candidate_embed_layers(vit_b_16())[-3:]
    ['encoder.ln', 'encoder', 'heads']

For torchvision's ViT-B/16 the right answer is ``"encoder.ln"`` (output ``[B, 197, 768]``,
reduced to ``[B, 768]`` by taking the class token). For a ResNet it is ``"avgpool"``
(``[B, 2048, 1, 1]``). For a HuggingFace ``AutoModelForSequenceClassification`` it is
usually ``"<base>.pooler"``.

Thread safety
-------------
The hook stashes the captured tensor in thread-local storage, so a server calling
``forward`` from N request threads never crosses embeddings between requests.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn

from guard.engine.monitor import GuardMonitor

logger = logging.getLogger("guard.integration")

EmbedReduce = Literal["auto", "none", "cls", "mean", "flatten"]


def candidate_embed_layers(model: nn.Module, *, max_results: int = 40) -> list[str]:
    """Names of submodules whose output could plausibly be the penultimate embedding.

    A discovery aid, not an oracle: it lists leaf-ish modules in forward order so you can
    see the shape of the network and choose. Pair it with a one-batch dry run and check the
    captured shape.
    """
    names = [name for name, _ in model.named_modules() if name]
    return names[-max_results:] if len(names) > max_results else names


def reduce_embedding(tensor: torch.Tensor, mode: EmbedReduce = "auto") -> torch.Tensor:
    """Reduce a captured activation to a 2-D ``[B, D]`` embedding.

    Args:
        tensor: whatever the hooked module emitted.
        mode: ``"none"`` requires it to already be ``[B, D]``; ``"cls"`` takes token 0 of a
            ``[B, T, D]`` sequence; ``"mean"`` averages over the token axis; ``"flatten"``
            flattens every trailing dimension; ``"auto"`` picks per rank — token-mean for
            3-D, spatial-mean for 4-D, flatten otherwise.

    ``"auto"`` averages rather than flattens for a reason: flattening a ``[B, 197, 768]``
    ViT activation produces a 151k-dimensional "embedding", whose reference covariance is a
    23 GB matrix. Averaging keeps D at 768, where the Mahalanobis math is cheap and the
    covariance is estimable from a realistic calibration set.
    """
    if tensor.ndim == 2:
        return tensor
    if mode == "none":
        raise ValueError(
            f"embed_reduce='none' requires a [B, D] activation, got {tuple(tensor.shape)}. "
            f"Use 'cls', 'mean', or 'flatten'."
        )
    if mode == "cls":
        if tensor.ndim != 3:
            raise ValueError(f"embed_reduce='cls' needs [B, T, D], got {tuple(tensor.shape)}")
        return tensor[:, 0]
    if mode == "mean":
        return tensor.flatten(2).mean(dim=2) if tensor.ndim > 3 else tensor.mean(dim=1)
    if mode == "flatten":
        return tensor.flatten(1)
    # auto
    if tensor.ndim == 3:  # [B, T, D] sequence  -> average over tokens
        return tensor.mean(dim=1)
    if tensor.ndim == 4:  # [B, C, H, W] feature map -> global average pool
        return tensor.flatten(2).mean(dim=2)
    return tensor.flatten(1)


def _default_logits(output: Any) -> torch.Tensor | None:
    """Best-effort extraction of the logits tensor from a model's return value."""
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "out", "output"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    return None


class GuardModule(nn.Module):
    """Wraps a model so every forward pass is observed by a :class:`GuardMonitor`.

    Args:
        model: the model to monitor. Not modified; only a forward hook is added.
        monitor: the engine that receives ``(logits, embeddings)``.
        embed_layer: dotted name of the submodule whose output is the embedding. Omit when
            the monitor was built with ``embed_dim=0`` (output-distribution monitoring only).
        embed_reduce: how to reduce a non-2-D activation. See :func:`reduce_embedding`.
        logits_from: callable mapping the model's return value to the logits tensor.
            Defaults to a heuristic covering plain tensors, ``.logits`` attributes, tuples,
            and dicts.
        observe_in_eval_only: skip monitoring while the model is in training mode. On by
            default — training-time activations are not what the baseline describes, and
            monitoring them produces drift alerts on your own fine-tuning job.

    Example::

        monitor = build_monitor(baseline, config, max_batch=64)
        model = GuardModule(vit_b_16().cuda().eval(), monitor, embed_layer="encoder.ln",
                            embed_reduce="cls")
        logits = model(images)   # monitored, identical output, no extra latency
    """

    def __init__(
        self,
        model: nn.Module,
        monitor: GuardMonitor,
        *,
        embed_layer: str | None = None,
        embed_reduce: EmbedReduce = "auto",
        logits_from: Callable[[Any], torch.Tensor | None] | None = None,
        observe_in_eval_only: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.monitor = monitor
        self.embed_layer = embed_layer
        self.embed_reduce: EmbedReduce = embed_reduce
        self.observe_in_eval_only = observe_in_eval_only
        self._logits_from = logits_from if logits_from is not None else _default_logits
        self._local = threading.local()
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        # A freshly constructed nn.Module defaults to training=True. Without this, wrapping
        # an already-.eval()'d model would leave the WRAPPER in training mode and
        # observe_in_eval_only would silently suppress every observation — GUARD installed,
        # dashboards empty, no error anywhere.
        self.train(model.training)

        if embed_layer is not None:
            try:
                target = model.get_submodule(embed_layer)
            except AttributeError as exc:
                raise ValueError(
                    f"no submodule named {embed_layer!r}. Candidates: "
                    f"{candidate_embed_layers(model)}"
                ) from exc
            self._handle = target.register_forward_hook(self._capture)
        elif monitor.embed_dim > 0:
            raise ValueError(
                f"monitor expects embeddings (embed_dim={monitor.embed_dim}) but no "
                f"embed_layer was given. Pass embed_layer=..., or build the monitor with "
                f"embed_dim=0 to monitor the output distribution only."
            )

    # ── capture ──────────────────────────────────────────────────────────────

    def _capture(self, module: nn.Module, args: Any, output: Any) -> None:
        """Forward hook: stash the penultimate activation for this thread."""
        del module, args
        tensor = output[0] if isinstance(output, (tuple, list)) and output else output
        if isinstance(tensor, torch.Tensor):
            # detach: GUARD must never extend the autograd graph's lifetime.
            self._local.embedding = tensor.detach()

    def _take_embedding(self) -> torch.Tensor | None:
        """Pop this thread's captured activation, reduced to ``[B, D]``."""
        tensor = getattr(self._local, "embedding", None)
        self._local.embedding = None
        if tensor is None:
            return None
        return reduce_embedding(tensor, self.embed_reduce)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped model, then observe its outputs. Returns the model's output as-is."""
        output = self.model(*args, **kwargs)
        try:
            self._observe(output)
        except Exception:
            logger.exception("GUARD integration failed; inference output is unaffected")
        return output

    def _observe(self, output: Any) -> None:
        """Extract the two tensors and hand them to the engine. Contained by the caller."""
        if not self.monitor.enabled:
            self._local.embedding = None
            return
        # Ask the *wrapped* model, not the wrapper: that is what the user put in eval mode,
        # and it stays correct even if someone flips the wrapper's flag independently.
        if self.observe_in_eval_only and self.model.training:
            self._local.embedding = None
            return

        logits = self._logits_from(output)
        embeddings = self._take_embedding()
        if logits is None:
            logger.debug("GUARD could not find logits in the model output; skipping step")
            return
        self.monitor.observe(logits, embeddings)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def detach_hook(self) -> None:
        """Remove the forward hook, leaving the wrapped model exactly as it was."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def unwrap(self) -> nn.Module:
        """The wrapped model, unmodified. Use this to checkpoint, copy, or export it."""
        return self.model

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """The **wrapped model's** state dict, with no ``model.`` prefix added.

        Delegating rather than inheriting matters for checkpoint compatibility. GUARD adds
        no parameters and no buffers, so a wrapped model's checkpoint should be
        byte-identical to an unwrapped one's. Without this, wrapping a model and then saving
        it silently renames every key to ``model.<orig>``, and the resulting checkpoint no
        longer loads into the bare model — a break that surfaces weeks later at deploy time.
        """
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        """Load into the wrapped model, accepting an unprefixed checkpoint."""
        return self.model.load_state_dict(state_dict, *args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        """Refuse to serialise, with an actionable message.

        A ``GuardModule`` owns live process state — a drain thread, a lock, CUDA streams and
        events — none of which is meaningful to copy. Left alone, ``copy.deepcopy(model)``
        (common in EMA and trainer code) fails with ``cannot pickle '_thread.lock' object``,
        which tells the caller nothing about what to do next.
        """
        raise TypeError(
            "GuardModule cannot be pickled or deep-copied: it owns a live monitor (drain "
            "thread, CUDA stream, pinned buffers). Serialise or copy the wrapped model "
            "instead — use guarded.unwrap(), and re-wrap the copy with its own monitor."
        )

    def close(self) -> None:
        """Detach the hook and shut the monitor down."""
        self.detach_hook()
        self.monitor.close()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped model, so the wrapper is transparent."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(super().__getattr__("model"), name)
