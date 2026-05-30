"""Versioned baseline persistence (P1-02).

A baseline is a trust anchor: every drift verdict is relative to it. So the store treats a
baseline file as a tamper-evident, version-bound artifact rather than a plain pickle.

* ``save`` serializes tensors + metadata, computes a content checksum, embeds it, and binds
  the file to the baseline's ``model_version``.
* ``load`` recomputes the checksum over the file's content and refuses to return a baseline
  whose bytes have changed, then runs :meth:`Baseline.validate` for structural sanity.

The checksum covers *content*, not the stored checksum string itself, so a baseline saved
with an empty ``checksum`` field and one reloaded from disk hash identically.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from guard.baseline.schema import Baseline

# Bump if the on-disk layout changes incompatibly; load rejects unknown versions.
FORMAT_VERSION = 1

# Order is fixed so the checksum is deterministic across runs and machines.
_META_FIELDS: tuple[str, ...] = (
    "model_version",
    "created_at",
    "sample_count",
    "num_classes",
    "embed_dim",
)
_TENSOR_FIELDS: tuple[str, ...] = (
    "class_distribution",
    "entropy_histogram",
    "entropy_bin_edges",
    "embed_mean",
    "embed_precision",
)


class BaselineIntegrityError(Exception):
    """Raised when a baseline file is corrupt, tampered, or version-mismatched."""


def _content_checksum(meta: dict[str, Any], tensors: dict[str, torch.Tensor]) -> str:
    """Deterministic SHA-256 over metadata + tensor bytes (dtype + shape + data).

    Hashing is order-stable and includes each field's name, dtype, and shape so that a
    change to any of them — not just the raw bytes — alters the digest.
    """
    h = hashlib.sha256()
    h.update(f"guard-baseline-v{FORMAT_VERSION}".encode())
    for name in _META_FIELDS:
        h.update(name.encode())
        h.update(repr(meta[name]).encode())
    for name in _TENSOR_FIELDS:
        t = tensors[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save(baseline: Baseline, path: str | Path) -> str:
    """Persist ``baseline`` to ``path`` with an embedded content checksum.

    Validates structure first (a malformed baseline is never written), computes the
    checksum, and returns it. The ``checksum`` field on the passed-in object is ignored;
    the file always carries a freshly computed digest.
    """
    baseline.validate()

    meta = {name: getattr(baseline, name) for name in _META_FIELDS}
    tensors = {name: getattr(baseline, name) for name in _TENSOR_FIELDS}
    checksum = _content_checksum(meta, tensors)

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "meta": meta,
        "tensors": {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
        "checksum": checksum,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return checksum


def load(path: str | Path, *, expected_model_version: str | None = None) -> Baseline:
    """Load and verify a baseline from ``path``.

    Verifies the format version, recomputes and checks the content checksum, optionally
    asserts the ``model_version``, reconstructs the :class:`Baseline`, and runs
    :meth:`Baseline.validate`. Any failure raises :class:`BaselineIntegrityError` (checksum
    / version / structure) so a bad baseline can never silently feed the detectors.
    """
    path = Path(path)
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(payload, dict) or "format_version" not in payload:
        raise BaselineIntegrityError(f"{path} is not a GUARD baseline file")
    if payload["format_version"] != FORMAT_VERSION:
        raise BaselineIntegrityError(
            f"unsupported baseline format_version {payload['format_version']!r} "
            f"(this build expects {FORMAT_VERSION})"
        )

    meta: dict[str, Any] = payload["meta"]
    tensors: dict[str, torch.Tensor] = payload["tensors"]
    stored = payload["checksum"]

    recomputed = _content_checksum(meta, tensors)
    if recomputed != stored:
        raise BaselineIntegrityError(
            f"checksum mismatch for {path}: file content does not match its embedded "
            f"checksum (expected {stored}, got {recomputed}); the baseline may be corrupt "
            f"or tampered"
        )

    if expected_model_version is not None and meta["model_version"] != expected_model_version:
        raise BaselineIntegrityError(
            f"model_version mismatch: baseline is {meta['model_version']!r}, "
            f"expected {expected_model_version!r}"
        )

    baseline = Baseline(
        model_version=meta["model_version"],
        created_at=meta["created_at"],
        sample_count=meta["sample_count"],
        num_classes=meta["num_classes"],
        embed_dim=meta["embed_dim"],
        class_distribution=tensors["class_distribution"],
        entropy_histogram=tensors["entropy_histogram"],
        entropy_bin_edges=tensors["entropy_bin_edges"],
        embed_mean=tensors["embed_mean"],
        embed_precision=tensors["embed_precision"],
        checksum=stored,
    )
    baseline.validate()
    return baseline
