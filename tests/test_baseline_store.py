"""P1-02: baseline save/load round-trip, checksum integrity, validation on load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from guard.baseline import Baseline, BaselineIntegrityError, load, save
from guard.baseline.store import FORMAT_VERSION, _content_checksum


def _realistic_baseline(c: int = 5, d: int = 8, seed: int = 0) -> Baseline:
    g = torch.Generator().manual_seed(seed)

    q = torch.rand(c, generator=g)
    q = q / q.sum()

    hist = torch.rand(16, generator=g)
    edges = torch.linspace(0.0, torch.log(torch.tensor(float(c))).item(), 17)

    mu = torch.randn(d, generator=g)
    # Symmetric PD precision matrix: A Aᵀ + I.
    a = torch.randn(d, d, generator=g)
    precision = a @ a.T + torch.eye(d)

    return Baseline(
        model_version="vit-b16@abc123",
        created_at="2026-05-30T12:00:00Z",
        sample_count=50_000,
        num_classes=c,
        embed_dim=d,
        class_distribution=q,
        entropy_histogram=hist,
        entropy_bin_edges=edges,
        embed_mean=mu,
        embed_precision=precision,
    )


def test_round_trip_reproduces_all_fields(tmp_path: Path) -> None:
    original = _realistic_baseline()
    path = tmp_path / "base.guard"
    checksum = save(original, path)

    loaded = load(path)

    # Scalar metadata exact.
    assert loaded.model_version == original.model_version
    assert loaded.created_at == original.created_at
    assert loaded.sample_count == original.sample_count
    assert loaded.num_classes == original.num_classes
    assert loaded.embed_dim == original.embed_dim
    # Checksum embedded and surfaced.
    assert loaded.checksum == checksum
    assert checksum  # non-empty

    # Tensors reproduced exactly.
    for field in (
        "class_distribution",
        "entropy_histogram",
        "entropy_bin_edges",
        "embed_mean",
        "embed_precision",
    ):
        assert torch.allclose(getattr(loaded, field), getattr(original, field))


def test_checksum_is_deterministic(tmp_path: Path) -> None:
    b = _realistic_baseline()
    c1 = save(b, tmp_path / "a.guard")
    c2 = save(b, tmp_path / "b.guard")
    assert c1 == c2


def test_round_trip_through_model_version_binding(tmp_path: Path) -> None:
    b = _realistic_baseline()
    path = tmp_path / "base.guard"
    save(b, path)
    # Correct version loads; wrong version is rejected.
    load(path, expected_model_version="vit-b16@abc123")
    with pytest.raises(BaselineIntegrityError, match="model_version mismatch"):
        load(path, expected_model_version="other@v2")


def test_tampered_tensor_raises(tmp_path: Path) -> None:
    path = tmp_path / "base.guard"
    save(_realistic_baseline(), path)

    # Tamper with the stored bytes without updating the embedded checksum.
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    payload["tensors"]["embed_mean"] = payload["tensors"]["embed_mean"] + 1.0
    torch.save(payload, path)

    with pytest.raises(BaselineIntegrityError, match="checksum mismatch"):
        load(path)


def test_tampered_metadata_raises(tmp_path: Path) -> None:
    path = tmp_path / "base.guard"
    save(_realistic_baseline(), path)

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    payload["meta"]["sample_count"] = 1  # change content, keep stale checksum
    torch.save(payload, path)

    with pytest.raises(BaselineIntegrityError, match="checksum mismatch"):
        load(path)


def test_unknown_format_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "base.guard"
    save(_realistic_baseline(), path)

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = FORMAT_VERSION + 1
    torch.save(payload, path)

    with pytest.raises(BaselineIntegrityError, match="format_version"):
        load(path)


def test_non_baseline_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.guard"
    torch.save({"hello": torch.zeros(3)}, path)
    with pytest.raises(BaselineIntegrityError, match="not a GUARD baseline"):
        load(path)


def test_save_rejects_invalid_baseline(tmp_path: Path) -> None:
    b = _realistic_baseline()
    b.class_distribution = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])  # sums to 2.5
    with pytest.raises(ValueError, match="sum to 1"):
        save(b, tmp_path / "bad.guard")


def test_load_runs_validate_even_when_checksum_matches(tmp_path: Path) -> None:
    """A file whose (consistent) content violates the schema must be rejected on load.

    We craft a payload with a *correct* checksum but a structurally invalid tensor, proving
    load() enforces Baseline.validate() and not just the checksum.
    """
    b = _realistic_baseline()
    meta = {
        "model_version": b.model_version,
        "created_at": b.created_at,
        "sample_count": b.sample_count,
        "num_classes": b.num_classes,
        "embed_dim": b.embed_dim,
    }
    tensors = {
        "class_distribution": torch.full((b.num_classes,), 0.5),  # sums to 2.5
        "entropy_histogram": b.entropy_histogram,
        "entropy_bin_edges": b.entropy_bin_edges,
        "embed_mean": b.embed_mean,
        "embed_precision": b.embed_precision,
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "meta": meta,
        "tensors": tensors,
        "checksum": _content_checksum(meta, tensors),  # honest checksum
    }
    path = tmp_path / "valid_checksum_bad_shape.guard"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="sum to 1"):
        load(path)
