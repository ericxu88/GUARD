"""Template test showing the two patterns Phase-1 work should follow.

1. Contract/CPU tests run everywhere (no marker) — this is where detector parity and
   property tests live.
2. Hardware tests carry @pytest.mark.gpu and are auto-skipped without CUDA (see conftest).

Delete or extend this file as real tickets land.
"""
from __future__ import annotations

import pytest
import torch

from guard.baseline.schema import Baseline


def _tiny_baseline(c: int = 3, d: int = 4) -> Baseline:
    return Baseline(
        model_version="test-v0",
        created_at="2026-01-01T00:00:00Z",
        sample_count=100,
        num_classes=c,
        embed_dim=d,
        class_distribution=torch.full((c,), 1.0 / c),
        entropy_histogram=torch.zeros(8),
        entropy_bin_edges=torch.linspace(0.0, 1.0, 9),
        embed_mean=torch.zeros(d),
        embed_precision=torch.eye(d),
    )


def test_baseline_validates() -> None:
    _tiny_baseline().validate()  # should not raise


def test_baseline_rejects_unnormalized_distribution() -> None:
    b = _tiny_baseline()
    b.class_distribution = torch.tensor([0.5, 0.5, 0.5])  # sums to 1.5
    with pytest.raises(ValueError):
        b.validate()


@pytest.mark.gpu
def test_cuda_is_visible() -> None:
    # Placeholder for Phase-2 hardware tests. Auto-skipped on CPU-only machines.
    assert torch.cuda.is_available()
