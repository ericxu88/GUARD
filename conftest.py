"""Pytest configuration: auto-skip GPU-marked tests when no CUDA device is present.

This is what makes the whole Phase-1 suite runnable in CPU-only CI while still letting a
GPU runner execute the Phase-2 overlap/memory tests. Mark hardware-dependent tests with
`@pytest.mark.gpu`.
"""
from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


CUDA = _cuda_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if CUDA:
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
