# GUARD

GPU-native, unsupervised anomaly & runtime drift monitor for ML inference. Detectors
(entropy, KL/JS divergence, embedding drift) run **on the GPU, concurrently with
inference** via a low-priority CUDA stream, for near-zero monitoring overhead.

## Status

Phase 1 — detectors, baselines, temporal change-point tests, config, export scaffolding.
All Phase-1 work is **CPU-testable** (no GPU required). The Phase-2 stream-overlap engine
and memory-safety tests require CUDA hardware (compute capability ≥ 8.0).

Phase-1 progress (see [`docs/PHASE_1_PLAN.md`](docs/PHASE_1_PLAN.md) for the full tracker):

- ✅ **P1-00** CI (ruff + mypy + `pytest -m "not gpu"`, opt-in GPU job)
- ✅ **P1-01** config schema (`guard/config.py`)
- ✅ **P1-02** versioned, checksummed baseline store (`guard/baseline/store.py`)
- ✅ **P1-03** entropy detector (`guard/detectors/entropy.py`)
- ✅ **P1-04** KL/JS divergence detector (`guard/detectors/divergence.py`)
- ✅ **P1-05** embedding-drift detector — Mahalanobis + streaming Welford (`guard/detectors/embedding.py`)
- ✅ **P1-06** temporal change-point tests — Page-Hinkley & CUSUM (`guard/temporal/`)
- ⬜ **P1-07..P1-09** baseline builder, drift-injection harness, Prometheus export

## Setup

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
```

Pin `torch` in `pyproject.toml` to the **exact** PyTorch + CUDA build your serving cluster
runs — CUDA stream / caching-allocator behavior is version-sensitive.

## Test

```bash
pytest -m "not gpu"          # full Phase-1 suite, runs anywhere (CI default)
pytest                       # everything, incl. GPU tests (needs CUDA)
ruff check . && mypy guard   # quality gates
```

## Monitored model

Engine is model-agnostic: any `nn.Module` emitting logits `[B, C]` plus a tappable
penultimate embedding `[B, D]`. Reference/demo target is **ViT-B/16** (`pip install
".[demo]"`). LLM and regression variants are documented extensions, not v1 scope.

## Layout

```
guard/
  detectors/   entropy, KL/JS, embedding-drift (+ base.py contract)
  baseline/    reference artifact schema, builder, versioned store
  temporal/    Page-Hinkley / CUSUM / ADWIN change-point tests
  engine/      Phase-2: CUDA stream overlap, ring buffer, memory safety
  export/      Prometheus / OpenTelemetry / alerts
docs/          design doc + phase plans
tests/         unit (CPU parity + property) and gpu-marked hardware tests
```
