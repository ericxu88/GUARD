# GUARD — Project Memory

GPU-native, unsupervised drift & anomaly monitor for ML inference. Detectors run on the
GPU concurrently with inference via a low-priority CUDA stream, so monitoring adds
near-zero latency. Full design: `docs/GUARD_design.md`. Current work: `docs/PHASE_1_PLAN.md`.

## Prime directives (NEVER violate — these define the product)

1. **Monitoring must never block, stall, or crash inference.** `observe()` is
   fire-and-forget. Every call site is wrapped in an exception firewall; a failing
   detector is disabled, never propagated to the serving path.
2. **No host synchronization in the hot path.** Inside `observe()` and any detector
   `compute()`: no `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `torch.cuda.synchronize()`,
   or printing a CUDA tensor. Scalars leave the GPU only via the batched async drain
   every `K` steps (Phase 2). A single stray `.item()` silently serializes inference —
   treat it as a build-breaking bug.
3. **Memory safety.** Any tensor captured from inference and read on the monitor stream
   MUST be protected with `tensor.record_stream(monitor_stream)` and/or copied into the
   owned ring buffer before the inference stream can reuse it. (Phase 2.)
4. **Every GPU detector has a CPU reference.** The detector's math is written with
   device-agnostic `torch` ops (runs on CPU or CUDA). Correctness tests assert the CUDA
   result matches the CPU reference within `DEFAULT_RTOL`/`DEFAULT_ATOL`
   (see `guard/baseline/schema.py`). This makes all of Phase 1 verifiable without a GPU.
5. **Bounded resources.** Fixed-size ring buffer, fixed baselines. No per-request
   allocation growth anywhere in the monitor.
6. **Deterministic tests.** Unit tests seed RNG and use fixed inputs. No flaky tolerances.

## Monitored-model assumption (changeable)

The engine is **model-agnostic**: it works with any `nn.Module` that emits logits
`[B, C]` and exposes a tappable penultimate embedding `[B, D]`. The reference/demo target
is **ViT-B/16** (torchvision) as an image classifier. Entropy/KL assume a categorical
output distribution; LLM (token-level) and regression variants are documented extensions,
not v1 scope.

## Commands

```bash
# Setup (pick one)
uv sync --extra dev            # preferred
pip install -e ".[dev]"        # fallback

# Tests
pytest -m "not gpu"            # full Phase-1 suite; runs anywhere, no GPU needed (CI default)
pytest                         # everything incl. GPU overlap/memory tests (needs CUDA)
pytest -m gpu                  # GPU-only tests

# Quality gates (must pass before any PR is "done")
ruff check . && ruff format --check .
mypy guard
```

## GPU availability rule

Phase 1 (detectors, temporal tests, baselines, config, export scaffolding) is **fully
CPU-testable** — build and verify it anywhere. Phase 2 (the stream-overlap engine and
memory-safety tests) can only be *verified* on real CUDA hardware (compute capability
≥ 8.0). Mark any test that requires CUDA with `@pytest.mark.gpu`; never let GPU-only
behavior leak into a non-`gpu` test. If no GPU is available, do not claim Phase-2 work is
done — it must be benchmarked on hardware (Nsight overlap proof + p99 overhead < 2%).

## Conventions

- Python ≥ 3.11, `from __future__ import annotations`, full type hints, `mypy` clean.
- Detector math: device-agnostic `torch` ops only; no NumPy in the hot path (NumPy is fine
  for the CPU *reference* and for scalar temporal tests).
- Numerical safety: add `eps` inside logs; clamp probabilities; never `log(0)`.
- Public API surfaces stay in `guard/<area>/__init__.py`; internals are private modules.
- Keep functions allocation-light; pre-allocate buffers, avoid Python loops over the batch.

## Definition of done (per ticket)

A ticket is done only when: code + types + docstrings complete; CPU-reference parity test
passes (for detectors); property tests pass; `ruff` + `mypy` clean; `pytest -m "not gpu"`
green; and the ticket's stated acceptance criteria in `docs/PHASE_1_PLAN.md` are all met.

## Out of scope for now

Phase-2 overlap engine, multi-GPU, Triton/TorchServe adapters, Grafana dashboards. Don't
build ahead of the plan; land Phase 1 first.
