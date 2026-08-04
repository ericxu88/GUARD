# Contributing to GUARD

## The one rule that outranks everything

**Monitoring must never block, stall, or crash inference.** GUARD is best-effort
observability attached to someone's production serving path. A change that makes the
monitor more accurate at the cost of touching that path is not a trade-off we make.

The full set of non-negotiables is in [`CLAUDE.md`](CLAUDE.md) — read it before your first
change. The two that catch people out:

- **No host synchronisation in the hot path.** Inside `observe()` or any detector
  `compute()`: no `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `torch.cuda.synchronize()`,
  no printing a CUDA tensor, and no Python branch on a *tensor value* (which forces an
  implicit sync). A single stray one silently serialises inference. Treat it as a
  build-breaking bug, not a performance nit.
- **Every GPU detector has a CPU reference.** Write detector math with device-agnostic
  `torch` ops so the identical code runs on CPU and CUDA, and assert parity within
  `DEFAULT_RTOL`/`DEFAULT_ATOL`. This is what makes the project verifiable without a GPU.

## Setup

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
```

## The gates

A change is done when all four pass:

```bash
ruff check . && ruff format --check .
mypy guard
pytest -m "not gpu"
python examples/quickstart.py
```

CI runs exactly these on Python 3.11 and 3.12. `pytest -m gpu` needs CUDA and runs on a
self-hosted runner; if your change touches `guard/engine/`, say in the PR whether you were
able to run it.

## Writing tests that mean something

The bar is: **would this test fail if the implementation were subtly wrong?** Assertions
that cannot fail (`assert x >= 0` on a norm) and "references" that call the production
function they are meant to check are worse than no test, because they buy false confidence.

Before opening a PR, mutate your implementation in a scratch copy and confirm your test
catches it. Several real bugs in this repo survived a green suite for exactly this reason —
the tests asserted invariants the code enforced by construction.

Tests must also be deterministic: seed every RNG, no wall-clock dependence, no network.

## Adding a detector

1. Implement the `Detector` protocol in `guard/detectors/base.py`. You need `name`,
   `metric_names` (a fixed tuple decided at construction — the engine allocates its summary
   buffer from it before the first batch), `compute()`, and `reference()`.
2. Move baseline tensors to the device with `DeviceCache`, not `.to(device)` inside
   `compute()`. A `[D, D]` precision matrix copied per batch is a 2.3 MB host→device
   transfer per inference request.
3. Add the name to `KNOWN_DETECTORS` in `guard/config.py` and to `build_detectors()` in
   `guard/detectors/__init__.py`.
4. Test: NumPy-reference parity, property bounds via Hypothesis, behaviour on degenerate
   input (B=1, constant batch, half precision), and a source scan proving no host-sync
   call appears in `compute()`.

## Changing anything in `guard/engine/`

This is the code that runs next to a live CUDA stream, and it is the code we can least
afford to get wrong. Extra expectations:

- Explain the CUDA ordering argument in a comment. "Why is this safe?" should be answerable
  from the file.
- Keep the CPU fallback path working. `guard/engine/streams.py` degrades every CUDA
  primitive to a no-op, which is what lets the CPU suite exercise the engine's real logic.
  If you add a CUDA call, add its no-op counterpart there rather than branching in
  `monitor.py`.
- Add both a CPU test in `tests/test_engine.py` and a hardware test in
  `tests/test_gpu_engine.py`.
- Nothing on the hot path may allocate per request.

## Performance claims

Do not put a number in the docs that you did not measure on the hardware you are describing.
The `p99 < 2%` figure is an SLO with a written procedure
([`docs/GPU_VERIFICATION.md`](docs/GPU_VERIFICATION.md)), not a result. If you measure it,
record the device, driver, torch build, and batch size alongside it — and publish the whole
saturation curve, not the best point on it.

## Commit and PR

- Explain *why* in the message; the diff already says what.
- Note which gates you ran, and which you could not (e.g. no GPU available).
- One logical change per PR.
