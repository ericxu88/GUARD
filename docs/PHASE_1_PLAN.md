# Phase 1 — Build Plan (GPU-independent slice)

**Goal of Phase 1:** every detector, the temporal change-point logic, baseline
management, config, and the export scaffolding — all of it correct and fully tested on
CPU. No CUDA hardware required. This de-risks the project's *math and contracts* before
Phase 2 tackles the harder, hardware-bound overlap engine.

**Why this order:** detectors are written with device-agnostic `torch` ops, so their
correctness is provable against CPU reference implementations. Phase 2 then wraps proven
detectors in the CUDA stream machinery, where the only new risk is *concurrency and memory
safety*, not *math*.

**Global definition of done** (applies to every ticket): code + type hints + docstrings;
`ruff check` and `ruff format --check` clean; `mypy guard` clean; `pytest -m "not gpu"`
green; acceptance criteria below met. See `CLAUDE.md` for the prime directives that all
code must respect.

## Progress

| Ticket | Status |
|---|---|
| P1-00 Scaffolding & CI | ✅ done |
| P1-01 Config schema | ✅ done |
| P1-02 Baseline store | ✅ done |
| P1-03 Entropy detector | ✅ done |
| P1-04 KL/JS divergence detector | ✅ done |
| P1-05 Embedding-drift detector | ✅ done |
| P1-06 Temporal change-point tests | ✅ done |
| P1-07 Baseline builder | ✅ done |
| P1-08 Drift-injection integration harness | ✅ done |
| P1-09 Export scaffolding | ✅ done |

---

## P1-00 — Scaffolding & CI ✅
**Files:** `.github/workflows/ci.yml` (or equivalent), pre-commit config (optional).
**Depends on:** nothing (skeleton already exists).
**Do:** wire CI to run `ruff`, `mypy guard`, and `pytest -m "not gpu"` on push. Add an
optional GPU job (manual/self-hosted runner) that runs the full suite.
**Acceptance:**
- CI green on the existing skeleton (`tests/test_contract_smoke.py` passes, gpu test skips).
- A failing `ruff`/`mypy`/test blocks the pipeline.

## P1-01 — Config schema ✅
**Files:** `guard/config.py`.
**Depends on:** P1-00.
**Do:** pydantic v2 models for the runtime config: enabled detectors + per-detector params,
window sizes, change-point thresholds (per metric), export endpoints, `drain_every_k`,
kill-switch flag. Load/validate from YAML.
**Required fields (lock these names):** `detectors: list[DetectorConfig]`,
`window_size: int`, `drain_every_k: int`, `thresholds: dict[str, ThresholdConfig]`,
`export: ExportConfig`, `enabled: bool`.
**Acceptance:**
- Valid YAML round-trips to model and back.
- Invalid input (negative window, unknown detector name) raises a clear `ValidationError`.
- 100% typed; mypy strict clean.

## P1-02 — Baseline store ✅
**Files:** `guard/baseline/store.py` (schema already in `schema.py`).
**Depends on:** P1-01.
**Do:** `save(baseline, path)` and `load(path) -> Baseline`. Serialize tensors + metadata;
compute and embed a content `checksum`; bind to `model_version`. `load` calls
`Baseline.validate()` and verifies the checksum.
**Acceptance:**
- save → load round-trip reproduces all fields exactly (tensors `allclose`).
- Tampered file (checksum mismatch) raises.
- Shape/normalization violations raise via `validate()`.

## P1-03 — Entropy detector ✅
**Files:** `guard/detectors/entropy.py`, `tests/test_entropy.py`.
**Depends on:** P1-02 (uses baseline entropy histogram for population comparison).
**Do:** per-sample Shannon entropy of softmax outputs; window mean + a chosen quantile;
optional population shift vs baseline entropy histogram. Device-agnostic torch ops.
Provide a NumPy CPU reference in the test.
**Acceptance:**
- Matches NumPy reference within `DEFAULT_RTOL`/`DEFAULT_ATOL`.
- Property tests (hypothesis): `0 ≤ H ≤ log C`; one-hot input → `H ≈ 0`; uniform → `H ≈ log C`.
- No host sync inside `compute` (assert via a no-`.item()` lint/review check).

## P1-04 — KL / JS divergence detector ✅
**Files:** `guard/detectors/divergence.py`, `tests/test_divergence.py`.
**Depends on:** P1-02 (baseline `Q`).
**Do:** windowed predicted-class distribution `P`; compute `KL(P‖Q)` and bounded `JS(P‖Q)`.
Numerically safe (eps in logs, clamp).
**Acceptance:**
- `KL ≥ 0`; `KL(P‖P) ≈ 0`; `0 ≤ JS ≤ log 2`; matches reference within tolerance.
- Stable when `P` has zero-probability classes (no NaN/inf).

## P1-05 — Embedding-drift detector ✅
**Files:** `guard/detectors/embedding.py`, `tests/test_embedding.py`.
**Depends on:** P1-02 (`μ_ref`, `Σ_ref⁻¹`).
**Do:** per-sample Mahalanobis distance to the reference Gaussian; streaming window
mean/covariance via Welford for a population drift score. (MMD with random Fourier
features is a stretch goal — separate ticket if pursued.)
**Acceptance:**
- Mahalanobis matches reference within tolerance; handles the identity-precision case.
- Welford streaming covariance matches a batch `torch.cov` within tolerance.
- No NaNs on degenerate (constant) batches.

## P1-06 — Temporal change-point tests ✅
**Files:** `guard/temporal/page_hinkley.py`, `guard/temporal/cusum.py`, `tests/test_temporal.py`.
**Depends on:** P1-01 (threshold config). Pure CPU / scalar streams — no torch needed.
**Do:** online Page-Hinkley and CUSUM over a scalar metric stream, with debounce + cooldown.
**Acceptance:**
- On a stationary seeded noise stream: no alert within N steps (false-positive guard).
- On an injected mean shift: alert fires within the documented lag and clears after recovery.
- O(1) state per stream; deterministic given seed.

## P1-07 — Baseline builder ✅
**Files:** `guard/baseline/compute.py`, `tests/test_baseline_builder.py`.
**Depends on:** P1-02..P1-05.
**Do:** given a dataloader of (logits, embeddings) over known-good data, compute `Q`,
entropy histogram, `μ_ref`, `Σ_ref⁻¹` (regularized inverse), and emit a valid `Baseline`.
**Acceptance:**
- Output passes `Baseline.validate()`.
- Deterministic given a fixed seed/input.
- Covariance inversion is regularized (no singular-matrix crash on low-rank input).

## P1-08 — Drift-injection integration harness ✅
**Files:** `tests/test_drift_injection.py`.
**Depends on:** P1-03..P1-07.
**Do:** synthetic stream generator: clean → gradual shift → abrupt shift → recovery. Run
each detector + change-point test end-to-end against it.
**Acceptance:**
- Each detector's change-point test fires within an expected window of the injected shift.
- Metrics return to baseline after recovery (alert clears).
- False-positive rate on the clean segment below the configured bound.

## P1-09 — Export scaffolding (no live wiring yet) ✅
**Files:** `guard/export/prometheus.py`, `tests/test_export.py`.
**Depends on:** P1-01.
**Do:** define the Prometheus metric objects (gauges/histograms named in the design doc)
and an `update(summary: dict[str, float])` entry point. A background drain thread is
Phase 2; here, just the registry + update path + self-telemetry metric stubs.
**Acceptance:**
- Metrics register without collision; `update` sets values; names match the design doc.
- Unit test scrapes the registry and finds expected metric families.

---

## Explicitly NOT in Phase 1
The CUDA stream overlap engine, ring buffer + `record_stream` memory management, the async
host-drain thread, exception firewall wiring into a live model, Triton/TorchServe adapters,
multi-GPU, and Grafana dashboards. Those are Phase 2+ and require GPU hardware to verify.

## Suggested first session for Claude Code
Start with **P1-00 → P1-01 → P1-03**. That produces CI, the config contract, and the first
fully-tested detector — a complete vertical slice you can review before continuing. Tell
the agent: "Implement P1-00 through P1-03 from docs/PHASE_1_PLAN.md, honoring CLAUDE.md.
Stop after P1-03 for review."

---

## Phase 1 is complete

All ten tickets are implemented, tested, and committed. `pytest -m "not gpu"` is green,
`ruff` and `mypy guard` are clean. The math and the contracts this phase existed to de-risk
are settled.

Work listed above as Phase 2+ has since landed in the repository and is no longer a plan:

- **Overlap engine** — `guard/engine/` (`monitor.py`, `ring_buffer.py`, `streams.py`,
  `memory.py`, `aggregator.py`), including the async host-drain thread.
- **Safety** — `guard/safety/firewall.py`: exception firewall, circuit breaker, kill-switch.
- **Integration** — `guard/integration/torch_module.py` (`GuardModule`). Triton and
  TorchServe adapters are still not implemented.
- **Observability** — `guard/export/` (Prometheus, OTel, alerts, fan-out) plus
  `deploy/grafana/guard-dashboard.json`.
- **Benchmarks** — `benchmarks/` (see `benchmarks/README.md`).

That code is **implemented but not verified on GPU hardware** — no CUDA device was available
during development, so kernel overlap, `record_stream` memory safety, and the p99 overhead
SLO are unmeasured. Do not call Phase 2 done until `docs/GPU_VERIFICATION.md` is filled in
with real numbers. Design-doc §18 and §20 track the same distinction.
