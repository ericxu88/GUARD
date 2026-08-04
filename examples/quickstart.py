#!/usr/bin/env python3
"""GUARD end to end in one file — build a baseline, serve, drift, alert.

Runs anywhere, CPU or GPU, in a few seconds and with no downloads. It walks the entire
lifecycle a real deployment goes through:

    1. calibrate  — run known-good data through the model, build a versioned baseline
    2. persist    — save it with a content checksum, load it back, verify integrity
    3. attach     — wrap the model; from here every forward pass is monitored
    4. serve      — in-distribution traffic, and nothing fires
    5. drift      — inject a covariate shift, and watch the change-point tests catch it
    6. recover    — traffic returns to normal and the alert clears
    7. scrape     — print what Prometheus would see

Run it::

    python examples/quickstart.py

On a CUDA machine add ``--device cuda`` and the detectors move onto a low-priority monitor
stream that overlaps inference. The numbers below stay the same; only the cost changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from guard import (
    Config,
    GuardModule,
    LoggingAlertSink,
    PrometheusExporter,
    build_baseline,
    build_monitor,
    load,
    save,
)

NUM_CLASSES = 20
EMBED_DIM = 32
BATCH = 16


class TinyClassifier(nn.Module):
    """A stand-in for the real thing: a backbone, a penultimate embedding, a head.

    GUARD is model-agnostic — all it needs is logits ``[B, C]`` and a tappable
    ``[B, D]`` activation. Swap this for ViT-B/16 (see ``examples/vit_demo.py``) and
    nothing else in this file changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, EMBED_DIM), nn.GELU()
        )
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 64]`` inputs to ``[B, NUM_CLASSES]`` logits."""
        return self.head(self.backbone(x))


def in_distribution(n: int, generator: torch.Generator) -> torch.Tensor:
    """Traffic that looks like what the model was calibrated on."""
    return torch.randn(n, 64, generator=generator)


def shifted(n: int, generator: torch.Generator, severity: float) -> torch.Tensor:
    """Covariate shift: the same input space, systematically displaced.

    This is the realistic failure — not garbage input, but plausible input from a
    different regime (a new camera, a new season, a new user cohort). The model keeps
    emitting confident-looking predictions; only its internal representation moves.
    """
    return torch.randn(n, 64, generator=generator) + severity


def banner(step: str, text: str) -> None:
    print(f"\n\033[1m{step}\033[0m  {text}\n" + "─" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--baseline", type=Path, default=Path("baselines/quickstart.guard"))
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(7)
    generator = torch.Generator().manual_seed(7)
    model = TinyClassifier().to(device).eval()

    # ── 1. calibrate ─────────────────────────────────────────────────────────
    banner("1/7", "Calibrating a baseline from known-good traffic")
    calibration = []
    probe: dict[str, torch.Tensor] = {}
    handle = model.backbone.register_forward_hook(
        lambda m, a, out: probe.__setitem__("z", out.detach())
    )
    with torch.no_grad():
        for _ in range(32):
            batch = in_distribution(BATCH, generator).to(device)
            logits = model(batch)
            calibration.append((logits.cpu(), probe["z"].cpu()))
    handle.remove()

    baseline = build_baseline(
        calibration,
        model_version="tiny-classifier@v1",
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
    )
    print(f"  samples          {baseline.sample_count}")
    print(f"  class dist Q     {baseline.class_distribution.shape} (sums to 1)")
    print(
        f"  entropy hist     {baseline.entropy_histogram.shape}, "
        f"{int(baseline.entropy_histogram.sum())} samples binned over [0, log C]"
    )
    print(f"  precision Σ⁻¹    {tuple(baseline.embed_precision.shape)}, regularised inverse")

    # ── 2. persist ───────────────────────────────────────────────────────────
    banner("2/7", "Saving and reloading — a baseline is a tamper-evident artifact")
    checksum = save(baseline, args.baseline)
    baseline = load(args.baseline, expected_model_version="tiny-classifier@v1")
    print(f"  wrote            {args.baseline}")
    print(f"  sha256           {checksum[:32]}…")
    print("  verified         checksum matched and structure validated on load")
    print("  a drift verdict is only as trustworthy as the reference it is measured")
    print("  against, so the store refuses to hand back a baseline whose bytes changed.")

    # ── 3. attach ────────────────────────────────────────────────────────────
    banner("3/7", "Attaching the monitor — no model surgery")
    config = Config.from_yaml(Path(__file__).parent / "guard.yaml")
    # The example runs a few hundred steps, so tighten the windows the shipped config
    # sizes for production traffic, and calibrate the threshold to this model.
    config = config.model_copy(update={"window_size": 4, "drain_every_k": 8})
    config.thresholds["embedding_mahalanobis_mean"].threshold = 4.0
    config.thresholds["embedding_mahalanobis_mean"].delta = 0.5

    exporter = PrometheusExporter()
    monitor = build_monitor(
        baseline,
        config,
        max_batch=BATCH,
        device=device,
        sink=exporter,
        alert_sink=LoggingAlertSink(),
        start_drain_thread=False,  # drive the drain explicitly so the demo is deterministic
    )
    guarded = GuardModule(model, monitor, embed_layer="backbone")
    print(f"  device           {monitor.device}")
    print(f"  detectors        {[d for d in ('entropy', 'divergence', 'embedding')]}")
    print(
        f"  metrics          {len(monitor.metric_names)} scalars/step: "
        f"{', '.join(monitor.metric_names[:3])}, …"
    )
    print(
        f"  ring buffer      {monitor.stats().ring_bytes / 1024:.1f} KiB, fixed for the "
        f"process lifetime"
    )
    if monitor.device.type != "cuda":
        print("  note             CPU device: detectors run inline. On CUDA they run on a")
        print("                   low-priority stream, overlapped with inference.")

    def serve(make_batch, steps: int) -> None:  # type: ignore[no-untyped-def]
        with torch.no_grad():
            for _ in range(steps):
                guarded(make_batch().to(device))
                # A real deployment runs the drain in its background thread; the demo
                # pumps it by hand so every step is deterministic and nothing is dropped
                # to backpressure.
                monitor.drain_pending()
        monitor.flush()

    def report(label: str) -> None:
        snapshot = monitor.aggregator.snapshot()
        latched = [d for d, on in monitor.aggregator.alert_states().items() if on]
        print(
            f"  {label:<12} entropy {snapshot['entropy_mean']:.3f}  "
            f"JS {snapshot['js_divergence']:.4f}  "
            f"mahalanobis {snapshot['embedding_mahalanobis_mean']:6.2f}  "
            f"mean-shift {snapshot['embedding_mean_shift']:6.2f}  "
            f"alerts {latched or '—'}"
        )

    # ── 4. serve ─────────────────────────────────────────────────────────────
    banner("4/7", "Serving in-distribution traffic — the quiet case")
    serve(lambda: in_distribution(BATCH, generator), 60)
    report("clean")
    fired_clean = monitor.aggregator.alerts_fired
    print(f"  alerts fired     {fired_clean}  (a monitor that cries wolf gets turned off)")

    # ── 5. drift ─────────────────────────────────────────────────────────────
    banner("5/7", "Injecting a covariate shift — gradual, then abrupt")
    for step in range(40):
        serve(lambda s=step: shifted(BATCH, generator, 0.06 * s), 1)  # type: ignore[misc]
    report("gradual")
    serve(lambda: shifted(BATCH, generator, 4.0), 40)
    report("abrupt")
    fired_drift = monitor.aggregator.alerts_fired - fired_clean
    print(f"  alerts fired     {fired_drift}")
    print("  the model never errored; it kept returning confident predictions on data")
    print("  it had never seen. That silence is exactly what GUARD exists to break.")

    # ── 6. recover ───────────────────────────────────────────────────────────
    banner("6/7", "Traffic returns to normal — the alert must clear on its own")
    serve(lambda: in_distribution(BATCH, generator), 300)
    report("recovered")
    latched = [d for d, on in monitor.aggregator.alert_states().items() if on]
    print(f"  latched alerts   {latched or '— cleared'}")
    print("  the population statistics forget exponentially (~69-batch half-life), so a")
    print("  recovered stream produces recovered metrics. With an infinite-memory")
    print("  estimator the drift episode would stay baked in and the alert could never")
    print("  clear — a monitor whose alarm never resets is a monitor people mute.")

    # ── 7. scrape ────────────────────────────────────────────────────────────
    banner("7/7", "What Prometheus sees")
    text = exporter.generate_latest().decode()
    for line in text.splitlines():
        if line.startswith("guard_") and not line.endswith("_created"):
            print(f"  {line}")

    stats = monitor.stats()
    print()
    print("─" * 72)
    print(
        f"engine: {stats.steps_observed} steps observed, {stats.rows_dispatched} rows "
        f"exported, {stats.observations_dropped} dropped, "
        f"{stats.mean_enqueue_us:.1f} µs/step on the inference thread"
    )
    print(f"failures: {stats.disabled_detectors or 'none — every detector healthy'}")
    guarded.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
