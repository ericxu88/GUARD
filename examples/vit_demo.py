#!/usr/bin/env python3
"""Monitor a real ViT-B/16 image classifier end to end: baseline, serve, corrupt, alert.

``examples/quickstart.py`` uses a toy MLP so it can run in two seconds. This one uses the
reference target from ``CLAUDE.md`` — torchvision's ViT-B/16, 1000 classes, a 768-wide
penultimate embedding — to show that the integration is three lines against a real network
and that the numbers behave sensibly at production width.

What it does::

    1. calibrate — push synthetic "clean" images through the model, build a baseline
    2. threshold — measure the clean Mahalanobis level and set the CUSUM boundary from it
    3. attach    — GuardModule(model, monitor, embed_layer="encoder.ln", embed_reduce="cls")
    4. serve     — clean traffic; nothing should fire
    5. corrupt   — additive Gaussian noise + a brightness shift; watch the drift metrics move

Run it::

    python examples/vit_demo.py                      # random weights, no downloads, ~15 s
    python examples/vit_demo.py --pretrained         # downloads ImageNet-1k weights (~330 MB)
    python examples/vit_demo.py --device cuda --batch 32 --steps 40

Weights are **random by default** so the example runs offline and in CI. A randomly
initialised classifier head emits near-zero logits, so its softmax is essentially uniform
and the output-distribution detectors (entropy, KL, JS) sit pinned at their no-information
values no matter what you feed the model. The embedding detector is the one carrying signal
on that path. ``--pretrained`` gives the head real weights and all three move.

Requires torchvision::

    pip install "guard-monitor[demo]"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from guard import (
    Config,
    GuardModule,
    LoggingAlertSink,
    PrometheusExporter,
    build_baseline,
    build_monitor,
)
from guard.detectors import mahalanobis
from guard.integration.torch_module import EmbedReduce

# ViT-B/16 facts, all verified against torchvision by `python examples/vit_demo.py`:
# the model takes 224x224 RGB, emits [B, 1000] logits, and `encoder.ln` — the final
# LayerNorm, the last thing before the classifier head — emits [B, 197, 768].
IMAGE_SIZE = 224
NUM_CLASSES = 1000
EMBED_DIM = 768
EMBED_LAYER = "encoder.ln"

# torchvision's ViT forward is `x = self.encoder(x); x = x[:, 0]; x = self.heads(x)`, so the
# vector the classifier actually consumes is token 0 of that [B, 197, 768] activation — the
# class token. embed_reduce="cls" takes exactly that. "mean" would monitor a different
# quantity than the head reads; "flatten" would ask for a 151296-wide embedding whose
# reference covariance is a 91 GB matrix.
EMBED_REDUCE: EmbedReduce = "cls"

# Corruption applied in normalised-input space (the model's own units, so a shift of 0.4 is
# 0.4 standard deviations of the ImageNet channel statistics). This is covariate shift, not
# garbage: a miscalibrated sensor and a noisier one. The model keeps returning predictions.
NOISE_STD = 0.5
BRIGHTNESS = 0.4

# Fewest batches per phase that still closes enough aggregation windows for the drift alert
# to clear its debounce. See the check in main().
MIN_STEPS = 6


def load_vit(pretrained: bool) -> tuple[nn.Module, str]:
    """Build torchvision's ViT-B/16. Returns the model and a version string for the baseline.

    torchvision is imported here, not at module scope, so the missing-dependency message is
    the first thing a reader sees rather than a traceback from an import they did not write.
    """
    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise SystemExit(
            "this example needs torchvision. Install it with:\n\n"
            '    pip install "guard-monitor[demo]"\n'
        ) from exc

    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        return vit_b_16(weights=weights), f"vit_b_16@{weights.name.lower()}"
    return vit_b_16(weights=None), "vit_b_16@random-init"


def synthetic_images(
    n: int,
    generator: torch.Generator,
    *,
    noise_std: float = 0.0,
    brightness: float = 0.0,
) -> torch.Tensor:
    """A batch of ``[n, 3, 224, 224]`` stand-in images, already in normalised-input space.

    Real images would be better and would need a dataset download, which would stop this
    example running offline. What matters for the demo is that "clean" is a stable
    distribution with spatial structure — a smooth low-frequency field, not white noise,
    so the patch embedding sees something a ViT can respond to — and that the corrupted
    version is the *same* distribution displaced, which is what covariate shift looks like.
    """
    coarse = torch.randn(n, 3, 16, 16, generator=generator)
    image = F.interpolate(coarse, size=IMAGE_SIZE, mode="bicubic", align_corners=False) * 0.6
    if noise_std > 0.0:
        image = image + torch.randn_like(image) * noise_std
    return image + brightness


def banner(step: str, text: str) -> None:
    print(f"\n\033[1m{step}\033[0m  {text}\n" + "─" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument("--batch", type=int, default=8, help="batch size (default: 8)")
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help=f"batches per phase; there are 3 (default: 12, minimum {MIN_STEPS})",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="download ImageNet-1k weights (~330 MB) instead of using random ones",
    )
    args = parser.parse_args()
    if args.steps < MIN_STEPS:
        # window_size=2 and the shipped debounce of 2 mean a firing needs two consecutive
        # flagged windows, i.e. four drifted steps. Below that the demo would end with no
        # alert and look broken when it is only starved of windows.
        parser.error(f"--steps must be >= {MIN_STEPS}: an alert needs enough closed windows")
    if args.batch < 1:
        parser.error("--batch must be >= 1")

    device = torch.device(args.device)
    torch.manual_seed(11)
    # A CPU generator keeps the input stream identical on every device, so --device cuda
    # reproduces the same numbers as --device cpu and any difference is GUARD's, not noise.
    generator = torch.Generator().manual_seed(11)
    started = time.perf_counter()

    model, model_version = load_vit(args.pretrained)
    model = model.to(device).eval()
    print(f"model            {model_version} on {device}")
    if not args.pretrained:
        print("                 RANDOM WEIGHTS — no download, nothing is classified correctly.")
        print("                 The head is untrained, so its softmax is ~uniform and entropy/")
        print("                 KL/JS carry no signal. Pass --pretrained to make them move.")
    print(f"batches          {args.steps} per phase x {args.batch} images")

    # ── 1. calibrate ─────────────────────────────────────────────────────────
    banner("1/5", "Calibrating a baseline from clean traffic")
    # A plain forward hook, only for calibration: the baseline is built offline from
    # (logits, embedding) pairs, and GuardModule is not involved yet.
    captured: dict[str, torch.Tensor] = {}
    handle = model.get_submodule(EMBED_LAYER).register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("z", output.detach()[:, 0])
    )
    calibration: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for _ in range(args.steps):
            logits = model(synthetic_images(args.batch, generator).to(device))
            calibration.append((logits.cpu(), captured["z"].cpu()))

    activation_shape = tuple(captured["z"].shape)
    print(f"  embed layer      {EMBED_LAYER!r} -> class token {activation_shape}")
    print(f"  logits           {tuple(logits.shape)}")

    baseline = build_baseline(
        calibration,
        model_version=model_version,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
    )
    print(f"  samples          {baseline.sample_count}")
    print(
        f"  precision Σ⁻¹    {tuple(baseline.embed_precision.shape)} "
        f"{str(baseline.embed_precision.dtype).removeprefix('torch.')}"
    )
    if baseline.sample_count < EMBED_DIM:
        print(
            f"  note             {baseline.sample_count} samples < D={EMBED_DIM}: the sample"
            f" covariance is\n"
            f"                   rank-deficient and the regularisation term carries it. Fine"
            f" for a demo,\n"
            f"                   inflated for production — calibrate on >> D samples."
        )

    # ── 2. threshold ─────────────────────────────────────────────────────────
    banner("2/5", "Setting the alert boundary from measured clean traffic")
    # The shipped guard.yaml threshold (8.0) was chosen for a 32-dim toy model. Mahalanobis
    # distance scales with the embedding dimension and with how well-conditioned the
    # reference covariance is, so a threshold is only meaningful next to the clean level it
    # was measured against. This is the calibration step every real deployment owes itself.
    held_out_n = args.batch * 4
    with torch.no_grad():
        model(synthetic_images(held_out_n, generator).to(device))
    handle.remove()  # calibration is over; from here the GuardModule owns the capture
    clean_level = float(
        mahalanobis(
            captured["z"].cpu().to(baseline.embed_mean.dtype),
            baseline.embed_mean,
            baseline.embed_precision,
        ).mean()
    )
    # CUSUM sums (x - running_mean - delta) and alarms when the sum passes `threshold`.
    # delta is the per-window wobble to ignore; threshold is the excess that must accumulate.
    cusum_delta = 0.25 * clean_level
    cusum_threshold = 2.0 * clean_level
    print(f"  clean level      mean Mahalanobis {clean_level:.1f} over {held_out_n} held-out")
    print("                   images the baseline never saw")
    print(f"  cusum delta      {cusum_delta:.1f}  (slack: ignore drift below 25% of clean)")
    print(f"  cusum threshold  {cusum_threshold:.1f}  (accumulated excess required to alarm)")

    config = Config.from_yaml(Path(__file__).parent / "guard.yaml")
    # The shipped config is sized for production traffic; this run is a few dozen steps.
    config = config.model_copy(update={"window_size": 2, "drain_every_k": 4})
    config.thresholds["embedding_mahalanobis_mean"].threshold = cusum_threshold
    config.thresholds["embedding_mahalanobis_mean"].delta = cusum_delta
    for detector in config.detectors:
        if detector.name == "embedding":
            # ||Σ_run·Σ_ref⁻¹ − I||_F is an O(D³) matmul, hence cov_every: 32 in the shipped
            # config. At this step count that would recompute it once; drop it to 4 so the
            # metric actually moves within the demo.
            detector.params["cov_every"] = 4

    # ── 3. attach ────────────────────────────────────────────────────────────
    banner("3/5", "Attaching the monitor")
    exporter = PrometheusExporter()
    monitor = build_monitor(
        baseline,
        config,
        max_batch=args.batch,
        device=device,
        sink=exporter,
        alert_sink=LoggingAlertSink(),
        start_drain_thread=False,  # drive the drain by hand so the demo is deterministic
    )
    guarded = GuardModule(model, monitor, embed_layer=EMBED_LAYER, embed_reduce=EMBED_REDUCE)
    print(f"  device           {monitor.device}")
    print(f"  metrics          {len(monitor.metric_names)}: {', '.join(monitor.metric_names)}")
    print(f"  ring buffer      {monitor.stats().ring_bytes / 1024:.0f} KiB, fixed for the process")
    print("  the model was not modified: one forward hook on encoder.ln, removed by close()")
    print("  alerts will read baseline=? — the checksum that identifies a baseline is stamped")
    print("  by save(), and this demo keeps it in memory. Persist it (quickstart.py stage 2)")
    print("  and every alert names the exact reference the verdict was measured against.")

    def serve(steps: int, **corruption: float) -> None:
        with torch.no_grad():
            for _ in range(steps):
                guarded(synthetic_images(args.batch, generator, **corruption).to(device))
                # Production runs this on the monitor's own drain thread; pumping it here
                # keeps every step accounted for and the output reproducible.
                monitor.drain_pending()
        monitor.flush()

    def report(label: str) -> None:
        snap = monitor.aggregator.snapshot()
        latched = [name for name, on in monitor.aggregator.alert_states().items() if on]
        print(
            f"  {label:<12} entropy {snap['entropy_mean']:.4f}  "
            f"JS {snap['js_divergence']:.5f}  "
            f"mahalanobis {snap['embedding_mahalanobis_mean']:8.1f}  "
            f"mean-shift {snap['embedding_mean_shift']:8.1f}  "
            f"alerts {latched or '—'}"
        )

    # ── 4. serve clean ───────────────────────────────────────────────────────
    banner("4/5", "Serving clean traffic — the quiet case")
    serve(args.steps)
    report("clean")
    fired_clean = monitor.aggregator.alerts_fired
    print(f"  alerts fired     {fired_clean}  (a monitor that cries wolf gets turned off)")

    # ── 5. corrupt ───────────────────────────────────────────────────────────
    banner("5/5", f"Corrupting the input — noise sigma={NOISE_STD}, brightness +{BRIGHTNESS}")
    serve(args.steps, noise_std=NOISE_STD, brightness=BRIGHTNESS)
    report("corrupted")
    print(f"  alerts fired     {monitor.aggregator.alerts_fired - fired_clean}")
    print("  no exception was raised and no prediction was refused. The model answered every")
    print("  batch with the same confidence it had on clean data — the failure mode GUARD")
    print("  exists to make visible.")

    print("\nWhat Prometheus would scrape")
    print("─" * 78)
    for line in exporter.generate_latest().decode().splitlines():
        # `<counter>_created` is a Prometheus client artefact (the counter's birth
        # timestamp), not a GUARD metric — drop it so the listing is only real signal.
        name = line.split(" ", 1)[0].split("{", 1)[0]
        if name.startswith("guard_") and not name.endswith("_created"):
            print(f"  {line}")

    stats = monitor.stats()
    print("\n" + "─" * 78)
    print(
        f"engine: {stats.steps_observed} steps, {stats.rows_dispatched} rows exported, "
        f"{stats.observations_dropped} dropped, {stats.mean_enqueue_us:.0f} µs/step on the "
        f"inference thread"
    )
    print(f"failures: {stats.disabled_detectors or 'none — every detector healthy'}")
    print(f"wall clock: {time.perf_counter() - started:.1f} s")

    if monitor.device.type != "cuda":
        print(
            "\nOn CUDA this changes. Here on CPU every stream and event is a no-op, so the\n"
            "detectors run inline on the calling thread and the µs/step above is the full\n"
            "cost of the detector math, paid serially. On a CUDA device observe() only\n"
            "records an event and enqueues kernels onto a low-priority stream that waits on\n"
            "it, so the math executes in whatever SM capacity inference leaves idle and the\n"
            "µs/step drops to the enqueue cost alone. The metric values are unchanged; only\n"
            "the cost is. The p99 < 2% overhead target is an SLO, not a measurement — verify\n"
            "it on your own hardware with benchmarks/ab_latency.py and confirm the overlap\n"
            "is real with benchmarks/nsight_overlap.sh."
        )

    guarded.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
