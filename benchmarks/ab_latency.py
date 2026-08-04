#!/usr/bin/env python3
"""A/B latency benchmark: the same model and load, GUARD off vs GUARD on.

This is the script that decides whether GUARD's central claim is true on *your* hardware
and *your* model. The design doc sets the SLO at **p99 latency increase < 2%** and
**throughput drop < 2%**; nothing in the repository asserts that number, because a number
you did not measure on the deployment you care about is marketing, not engineering.

Methodology (design doc §14), and why each part matters:

* **CUDA-event timing, not ``time.time()``.** Kernel launches are asynchronous, so a host
  timer straddling ``forward()`` measures launch overhead, not execution. Events are
  recorded *in the stream* and read after a synchronisation at the measurement boundary —
  never inside the timed region.
* **Warm-up is discarded.** The first iterations pay for allocator growth, cuDNN autotuning,
  and lazy module init. Including them would flatter or wreck the result at random.
* **Interleaved A/B.** Off and on are measured in alternating blocks rather than one after
  the other, so a thermal ramp or a noisy neighbour lands on both arms equally.
* **Percentiles, not means.** Serving is judged on tails. A mean hides exactly the stalls a
  monitoring hook is likely to introduce.

Usage::

    python benchmarks/ab_latency.py                       # synthetic model, quick
    python benchmarks/ab_latency.py --model vit_b_16 --batch 32 --iters 500
    python benchmarks/ab_latency.py --json results.json   # machine-readable output

Exit status is non-zero when the measured p99 overhead exceeds ``--slo`` (default 2%), so
this can gate a GPU CI job.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guard.baseline.compute import build_baseline
from guard.config import Config, DetectorConfig
from guard.engine.monitor import build_monitor
from guard.integration.torch_module import GuardModule


class SyntheticModel(nn.Module):
    """A transformer-ish stack with a tappable penultimate embedding.

    Deliberately not a toy: the overlap story depends on inference kernels being large
    enough to leave gaps, so a benchmark against a two-layer MLP would report an overhead
    that no real deployment will ever see.
    """

    def __init__(self, num_classes: int = 1000, embed_dim: int = 768, depth: int = 12) -> None:
        super().__init__()
        nhead = next(h for h in (12, 8, 6, 4, 2, 1) if embed_dim % h == 0)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T, D]``; returns logits ``[B, C]``."""
        h = self.norm(self.encoder(x))
        return self.head(h[:, 0])


@dataclass
class Arm:
    """Latency statistics for one arm of the A/B."""

    label: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_qps: float
    samples: int


@dataclass
class Report:
    """The full benchmark result, JSON-serialisable."""

    device: str
    model: str
    batch: int
    iterations: int
    off: Arm
    on: Arm
    p50_overhead_pct: float
    p95_overhead_pct: float
    p99_overhead_pct: float
    throughput_drop_pct: float
    slo_pct: float
    passed: bool
    monitor_enqueue_us: float
    monitor_gpu_us: float
    ring_bytes: int
    notes: list[str]


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; avoids interpolating across a bimodal latency distribution."""
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _time_block(model: nn.Module, batch: torch.Tensor, iterations: int, cuda: bool) -> list[float]:
    """Per-iteration latency in milliseconds, timed with CUDA events where available."""
    timings: list[float] = []
    if cuda:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for i in range(iterations):
            starts[i].record()
            model(batch)
            stops[i].record()
        # One synchronisation, at the measurement boundary — never inside the loop.
        torch.cuda.synchronize()
        timings = [starts[i].elapsed_time(stops[i]) for i in range(iterations)]
    else:
        import time

        for _ in range(iterations):
            start = time.perf_counter()
            model(batch)
            timings.append((time.perf_counter() - start) * 1e3)
    return timings


def _arm(label: str, timings: list[float], batch_size: int) -> Arm:
    total_s = sum(timings) / 1e3
    return Arm(
        label=label,
        p50_ms=_percentile(timings, 0.50),
        p95_ms=_percentile(timings, 0.95),
        p99_ms=_percentile(timings, 0.99),
        mean_ms=statistics.fmean(timings),
        throughput_qps=(len(timings) * batch_size / total_s) if total_s > 0 else math.nan,
        samples=len(timings),
    )


def _build_model(
    name: str, num_classes: int, embed_dim: int, depth: int, device: torch.device
) -> Any:
    """Build the model under test and return ``(model, embed_layer, embed_reduce, input)``."""
    if name == "vit_b_16":
        from torchvision.models import vit_b_16

        model = vit_b_16().to(device).eval()
        return model, "encoder.ln", "cls", lambda b: torch.randn(b, 3, 224, 224, device=device)
    model = (
        SyntheticModel(num_classes=num_classes, embed_dim=embed_dim, depth=depth).to(device).eval()
    )
    return model, "norm", "cls", lambda b: torch.randn(b, 197, embed_dim, device=device)


@torch.no_grad()
def run(args: argparse.Namespace) -> Report:
    """Execute the interleaved A/B and return the report."""
    device = torch.device(args.device)
    notes: list[str] = []
    if device.type != "cuda":
        notes.append(
            "Not running on CUDA: detectors execute inline on the calling thread, so this "
            "measures worst-case serial cost, NOT the overlapped cost GUARD is designed for."
        )
    torch.manual_seed(0)

    model, embed_layer, embed_reduce, make_input = _build_model(
        args.model, args.num_classes, args.embed_dim, args.depth, device
    )
    batch = make_input(args.batch)

    # Baseline from a short clean run, so the detectors do real work against real statistics.
    with torch.no_grad():
        calib = []
        probe: dict[str, torch.Tensor] = {}
        handle = model.get_submodule(embed_layer).register_forward_hook(
            lambda m, a, out: probe.__setitem__("z", out.detach())
        )
        for _ in range(4):
            logits = model(batch)
            embeddings = probe["z"]
            if embeddings.ndim == 3:
                embeddings = embeddings[:, 0]
            calib.append((logits.float().cpu(), embeddings.float().cpu()))
        handle.remove()
    num_classes = calib[0][0].shape[1]
    embed_dim = calib[0][1].shape[1]
    baseline = build_baseline(
        calib, model_version=f"{args.model}@bench", num_classes=num_classes, embed_dim=embed_dim
    )

    config = Config(
        window_size=args.batch,
        drain_every_k=args.drain_every_k,
        detectors=[
            DetectorConfig(name="entropy"),
            DetectorConfig(name="divergence"),
            DetectorConfig(name="embedding", params={"cov_every": args.cov_every}),
        ],
    )
    monitor = build_monitor(baseline, config, max_batch=args.batch, device=device)
    guarded = GuardModule(model, monitor, embed_layer=embed_layer, embed_reduce=embed_reduce)

    cuda = device.type == "cuda"
    # Warm-up both arms: allocator growth and autotuning must not land in the measurement.
    monitor.disable()
    _time_block(model, batch, args.warmup, cuda)
    monitor.enable()
    _time_block(guarded, batch, args.warmup, cuda)

    off: list[float] = []
    on: list[float] = []
    block = max(1, args.iters // args.blocks)
    for _ in range(args.blocks):
        monitor.disable()
        off += _time_block(model, batch, block, cuda)
        monitor.enable()
        on += _time_block(guarded, batch, block, cuda)

    stats = monitor.stats()
    guarded.close()

    off_arm = _arm("guard-off", off, args.batch)
    on_arm = _arm("guard-on", on, args.batch)

    def pct(a: float, b: float) -> float:
        return (b - a) / a * 100.0 if a > 0 else math.nan

    p99 = pct(off_arm.p99_ms, on_arm.p99_ms)
    drop = pct(on_arm.throughput_qps, off_arm.throughput_qps)
    if stats.observations_dropped:
        notes.append(f"{stats.observations_dropped} observations were dropped — check shapes.")
    if stats.summaries_dropped:
        notes.append(
            f"{stats.summaries_dropped} summaries dropped under backpressure; raise "
            f"max_pending_drains or drain_every_k if you need every window exported."
        )
    if stats.disabled_detectors:
        notes.append(f"detectors disabled by the firewall: {list(stats.disabled_detectors)}")

    return Report(
        device=str(device),
        model=args.model,
        batch=args.batch,
        iterations=len(on),
        off=off_arm,
        on=on_arm,
        p50_overhead_pct=pct(off_arm.p50_ms, on_arm.p50_ms),
        p95_overhead_pct=pct(off_arm.p95_ms, on_arm.p95_ms),
        p99_overhead_pct=p99,
        throughput_drop_pct=drop,
        slo_pct=args.slo,
        passed=bool(p99 <= args.slo and drop <= args.slo),
        monitor_enqueue_us=stats.mean_enqueue_us,
        monitor_gpu_us=stats.gpu_time_us,
        ring_bytes=stats.ring_bytes,
        notes=notes,
    )


def _print(report: Report) -> None:
    print(f"\nGUARD A/B latency — {report.model} on {report.device}, batch={report.batch}")
    print("=" * 78)
    header = f"{'arm':<12}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'mean ms':>10}{'qps':>12}"
    print(header)
    for arm in (report.off, report.on):
        print(
            f"{arm.label:<12}{arm.p50_ms:>10.3f}{arm.p95_ms:>10.3f}{arm.p99_ms:>10.3f}"
            f"{arm.mean_ms:>10.3f}{arm.throughput_qps:>12.1f}"
        )
    print("-" * 78)
    print(
        f"overhead   p50 {report.p50_overhead_pct:+.2f}%   p95 {report.p95_overhead_pct:+.2f}%   "
        f"p99 {report.p99_overhead_pct:+.2f}%   throughput {report.throughput_drop_pct:+.2f}%"
    )
    print(
        f"monitor    enqueue {report.monitor_enqueue_us:.1f} us/step (CPU cost to the "
        f"inference thread)   ring {report.ring_bytes / 2**20:.1f} MiB"
    )
    verdict = "PASS" if report.passed else "FAIL"
    print(f"SLO        p99 and throughput within {report.slo_pct:.1f}%  ->  {verdict}")
    for note in report.notes:
        print(f"note       {note}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="synthetic", choices=["synthetic", "vit_b_16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iters", type=int, default=300, help="measured iterations per arm")
    parser.add_argument("--blocks", type=int, default=6, help="interleaved A/B blocks")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12, help="synthetic model layer count")
    parser.add_argument("--drain-every-k", type=int, default=32)
    parser.add_argument("--cov-every", type=int, default=32)
    parser.add_argument("--slo", type=float, default=2.0, help="max allowed %% p99 overhead")
    parser.add_argument("--json", type=Path, help="write the report as JSON to this path")
    args = parser.parse_args()

    report = run(args)
    _print(report)
    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2))
        print(f"wrote {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
