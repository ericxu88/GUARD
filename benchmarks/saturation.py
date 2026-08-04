#!/usr/bin/env python3
"""Saturation sweep: where does overlap stop being free?

GUARD's overhead is not a constant — it is a function of how much SM capacity inference
leaves idle. At small batch sizes a transformer forward pass is launch-bound and latency-
bound, there is plenty of spare capacity, and the monitor kernels genuinely disappear into
the gaps. At large batch sizes every SM is busy and the monitor's kernels queue behind
inference, adding a small serial tail.

The design doc is explicit that this curve must be published honestly rather than quoted at
its best point (§14.5, §19). This script produces it: the same A/B measurement as
``ab_latency.py``, repeated across a batch-size sweep, printed as a table and optionally
written as JSON for a plot.

Usage::

    python benchmarks/saturation.py --batches 1,2,4,8,16,32,64,128
    python benchmarks/saturation.py --model vit_b_16 --json sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.ab_latency import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="synthetic", choices=["synthetic", "vit_b_16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--drain-every-k", type=int, default=32)
    parser.add_argument("--cov-every", type=int, default=32)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    rows: list[dict[str, Any]] = []

    print(f"\nGUARD saturation sweep — {args.model} on {args.device}")
    print("=" * 84)
    print(
        f"{'batch':>7}{'off p99 ms':>13}{'on p99 ms':>12}{'p99 %':>9}"
        f"{'qps off':>11}{'qps on':>11}{'thr %':>9}{'enq us':>9}"
    )
    for batch in batches:
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.batch = batch
        sweep_args.slo = float("inf")  # the sweep reports, it does not gate
        report = run(sweep_args)
        rows.append(
            {
                "batch": batch,
                "p99_off_ms": report.off.p99_ms,
                "p99_on_ms": report.on.p99_ms,
                "p99_overhead_pct": report.p99_overhead_pct,
                "qps_off": report.off.throughput_qps,
                "qps_on": report.on.throughput_qps,
                "throughput_drop_pct": report.throughput_drop_pct,
                "enqueue_us": report.monitor_enqueue_us,
                "ring_bytes": report.ring_bytes,
            }
        )
        print(
            f"{batch:>7}{report.off.p99_ms:>13.3f}{report.on.p99_ms:>12.3f}"
            f"{report.p99_overhead_pct:>9.2f}{report.off.throughput_qps:>11.1f}"
            f"{report.on.throughput_qps:>11.1f}{report.throughput_drop_pct:>9.2f}"
            f"{report.monitor_enqueue_us:>9.1f}"
        )

    worst = max(rows, key=lambda r: r["p99_overhead_pct"])
    print("-" * 84)
    print(
        f"worst case: batch={worst['batch']} -> p99 +{worst['p99_overhead_pct']:.2f}%, "
        f"throughput {worst['throughput_drop_pct']:+.2f}%"
    )
    print(
        "Publish this whole curve, not just the best point. Overhead grows as inference "
        "saturates the SMs; that is the honest shape of the trade-off."
    )
    if args.json:
        args.json.write_text(json.dumps({"device": args.device, "rows": rows}, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
