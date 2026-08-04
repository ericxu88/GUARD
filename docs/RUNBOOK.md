# GUARD Operational Runbook

For the person who owns GUARD in production: the one-time calibration work, the baseline
lifecycle, what to do when a metric pages, what to do when GUARD itself misbehaves, and how
to turn it off.

Everything below is written against the shipped API. Config keys come from
`guard/config.py`, metric names from `guard/export/prometheus.py`, engine knobs from
`guard/engine/monitor.py`. Where a number is illustrative rather than universal, it says so.

**One thing to internalise before anything else:** GUARD reports *behavioural change
relative to a baseline*. It does not measure accuracy, and it cannot tell you whether a
change is bad. Every alert is a question ("why did this move?"), never a verdict.

---

## 1. Calibrating thresholds

This is the single most important operational task, and it is the one the shipped defaults
cannot do for you. The thresholds in `examples/guard.yaml` are illustrative: they were
chosen so the example runs, not so they fit your model. A threshold that is too low pages
on ordinary traffic until someone mutes the alert; one that is too high never fires and you
find out about drift from a customer.

### 1.1 What a threshold actually applies to

Three facts decide how you calibrate, plus one trap.

**`thresholds` is keyed by METRIC name, not detector name.** A detector emits several
metrics and each one gets its own entry:

```yaml
thresholds:
  js_divergence: {...}              # emitted by the "divergence" detector
  entropy_mean: {...}               # emitted by the "entropy" detector
  embedding_mahalanobis_mean: {...} # emitted by the "embedding" detector
```

The valid keys are exactly the metric names the enabled detectors emit. With all three
detectors on, that set is:

| Metric name | Detector |
|---|---|
| `entropy_mean` | entropy |
| `entropy_quantile` | entropy |
| `entropy_pop_shift` | entropy |
| `kl_divergence` | divergence |
| `js_divergence` | divergence |
| `embedding_mahalanobis_mean` | embedding |
| `embedding_mean_shift` | embedding |
| `embedding_cov_drift` | embedding (only when `cov_every > 0`) |

Print the authoritative list for your own config instead of trusting this table:

```python
monitor = build_monitor(baseline, config, max_batch=64)
print(monitor.metric_names)
```

A metric with **no** `thresholds` entry is still exported to Prometheus but can never page
anyone. That is the right default for diagnostic signals — keep `entropy_pop_shift` and
`embedding_cov_drift` on a dashboard, not on a rota.

**A misspelled metric name is silently ignored.** `Config` rejects unknown *top-level* keys
and unknown detector names, but `thresholds` is an open mapping, so
`entropy_men: {threshold: 2.0}` validates cleanly — and `MetricAggregator` then discards it
because it matches no metric. You get no error, no warning, and no alerting on that metric.
Verify after every config change:

```python
ignored = set(config.thresholds) - set(monitor.metric_names)
assert not ignored, f"these threshold entries alert on nothing: {sorted(ignored)}"
```

Put that assertion in your startup path. It is three lines and it is the difference between
a configured alert and the belief that you have one.

**`threshold` is a bound on the change-point statistic, not on the metric.** Both tests
accumulate:

- Page-Hinkley: `m_t = Σ_{i≤t}(x_i − x̄_i − delta)` and `PH_t = m_t − min_{i≤t} m_i` — the
  running total of how far the stream has exceeded its own running mean, minus the slack
  `delta`, measured from its lowest point so far.
- CUSUM: `S_t = max(0, S_{t-1} + (x_t − target_t − delta))`, with `target` the running mean.

So "set `threshold` to the metric's p99.9" is dimensionally wrong. `delta` is the value
that lives in metric units — it is the per-window slack that clean noise must not exceed —
and `threshold` is how much slack-exceeding evidence must pile up before you get paged.
Calibrate `delta` from the metric's clean distribution, then calibrate `threshold` from the
statistic those clean values actually produce.

**The tests see tumbling window means, not raw per-step values.** `MetricAggregator` keeps
two windows: a rolling one whose mean is exported to Prometheus, and a tumbling one (reset
every `window_size` steps) whose mean feeds the change-point test. Calibrate on tumbling
means of exactly your production `window_size`, or your numbers describe a different
distribution than the one that will page you.

### 1.2 The procedure

**Step 1 — capture known-good traffic.** Use the same tap that fed `build_baseline`: a
forward hook on the embedding layer plus the model's logits. Save a list of
`(logits [B, C], embeddings [B, D])` CPU tensors. Capture *more* than the baseline needs,
because you are about to split it: enough batches that
`batches / window_size` is in the hundreds of windows on each side of the split. With
`window_size: 32`, 200 held-out windows means 6 400 captured batches.

The capture must span whatever normal variation your traffic has — a full daily cycle at
minimum. Calibrating on one quiet hour produces thresholds that page every morning.

**Step 2 — split it.** Fit on the first ~60%, hold out the rest. The held-out segment is
the only thing that tells you the false-positive rate you just bought; measuring it on the
data you fitted is measuring nothing.

**Step 3 — run the calibrator.** It is offline, CPU-only, and uses the real detectors
(`build_detectors`) and the real tests (`PageHinkley` / `Cusum`), so the numbers it produces
are the numbers production will produce. Save it as `calibrate_thresholds.py`.

```python
#!/usr/bin/env python3
"""Derive GUARD thresholds from a capture of known-good traffic.

Usage:
    python calibrate_thresholds.py --baseline baselines/prod.guard \
        --config guard.yaml --capture clean.pt

``clean.pt`` is ``torch.save(pairs, "clean.pt")`` where ``pairs`` is a list of
``(logits [B, C], embeddings [B, D])`` CPU tensors captured from production traffic you
believe is healthy — the same tap that fed ``build_baseline``.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from pathlib import Path

import torch

from guard import Config, Cusum, PageHinkley, build_detectors, load
from guard.detectors.base import Detector

Batches = Sequence[tuple[torch.Tensor, torch.Tensor]]
Test = PageHinkley | Cusum

# Windows of sustained, slack-exceeding excess required before paging.
PATIENCE = 5
# Headroom over the worst statistic clean traffic produced.
MARGIN = 3.0


def per_step_metrics(detectors: Sequence[Detector], batches: Batches) -> list[dict[str, float]]:
    """One dict of metric values per batch — the same numbers the engine drains."""
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for logits, embeddings in batches:
            row: dict[str, float] = {}
            for detector in detectors:
                for name, value in detector.compute(logits, embeddings).scores.items():
                    row[name] = float(value)
            rows.append(row)
    return rows


def windowed(rows: Sequence[dict[str, float]], metric: str, window_size: int) -> list[float]:
    """Tumbling-window means: exactly what a change-point test is fed in production."""
    values = [row[metric] for row in rows if metric in row]
    return [
        statistics.fmean(values[i : i + window_size])
        for i in range(0, len(values) - window_size + 1, window_size)
    ]


def make_test(method: str, threshold: float, delta: float, **gate: int) -> Test:
    cls = PageHinkley if method == "page_hinkley" else Cusum
    return cls(threshold=threshold, delta=delta, **gate)


def calibrate(series: Sequence[float], method: str) -> tuple[float, float]:
    """Return ``(delta, threshold)`` for one metric from its clean windowed values."""
    delta = max(series) - statistics.median(series)
    probe = make_test(method, threshold=1e12, delta=delta)  # alarming effectively off
    clean_peak = max(probe.update(x).statistic for x in series)
    return delta, max(MARGIN * clean_peak, PATIENCE * delta)


def count_alarms(series: Sequence[float], test: Test) -> int:
    return sum(1 for x in series if test.update(x).alarm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--split", type=float, default=0.6, help="fraction used to fit")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    baseline = load(args.baseline)
    pairs: Batches = torch.load(args.capture, weights_only=True)
    cut = int(len(pairs) * args.split)
    fit_rows = per_step_metrics(build_detectors(config, baseline), pairs[:cut])
    held_rows = per_step_metrics(build_detectors(config, baseline), pairs[cut:])

    print(f"# baseline {baseline.model_version} ({baseline.sample_count} samples)")
    print("thresholds:")
    for metric, cfg in config.thresholds.items():
        series = windowed(fit_rows, metric, config.window_size)
        if not series:
            print(f"  # {metric}: no data — is its detector enabled?")
            continue
        delta, threshold = calibrate(series, cfg.method)
        held = windowed(held_rows, metric, config.window_size)
        gate = {"debounce": cfg.debounce, "cooldown": cfg.cooldown}
        false_alarms = count_alarms(held, make_test(cfg.method, threshold, delta, **gate))
        print(f"  {metric}:")
        print(f"    method: {cfg.method}")
        print(f"    threshold: {threshold:.6g}")
        print(f"    delta: {delta:.6g}")
        print(f"    debounce: {cfg.debounce}")
        print(f"    cooldown: {cfg.cooldown}")
        print(
            f"    # clean median {statistics.median(series):.6g}, "
            f"max {max(series):.6g}, {false_alarms} false alarm(s) "
            f"over {len(held)} held-out windows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`calibrate()` uses `max(series)` rather than a p99.9 quantile because a few hundred windows
is not enough to estimate a 99.9th percentile — on 300 windows the empirical p99.9 *is*
roughly the maximum. If your capture yields tens of thousands of windows, switch to a true
quantile (`statistics.quantiles(series, n=1000)[-1]`) so that one freak window does not set
your slack for you.

Note the two constants at the top, which are the only real judgement calls:

- `PATIENCE = 5` — the floor on `threshold`, expressed as "a `delta`-sized excursion
  sustained for 5 windows". Raise it to be slower and quieter, lower it to be faster and
  noisier.
- `MARGIN = 3.0` — headroom over the largest statistic clean traffic actually reached. If
  your clean capture is short or unrepresentative, this is the only thing standing between
  you and a 3 a.m. page. Widen it rather than narrowing it.

**Step 4 — read the false-alarm column.** Anything above zero on the held-out segment means
the threshold is too tight. Do not "fix" it by raising `debounce`: debounce requires
*consecutive* flagged windows, which suppresses isolated spikes but not a systematically
low threshold.

**Step 5 — paste the emitted YAML into your config and restart.** The script prints a valid
`thresholds:` block. Diff it against what you are running; a threshold that moved by an
order of magnitude means either the traffic or the baseline changed, and you want to know
which before you adopt it.

### 1.3 Sanity checks on the calibrated numbers

- `js_divergence` is bounded in `[0, log 2] ≈ [0, 0.693]`. A clean median anywhere near the
  top of that range means `P` and `Q` barely overlap — your baseline does not describe this
  traffic, and no threshold will fix that.
- `entropy_mean` is bounded in `[0, log C]`. Compare the clean median to `log C`: a
  well-trained classifier sits well below it.
- `embedding_mahalanobis_mean` should sit near `√D` when the reference Gaussian fits (for a
  well-specified reference, `E[d²] = D`). Substantially above `√D` on clean traffic is the
  under-sampled-covariance symptom described in §2.4.
- `embedding_cov_drift` and `entropy_pop_shift` have a **non-zero clean floor**, because
  both compare a finite live sample against a much larger reference. `entropy_pop_shift`
  compares a per-batch histogram of `B` samples against a baseline of tens of thousands;
  the total-variation distance between them is bounded away from 0 by batch size alone.
  Never assume either metric should read 0 — measure the floor and calibrate above it.

### 1.4 Re-calibrate when

Any of these invalidates the numbers: a new baseline, a new model version, a change to
`window_size` (it changes what the tests are fed), a change to `detectors[].params`
(`quantile`, `cov_every`), or a change in serving batch size (it changes the noise in every
per-batch statistic).

---

## 2. Baseline lifecycle

### 2.1 Build

```python
from guard import build_baseline, save

baseline = build_baseline(
    calibration_batches,               # iterable of (logits [B, C], embeddings [B, D])
    model_version="vit-b16@abc123",    # bind it to the exact model artifact
    num_classes=1000,
    embed_dim=768,
)
checksum = save(baseline, "baselines/vit-b16-abc123.guard")
```

`build_baseline` accumulates in float64 on CPU and produces five reference tensors:
`class_distribution` (Q), `entropy_histogram` + `entropy_bin_edges`, `embed_mean` (μ_ref),
and `embed_precision` (Σ_ref⁻¹, a regularised inverse). `sample_count` records how many
samples went in. It runs offline; never on a serving process.

Source the calibration data from held-out validation data or from *vetted* recent
production traffic. "Vetted" is doing real work in that sentence — see §2.5.

### 2.2 Bind to a model version

`model_version` is a free-form string, and the discipline is entirely yours: make it
identify the exact weights, not the model family. `vit-b16@<git-sha-or-artifact-digest>` is
right; `vit-b16` is not, because it cannot distinguish the model you calibrated from the one
you deployed.

The binding is enforced on load:

```python
baseline = load("baselines/vit-b16-abc123.guard",
                expected_model_version="vit-b16@abc123")
```

Pass `expected_model_version` from the same place your serving code gets its model
identifier. A `BaselineIntegrityError` at startup is a good outcome; silently monitoring
model B against model A's baseline is not.

The first 12 characters of the baseline's checksum are stamped onto every alert as
`baseline_version` (`build_monitor` does this from `baseline.checksum`), so a firing is
always traceable to the reference that judged it.

### 2.3 Save, load, and integrity

`save` validates the structure first (a malformed baseline is never written), computes a
SHA-256 over metadata + tensor bytes, embeds it, and returns it. `load` recomputes that
digest over the file's content and raises `BaselineIntegrityError` if it does not match,
then re-runs `Baseline.validate()`.

`BaselineIntegrityError` on load means one of four things:

1. **Checksum mismatch** — the file's bytes changed after it was written. Corrupt transfer,
   truncated download, or tampering. Re-fetch from the registry; do not "fix" it.
2. **`model_version` mismatch** — you passed `expected_model_version` and it disagreed.
   Deployment wiring bug: the wrong baseline is being shipped with this model.
3. **Unsupported `format_version`** — the file was written by a different GUARD build.
   Rebuild the baseline with the version you are deploying.
4. **Structure violation** — `validate()` rejected it (Q does not sum to 1, precision is not
   symmetric positive-definite, the entropy histogram is empty). Rebuild it; do not patch
   the file.

Treat the `.guard` file as an immutable, versioned artifact stored beside the model in your
registry or object store. Never edit one in place. Never hand-write one.

**Fail closed on a bad baseline.** `load` raises in your startup path, before any monitor
exists. Let it kill the process if monitoring is required, or catch it and start serving
with GUARD disabled — but make the choice explicitly and log it loudly. What you must not
do is fall back to some other baseline file.

### 2.4 How many calibration samples

The binding constraint is the embedding covariance. `build_baseline` estimates a `D × D`
covariance from `N` samples and inverts it. When `N` is not comfortably larger than `D`, the
estimate is rank-deficient or ill-conditioned; the regulariser
(`Σ + λI`, `λ = reg_factor · max(diag Σ)`, default `reg_factor = 1e-4`) keeps the inverse
from blowing up, but it cannot invent information that is not in the data. The result is a
precision matrix that assigns huge distances to directions it never observed — so
`embedding_mahalanobis_mean` and `embedding_cov_drift` read high on traffic that is
perfectly clean.

**Guidance: `N ≥ 20·D` as a hard floor, `N ≥ 50·D` for comfort.** For ViT-B/16 with
`D = 768` that is 15 000 samples minimum and 40 000 preferred.

A measurement of the effect, on synthetic Gaussian data with `D = 32`, batch 64, all
traffic drawn from the same clean distribution the baseline was built on
(`√D = 5.66`, so a perfectly specified reference gives `embedding_mahalanobis_mean ≈ 5.66`):

| Calibration samples | N/D | `embedding_mahalanobis_mean` | `embedding_cov_drift` |
|---:|---:|---:|---:|
| 64 | 2 | 8.17 | 14.50 |
| 320 | 10 | 6.00 | 2.60 |
| 1 600 | 50 | 5.72 | 0.95 |
| 6 400 | 200 | 5.62 | 0.54 |

This is synthetic and illustrative — your embeddings are not isotropic Gaussians and your
numbers will differ. What generalises is the shape: at `N/D = 2` the "clean" Mahalanobis
distance is inflated by ~45% and the covariance-drift statistic is nonsense; by `N/D = 50`
both have essentially converged. Also note the residual `embedding_cov_drift = 0.54` at
`N/D = 200`: that floor comes from the *live* estimator, which is exponentially weighted
over roughly the last 100 batches, not from the baseline. It never reaches zero.

Diagnostic: build the baseline, then run the calibrator from §1.2 on a **held-out clean**
segment. If `embedding_mahalanobis_mean` sits well above `√D`, or `embedding_cov_drift` is
large compared with `√D`, you under-sampled. Collect more calibration data. Do not
compensate by raising thresholds — that hides the real drift you are trying to catch.

If you genuinely cannot get enough samples, the honest options are to reduce `D` (a
`mean`-reduced embedding rather than a `flatten`-reduced one — see `reduce_embedding` in
`guard/integration/torch_module.py`), or to disable the `embedding` detector and monitor the
output distribution only.

### 2.5 Re-baselining

**The trap, stated plainly:** a stale baseline alarms on legitimate evolution, and a
too-fresh one masks real drift. Rebuild too rarely and you page on last quarter's traffic
mix; rebuild automatically from recent production and you calibrate the alarm to the fire —
if the model has been quietly degrading for a week, a baseline built from that week declares
the degradation normal.

There is no threshold setting that resolves this. It is a process control, and the process
is:

1. **Scheduled recompute.** On a cadence (monthly is a reasonable starting point) build a
   candidate baseline from recent, *vetted* production traffic. Vetted means someone
   confirmed the model was behaving correctly over that window using evidence GUARD does
   not have — labelled evaluation, business metrics, a spot audit.
2. **Diff against the incumbent.** Run the §1.2 calibrator twice on the same held-out clean
   capture, once per baseline, and compare the resulting clean medians and thresholds.
   Large moves are the interesting part.
3. **Human approval.** A named person signs off on the diff, recording *why* the shift is
   legitimate evolution. If nobody can articulate the why, that is your answer: do not adopt
   it, investigate instead.
4. **Versioned rollout.** A new filename and a new artifact — never an overwrite. Its
   checksum changes, so `baseline_version` on every subsequent alert changes with it, which
   is how you tell later which reference produced a firing. (`model_version` only changes if
   the *model* changed.) Deploy it like any other config change: canary first, keep the
   previous file, and be able to roll back to it.
5. **Re-calibrate thresholds.** A new baseline changes every metric's clean distribution.
   Thresholds calibrated against the old one are meaningless. §1.

**Never auto-adopt a new baseline.** Nothing in GUARD does this, and nothing you build
around it should.

After a rollout, expect a transient: the live population statistics inside the embedding
detector (`WelfordCovariance`, exponentially weighted with a ~69-batch half-life) still
carry the old regime for a few hundred batches. `MetricAggregator.reset()` clears windows,
test state, and latches if you need a clean slate; a process restart clears everything.

---

## 3. Responding to an alert

An alert arrives as an `Alert` (logged by `LoggingAlertSink`, POSTed by
`WebhookAlertSink`) carrying `metric`, `detector`, `step`, `value`, `statistic`,
`threshold`, `method`, `model_version`, and `baseline_version`. In Prometheus, the same
firing latches `guard_drift_alert{detector="..."}` at 1 for 10 windows
(`DEFAULT_ALERT_HOLD` in `guard/engine/aggregator.py`), decaying to 0 once the stream is
quiet — so the alert clears on recovery without a manual reset.

Note the label is the **detector**, not the metric. `entropy_mean` and `entropy_pop_shift`
both raise `guard_drift_alert{detector="entropy"}`; only the `Alert` payload tells you which
metric fired.

### Before anything else: four questions for every alert

1. **Is the baseline the right one?** Compare `baseline_version` and `model_version` in the
   alert against what is actually deployed. A model rollout that shipped without its
   baseline produces drift alerts on a perfectly healthy model.
2. **Did anything else change at the same time?** Deploys, preprocessing changes, a new
   client, a traffic-routing change, a batch-size change. Correlate the alert timestamp with
   your deploy log before you look at the model.
3. **Are the other metrics moving?** A single metric moving alone means something different
   from all of them moving together. §3.9 has the combination table.
4. **Is GUARD itself healthy?** Check `guard_detectors_disabled`,
   `guard_observations_dropped_total`, and `guard_summaries_dropped_total` — §4. A partially
   broken monitor produces metric streams with holes, and a holed stream can page.

---

### 3.1 `guard_entropy_mean`

**What it means.** Windowed mean Shannon entropy of the softmax outputs, in nats, bounded in
`[0, log C]`. It measures how uncertain the model is, averaged over the window. Rising
entropy means the model is becoming less confident — usually the earliest available signal
of input shift, and it typically moves before accuracy does.

**Most likely benign cause.** The traffic mix genuinely got harder: a new customer segment,
a seasonal shift, a class that was rare in calibration becoming common. The model is
behaving correctly on genuinely more ambiguous input.

**Most likely real cause.** Covariate shift the model cannot represent — an upstream
preprocessing change (resize, normalisation constants, colour space, tokeniser version), a
sensor or capture-pipeline change, or input corruption. Entropy rising with
`embedding_mahalanobis_mean` is the classic input-shift signature.

**First three things to check.**
1. `embedding_mahalanobis_mean` and `embedding_mean_shift` over the same window. Both up
   ⇒ the inputs moved, look upstream. Entropy up alone ⇒ suspect the output layer or a
   change in the class mix.
2. The upstream preprocessing pipeline's deploy history. This is the single most common real
   cause and the cheapest to rule out.
3. Sample the actual inputs from the alerting window and look at them. Entropy tells you the
   model is unsure; only the data tells you why.

---

### 3.2 `guard_entropy_quantile` / `guard_entropy_p99`

**What it means.** The configured tail quantile (`detectors[].params.quantile`, default
0.99) of per-sample entropy within the window. Where `entropy_mean` describes the whole
population, this describes the worst tail — it moves when a *subset* of traffic goes
out-of-distribution while the bulk stays fine.

`guard_entropy_p99` is an alias fed **only** when the configured quantile really is 0.99;
`guard_entropy_quantile` always carries the true value. If you set `quantile: 0.95`, the p99
gauge stays at 0 by design and your dashboard should use `guard_entropy_quantile`.

**Most likely benign cause.** A small, legitimately hard cohort appearing in the mix —
edge-case inputs that were always going to be uncertain and simply became more frequent.

**Most likely real cause.** A subpopulation of genuinely OOD traffic: one broken client, one
misconfigured device fleet, one region with a different capture pipeline. The tail moves
first because the bad cohort is a minority.

**First three things to check.**
1. Is `entropy_mean` flat while the quantile rises? That gap *is* the finding: a subset, not
   the whole stream. Look for a per-client/per-region/per-device split in your request logs.
2. `entropy_pop_shift` — a change in histogram shape rather than location supports the
   "second mode appeared" reading.
3. Whether the alert coincides with a traffic-source change (new integration, new API key,
   a client rolling out a new app version).

---

### 3.3 `guard_entropy_pop_shift`

**What it means.** Total-variation distance in `[0, 1]` between the batch's entropy
histogram (binned on the baseline's edges) and the baseline entropy histogram. Unlike mean
and quantile, this sees the *shape* of the confidence distribution — it catches a
distribution that split into two modes while its mean stayed put.

This metric has **no `thresholds` entry in the shipped config** and therefore never pages.
That is deliberate; it is a diagnostic, and it has a substantial non-zero floor on clean
traffic (a `B`-sample histogram compared against a reference of tens of thousands cannot
match exactly). Read it alongside the others, not on its own.

**Most likely benign cause.** Small serving batch size. The floor falls roughly as `1/√B`,
so a small-batch server reads a large distance no matter how healthy it is. Measured on
synthetic clean traffic (`C = 10`, 64-bin baseline histogram built from 25 600 samples,
every batch drawn from the same distribution as the baseline):

| Batch size | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| median `entropy_pop_shift` | 0.61 | 0.44 | 0.33 | 0.24 | 0.16 | 0.12 | 0.09 |

Your floor depends on `C`, the bin count, and how peaked the entropy distribution is —
measure it rather than borrowing this table.

**Most likely real cause.** A bimodal confidence distribution: part of the traffic handled
confidently, part not — the same "one bad cohort" story as §3.2, seen from a different angle.

**First three things to check.**
1. What the clean floor is for your batch size — measure it before reading anything into the
   value.
2. Whether `entropy_mean` and `entropy_quantile` are diverging from each other; shape change
   without location change is the signal this metric exists for.
3. The baseline's `entropy_histogram` sample count (`baseline.sample_count`). A histogram
   built from too few samples is noisy, and every comparison against it inherits the noise.

---

### 3.4 `guard_kl_divergence`

**What it means.** `KL(P‖Q)` between the batch-mean softmax `P` and the baseline class
distribution `Q`. Unbounded, asymmetric, and heavily penalises classes that `Q` considers
nearly impossible but `P` predicts.

**Prefer `js_divergence` for alerting.** KL's unbounded range makes thresholds unstable —
one rare class appearing can move it by an order of magnitude. Keep KL as a diagnostic: when
JS fires, KL's magnitude tells you whether the shift is concentrated in low-probability
classes (large KL relative to JS) or spread across common ones.

**Most likely benign cause.** A rare class showing up at all. `Q` gives it near-zero mass,
so any real prediction of it produces a large KL contribution.

**Most likely real cause.** Same as JS (§3.5) — this is the same shift measured on a less
stable scale.

**First three things to check.**
1. `js_divergence` in the same window. If JS is quiet and KL is loud, the movement is in the
   tail of `Q`, and it is probably not what you want to page on.
2. The per-class prediction histogram now versus the baseline `Q` — which classes actually
   moved.
3. Whether the class taxonomy changed (a label added, merged, or deprecated upstream) since
   the baseline was built.

---

### 3.5 `guard_js_divergence`

**What it means.** Jensen-Shannon divergence between the batch-mean softmax `P` and the
baseline class distribution `Q`. Symmetric and bounded in `[0, log 2] ≈ [0, 0.693]` nats,
which is exactly why it is the preferred alerting signal: the bounded range makes a
threshold mean the same thing over time. It detects **prior / label shift** — the mix of
things the model predicts has changed.

**Most likely benign cause.** The real-world class mix moved for a legitimate reason:
seasonality, a marketing campaign, a new market, a product-catalogue change. The model is
correct; the world is different.

**Most likely real cause.** Something upstream is routing different traffic to this model
(a router change, a filter that stopped filtering, a new client sending a different
workload), or the model has developed a prediction bias — collapsing toward a dominant
class, which shows up as `P` concentrating.

**First three things to check.**
1. The top-k predicted class frequencies now versus `Q`. A single class gaining most of the
   mass reads very differently from a broad reshuffle: concentration suggests a model or
   input problem, reshuffling suggests a traffic-mix change.
2. Request volume and routing by source. A shift in `P` with a matching shift in who is
   calling you is a traffic story, not a model story.
3. `entropy_mean` in the same window. JS up with entropy *down* means the model became more
   confident about a different mix — often a collapse toward one class. JS up with entropy
   up means it is seeing input it cannot classify.

---

### 3.6 `guard_embedding_mahalanobis_mean`

**What it means.** Mean per-sample Mahalanobis distance from the penultimate embedding to
the reference Gaussian `(μ_ref, Σ_ref)`. This is GUARD's most direct **covariate shift**
signal: it measures the inputs' representation moving, independently of what the model
predicts. On clean, well-calibrated traffic it sits near `√D`.

**Most likely benign cause.** An under-sampled baseline covariance (§2.4). This is the most
common false positive in the whole system, it is present from the first day, and it is
diagnosed by comparing the clean value against `√D`, not by staring at the alert.

**Most likely real cause.** The input distribution genuinely moved: a preprocessing change,
a new data source, a domain shift. This is exactly what the metric is for, and it typically
fires before entropy or JS do — the representation moves before the prediction does.

**First three things to check.**
1. `embedding_mean_shift` alongside it. Both up ⇒ the whole population moved (a systematic
   change: preprocessing, a new source). Mahalanobis up with mean-shift flat ⇒ the
   population centre is unchanged but individual samples are further out — a spread or
   outlier problem, e.g. one bad cohort mixed into normal traffic.
2. Whether the clean-traffic value was ever near `√D`. If it was already inflated before the
   alert, you are looking at a baseline problem, not a traffic problem. Re-read §2.4.
3. The embedding tap itself: is `embed_layer` still pointing at the same module, and is
   `embed_reduce` still doing the same reduction, after your last model change? A changed tap
   silently changes the entire embedding space and every distance in it.

---

### 3.7 `guard_embedding_mean_shift`

**What it means.** Mahalanobis distance from the *running* embedding mean (exponentially
weighted, ~69-batch half-life) to `μ_ref`. Where `mahalanobis_mean` averages per-sample
distances, this measures how far the population *centre* has travelled — it is much less
noisy and much more specific to systematic shift.

Because the estimator forgets exponentially, this metric returns toward its clean level once
traffic recovers. That is deliberate: an infinite-memory estimator would keep a drift
episode baked in forever, and an alert that can never clear is an alert people mute.

**Most likely benign cause.** A slow, real population change that your business already
knows about — new geography, new device generation, a gradual catalogue shift.

**Most likely real cause.** A systematic upstream change applied to every input:
normalisation constants, image resize/crop policy, colour-space handling, tokeniser or
vocabulary version. These shift the entire population coherently, which is precisely what
this metric is most sensitive to.

**First three things to check.**
1. The upstream preprocessing deploy log, again. A coherent whole-population shift almost
   always has a single upstream cause.
2. Whether the shift is monotone (a ramp over hours/days ⇒ real-world evolution ⇒
   re-baselining conversation, §2.5) or a step (a step at a timestamp ⇒ a deploy).
3. `embedding_cov_drift` in the same window: mean-shift alone means the population
   translated; mean-shift plus cov-drift means its shape changed too, which is the more
   serious finding.

---

### 3.8 `guard_embedding_cov_drift`

**What it means.** `‖Σ_run·Σ_ref⁻¹ − I‖_F` — how much the live embedding covariance differs
from the reference in shape, not just location. Zero when they match. Recomputed every
`cov_every` steps (default 32) because it is an `O(D³)` matmul; the reported value is the
most recent computation, which is the right semantics for a cumulative statistic.

Like `entropy_pop_shift`, this has **no `thresholds` entry in the shipped config** and does
not page. It has a non-zero clean floor set by the live estimator's effective sample count
(batch size × ~100 batches) and by `D`; see the table in §2.4.

**Most likely benign cause.** An under-sampled baseline covariance (§2.4), or simply the
metric's natural floor for your `D` and batch size. Both are constants, not events.

**Most likely real cause.** The structure of the representation changed: a genuinely
different data regime, or a new mode in the traffic that occupies directions the reference
never saw.

**First three things to check.**
1. Your measured clean floor. If you have not measured it, you cannot interpret the value at
   all.
2. `N/D` for the baseline (`baseline.sample_count / baseline.embed_dim`). Below ~20, this
   metric is not trustworthy.
3. `cov_every`. If someone set it to 1, the metric updates every step and costs `O(D³)` per
   batch — check `guard_monitor_overhead_us` and `guard_ring_utilization` before assuming the
   value is the problem.

---

### 3.9 Reading the combination

The single most useful triage step is looking at which metrics moved together.

| Pattern | Most likely reading |
|---|---|
| Embedding metrics up, entropy and JS flat | Input distribution moved; the model absorbed it so far. Leading indicator — investigate, do not panic. |
| Entropy up + embedding up, JS flat | Genuine covariate shift the model cannot handle. The serious case. |
| JS up, entropy down, embedding flat | Prediction collapse toward a dominant class, with unchanged inputs. Suspect the model or its output layer. |
| JS up, embedding up, entropy up | Everything moved. Check for a wholesale traffic-routing or preprocessing change first. |
| Everything up at one timestamp, sharply | A deploy. Correlate before investigating the model. |
| One metric up, all others perfectly flat | Suspect the monitor before the model — §4. |

---

## 4. When GUARD itself is unhealthy

GUARD is designed to fail quietly and never harm serving. The cost of that is that its own
failures are invisible unless you watch for them. These are the signals.

### 4.1 A detector was disabled by the circuit breaker

`Firewall` (`guard/safety/firewall.py`) wraps every detector call. A detector that raises is
counted; at `max_detector_failures` (default 3, a `GuardMonitor`/`build_monitor` kwarg) its
breaker trips and the firewall stops calling it entirely. The metric slice stays NaN, the
aggregator skips it, and the metric simply stops updating in Prometheus. Inference is
untouched — that is the whole design — but you are now flying with one instrument dead.

The breaker counts **total** failures, not consecutive ones. A detector that fails
intermittently produces a metric stream with holes, and a drift verdict computed from a
holed stream is worse than no verdict.

**Signals.** `guard_detectors_disabled > 0`. A detector's gauges going flat while
`guard_steps_observed_total` keeps climbing. One `ERROR` log record with a traceback (the
*first* failure only — a broken detector must not flood the serving logs) plus one `WARNING`
on the trip, both on the `guard.engine` logger, since that is the logger the engine hands
its firewall.

**Inspect it:**

```python
fw = monitor.firewall
fw.disabled                       # frozenset of disabled component names
fw.is_disabled("embedding")       # bool
fw.failure_count("embedding")     # total failures recorded
fw.last_error("embedding")        # the most recent exception object
monitor.stats().disabled_detectors  # same set, as a sorted tuple
```

Component names are the detector names (`"entropy"`, `"divergence"`, `"embedding"`) plus two
special ones:

- **`"engine"`** — the enqueue path itself. If this trips, *all* monitoring stops. Note that
  `monitor.enabled` still reports `True` (the kill-switch was not flipped) and
  `guard_detectors_disabled` freezes at its last published value, because publishing happens
  on the drain and there are no more drains. The reliable external signal is
  `guard_steps_observed_total` flatlining while the server is demonstrably serving traffic.
  Alert on that.
- **`"aggregator"`** — the host-side dispatch of drained buffers. If this trips, detectors
  keep running on the GPU but nothing reaches the exporter or the change-point tests:
  `stats().rows_dispatched` stops growing while `steps_observed` climbs.

**Recovery.** Fix the underlying cause (`last_error` has the exception), then re-arm:

```python
monitor.firewall.reset("embedding")   # one component
monitor.firewall.reset()              # all of them
```

Re-arming without fixing the cause just burns the failure budget again.

`guard_detector_failures_total{detector}` is fed by the engine: `_publish_telemetry` reads
`firewall.failure_count(...)` on each drain and reports the **delta** to the sink. That
gives you the failure *rate* per detector, which is the signal that tells you a detector is
degrading before its breaker trips — `guard_detectors_disabled` only moves after the budget
is already spent. In-process, `monitor.firewall.failure_count(name)` and
`monitor.firewall.last_error(name)` are the same information without the scrape delay.

### 4.2 Summaries dropped under backpressure (`guard_summaries_dropped_total`)

Every `drain_every_k` steps the engine enqueues one async device→host copy of the scalar
summary block into a pinned buffer taken from a pool of `max_pending_drains` (default 4).
If every buffer is still in flight — because the drain thread has not consumed them — the
copy is **skipped and the whole block is dropped**, counted in units of `drain_every_k`. It
is never buffered without bound, and it never blocks.

**What it means.** The host-side drain is not keeping up with inference. Metrics are
sampled rather than complete: the change-point tests see fewer windows, so detection is
slower and less reliable. Inference is unaffected.

**Signals.** `rate(guard_summaries_dropped_total[5m]) > 0`. `stats().summaries_dropped`
climbing. `stats().drains_enqueued − stats().drains_completed` persistently at
`max_pending_drains`.

**Tuning knobs**, in the order to try them:

1. **`drain_every_k`** (config key, also a `GuardMonitor` kwarg) — raise it. Fewer, larger
   copies per unit time; the direct fix for drain-rate pressure. The cost is metric
   staleness: a value can be up to `drain_every_k` steps old before it is exported.
2. **`max_pending_drains`** (`GuardMonitor`/`build_monitor` kwarg, default 4) — raise it to
   absorb bursts. Costs one pinned host buffer of `drain_every_k × n_metrics × 4` bytes per
   slot; small, but pinned memory is a finite system resource.
3. **`drain_poll_interval`** (kwarg, default 0.05 s) — lower it if the drain thread is
   sleeping through completed copies. Costs CPU on a background thread.

If you drive the drain yourself (`start_drain_thread=False`), drops mean *your* loop is not
calling `monitor.drain_pending()` often enough — that is the knob, not the config.

### 4.3 Observations dropped (`guard_observations_dropped_total`)

`observe()` rejects a batch — silently, by design, counted rather than raised — when:

- logits are not rank-2, or their width is not `num_classes`;
- the batch size is outside `[1, max_batch]` — **the batch is dropped entirely, never
  truncated**;
- logits or embeddings are on a different device than the monitor;
- embeddings are not rank-2, their width is not `embed_dim`, or their batch size does not
  match the logits';
- a CUDA Graph capture is in progress (§6.3).

Only the *first* rejection is logged (with the reason); the rest are counted only. So the
one log line at startup is the whole diagnosis — find it.

**The most common real cause is dynamic batching.** A server that batches up to 64 will
occasionally produce a batch of 65, and every one of those is dropped. Set `max_batch` to
the server's true maximum, not its nominal one. The cost is memory: the ring buffer is
`ring_slots × max_batch × (num_classes + embed_dim) × 4` bytes, reported as
`stats().ring_bytes`.

**Signals.** `guard_observations_dropped_total` climbing. `stats().observations_dropped`.
A one-off `WARNING` from `guard.engine` naming the reason.

### 4.4 Ring utilisation (`guard_ring_utilization`)

The fraction of used ring slots whose monitor-stream work has not yet finished, polled with
a non-blocking event query. Persistently near 1.0 means the monitor stream is falling behind
inference — overlap has stopped being free on this workload. Raising `ring_slots` (default
8) buys pipelining depth, not throughput; if the detectors genuinely cost more than the gaps
inference leaves, the real fixes are raising `cov_every`, disabling the `embedding`
detector, or accepting the overhead. Measure it with `benchmarks/ab_latency.py` rather than
guessing.

### 4.5 Overhead telemetry

`guard_monitor_overhead_us` is the EMA (≈100 steps) of the CPU time `observe()` adds to the
inference thread. This is the honest cost metric — it is what inference actually pays. There
is no universal "good" value; record yours on day one from a healthy deployment and alert on
departures from it. A sustained rise means something on the enqueue path started doing real
work, and the classic cause is a host synchronisation (`.item()`, `.cpu()`, `synchronize()`)
sneaking into a detector, which serialises the inference stream.

`guard_monitor_gpu_time_us` measures monitor-stream GPU time but requires
`enable_gpu_timing=True`, which records two CUDA events per `drain_every_k` steps. It is off
by default. Turn it on to investigate, not to run.

---

## 5. Turning it off

Three mechanisms, in increasing order of blast radius.

### 5.1 `GUARD_DISABLED` environment variable — the kill-switch

Set `GUARD_DISABLED` to any of `1`, `true`, `yes`, `on` (case-insensitive, surrounding
whitespace ignored) and every `GuardMonitor` in the process stops monitoring. It is checked
on **every** read of the enabled flag, not cached at startup, and it is ANDed with the
runtime flag: GUARD runs only if the flag is set *and* the env kill-switch is not engaged.

```bash
# In a container/pod spec, then restart the pod:
GUARD_DISABLED=1
```

`observe()` returns immediately, so the cost collapses to one property read per forward
pass. The wrapped model's output is byte-identical either way. Nothing else changes: the
process still starts, `/metrics` still serves (with frozen values), and no exception is
raised anywhere.

This is the switch to reach for during an incident when you suspect GUARD and want it gone
without shipping a new image.

### 5.2 `GuardMonitor.disable()` — runtime, in-process

```python
monitor.disable()          # stop observing; everything else stays up
monitor.enable()           # resume (still subject to GUARD_DISABLED)
monitor.enabled            # current state: runtime flag AND env AND not closed
```

Use this from an admin endpoint or a debugger when you cannot restart the process. It stops
`observe()` doing work; it does **not** stop the drain thread, close sinks, or free the ring
buffer. `enable()` brings it back. Note that the change-point tests keep whatever state they
had, so the first windows after re-enabling are compared against pre-disable history.

### 5.3 `enabled: false` in the config

```yaml
enabled: false
```

Passed through `build_monitor` → `GuardMonitor.from_config` as the *initial* kill-switch
state. It is the declarative equivalent of starting with `disable()` already called, and
`monitor.enable()` can still turn it on afterwards. Use it to ship a build with GUARD off by
default; use the env var for incident response.

### 5.4 Shutting down cleanly

```python
monitor.close()      # stop drain thread, flush in-flight work, close alert sinks
guarded.close()      # GuardModule: remove the forward hook, then monitor.close()
```

`close()` is idempotent and synchronises the monitor stream, so do not call it from a
request path. `GuardModule.detach_hook()` removes the forward hook and leaves the wrapped
model exactly as it was, if you want to un-instrument without shutting the monitor down.

### 5.5 What the config's `export:` block does and does not do

When you call `build_monitor(...)` **without** a `sink`, it reads `export.enabled` and
`export.namespace` and builds the sink for you — a `PrometheusExporter` in that namespace,
or a `NullSink` when `enabled: false`. Passing an explicit `sink=` overrides the block
entirely.

Two keys are still yours to act on. `export.prometheus_port` is **not** bound
automatically: a library should not open a listening socket behind your back, so the
endpoint only exists once you ask for it. And `export.push_gateway_url` is declarative —
nothing reads it; wire your own Pushgateway client if you run batch jobs that are never
scraped.

```python
monitor = build_monitor(baseline, config, max_batch=64)   # sink built from config.export
if config.export.enabled:
    monitor.sink.start_http_server(config.export.prometheus_port)   # you bind the port
```

---

## 6. Known limitations

State these to anyone who asks GUARD a question it cannot answer.

### 6.1 Drift is not accuracy loss

GUARD is unsupervised. It compares live behaviour to a baseline and reports *change*. It has
no labels and therefore no notion of correctness. Behavioural change usually precedes
degradation, which is what makes it valuable as a leading indicator — but "usually" is not
"always", in either direction:

- A model can drift substantially and remain accurate (the world changed; the model
  generalised).
- A model can degrade with every GUARD metric flat (a subtle label-quality problem, an
  adversarial input that stays in-distribution in embedding space).

Pair GUARD with periodic labelled evaluation. An alert is a prompt to investigate, never
grounds for a rollback on its own.

### 6.2 Overhead is workload-dependent

The design targets p99 latency overhead **< 2%**. That is an SLO to be verified on your
hardware and your workload, not a measured result of this build, and this repository does
not ship a benchmark number.

The mechanism is why: monitor kernels run on a low-priority CUDA stream and execute in
whatever SM capacity inference leaves free. When inference has headroom (small batches,
memory-bound layers, inter-layer gaps) monitoring is nearly free. When inference saturates
every SM, monitoring kernels queue and add a small serial tail. The claim rests on
monitoring FLOPs being minuscule next to the model's, not on the scheduler working magic.

Measure it before you promise it: `benchmarks/ab_latency.py` for the A/B p50/p95/p99 delta,
`benchmarks/saturation.py` for the batch-size sweep where overlap degrades, and
`benchmarks/nsight_overlap.sh` for the Nsight Systems timeline that proves monitor kernels
interleave with inference kernels rather than trailing them. Publish the saturation curve
including the worst case.

On a non-CUDA device there is no overlap at all: streams and events degrade to no-ops and
the detectors run **inline** on the calling thread. CPU deployments pay the full detector
cost synchronously.

### 6.3 CUDA Graph capture drops observations, by design

If your serving path captures the forward pass into a CUDA Graph, per-step Python hooks do
not fire and recording an event or switching streams inside the capture corrupts the graph
(design doc §6.4). `observe()` detects an active capture via
`torch.cuda.is_current_stream_capturing()` and **drops the observation**, counting it in
`guard_observations_dropped_total`. Dropping is the only safe answer; corrupting the graph
would violate prime directive #1.

The consequence is concrete: a server that graph-captures every step gets **zero**
monitoring while producing no error. If `guard_observations_dropped_total` climbs in
lockstep with your request count and `guard_steps_observed_total` does not move, graph
capture is the first thing to check.

Working around it means either capturing the detector kernels into the same graph against
static output tensors, or writing outputs to a fixed buffer the monitor reads after the
graph replays. Neither is implemented here. Flag graph capture at integration time, not in
production.

### 6.4 The tests signal a level CHANGE, and fire once per regime transition

Both `PageHinkley` and `Cusum` are **one-sided upward** change-point tests over a scalar
stream. Two consequences.

**They detect changes, not levels.** A stream that has been elevated since before the
process started looks stationary and produces no alarm. Restarting a serving pod during an
ongoing drift episode therefore *clears* the alert — the new process's tests see a flat,
already-drifted stream. Compare against the baseline-relative metric values on your
dashboard, not only against the alert gauge.

**The unsupervised CUSUM mode fires once per transition.** `Cusum.from_config` does not set
`target`, so the target is the stream's own running mean. When a shift accumulates enough
evidence the test fires and the statistic resets to 0 — but the running mean keeps tracking
the stream, so on a sustained plateau it converges to the new level, the increments go
non-positive, and no further alarms come. You get one alarm at the transition, not a
continuous alert during the plateau. `guard_drift_alert{detector}` latches for 10 windows
after a firing, which is what makes the firing visible to a scrape at all; it then decays to
0 even though the drift persists.

If you need an alert that stays lit for as long as the shift lasts, add a static
Alertmanager rule on the metric gauge itself, using the clean maximum the §1 calibrator
reported as the bound. Use the change-point alert for the transition edge and the static
rule for the plateau; they answer different questions.

**They only detect upward movement.** A metric that *falls* under drift will never fire.
This is not hypothetical: a shift that makes the model more confident lowers `entropy_mean`,
and no threshold configuration will catch it, because both tests accumulate only positive
excursions. If downward movement matters for your model, alert on it separately with a
static Alertmanager rule against `guard_entropy_mean`.

### 6.5 Other things worth knowing

- **`alert_hold` is not a config key.** The 10-window latch is `DEFAULT_ALERT_HOLD` in
  `guard/engine/aggregator.py`. To change it, build a `MetricAggregator` yourself and pass it
  to `build_monitor(..., aggregator=...)`.
- **Detection lag has a floor of `window_size × debounce` steps**, plus up to
  `drain_every_k` steps of drain latency and one scrape interval. With the shipped
  `window_size: 32`, `debounce: 3`, `drain_every_k: 32`, that floor is ~130 batches. It is a
  floor, not an estimate: the change-point statistic must also accumulate past `threshold`,
  which takes more windows the smaller the shift. Shrinking these numbers buys speed and
  costs false positives.
- **Population statistics forget.** The embedding detector's running mean and covariance are
  exponentially weighted with a ~69-batch half-life, so `embedding_mean_shift` and
  `embedding_cov_drift` recover on their own after an episode ends. They are not cumulative
  records of everything that ever happened.
- **`observe_in_eval_only` defaults to `True`.** `GuardModule` skips monitoring while the
  wrapped model is in training mode. If your dashboards are empty, confirm the model is
  actually in `.eval()`.
- **Half precision is promoted.** The entropy and divergence detectors promote fp16/bf16
  inputs to fp32 before computing, because `eps = 1e-12` underflows in fp16 and would put
  `log(0)` back into the math. The ring buffer also stores float32 by default regardless of
  the model's compute dtype, so the detectors read fp32 in the normal path.
- **One `GuardMonitor` per device.** There is no multi-GPU aggregation on the hot path; give
  each device its own monitor, ring buffer, and baseline, and aggregate in Prometheus. Note
  that `PrometheusExporter` adds **no device label** — the only labelled metrics are
  `guard_drift_alert{detector}` and `guard_detector_failures_total{detector}` — so distinguish
  devices with a separate scrape target per monitor, a distinct `namespace`, or a
  target-level label added by Prometheus.

---

## Appendix: quick reference

**Config keys** (`guard/config.py`) — `enabled`, `window_size`, `drain_every_k`,
`detectors[].{name,enabled,params}`, `thresholds.<METRIC>.{method,threshold,delta,debounce,cooldown}`,
`export.{enabled,prometheus_port,namespace,push_gateway_url}`.

**Engine knobs not in the config** (`GuardMonitor` / `build_monitor` kwargs) — `max_batch`,
`device`, `dtype`, `ring_slots`, `max_pending_drains`, `max_detector_failures`,
`enable_gpu_timing`, `stream_priority`, `start_drain_thread`, `drain_poll_interval`,
`sink`, `alert_sink`, `aggregator`. (`model_version` and `baseline_version` are also
`GuardMonitor` kwargs, but `build_monitor` fills them from the baseline — pass them yourself
only when constructing `GuardMonitor` directly.)

**Health at a glance:**

```python
s = monitor.stats()
s.enabled, s.device, s.steps_observed, s.observations_dropped, s.summaries_dropped
s.drains_enqueued, s.drains_completed, s.rows_dispatched, s.ring_utilization
s.last_enqueue_us, s.mean_enqueue_us, s.gpu_time_us, s.disabled_detectors
s.ring_bytes, s.pinned_bytes
```

**Alerts to define in Alertmanager** (none of these are shipped; define them yourself):

| Condition | Meaning |
|---|---|
| `guard_drift_alert{detector=~".+"} == 1` | A change-point test fired. |
| `guard_detectors_disabled > 0` | A detector's breaker tripped. |
| `rate(guard_steps_observed_total[5m]) == 0` while serving | Monitoring stopped entirely. |
| `rate(guard_observations_dropped_total[5m]) > 0` | Batches rejected — shape, `max_batch`, or graph capture. |
| `rate(guard_summaries_dropped_total[5m]) > 0` | Export backpressure; §4.2. |
| `guard_ring_utilization > 0.9` for 10m | Monitor stream falling behind inference. |
| `guard_monitor_overhead_us` above your own baseline | Something started synchronising on the hot path. |

**Related documents** — `docs/GUARD_design.md` (architecture and rationale),
`examples/guard.yaml` (annotated config), `examples/quickstart.py` (the whole lifecycle in
one runnable file), `CLAUDE.md` (the invariants this system is built on).
