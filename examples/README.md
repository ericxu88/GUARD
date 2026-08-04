# GUARD examples

Two runnable programs and the config they share. Both work on CPU, need no dataset
download, and print the whole lifecycle a monitored deployment goes through: calibrate a
baseline, attach the monitor, serve clean traffic, inject drift, watch the alert fire.

| File | Model | Runtime (CPU) | Downloads | Shows |
|---|---|---|---|---|
| [`quickstart.py`](quickstart.py) | 32-dim toy MLP | ~2 s | none | the full lifecycle incl. persistence and alert recovery |
| [`vit_demo.py`](vit_demo.py) | torchvision ViT-B/16 | ~15 s | none by default | the same integration against a real 1000-class network |
| [`guard.yaml`](guard.yaml) | — | — | — | every config key, with the reasoning for each default |

```bash
python examples/quickstart.py
python examples/vit_demo.py           # needs torchvision: pip install "guard-monitor[demo]"
```

---

## `quickstart.py`

Seven numbered stages against a deliberately tiny model, so nothing in the output is the
model's fault. It is the file to read first.

1. **calibrate** — 32 batches of known-good data → `build_baseline(...)`
2. **persist** — `save()` returns a SHA-256; `load(..., expected_model_version=...)` verifies
   it and refuses a baseline whose bytes changed
3. **attach** — `GuardModule(model, monitor, embed_layer="backbone")`
4. **serve** — 60 in-distribution batches; nothing fires
5. **drift** — 40 batches of gradual covariate shift, then 40 abrupt; alerts fire
6. **recover** — 300 clean batches; the latched alert clears on its own
7. **scrape** — the Prometheus text exposition the exporter would serve

Options: `--device cpu|cuda`, `--baseline PATH` (default `baselines/quickstart.guard`).

Output shape. The seed is fixed at 7, so the metric values below reproduce exactly; only
the µs/step line is machine-dependent.

```
4/7  Serving in-distribution traffic — the quiet case
────────────────────────────────────────────────────────────────────────
  clean        entropy 2.988  JS 0.0000  mahalanobis   5.80  mean-shift   0.31  alerts —
  alerts fired     0  (a monitor that cries wolf gets turned off)
...
  gradual      entropy 2.973  JS 0.0056  mahalanobis  21.90  mean-shift   4.32  alerts ['embedding']
  abrupt       entropy 2.894  JS 0.0288  mahalanobis  51.36  mean-shift  23.24  alerts ['divergence']
  recovered    entropy 2.988  JS 0.0000  mahalanobis   5.67  mean-shift   0.94  alerts —
...
  guard_embedding_mahalanobis_mean 5.673207640647888
  guard_drift_alert{detector="embedding"} 0.0

engine: 440 steps observed, 440 rows exported, 0 dropped, <n> µs/step on the inference thread
failures: none — every detector healthy
```

The two things worth watching: `mahalanobis` rises ~9x during stage 5 and returns to its
clean level in stage 6, and `alerts` latches then clears. An alert that never clears is an
alert people mute.

## `vit_demo.py`

The reference target from `CLAUDE.md`: torchvision's ViT-B/16, 1000 classes, a 768-wide
class-token embedding. Identical integration to `quickstart.py`, at production width.

```bash
python examples/vit_demo.py                                   # random weights, offline
python examples/vit_demo.py --pretrained                      # ImageNet-1k weights (~330 MB)
python examples/vit_demo.py --device cuda --batch 32 --steps 40
```

Options: `--device` (default `cpu`), `--batch` (default 8), `--steps` (batches per phase,
default 12, minimum 6 — below that too few aggregation windows close for the alert to clear
its debounce, and the run would look broken when it is only starved), `--pretrained`.

What it establishes that `quickstart.py` cannot:

- **`embed_layer="encoder.ln"`, `embed_reduce="cls"` is the correct tap for this model.**
  `encoder.ln` emits `[B, 197, 768]`; torchvision's ViT forward is
  `x = self.encoder(x); x = x[:, 0]; x = self.heads(x)`, so token 0 is exactly the vector
  the classifier head consumes. `embed_reduce="cls"` takes it. The demo prints the captured
  shape so you can see it rather than trust it.
- **Thresholds are not portable.** Mahalanobis distance scales with embedding dimension and
  with how well-conditioned the reference covariance is. The demo measures the clean level
  on held-out images the baseline never saw and derives the CUSUM boundary from it
  (`delta = 0.25 x clean`, `threshold = 2.0 x clean`) instead of reusing `guard.yaml`'s 8.0,
  which was chosen for a 32-dim toy. Do the same with your own traffic.

Weights are **random by default** so the example runs offline and in CI. An untrained
classifier head emits near-zero logits, so its softmax is essentially uniform: entropy sits
pinned at `log(1000) = 6.9078` and KL/JS at 0 no matter what you feed the model. On that
path the embedding detector is the only one carrying signal — which the demo says out loud
rather than hiding. `--pretrained` gives the head real weights and the output-distribution
metrics move too.

Output shape:

```
1/5  Calibrating a baseline from clean traffic
──────────────────────────────────────────────────────────────────────────────
  embed layer      'encoder.ln' -> class token (8, 768)
  logits           (8, 1000)
  samples          96
  precision Σ⁻¹    (768, 768) float32
...
4/5  Serving clean traffic — the quiet case
  clean        entropy 6.9078  JS 0.00000  mahalanobis    394.1  mean-shift     64.2  alerts —
  alerts fired     0

5/5  Corrupting the input — noise sigma=0.5, brightness +0.4
  corrupted    entropy 6.9078  JS 0.00000  mahalanobis   2473.6  mean-shift   1271.7  alerts ['embedding']
  alerts fired     1
```

With `--pretrained` the output-distribution detectors wake up and both alerts fire:

```
  clean        entropy 4.4851  JS 0.00371  mahalanobis    613.6  mean-shift    117.1  alerts —
  corrupted    entropy 4.8554  JS 0.38467  mahalanobis   4228.3  mean-shift   2144.2  alerts ['divergence', 'embedding']
```

Both examples end with a note on what changes under CUDA: on CPU every stream and event is
a no-op and the detectors run inline, so the printed µs/step is the whole cost of the
detector math paid serially on the calling thread. On CUDA `observe()` only records an event
and enqueues kernels onto a low-priority stream, so the math runs in the SM capacity
inference leaves idle and the µs/step drops to the enqueue cost. **The metric values are
identical; only the cost changes.** The p99 < 2 % overhead figure in the design doc is an
SLO to be verified on hardware — see `benchmarks/ab_latency.py` and
`benchmarks/nsight_overlap.sh` — not something these examples measure.

---

## Drop-in integration

The whole surface, against a model you already have and a baseline you already built (see
stages 1-2 of `quickstart.py` for how to build one):

```python
from guard import (Config, GuardModule, LoggingAlertSink, PrometheusExporter,
                   build_monitor, load)

baseline = load("baselines/vit-b16.guard", expected_model_version="vit_b_16@imagenet1k_v1")
config = Config.from_yaml("guard.yaml")

exporter = PrometheusExporter(namespace=config.export.namespace)
exporter.start_http_server(config.export.prometheus_port)     # serves /metrics

monitor = build_monitor(baseline, config, max_batch=64, device="cuda",
                        sink=exporter, alert_sink=LoggingAlertSink())
model = GuardModule(model.eval(), monitor, embed_layer="encoder.ln", embed_reduce="cls")

logits = model(images)     # identical output, identical dtype, now monitored
model.close()              # on shutdown: removes the hook, stops the drain thread
```

Notes on the three things people get wrong:

- **`embed_layer`** is the submodule whose *output* is the feature vector the classifier
  head reads — `"encoder.ln"` for torchvision ViT, `"avgpool"` for a ResNet, usually
  `"<base>.pooler"` for a HuggingFace sequence classifier. Call
  `guard.integration.candidate_embed_layers(model)` to list plausible names, then do one
  dry-run batch and check the shape.
- **`max_batch`** must be at least the largest batch you will ever serve. Bigger batches are
  dropped and counted in `guard_observations_dropped_total`, never truncated and never
  raised.
- **thresholds** in `guard.yaml` are illustrative. Calibrate each one above the p99.9 you
  observe on your own known-good traffic, exactly as `vit_demo.py` stage 2 does.

`model(images)` returns whatever the wrapped model returned, unchanged, on the same code
path, even if every detector is broken — `observe()` is fire-and-forget and each call site
sits behind an exception firewall. Set `GUARD_DISABLED=1` in the environment to neutralise
monitoring in a running deployment without a code change.
