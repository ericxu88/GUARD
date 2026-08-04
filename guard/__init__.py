"""GUARD: GPU-native unsupervised anomaly & runtime drift monitor.

Typical use — build a baseline offline, then attach the monitor to a live model::

    from guard import (Config, GuardModule, LoggingAlertSink, PrometheusExporter,
                       build_baseline, build_monitor, save)

    baseline = build_baseline(calibration_batches, model_version="vit-b16@abc123",
                              num_classes=1000, embed_dim=768)
    save(baseline, "baselines/vit-b16.guard")

    config = Config.from_yaml("guard.yaml")
    monitor = build_monitor(baseline, config, max_batch=64,
                            sink=PrometheusExporter(), alert_sink=LoggingAlertSink())
    model = GuardModule(model, monitor, embed_layer="encoder.ln", embed_reduce="cls")

Everything after that is automatic: ``model(x)`` is monitored on a low-priority CUDA
stream, metrics land in Prometheus, and change-point alerts fire when live behaviour
departs from the baseline.
"""

from guard.baseline.compute import build_baseline
from guard.baseline.schema import Baseline
from guard.baseline.store import BaselineIntegrityError, load, save
from guard.config import Config, DetectorConfig, ExportConfig, ThresholdConfig
from guard.detectors import (
    DivergenceDetector,
    EmbeddingDriftDetector,
    EntropyDetector,
    build_detectors,
)
from guard.engine import GuardMonitor, MonitorStats, build_monitor
from guard.export import (
    Alert,
    LoggingAlertSink,
    PrometheusExporter,
    WebhookAlertSink,
)
from guard.integration import GuardModule
from guard.safety import Firewall, KillSwitch
from guard.temporal import Cusum, PageHinkley

__version__ = "0.2.0"

__all__ = [
    "Alert",
    "Baseline",
    "BaselineIntegrityError",
    "Config",
    "Cusum",
    "DetectorConfig",
    "DivergenceDetector",
    "EmbeddingDriftDetector",
    "EntropyDetector",
    "ExportConfig",
    "Firewall",
    "GuardModule",
    "GuardMonitor",
    "KillSwitch",
    "LoggingAlertSink",
    "MonitorStats",
    "PageHinkley",
    "PrometheusExporter",
    "ThresholdConfig",
    "WebhookAlertSink",
    "__version__",
    "build_baseline",
    "build_detectors",
    "build_monitor",
    "load",
    "save",
]
