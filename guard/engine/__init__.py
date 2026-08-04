"""The overlap engine: GPU-resident capture, detector dispatch, and async scalar drain."""

from guard.engine.aggregator import MetricAggregator
from guard.engine.memory import HostBuffer, HostBufferPool
from guard.engine.monitor import GuardMonitor, MonitorStats, build_monitor
from guard.engine.ring_buffer import RingBuffer
from guard.engine.streams import (
    LOW_PRIORITY,
    MonitorEvent,
    MonitorStream,
    is_cuda,
    resolve_device,
)

__all__ = [
    "LOW_PRIORITY",
    "GuardMonitor",
    "HostBuffer",
    "HostBufferPool",
    "MetricAggregator",
    "MonitorEvent",
    "MonitorStats",
    "MonitorStream",
    "RingBuffer",
    "build_monitor",
    "is_cuda",
    "resolve_device",
]
