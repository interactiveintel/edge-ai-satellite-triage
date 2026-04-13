"""Bandwidth, power, and TOPS/Watt metrics for the triage pipeline.

Tracks before/after data volume, cumulative power draw, and per-tile statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import config
from .triage import TriageResult


@dataclass
class SessionMetrics:
    """Accumulates metrics across a batch of tiles."""

    tiles_processed: int = 0
    tiles_kept: int = 0
    tiles_filtered: int = 0
    total_input_bytes: int = 0
    total_output_bytes: int = 0
    total_power_joules: float = 0.0
    total_inference_ms: float = 0.0
    per_tile: list[dict[str, Any]] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time, repr=False)

    # ── Derived properties ──────────────────────────────────────

    @property
    def bandwidth_saved_percent(self) -> float:
        if self.total_input_bytes == 0:
            return 0.0
        return (1.0 - self.total_output_bytes / self.total_input_bytes) * 100.0

    @property
    def avg_power_watts(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self.total_power_joules / elapsed

    @property
    def avg_inference_ms(self) -> float:
        if self.tiles_processed == 0:
            return 0.0
        return self.total_inference_ms / self.tiles_processed

    @property
    def keep_rate(self) -> float:
        if self.tiles_processed == 0:
            return 0.0
        return self.tiles_kept / self.tiles_processed

    def summary(self) -> dict[str, Any]:
        return {
            "tiles_processed": self.tiles_processed,
            "tiles_kept": self.tiles_kept,
            "tiles_filtered": self.tiles_filtered,
            "keep_rate": round(self.keep_rate, 3),
            "bandwidth_saved_percent": round(self.bandwidth_saved_percent, 1),
            "total_input_MB": round(self.total_input_bytes / 1e6, 2),
            "total_output_MB": round(self.total_output_bytes / 1e6, 2),
            "avg_power_watts": round(self.avg_power_watts, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
        }


class MetricsCollector:
    """Collect and aggregate triage metrics across a session.

    Usage::

        collector = MetricsCollector()
        for tile in tiles:
            result = engine.process_tile(tile)
            collector.record(tile, result)
        print(collector.metrics.summary())
    """

    def __init__(self) -> None:
        self.metrics = SessionMetrics()

    def record(self, input_tile: np.ndarray, result: TriageResult) -> None:
        """Record metrics for one tile."""
        input_bytes = input_tile.nbytes
        # Kept tiles transmit full res; filtered tiles transmit a 64x64 JPEG thumbnail (~4 KB)
        output_bytes = input_bytes if result.keep else 4096

        self.metrics.tiles_processed += 1
        if result.keep:
            self.metrics.tiles_kept += 1
        else:
            self.metrics.tiles_filtered += 1
        self.metrics.total_input_bytes += input_bytes
        self.metrics.total_output_bytes += output_bytes
        self.metrics.total_inference_ms += result.cnn_results.get("inference_ms", 0.0)
        self.metrics.total_power_joules += (
            result.power_used_watts * result.cnn_results.get("inference_ms", 50.0) / 1000.0
        )
        self.metrics.per_tile.append({
            "keep": result.keep,
            "score": result.final_score,
            "cloud": result.cnn_results.get("cloud_fraction", 0),
            "power_w": result.power_used_watts,
            "bw_saved": result.bandwidth_saved_percent,
            "agent": result.agent_decision is not None,
        })

    def reset(self) -> None:
        self.metrics = SessionMetrics()
