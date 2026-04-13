"""Power monitoring, timing helpers, and logging for edge triage on Jetson hardware."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from .config import config

logger = logging.getLogger(__name__)


# ── Power reading dataclass ─────────────────────────────────────────────────


@dataclass
class PowerReading:
    """Single snapshot of board power state."""

    timestamp: float
    power_watts: float
    gpu_utilization_pct: float | None = None
    cpu_utilization_pct: float | None = None
    temp_celsius: float | None = None


# ── PowerMonitor ────────────────────────────────────────────────────────────


class PowerMonitor:
    """Read real-time power from Jetson ``tegrastats`` / ``jtop``, with CPU-timing fallback.

    Usage::

        monitor = PowerMonitor()

        # One-shot reading
        watts = monitor.get_current_power()

        # Context-manager for a timed measurement window
        with monitor.measure() as session:
            run_inference(tile)
        print(session.avg_power_watts, session.elapsed_seconds)
    """

    def __init__(self, interval_ms: int | None = None) -> None:
        self._interval_ms = interval_ms or config.JETSON_TEGRASTATS_INTERVAL_MS
        self._backend = self._detect_backend()
        self._readings: list[PowerReading] = []
        self._sampling = False
        self._thread: threading.Thread | None = None
        self._fallback_warned = False
        if self._backend == "cpu_timer":
            logger.warning(
                "PowerMonitor: no hardware power source (tegrastats/jtop) found. "
                "Power readings will be 0.0 W. Install jetson-stats or run on Jetson "
                "for real measurements."
            )
        else:
            logger.info("PowerMonitor initialised — backend: %s", self._backend)

    # ── Backend detection ───────────────────────────────────────

    @staticmethod
    def _detect_backend() -> str:
        """Pick the best available power-reading backend."""
        if shutil.which("tegrastats"):
            return "tegrastats"
        try:
            import jtop as _jtop  # noqa: F401
            return "jtop"
        except ImportError:
            pass
        return "cpu_timer"

    @property
    def is_hardware_backed(self) -> bool:
        """True if reading from real hardware, not CPU-timer fallback."""
        return self._backend != "cpu_timer"

    @property
    def backend(self) -> str:
        return self._backend

    # ── Public API ──────────────────────────────────────────────

    def get_current_power(self) -> float:
        """Return instantaneous board power in Watts (0.0 on fallback)."""
        if self._backend == "tegrastats":
            return self._read_tegrastats()
        if self._backend == "jtop":
            return self._read_jtop()
        return 0.0  # CPU-timer fallback has no real power data

    def get_avg_power(self) -> float:
        """Average power across all readings collected during a ``measure()`` window."""
        if not self._readings:
            return 0.0
        return sum(r.power_watts for r in self._readings) / len(self._readings)

    @contextmanager
    def measure(self) -> Generator[_MeasureSession, None, None]:
        """Context-manager that samples power in the background while user code runs.

        Yields a :class:`_MeasureSession` whose ``avg_power_watts`` and
        ``elapsed_seconds`` are populated on exit.
        """
        session = _MeasureSession()
        self._readings.clear()
        self._sampling = True

        if self._backend != "cpu_timer":
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

        session._start = time.perf_counter()
        try:
            yield session
        finally:
            session._end = time.perf_counter()
            self._sampling = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)
                self._thread = None

            session.elapsed_seconds = session._end - session._start
            session.avg_power_watts = self.get_avg_power()
            session.readings = list(self._readings)

    # ── Internal sampling loop ──────────────────────────────────

    def _sample_loop(self) -> None:
        interval = self._interval_ms / 1000.0
        while self._sampling:
            try:
                watts = self.get_current_power()
                self._readings.append(
                    PowerReading(timestamp=time.time(), power_watts=watts)
                )
            except Exception:
                logger.debug("Power sample failed", exc_info=True)
            time.sleep(interval)

    # ── tegrastats parser ───────────────────────────────────────

    def _read_tegrastats(self) -> float:
        """Run one shot of ``tegrastats`` and parse total board power (VDD_IN / POM_5V_IN)."""
        try:
            proc = subprocess.run(
                ["tegrastats", "--interval", str(self._interval_ms)],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout else ""
            return self._parse_tegrastats_power(line)
        except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
            logger.warning("tegrastats read failed — returning 0.0 W")
            return 0.0

    @staticmethod
    def _parse_tegrastats_power(line: str) -> float:
        """Extract board-level power in Watts from a tegrastats output line.

        Handles both Orin-style ``VDD_IN 5500mW/5500mW`` and
        older ``POM_5V_IN 4800/4800`` formats.
        """
        # Match patterns like "VDD_IN 5500mW/5500mW" or "POM_5V_IN 4800/4800"
        for pattern in (
            r"VDD_IN\s+(\d+)mW",
            r"POM_5V_IN\s+(\d+)",
            r"VDD_CPU_GPU_CV\s+(\d+)mW",
        ):
            match = re.search(pattern, line)
            if match:
                return int(match.group(1)) / 1000.0
        return 0.0

    # ── jtop reader ─────────────────────────────────────────────

    @staticmethod
    def _read_jtop() -> float:
        """Read power via the ``jetson-stats`` jtop library."""
        try:
            from jtop import jtop

            with jtop() as jetson:
                # jtop.power returns dict; total board power is under "tot"
                power_info = jetson.power
                if "tot" in power_info:
                    return power_info["tot"].get("power", 0) / 1000.0
                # Fallback: sum all rail values
                total_mw = sum(
                    v.get("power", 0)
                    for v in power_info.values()
                    if isinstance(v, dict)
                )
                return total_mw / 1000.0
        except Exception:
            logger.warning("jtop read failed — returning 0.0 W")
            return 0.0


# ── Measure session (returned by PowerMonitor.measure()) ────────────────────


@dataclass
class _MeasureSession:
    """Accumulates stats during a ``PowerMonitor.measure()`` window."""

    avg_power_watts: float = 0.0
    elapsed_seconds: float = 0.0
    readings: list[PowerReading] = field(default_factory=list)
    _start: float = field(default=0.0, repr=False)
    _end: float = field(default=0.0, repr=False)

    @property
    def tops_per_watt(self) -> float:
        """Estimated TOPS/Watt (placeholder — real value comes from TensorRT profiler)."""
        if self.avg_power_watts <= 0:
            return 0.0
        # Orin Nano INT8 peak is ~100 TOPS at ~15 W ≈ 6.67 TOPS/W board-level.
        # Per-accelerator figures (DLA) reach 40+ TOPS/W.
        # This is a placeholder; replace with actual TensorRT engine stats.
        return 0.0


# ── Convenience: AgentPowerGuard ────────────────────────────────────────────


class AgentPowerGuard:
    """Enforce a hard power ceiling during inference/agent reasoning.

    Usage::

        guard = AgentPowerGuard()
        with guard.enforce_budget(max_watts=15.0):
            run_expensive_stuff()
    """

    def __init__(self) -> None:
        self._monitor = PowerMonitor()

    @contextmanager
    def enforce_budget(self, max_watts: float | None = None) -> Generator[None, None, None]:
        """Log a warning if average power exceeds *max_watts* during the block."""
        budget = max_watts or config.POWER_BUDGET_WATTS
        with self._monitor.measure() as session:
            yield
        if session.avg_power_watts > budget:
            logger.warning(
                "Power budget exceeded: %.1f W avg vs %.1f W budget",
                session.avg_power_watts,
                budget,
            )
