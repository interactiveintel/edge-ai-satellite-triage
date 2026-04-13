"""Core triage pipeline — ingest -> CNN inference -> agentic reasoning -> decision.

Designed for <20 W total on NVIDIA Jetson Orin Nano / AGX.
Includes timeout enforcement, error recovery, audit logging, and provenance.
"""

from __future__ import annotations

import logging
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agent import AgentDecision, EdgeAgent
from .audit import AuditLogger
from .config import config
from .data_ingest import ImageIngestor
from .inference import QuantizedInferencer
from .utils import AgentPowerGuard, PowerMonitor

logger = logging.getLogger(__name__)


# ── Timeout helper ──────────────────────────────────────────────────────────


class _InferenceTimeout(Exception):
    """Raised when inference exceeds MAX_INFERENCE_MS."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _InferenceTimeout(f"Inference exceeded {config.MAX_INFERENCE_MS} ms ceiling")


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class TriageResult:
    """Full decision record with provenance for audit trail."""

    # Decision
    keep: bool
    final_score: float
    bandwidth_saved_percent: float
    power_used_watts: float

    # CNN layer
    cnn_results: dict[str, Any]

    # Agent layer
    agent_decision: AgentDecision | None = None
    explanation: str = ""
    actions: list[str] = field(default_factory=list)

    # Provenance
    tile_id: str = ""
    scene_id: str = ""
    processing_timestamp_utc: str = ""
    input_hash: str = ""
    inference_timed_out: bool = False
    error: str | None = None


# ── Engine ──────────────────────────────────────────────────────────────────


class EdgeTriageEngine:
    """Main edge AI triage engine with agentic reasoning, audit, and error recovery.

    Usage::

        engine = EdgeTriageEngine()
        result = engine.process_tile(image_array, {"context": "Wildfire pass"})
    """

    def __init__(self, audit: bool | None = None) -> None:
        self.ingestor = ImageIngestor()
        self.inferencer = QuantizedInferencer()
        self.agent = EdgeAgent() if config.AGENT_ENABLED else None
        self.power_monitor = PowerMonitor()
        self.power_guard = AgentPowerGuard()

        enable_audit = audit if audit is not None else config.AUDIT_ENABLED
        self._audit: AuditLogger | None = AuditLogger() if enable_audit else None

    def process_tile(
        self,
        image: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> TriageResult:
        """Full pipeline: ingest -> CNN -> optional agent -> final decision.

        Parameters
        ----------
        image : np.ndarray
            Raw image array (H, W, C), (C, H, W), or (H, W) grayscale.
        metadata : dict, optional
            Mission context.  Recognised keys:
            ``scene_id``, ``tile_id``, ``tile_coords``, ``acquisition_time``,
            ``context``.
        """
        if metadata is None:
            metadata = {"context": "Standard Earth observation pass"}

        tile_id = metadata.get("tile_id") or uuid.uuid4().hex[:12]
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # ── 1. Preprocess (with error recovery) ────────────────
        try:
            tile = self.ingestor.preprocess(image)
        except Exception as exc:
            logger.error("Tile %s preprocess failed: %s", tile_id, exc)
            return self._error_result(tile_id, ts_utc, metadata, f"preprocess: {exc}")

        # ── 2. CNN inference (with timeout) ─────────────────────
        cnn_results: dict[str, Any]
        timed_out = False
        try:
            cnn_results = self._infer_with_timeout(tile)
        except _InferenceTimeout:
            logger.warning("Tile %s: inference timed out (>%.0f ms)", tile_id, config.MAX_INFERENCE_MS)
            cnn_results = {"cloud_fraction": 0.5, "anomaly_score": 0.0,
                           "value_score": 0.0, "inference_ms": config.MAX_INFERENCE_MS,
                           "power_watts": 0.0, "backend": "timeout"}
            timed_out = True
        except Exception as exc:
            logger.error("Tile %s inference failed: %s", tile_id, exc)
            return self._error_result(tile_id, ts_utc, metadata, f"inference: {exc}")

        # ── 3. Agentic reasoning ────────────────────────────────
        agent_decision: AgentDecision | None = None
        with self.power_guard.enforce_budget(max_watts=config.POWER_BUDGET_WATTS):
            if (
                self.agent
                and not timed_out
                and cnn_results.get("value_score", 0) > config.AGENT_ACTIVATION_THRESHOLD
            ):
                try:
                    agent_decision = self.agent.reason_and_decide(cnn_results, metadata)
                except Exception as exc:
                    logger.warning("Tile %s agent failed: %s — falling back to CNN", tile_id, exc)

        # ── 4. Combine into final decision ──────────────────────
        if agent_decision:
            final_keep = agent_decision.keep
            final_score = agent_decision.final_score
            explanation = agent_decision.explanation
            actions = agent_decision.actions
        else:
            raw_score = cnn_results.get("value_score", 0)
            final_keep = raw_score > config.KEEP_SCORE_THRESHOLD
            final_score = raw_score
            explanation = "CNN-only decision (agent disabled or low-value tile)"
            actions = []

        # Guard against NaN/Inf from bad sensor data — default to safe FILTER
        if not np.isfinite(final_score):
            final_score = 0.0
            final_keep = False
            explanation = "NaN/Inf detected in scores — filtering as safety measure"

        # ── 5. Bandwidth + power ────────────────────────────────
        cloud = cnn_results.get("cloud_fraction", 0)
        bandwidth_saved = 95.0 if not final_keep else max(60.0, 85.0 - cloud * 30)
        power_used = self.power_monitor.get_avg_power()

        result = TriageResult(
            keep=final_keep,
            final_score=final_score,
            bandwidth_saved_percent=bandwidth_saved,
            power_used_watts=power_used,
            cnn_results=cnn_results,
            agent_decision=agent_decision,
            explanation=explanation,
            actions=actions,
            tile_id=tile_id,
            scene_id=metadata.get("scene_id", "unknown"),
            processing_timestamp_utc=ts_utc,
            input_hash=self._fast_hash(image),
            inference_timed_out=timed_out,
        )

        # ── 6. Audit log ───────────────────────────────────────
        if self._audit:
            try:
                self._audit.log_decision(image.tobytes(), result, metadata)
            except Exception:
                logger.warning("Audit write failed for tile %s", tile_id, exc_info=True)

        logger.info(
            "Triage %s: %s | score=%.3f | power=%.1fW | bw_saved=%.1f%% | backend=%s",
            tile_id, "KEEP" if final_keep else "FILTER",
            final_score, power_used, bandwidth_saved,
            cnn_results.get("backend", "?"),
        )

        return result

    # ── Helpers ─────────────────────────────────────────────────

    def _infer_with_timeout(self, tile: np.ndarray) -> dict[str, Any]:
        """Run inference with a wall-clock timeout ceiling."""
        timeout_sec = config.MAX_INFERENCE_MS / 1000.0
        # SIGALRM only works on Unix; fall back to post-check on other platforms
        import threading
        use_signal = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
        if use_signal:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
            t0 = time.perf_counter()
            result = self.inferencer.infer(tile)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if not use_signal and elapsed_ms > config.MAX_INFERENCE_MS:
                logger.warning("Inference took %.0f ms (limit %.0f ms)", elapsed_ms, config.MAX_INFERENCE_MS)
            return result
        finally:
            if use_signal:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

    @staticmethod
    def _fast_hash(arr: np.ndarray) -> str:
        """xxHash-style fast hash for provenance (falls back to md5 for speed)."""
        import hashlib
        return hashlib.md5(arr.tobytes()).hexdigest()

    @staticmethod
    def _error_result(
        tile_id: str, ts: str, metadata: dict[str, Any], error_msg: str,
    ) -> TriageResult:
        """Return a safe FILTER result when processing fails."""
        return TriageResult(
            keep=False,
            final_score=0.0,
            bandwidth_saved_percent=95.0,
            power_used_watts=0.0,
            cnn_results={"backend": "error", "cloud_fraction": 0, "anomaly_score": 0,
                         "value_score": 0, "inference_ms": 0, "power_watts": 0},
            explanation=f"Error during processing: {error_msg}",
            tile_id=tile_id,
            scene_id=metadata.get("scene_id", "unknown"),
            processing_timestamp_utc=ts,
            error=error_msg,
        )


def triage_image(
    image: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> TriageResult:
    """Convenience one-shot function for quick testing."""
    return EdgeTriageEngine(audit=False).process_tile(image, metadata)
