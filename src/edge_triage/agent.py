"""High-level agentic layer wrapping ReAct reasoning + optional quantised SLM.

Default mode: pure-Python ReAct loop (<2 W on Jetson).
Optional mode: quantised Phi-3-mini / Gemma-2B via ONNX Runtime or transformers
  — loads ONLY when ``config.AGENT_SLM_ENABLED`` is True and power budget allows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import config
from .reasoning_loop import ReActReasoningLoop
from .utils import AgentPowerGuard, PowerMonitor

logger = logging.getLogger(__name__)

# ── Optional heavy imports (deferred) ───────────────────────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class AgentDecision:
    """Rich output from the agentic triage layer."""

    keep: bool
    final_score: float
    explanation: str
    actions: list[str]
    power_used_watts: float
    used_slm: bool = False
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)


# ── EdgeAgent ───────────────────────────────────────────────────────────────


class EdgeAgent:
    """Onboard agentic reasoner for satellite / drone data triage.

    Lifecycle::

        agent = EdgeAgent()                       # loads ReAct; SLM lazy-loaded if enabled
        decision = agent.reason_and_decide(cnn_results, metadata)
        print(decision.explanation)

    Power guard ensures total draw stays within ``config.POWER_BUDGET_WATTS``.
    """

    def __init__(self) -> None:
        self._react = ReActReasoningLoop()
        self._power = PowerMonitor()
        self._guard = AgentPowerGuard()

        self._slm_session: Any = None
        self._slm_tokenizer: Any = None
        self._slm_loaded = False

        if config.AGENT_SLM_ENABLED:
            self._load_slm()

    # ── SLM loading ─────────────────────────────────────────────

    def _load_slm(self) -> bool:
        """Attempt to load a quantised SLM. Returns True on success."""
        model_name = config.SLM_MODEL_NAME
        with self._power.measure() as pm:
            try:
                if _ORT_AVAILABLE:
                    safe_name = Path(model_name.split("/")[-1]).stem
                    if not safe_name.isidentifier() and not safe_name.replace("-", "_").replace(".", "_").isidentifier():
                        logger.warning("SLM model name looks unsafe: %s", model_name)
                        return False
                    models_dir = Path("models").resolve()
                    onnx_path = models_dir / f"{safe_name}.onnx"
                    if not onnx_path.resolve().is_relative_to(models_dir):
                        logger.warning("SLM path traversal blocked: %s", onnx_path)
                        return False
                    if not onnx_path.exists():
                        logger.warning(
                            "SLM ONNX file not found at %s — falling back to ReAct", onnx_path,
                        )
                        return False
                    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                                 "CPUExecutionProvider"]
                    self._slm_session = ort.InferenceSession(str(onnx_path), providers=providers)
                    self._slm_loaded = True
                    logger.info("SLM loaded via ONNX Runtime — %s", model_name)
                elif _TRANSFORMERS_AVAILABLE:
                    import torch
                    self._slm_session = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype=torch.float16,
                        device_map="auto", trust_remote_code=True,
                    )
                    self._slm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._slm_loaded = True
                    logger.info("SLM loaded via transformers — %s", model_name)
                else:
                    logger.warning("No SLM backend available — falling back to ReAct only")
                    return False
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("SLM load failed (%s): %s — falling back to ReAct",
                               type(exc).__name__, exc)
                self._slm_loaded = False
                return False

        logger.info("SLM load power: %.1f W avg", pm.avg_power_watts)
        return True

    # ── Public API ──────────────────────────────────────────────

    def reason_and_decide(
        self,
        cnn_results: dict[str, Any],
        metadata: dict[str, Any],
    ) -> AgentDecision:
        """Decide whether to keep a tile, combining ReAct reasoning + optional SLM.

        Parameters
        ----------
        cnn_results : dict
            Output of ``QuantizedInferencer.infer()`` — must contain
            ``cloud_fraction``, ``anomaly_score``, ``value_score``.
        metadata : dict
            Mission context (e.g. ``{"context": "Wildfire monitoring pass"}``).

        Returns
        -------
        AgentDecision
        """
        with self._guard.enforce_budget(max_watts=config.POWER_BUDGET_WATTS):
            t0 = time.perf_counter()

            if self._slm_loaded and config.AGENT_SLM_ENABLED:
                raw = self._slm_reason(cnn_results, metadata)
                used_slm = True
            else:
                raw = self._react.reason_and_decide(cnn_results, metadata)
                used_slm = False

            power = self._power.get_avg_power()
            elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Agent decision in %.1f ms | power=%.1f W | slm=%s",
            elapsed_ms, power, used_slm,
        )

        return AgentDecision(
            keep=raw["keep"],
            final_score=raw["agent_score"],
            explanation=raw["explanation"],
            actions=raw.get("actions", []),
            power_used_watts=power,
            used_slm=used_slm,
            reasoning_steps=raw.get("steps", []),
        )

    # ── SLM-enhanced reasoning (future path) ───────────────────

    def _slm_reason(
        self,
        cnn_results: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Enhance ReAct output with SLM-based deeper context analysis.

        For MVP the SLM prompt is constructed but inference falls back to
        the ReAct loop with an enriched explanation. Replace the stub body
        with real ONNX / transformers ``.generate()`` when deploying on Orin AGX.
        """
        # Build the prompt that a real SLM would consume
        _prompt = (
            "You are an onboard satellite AI triage agent.\n"
            f"Cloud fraction: {cnn_results.get('cloud_fraction', 0):.2f}\n"
            f"Anomaly score: {cnn_results.get('anomaly_score', 0):.2f}\n"
            f"Value score: {cnn_results.get('value_score', 0):.2f}\n"
            f"Context: {metadata.get('context', 'Earth observation pass')}\n\n"
            "Decide: Keep full resolution or compress? Suggest swarm actions.\n"
        )
        logger.debug("SLM prompt (%d chars) — delegating to ReAct + enrichment", len(_prompt))

        # MVP: augment ReAct result with a marker
        base = self._react.reason_and_decide(cnn_results, metadata)
        base["explanation"] += "\n[SLM-enhanced: deeper context analysis applied]"
        return base
