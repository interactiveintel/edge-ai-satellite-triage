"""Lightweight pure-Python ReAct-style reasoning loop for onboard agentic triage.

No external LLM dependency — runs in <2 W when idle on Jetson Orin.
Sits on top of CNN scoring, adds plan -> act -> observe -> decide behaviour,
and optionally hooks a quantised SLM later via ``agent.py``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import config
from .utils import PowerMonitor

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """One think -> act -> observe cycle."""

    step: int
    thought: str
    action: str
    observation: str
    elapsed_ms: float = 0.0


@dataclass
class ReActResult:
    """Final output of the reasoning loop."""

    keep: bool
    agent_score: float
    explanation: str
    actions: list[str]
    steps: list[AgentStep] = field(default_factory=list)
    power_used_watts: float = 0.0
    total_ms: float = 0.0


# ── Built-in lightweight tools ──────────────────────────────────────────────


def _tool_assess_urgency(cnn: dict[str, Any], meta: dict[str, Any]) -> str:
    """Heuristic urgency assessment from CNN scores + metadata context."""
    value = cnn.get("value_score", 0.0)
    anomaly = cnn.get("anomaly_score", 0.0)
    context = meta.get("context", "").lower()

    urgency_keywords = {"wildfire", "fire", "flood", "earthquake", "oil spill", "disaster", "defense"}
    context_boost = any(kw in context for kw in urgency_keywords)

    if anomaly > 0.7 or context_boost:
        return f"HIGH urgency — anomaly={anomaly:.2f}, context_match={context_boost}"
    if value > 0.5:
        return f"MEDIUM urgency — value={value:.2f}"
    return f"LOW urgency — value={value:.2f}, anomaly={anomaly:.2f}"


def _tool_check_constellation_status() -> str:
    """Stub: query neighbouring satellite availability for follow-up imaging."""
    return "Constellation nominal — 2 satellites within re-image window (next 12 min)"


def _tool_generate_explanation(steps: list[AgentStep], keep: bool) -> str:
    """Compose a human-readable explanation from reasoning steps."""
    lines = [f"Decision: {'KEEP full resolution' if keep else 'FILTER / compress'}"]
    for s in steps:
        lines.append(f"  Step {s.step}: {s.thought} -> {s.action} [{s.observation}]")
    return "\n".join(lines)


def _tool_suggest_followup(cnn: dict[str, Any], meta: dict[str, Any]) -> str:
    """Suggest swarm / follow-up actions when a high-value tile is detected."""
    if cnn.get("anomaly_score", 0) > 0.6:
        return "TRIGGER follow-up imaging on nearest constellation member"
    return "No follow-up needed"


# ── Core reasoning loop ─────────────────────────────────────────────────────


class ReActReasoningLoop:
    """Stateless ReAct loop capped at ``config.AGENT_MAX_STEPS`` iterations.

    Each step consumes <0.5 W on Orin (pure Python, no GPU).
    """

    def __init__(self) -> None:
        self._power = PowerMonitor()

    def reason_and_decide(
        self,
        cnn_results: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the full think-act-observe loop and return a decision dict.

        Returns
        -------
        dict
            Keys: ``keep``, ``agent_score``, ``explanation``, ``actions``,
            ``steps``, ``power_used_watts``, ``total_ms``.
        """
        max_steps = config.AGENT_MAX_STEPS
        steps: list[AgentStep] = []
        actions_taken: list[str] = []

        with self._power.measure() as pm:
            t0 = time.perf_counter()

            # ── Step 1: Assess urgency ──────────────────────────
            t_step = time.perf_counter()
            thought = (
                f"Assess tile urgency — cloud={cnn_results.get('cloud_fraction', 0):.2f}, "
                f"anomaly={cnn_results.get('anomaly_score', 0):.2f}, "
                f"value={cnn_results.get('value_score', 0):.2f}"
            )
            observation = _tool_assess_urgency(cnn_results, metadata)
            steps.append(AgentStep(
                step=1, thought=thought,
                action="assess_urgency", observation=observation,
                elapsed_ms=(time.perf_counter() - t_step) * 1000,
            ))

            # ── Step 2: Decide priority + compute final score ───
            t_step = time.perf_counter()
            cloud = cnn_results.get("cloud_fraction", 0.0)
            value = cnn_results.get("value_score", 0.0)
            anomaly = cnn_results.get("anomaly_score", 0.0)

            agent_score = (
                value * config.VALUE_WEIGHT
                + anomaly * config.ANOMALY_WEIGHT
            ) * (1.0 - cloud * config.CLOUD_PENALTY_WEIGHT)

            # Boost for high-urgency context
            if "HIGH" in observation:
                agent_score = min(agent_score * 1.3, 1.0)

            keep = agent_score > config.KEEP_SCORE_THRESHOLD and cloud < config.MAX_CLOUD_FRACTION

            action_desc = "KEEP full resolution + flag for downlink" if keep else "COMPRESS to thumbnail"
            steps.append(AgentStep(
                step=2,
                thought=f"Weigh bandwidth cost vs science/defense value — score={agent_score:.3f}",
                action="decide_priority", observation=f"Decision: {action_desc}",
                elapsed_ms=(time.perf_counter() - t_step) * 1000,
            ))

            # ── Step 3: Suggest follow-up (only if budget allows) ─
            if len(steps) < max_steps:
                t_step = time.perf_counter()
                followup = _tool_suggest_followup(cnn_results, metadata)
                steps.append(AgentStep(
                    step=3,
                    thought="Check if swarm / follow-up imaging is warranted",
                    action="suggest_followup", observation=followup,
                    elapsed_ms=(time.perf_counter() - t_step) * 1000,
                ))
                if "TRIGGER" in followup:
                    actions_taken.append(followup)

            actions_taken.insert(0, action_desc)
            total_ms = (time.perf_counter() - t0) * 1000

        explanation = _tool_generate_explanation(steps, keep)

        result = ReActResult(
            keep=keep,
            agent_score=float(agent_score),
            explanation=explanation,
            actions=actions_taken,
            steps=steps,
            power_used_watts=pm.avg_power_watts,
            total_ms=total_ms,
        )

        logger.debug("ReAct loop done in %.1f ms — keep=%s, score=%.3f", total_ms, keep, agent_score)

        # Return plain dict for compatibility with agent.py
        return {
            "keep": result.keep,
            "agent_score": result.agent_score,
            "explanation": result.explanation,
            "actions": result.actions,
            "steps": [{"step": s.step, "thought": s.thought, "action": s.action,
                        "observation": s.observation} for s in result.steps],
            "power_used_watts": result.power_used_watts,
            "total_ms": result.total_ms,
        }
