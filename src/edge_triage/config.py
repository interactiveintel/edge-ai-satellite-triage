"""Hardware constants, power budgets, and model paths for edge triage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TriageConfig:
    """Runtime configuration for the edge triage pipeline.

    Mutable so the Streamlit dashboard and CLI flags can override defaults.
    All power values in Watts. All performance targets reflect
    NVIDIA Jetson Orin Nano / AGX class hardware.

    Raises ``ValueError`` on construction if any value is out of valid range.
    """

    # ── Power budget ────────────────────────────────────────────
    POWER_BUDGET_WATTS: float = 15.0
    POWER_MODE_LOW_WATTS: float = 7.0
    POWER_MODE_HIGH_WATTS: float = 25.0

    # ── Performance targets ─────────────────────────────────────
    TARGET_TOPS_WATT: float = 40.0          # INT8 on Orin Nano ≈ 40+ TOPS/W
    MAX_INFERENCE_MS: float = 50.0           # per-tile latency ceiling
    TARGET_BANDWIDTH_SAVINGS: float = 0.85   # 85 % downlink reduction

    # ── Model paths (relative to project root) ─────────────────
    MODEL_DIR: Path = field(default_factory=lambda: Path("models"))
    DEFAULT_MODEL: str = "cloud_mask_mobilenet_int8.onnx"
    FALLBACK_MODEL: str = "cloud_mask_mobilenet_fp16.onnx"

    # ── Triage thresholds ───────────────────────────────────────
    KEEP_SCORE_THRESHOLD: float = 0.65
    MAX_CLOUD_FRACTION: float = 0.85
    CLOUD_PENALTY_WEIGHT: float = 0.8
    VALUE_WEIGHT: float = 0.7
    ANOMALY_WEIGHT: float = 0.3

    # ── Agentic reasoning ─────────────────────────────────────
    AGENT_ENABLED: bool = True
    AGENT_ACTIVATION_THRESHOLD: float = 0.6
    AGENT_MAX_STEPS: int = 3
    AGENT_SLM_ENABLED: bool = False   # True only on AGX Orin 64 GB / Thor
    SLM_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"

    # ── Dual-use mode ──────────────────────────────────────────
    MODE: str = "space"               # "space" | "ground"

    # ── Image ingest defaults ───────────────────────────────────
    TILE_SIZE: int = 256
    INPUT_CHANNELS: int = 13          # Sentinel-2 multispectral
    RGB_CHANNELS: tuple[int, ...] = (4, 3, 2)  # Sentinel-2 true-colour band indices

    # ── Hardware detection ──────────────────────────────────────
    JETSON_TEGRASTATS_PATH: str = "/usr/bin/tegrastats"
    JETSON_TEGRASTATS_INTERVAL_MS: int = 100

    # ── Audit / provenance ─────────────────────────────────────
    AUDIT_LOG_PATH: Path = field(default_factory=lambda: Path("logs/audit.jsonl"))
    AUDIT_ENABLED: bool = True

    def __post_init__(self) -> None:
        """Validate all config values are within sane bounds."""
        errors: list[str] = []

        def _check(name: str, val: float, lo: float, hi: float) -> None:
            if not (lo <= val <= hi):
                errors.append(f"{name}={val} out of range [{lo}, {hi}]")

        _check("POWER_BUDGET_WATTS", self.POWER_BUDGET_WATTS, 1.0, 100.0)
        _check("POWER_MODE_LOW_WATTS", self.POWER_MODE_LOW_WATTS, 1.0, 50.0)
        _check("POWER_MODE_HIGH_WATTS", self.POWER_MODE_HIGH_WATTS, 1.0, 150.0)
        _check("TARGET_TOPS_WATT", self.TARGET_TOPS_WATT, 0.1, 1000.0)
        _check("MAX_INFERENCE_MS", self.MAX_INFERENCE_MS, 1.0, 60000.0)
        _check("TARGET_BANDWIDTH_SAVINGS", self.TARGET_BANDWIDTH_SAVINGS, 0.0, 1.0)
        _check("KEEP_SCORE_THRESHOLD", self.KEEP_SCORE_THRESHOLD, 0.0, 1.0)
        _check("MAX_CLOUD_FRACTION", self.MAX_CLOUD_FRACTION, 0.0, 1.0)
        _check("CLOUD_PENALTY_WEIGHT", self.CLOUD_PENALTY_WEIGHT, 0.0, 1.0)
        _check("VALUE_WEIGHT", self.VALUE_WEIGHT, 0.0, 1.0)
        _check("ANOMALY_WEIGHT", self.ANOMALY_WEIGHT, 0.0, 1.0)
        _check("AGENT_ACTIVATION_THRESHOLD", self.AGENT_ACTIVATION_THRESHOLD, 0.0, 1.0)

        if self.AGENT_MAX_STEPS < 1 or self.AGENT_MAX_STEPS > 20:
            errors.append(f"AGENT_MAX_STEPS={self.AGENT_MAX_STEPS} out of range [1, 20]")
        if self.TILE_SIZE < 16 or self.TILE_SIZE > 4096:
            errors.append(f"TILE_SIZE={self.TILE_SIZE} out of range [16, 4096]")
        if self.INPUT_CHANNELS < 1 or self.INPUT_CHANNELS > 256:
            errors.append(f"INPUT_CHANNELS={self.INPUT_CHANNELS} out of range [1, 256]")
        if self.MODE not in ("space", "ground"):
            errors.append(f"MODE='{self.MODE}' must be 'space' or 'ground'")

        if errors:
            raise ValueError("Invalid TriageConfig:\n  " + "\n  ".join(errors))

    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / self.DEFAULT_MODEL


# Module-level singleton — import as `from edge_triage.config import config`
config = TriageConfig()
