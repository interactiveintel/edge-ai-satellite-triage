"""Persistent JSON Lines audit log for triage decisions with HMAC integrity.

Every tile decision is written to ``config.AUDIT_LOG_PATH`` as a single JSON line
with full provenance: input hash, scene metadata, CNN scores, agent reasoning,
final decision, timestamps, hardware info, and an HMAC-SHA256 authentication tag.

HMAC integrity: each record is authenticated with a keyed hash so that
tampering is detectable. The key comes from ``EDGE_TRIAGE_AUDIT_KEY`` env var
or is derived from the machine identity. Verify with :meth:`AuditLogger.verify_log`.

Gov/defense traceability: any decision can be reconstructed from the log, and
the integrity of the entire log can be cryptographically verified.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

from .config import config

logger = logging.getLogger(__name__)


def _input_hash(data: bytes, algo: str = "sha256") -> str:
    """Return hex digest of raw tile bytes for integrity tracking."""
    return hashlib.new(algo, data).hexdigest()


def _get_audit_key() -> bytes:
    """Derive HMAC key from env var or machine identity.

    Priority:
      1. ``EDGE_TRIAGE_AUDIT_KEY`` environment variable
      2. Deterministic derivation from platform identity (fallback)

    For production gov/defense deployment, always set the env var to a
    securely generated key (e.g. 32-byte hex: ``openssl rand -hex 32``).
    """
    env_key = os.environ.get("EDGE_TRIAGE_AUDIT_KEY")
    if env_key:
        return env_key.encode("utf-8")
    # Deterministic fallback — stable across restarts on the same machine
    identity = f"{platform.node()}-{platform.machine()}-edge-triage-audit"
    return hashlib.sha256(identity.encode()).digest()


class AuditLogger:
    """Append-only JSON Lines audit log with HMAC integrity.

    Each call to :meth:`log_decision` writes one line containing the full
    decision record followed by a tab and an HMAC-SHA256 tag.  The file is
    flushed after every write so that power loss does not lose the last record.

    Format per line::

        {json_record}\\t{hmac_hex}

    Usage::

        audit = AuditLogger()
        audit.log_decision(tile_bytes, result, metadata)
        # Later: verify integrity
        results = AuditLogger.verify_log(Path("logs/audit.jsonl"))
    """

    def __init__(self, path: Path | None = None, hmac_key: bytes | None = None) -> None:
        self._path = path or config.AUDIT_LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._hmac_key = hmac_key or _get_audit_key()
        self._hw_info = self._detect_hardware()
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        logger.info("Audit log open: %s (HMAC enabled)", self._path)

    def log_decision(
        self,
        tile_raw_bytes: bytes,
        result: Any,
        metadata: dict[str, Any],
    ) -> None:
        """Write one HMAC-authenticated decision record."""
        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "timestamp_mono": time.monotonic(),
            "input_hash_sha256": _input_hash(tile_raw_bytes),
            "input_bytes": len(tile_raw_bytes),
            "scene_id": metadata.get("scene_id", "unknown"),
            "tile_id": metadata.get("tile_id", "unknown"),
            "tile_coords": metadata.get("tile_coords"),
            "acquisition_time": metadata.get("acquisition_time"),
            "context": metadata.get("context", ""),
            "mode": config.MODE,
            "keep": result.keep,
            "final_score": round(result.final_score, 6),
            "bandwidth_saved_pct": round(result.bandwidth_saved_percent, 2),
            "power_watts": round(result.power_used_watts, 3),
            "cnn_backend": result.cnn_results.get("backend", "unknown"),
            "cnn_cloud_fraction": round(result.cnn_results.get("cloud_fraction", 0), 6),
            "cnn_anomaly_score": round(result.cnn_results.get("anomaly_score", 0), 6),
            "cnn_value_score": round(result.cnn_results.get("value_score", 0), 6),
            "cnn_inference_ms": round(result.cnn_results.get("inference_ms", 0), 3),
            "agent_used": result.agent_decision is not None,
            "agent_slm_used": (
                result.agent_decision.used_slm if result.agent_decision else False
            ),
            "explanation": result.explanation[:500],
            "actions": result.actions,
            "hardware": self._hw_info,
            "software_version": _get_version(),
        }
        line = json.dumps(record, separators=(",", ":"), default=str)
        mac = hmac_mod.new(self._hmac_key, line.encode("utf-8"), "sha256").hexdigest()
        self._fh.write(f"{line}\t{mac}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def verify_log(
        path: Path,
        hmac_key: bytes | None = None,
    ) -> list[tuple[int, bool, str]]:
        """Verify HMAC integrity of every record in an audit log.

        Parameters
        ----------
        path : Path
            Path to the audit JSONL file.
        hmac_key : bytes, optional
            Key used when the log was written.  Defaults to :func:`_get_audit_key`.

        Returns
        -------
        list of (line_number, passed, detail)
        """
        key = hmac_key or _get_audit_key()
        results: list[tuple[int, bool, str]] = []
        with open(path, encoding="utf-8") as f:
            for i, raw_line in enumerate(f, 1):
                raw_line = raw_line.rstrip("\n")
                if not raw_line:
                    continue
                if "\t" not in raw_line:
                    results.append((i, False, "no HMAC tag (legacy record?)"))
                    continue
                content, stored_mac = raw_line.rsplit("\t", 1)
                expected = hmac_mod.new(key, content.encode("utf-8"), "sha256").hexdigest()
                if hmac_mod.compare_digest(stored_mac, expected):
                    results.append((i, True, "ok"))
                else:
                    results.append((i, False, "HMAC mismatch — record may be tampered"))
        return results

    @staticmethod
    def _detect_hardware() -> dict[str, str]:
        return {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        }


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"
