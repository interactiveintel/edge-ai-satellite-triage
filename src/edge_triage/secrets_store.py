"""Local secrets storage for API keys (FIRMS MAP_KEY, future Planet/Maxar, etc.).

Design goals:
  - Human-readable JSON file under ``~/.edge_triage/secrets.json`` (or an
    override path) so it survives across dashboard sessions.
  - File permissions hardened to 0600 on POSIX (owner-only read/write).
  - Never log key values; redact when printing.
  - Env-var fallback so CI/CD and deploys still work without a file.
  - No third-party deps.

Key lookup priority (highest wins):
  1. Environment variable (e.g. ``EDGE_TRIAGE_FIRMS_KEY``)
  2. Secrets file (``SecretsStore.get()``)

The file lives outside the repo and is excluded via .gitignore.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Known keys ──────────────────────────────────────────────────────────────

# Stable short names that the UI / callers use.
KNOWN_KEYS: dict[str, dict[str, Any]] = {
    "firms": {
        "env_var": "EDGE_TRIAGE_FIRMS_KEY",
        "label": "NASA FIRMS MAP_KEY",
        "description": (
            "Free key for NASA FIRMS active fire detections. "
            "Get one at https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        ),
        "signup_url": "https://firms.modaps.eosdis.nasa.gov/api/map_key/",
        "test_endpoint": (
            "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={key}"
        ),
    },
    # Room to grow — planet, maxar, sentinelhub, etc.
}


def _default_path() -> Path:
    """Where secrets live by default."""
    override = os.environ.get("EDGE_TRIAGE_SECRETS_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".edge_triage" / "secrets.json"


def _redact(value: str) -> str:
    """Redact an API key for logging — keep first 3 + last 2 chars."""
    if not value:
        return "<empty>"
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}…{value[-2:]}"


# ── Store ───────────────────────────────────────────────────────────────────


class SecretsStore:
    """Tiny JSON-backed secrets vault — NOT a substitute for a real KMS.

    This is for developer convenience on a single workstation. For production
    deployment (Jetson, cloud), provide keys via env vars or a mounted secret.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _default_path()

    # ── Read ────────────────────────────────────────────────────

    def _read_all(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning("Secrets file %s is malformed — ignoring", self.path)
                return {}
            return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read secrets file %s: %s", self.path, exc)
            return {}

    def get(self, name: str) -> str:
        """Return the key value, checking env var first then the file.

        Empty string if not set anywhere.
        """
        info = KNOWN_KEYS.get(name, {})
        env_var = info.get("env_var")
        if env_var:
            env_val = os.environ.get(env_var, "").strip()
            if env_val:
                return env_val
        return self._read_all().get(name, "").strip()

    def source(self, name: str) -> str:
        """Return where the current value came from: 'env', 'file', or 'unset'."""
        info = KNOWN_KEYS.get(name, {})
        env_var = info.get("env_var")
        if env_var and os.environ.get(env_var, "").strip():
            return "env"
        if self._read_all().get(name, "").strip():
            return "file"
        return "unset"

    def all_status(self) -> dict[str, dict[str, Any]]:
        """Snapshot of every known secret: name → {set, source, redacted}."""
        out: dict[str, dict[str, Any]] = {}
        for name, info in KNOWN_KEYS.items():
            val = self.get(name)
            out[name] = {
                "label": info["label"],
                "description": info["description"],
                "signup_url": info.get("signup_url", ""),
                "set": bool(val),
                "source": self.source(name),
                "redacted": _redact(val) if val else "",
            }
        return out

    # ── Write ───────────────────────────────────────────────────

    def set(self, name: str, value: str) -> None:
        """Persist a key to the local file with 0600 permissions."""
        value = value.strip()
        if not value:
            # Treat blank as delete
            self.delete(name)
            return

        if name not in KNOWN_KEYS:
            logger.warning("Unknown secret name '%s' — saving anyway", name)

        data = self._read_all()
        data[name] = value

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

        # Harden permissions on POSIX
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
            os.chmod(self.path.parent, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except (OSError, NotImplementedError):
            pass  # Windows or other

        logger.info("Saved secret %s (%s) to %s", name, _redact(value), self.path)

    def delete(self, name: str) -> None:
        data = self._read_all()
        if name in data:
            data.pop(name)
            if data:
                self.path.write_text(
                    json.dumps(data, indent=2, sort_keys=True), encoding="utf-8",
                )
            else:
                try:
                    self.path.unlink()
                except OSError:
                    pass
            logger.info("Removed secret %s", name)

    # ── Validation ──────────────────────────────────────────────

    def test(self, name: str) -> tuple[bool, str]:
        """Try a trivial API call to verify the key. Returns (ok, message)."""
        value = self.get(name)
        if not value:
            return False, "Key is not set"

        info = KNOWN_KEYS.get(name, {})
        endpoint = info.get("test_endpoint", "")
        if not endpoint:
            return True, "Key present (no test endpoint defined)"

        url = endpoint.format(key=value)
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "edge-triage/0.1"})
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                body = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            return False, f"Request failed: {type(exc).__name__}: {exc}"

        # FIRMS mapkey_status returns an XML blob containing <current_transactions>N</...>
        # Invalid keys trigger HTML error pages or explicit error text.
        lowered = body.lower()
        if "invalid" in lowered and "map" in lowered:
            return False, "Server rejected key as invalid"
        if "current_transactions" in lowered or "<mapkey>" in lowered or "transaction_limit" in lowered:
            return True, "Key verified OK"
        # Fallback: HTTP 200 with non-error body
        return True, "Key appears valid (server returned 200)"


# Convenience singleton — import as ``from edge_triage.secrets_store import secrets``
secrets = SecretsStore()
