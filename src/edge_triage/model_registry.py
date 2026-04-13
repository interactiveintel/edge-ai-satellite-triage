"""Model registry — versioned model management with SHA-256 integrity checks.

Tracks model versions in ``models/registry.json``.  Each entry stores the
filename, checksum, training metadata, and an ``active`` flag.  The inference
engine loads whichever version is marked active.

Gov/defense requirement: every deployed model is traceable to a training run,
dataset, and evaluation metric.  Rollback is always one command away.

Usage::

    from edge_triage.model_registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(Path("models/cloud_mask_best.pt"), version="1.0.0",
                 dataset="eurosat", epochs=20, val_loss=0.023)
    reg.activate("1.0.0")
    active = reg.get_active()
    reg.verify_all()
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"


class ModelRegistry:
    """File-based model version registry with integrity verification.

    Parameters
    ----------
    models_dir : Path, optional
        Directory containing model files and the registry JSON.
        Defaults to ``config.MODEL_DIR``.
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self._dir = Path(models_dir) if models_dir else config.MODEL_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._dir / REGISTRY_FILENAME
        self._versions: list[dict[str, Any]] = self._load()

    # ── Persistence ────────────────────────────────────────────

    def _load(self) -> list[dict[str, Any]]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Registry load failed: %s — starting fresh", exc)
        return []

    def _save(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._versions, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    # ── Public API ─────────────────────────────────────────────

    def register(
        self,
        model_path: Path,
        version: str,
        *,
        dataset: str = "",
        epochs: int = 0,
        val_loss: float = 0.0,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
        activate: bool = True,
    ) -> dict[str, Any]:
        """Register a new model version.

        Copies the model file into the registry directory with a versioned
        name, computes its SHA-256 checksum, and records training metadata.

        Parameters
        ----------
        model_path : Path
            Path to the model file (ONNX, .pt, .engine).
        version : str
            Semantic version string (e.g. ``"1.2.0"``).
        dataset : str
            Training dataset name.
        epochs : int
            Number of training epochs.
        val_loss : float
            Best validation loss achieved.
        metrics : dict, optional
            Additional evaluation metrics.
        notes : str
            Free-text notes for this version.
        activate : bool
            If True, mark this version as the active deployment model.

        Returns
        -------
        dict
            The registry entry for the newly registered model.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Check for duplicate version
        for v in self._versions:
            if v["version"] == version:
                raise ValueError(f"Version {version} already registered — use a new version string")

        sha256 = self._checksum(model_path)

        # Copy into registry directory with versioned name
        ext = model_path.suffix
        registered_name = f"cloud_mask_v{version}{ext}"
        dest = self._dir / registered_name
        if model_path.resolve() != dest.resolve():
            shutil.copy2(model_path, dest)

        entry: dict[str, Any] = {
            "version": version,
            "filename": registered_name,
            "sha256": sha256,
            "size_bytes": dest.stat().st_size,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_dataset": dataset,
            "training_epochs": epochs,
            "val_loss": round(val_loss, 6),
            "metrics": metrics or {},
            "active": False,
            "notes": notes,
        }
        self._versions.append(entry)

        if activate:
            self._set_active(version)

        self._save()
        logger.info("Registered model v%s (%s, %d bytes, sha256=%s…)",
                     version, registered_name, entry["size_bytes"], sha256[:16])
        return entry

    def activate(self, version: str) -> None:
        """Mark a specific version as the active deployment model."""
        self._set_active(version)
        self._save()
        logger.info("Activated model v%s", version)

    def get_active(self) -> dict[str, Any] | None:
        """Return the currently active model entry, or None."""
        for v in self._versions:
            if v.get("active"):
                return v
        return None

    def get_active_path(self) -> Path | None:
        """Return the filesystem path to the active model, or None."""
        active = self.get_active()
        if active:
            return self._dir / active["filename"]
        return None

    def rollback(self, to_version: str | None = None) -> dict[str, Any]:
        """Roll back to a specific version, or the previous one.

        Raises ``ValueError`` if no previous version exists.
        """
        if to_version:
            self.activate(to_version)
            return self.get_active()  # type: ignore[return-value]

        if len(self._versions) < 2:
            raise ValueError("No previous version to roll back to")

        # Find current active, activate the one before it
        for i, v in enumerate(self._versions):
            if v.get("active") and i > 0:
                self.activate(self._versions[i - 1]["version"])
                return self.get_active()  # type: ignore[return-value]

        # Fallback: activate second-to-last
        self.activate(self._versions[-2]["version"])
        return self.get_active()  # type: ignore[return-value]

    def list_versions(self) -> list[dict[str, Any]]:
        """Return all registered versions (newest last)."""
        return list(self._versions)

    def verify_all(self) -> list[tuple[str, bool, str]]:
        """Verify SHA-256 integrity of every registered model.

        Returns a list of ``(version, passed, detail)`` tuples.
        """
        results: list[tuple[str, bool, str]] = []
        for v in self._versions:
            path = self._dir / v["filename"]
            if not path.exists():
                results.append((v["version"], False, "file missing"))
                continue
            actual = self._checksum(path)
            if actual == v["sha256"]:
                results.append((v["version"], True, "ok"))
            else:
                results.append((v["version"], False,
                                f"sha256 mismatch: expected {v['sha256'][:16]}…, got {actual[:16]}…"))
        return results

    # ── Internal ───────────────────────────────────────────────

    def _set_active(self, version: str) -> None:
        found = False
        for v in self._versions:
            if v["version"] == version:
                v["active"] = True
                found = True
            else:
                v["active"] = False
        if not found:
            raise ValueError(f"Version not found: {version}")

    @staticmethod
    def _checksum(path: Path) -> str:
        """Compute SHA-256 of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
