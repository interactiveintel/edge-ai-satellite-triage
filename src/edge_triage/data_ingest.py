"""Image ingest, tile splitting, and preprocessing for edge triage.

Supports TIFF (Sentinel-2 multispectral), JPEG, PNG, and raw numpy arrays.
Designed for zero-copy, memory-efficient operation on Jetson Orin.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import config

logger = logging.getLogger(__name__)

# Optional heavy imports — deferred so the module loads fast on constrained HW.
try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None  # type: ignore[assignment,misc]

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class ImageIngestor:
    """Load, tile, and normalise satellite / drone imagery for inference.

    Usage::

        ingestor = ImageIngestor(tile_size=256)
        for tile in ingestor.load_and_tile("scene.tif"):
            result = engine.process_tile(tile)
    """

    def __init__(self, tile_size: int | None = None, channels: int | None = None) -> None:
        self.tile_size = tile_size or config.TILE_SIZE
        self.channels = channels or config.INPUT_CHANNELS

    # ── Public API ──────────────────────────────────────────────

    def load(self, source: str | Path | np.ndarray) -> np.ndarray:
        """Load an image from *source* and return as ``(H, W, C)`` float32 array."""
        if isinstance(source, np.ndarray):
            return self._ensure_float32(source)
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            return self._load_tiff(path)
        if suffix in (".jpg", ".jpeg", ".png", ".bmp"):
            return self._load_standard(path)
        if suffix == ".npy":
            return self._ensure_float32(np.load(path, allow_pickle=False))
        raise ValueError(f"Unsupported format: {suffix}")

    def load_and_tile(self, source: str | Path | np.ndarray) -> Iterator[np.ndarray]:
        """Load an image and yield non-overlapping tiles of ``(tile_size, tile_size, C)``."""
        image = self.load(source)
        yield from self.tile(image)

    def tile(self, image: np.ndarray) -> Iterator[np.ndarray]:
        """Split ``(H, W, C)`` array into non-overlapping tiles.

        Partial edge tiles are zero-padded to ``tile_size``.
        Accepts 2D grayscale (H, W) or 3D (H, W, C).
        """
        image = self._ensure_3d(image)
        h, w = image.shape[:2]
        ts = self.tile_size
        for y in range(0, h, ts):
            for x in range(0, w, ts):
                patch = image[y : y + ts, x : x + ts]
                if patch.shape[0] != ts or patch.shape[1] != ts:
                    padded = np.zeros((ts, ts, image.shape[2]), dtype=image.dtype)
                    padded[: patch.shape[0], : patch.shape[1]] = patch
                    patch = padded
                yield patch

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Normalise a tile to ``[0, 1]`` float32 and convert to ``(C, H, W)``.

        Input can be:
        - ``(H, W)`` grayscale
        - ``(H, W, C)`` channel-last (any C: 3 for RGB, 13 for Sentinel-2)
        - ``(C, H, W)`` channel-first (already in PyTorch layout)
        - uint8 (0–255), uint16 (0–10000 Sentinel-2), or float32
        """
        arr = self._ensure_float32(image)
        arr = self._ensure_3d(arr)
        arr = self._to_channel_first(arr)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _ensure_3d(arr: np.ndarray) -> np.ndarray:
        """Promote 2D (H, W) to 3D (H, W, 1)."""
        if arr.ndim == 2:
            return arr[:, :, np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
        return arr

    @staticmethod
    def _to_channel_first(arr: np.ndarray) -> np.ndarray:
        """Convert ``(H, W, C)`` to ``(C, H, W)``.

        Distinguishes layout by checking which axis is smallest.  For the
        ambiguous case where the smallest dimension equals the largest (e.g.
        a cube), we assume channel-last.  If the data is *already*
        channel-first (smallest dim is axis 0), we leave it alone.
        """
        # Already channel-first: axis 0 is the smallest
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            return arr
        # Channel-last: axis 2 is the smallest (or equal — default assumption)
        return np.transpose(arr, (2, 0, 1))

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _ensure_float32(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        if arr.dtype == np.uint16:
            return np.clip(arr.astype(np.float32) / 10000.0, 0.0, 1.0)
        if arr.dtype != np.float32:
            return arr.astype(np.float32)
        return arr

    @staticmethod
    def _load_tiff(path: Path) -> np.ndarray:
        """Load a GeoTIFF / multi-band TIFF via PIL or OpenCV."""
        if PILImage is not None:
            img = PILImage.open(path)
            return np.array(img, dtype=np.float32)
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"cv2 could not read {path}")
            return img.astype(np.float32)
        raise ImportError("Install Pillow or opencv-python to load TIFF files")

    @staticmethod
    def _load_standard(path: Path) -> np.ndarray:
        """Load JPEG / PNG via PIL (preferred) or OpenCV."""
        if PILImage is not None:
            img = PILImage.open(path).convert("RGB")
            return np.array(img, dtype=np.float32) / 255.0
        if cv2 is not None:
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"cv2 could not read {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.float32) / 255.0
        raise ImportError("Install Pillow or opencv-python to load images")
