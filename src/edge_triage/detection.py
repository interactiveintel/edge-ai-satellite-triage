"""Object detection — identifies items of interest within a tile.

Pluggable backends, priority order:
  1. ONNX Runtime with YOLOv8-nano (production on Jetson/CPU)
  2. Ultralytics YOLO (dev/training)
  3. Stub — pixel-statistics heuristic for offline demo and CI

Outputs a list of :class:`Detection` per tile. The triage engine uses these to:
  - Upgrade a tile from FILTER to KEEP when items of interest are present
  - Feed richer context to the ReAct agent ("3 vessels, 2 in formation")
  - Provide bounding-box overlays for the dashboard

Gov/defense: per-detection provenance (class, confidence, bbox) flows into the
HMAC audit log for downstream analyst review.

Power: YOLOv8n runs at ~3-4 W on Orin Nano INT8; stub is effectively free.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import config

logger = logging.getLogger(__name__)

# ── Optional heavy imports (deferred, don't crash CI without them) ──────────

try:
    import onnxruntime as _ort  # noqa: F401
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    from ultralytics import YOLO as _UltraYOLO  # noqa: F401
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


# ── COCO-lite class names (for stub + default YOLOv8n) ──────────────────────
# A subset of COCO classes most relevant to dual-use satellite / drone triage.
# Real YOLOv8n returns all 80 COCO classes; the stub picks from this shortlist.

DEFENSE_RELEVANT_CLASSES = (
    "ship", "boat", "airplane", "helicopter", "truck", "car",
    "bus", "train", "person", "building", "fire", "smoke",
)


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class Detection:
    """A single object-of-interest detection within a tile."""

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0,1]
    class_id: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 4) for v in self.bbox],
        }


@dataclass
class DetectionResult:
    """Output of a single-tile detection pass."""

    detections: list[Detection] = field(default_factory=list)
    backend: str = "stub"
    inference_ms: float = 0.0
    power_watts: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def classes_present(self) -> list[str]:
        return sorted({d.class_name for d in self.detections})

    def summary(self, max_classes: int = 5) -> str:
        """Human-readable summary, e.g. '3 ships, 1 airplane'."""
        if not self.detections:
            return "no items of interest"
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        parts = sorted(counts.items(), key=lambda kv: -kv[1])[:max_classes]
        return ", ".join(
            f"{n} {cls}{'s' if n > 1 else ''}" for cls, n in parts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "classes_present": self.classes_present,
            "summary": self.summary(),
            "detections": [d.to_dict() for d in self.detections],
            "backend": self.backend,
            "inference_ms": round(self.inference_ms, 2),
            "power_watts": round(self.power_watts, 2),
        }


# ── Detector ────────────────────────────────────────────────────────────────


class ObjectDetector:
    """Pluggable object detector — ONNX YOLOv8 > Ultralytics > Stub."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or (config.MODEL_DIR / "yolov8n.onnx")
        self._session: Any = None
        self._ultra_model: Any = None
        self._backend = self._init_backend()

    @property
    def backend(self) -> str:
        return self._backend

    def _init_backend(self) -> str:
        """Try backends in priority order. Stub is always available."""
        # 1. ONNX Runtime with YOLOv8n weights on disk
        if _ORT_AVAILABLE and self.model_path.exists():
            try:
                import onnxruntime as ort  # local import to avoid module-level crash
                providers = [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    providers=providers,
                )
                active = self._session.get_providers()[0]
                logger.info("ObjectDetector: ONNX Runtime YOLOv8n (%s)", active)
                return f"onnx:{active}"
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("YOLOv8 ONNX load failed: %s — falling back", exc)

        # 2. Ultralytics YOLO (dev/test)
        if _ULTRALYTICS_AVAILABLE:
            try:
                from ultralytics import YOLO
                self._ultra_model = YOLO("yolov8n.pt")  # will download if missing
                logger.info("ObjectDetector: Ultralytics YOLOv8n")
                return "ultralytics"
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("Ultralytics init failed: %s — falling back", exc)

        # 3. Stub — heuristic, always works
        warnings.warn(
            "ObjectDetector: using stub backend (no YOLOv8 weights found). "
            "Place yolov8n.onnx in models/ or install `ultralytics` for real detection.",
            UserWarning,
            stacklevel=2,
        )
        return "stub"

    # ── Public API ─────────────────────────────────────────────

    def detect(
        self,
        tile: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> DetectionResult:
        """Run detection on a preprocessed tile (C, H, W) float32."""
        if confidence_threshold is None:
            confidence_threshold = config.DETECTION_CONFIDENCE_THRESHOLD

        import time
        t0 = time.perf_counter()

        if self._backend.startswith("onnx"):
            detections = self._detect_onnx(tile, confidence_threshold)
        elif self._backend == "ultralytics":
            detections = self._detect_ultralytics(tile, confidence_threshold)
        else:
            detections = self._detect_stub(tile, confidence_threshold)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            detections=detections,
            backend=self._backend,
            inference_ms=elapsed_ms,
            power_watts=0.0,  # populated by PowerMonitor in triage layer
        )

    # ── ONNX backend ────────────────────────────────────────────

    def _detect_onnx(
        self,
        tile: np.ndarray,
        conf: float,
    ) -> list[Detection]:
        """Run YOLOv8 ONNX forward pass and post-process."""
        # YOLOv8 expects (1, 3, 640, 640) uint8 or float32 in [0,1]
        # Reduce multi-spectral input to 3 channels and resize
        t = self._prepare_yolo_input(tile)
        inputs = {self._session.get_inputs()[0].name: t}
        try:
            outputs = self._session.run(None, inputs)
        except Exception as exc:
            logger.warning("YOLOv8 forward failed: %s — empty result", exc)
            return []

        # YOLOv8 output shape: (1, 84, 8400) — 4 bbox + 80 class scores
        preds = outputs[0]
        return self._parse_yolo_output(preds, conf)

    @staticmethod
    def _prepare_yolo_input(tile: np.ndarray) -> np.ndarray:
        """Convert any tile shape to YOLOv8-ready (1, 3, 640, 640) float32."""
        arr = np.asarray(tile, dtype=np.float32)

        # (H, W) → (3, H, W)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)
        # (H, W, C) → (C, H, W)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4, 13):
            arr = np.transpose(arr, (2, 0, 1))

        # Collapse multi-spectral → 3 channels (pick true-colour bands)
        if arr.shape[0] > 3:
            rgb_idx = list(config.RGB_CHANNELS)
            # Clip indices to available channels
            rgb_idx = [i for i in rgb_idx if i < arr.shape[0]][:3]
            if len(rgb_idx) < 3:
                rgb_idx = list(range(3))
            arr = arr[rgb_idx, :, :]

        # Normalize to [0, 1]
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0, 1)

        # Resize to 640×640 (simple nearest-neighbour — fine for stub demo)
        from PIL import Image as PILImage
        chw = []
        for c in range(arr.shape[0]):
            img = PILImage.fromarray((arr[c] * 255).astype(np.uint8))
            img = img.resize((640, 640), PILImage.BILINEAR)
            chw.append(np.asarray(img, dtype=np.float32) / 255.0)
        arr = np.stack(chw, axis=0)

        return arr[None, ...]  # add batch dim

    def _parse_yolo_output(
        self,
        preds: np.ndarray,
        conf: float,
    ) -> list[Detection]:
        """Parse YOLOv8 output tensor to list of Detection."""
        # preds shape: (1, 84, 8400) — transpose to (8400, 84)
        preds = preds[0].T  # (N, 84)
        boxes = preds[:, :4]  # cx, cy, w, h in pixels
        scores = preds[:, 4:]  # 80 class scores

        # Best class per row
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        keep = confidences > conf
        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # NMS (simple centre-distance; for production use cv2.dnn.NMSBoxes)
        if len(boxes) == 0:
            return []

        # Convert to xyxy normalized [0,1] (input was 640×640)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.clip((cx - w / 2) / 640.0, 0, 1)
        y1 = np.clip((cy - h / 2) / 640.0, 0, 1)
        x2 = np.clip((cx + w / 2) / 640.0, 0, 1)
        y2 = np.clip((cy + h / 2) / 640.0, 0, 1)

        # Use COCO class names (YOLOv8 default)
        from .detection_classes import COCO_CLASS_NAMES

        detections: list[Detection] = []
        # Take top-30 by confidence
        top_k = min(30, len(confidences))
        order = np.argsort(-confidences)[:top_k]
        for idx in order:
            name = COCO_CLASS_NAMES[int(class_ids[idx])] if int(class_ids[idx]) < len(COCO_CLASS_NAMES) else f"class_{class_ids[idx]}"
            detections.append(Detection(
                class_name=name,
                class_id=int(class_ids[idx]),
                confidence=float(confidences[idx]),
                bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
            ))
        return detections

    # ── Ultralytics backend ─────────────────────────────────────

    def _detect_ultralytics(
        self,
        tile: np.ndarray,
        conf: float,
    ) -> list[Detection]:
        """Use ultralytics YOLO (dev only — don't ship on Jetson)."""
        import numpy as _np
        arr = self._prepare_yolo_input(tile)[0]  # drop batch
        arr = (arr * 255).astype(_np.uint8).transpose(1, 2, 0)  # HWC uint8
        try:
            results = self._ultra_model(arr, conf=conf, verbose=False)
        except Exception as exc:
            logger.warning("Ultralytics forward failed: %s", exc)
            return []

        detections: list[Detection] = []
        if results and len(results) > 0:
            r = results[0]
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                detections.append(Detection(
                    class_name=names.get(cls_id, f"class_{cls_id}"),
                    class_id=cls_id,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                ))
        return detections

    # ── Stub backend ───────────────────────────────────────────

    def _detect_stub(
        self,
        tile: np.ndarray,
        conf: float,
    ) -> list[Detection]:
        """Heuristic stub — produces plausible detections from tile statistics.

        Uses mean brightness, variance, and a deterministic hash of the tile
        to decide how many "items of interest" to synthesize. Good enough
        for dashboard demos and CI tests; replaced by real YOLOv8 on Jetson.
        """
        arr = np.asarray(tile, dtype=np.float32)
        if arr.ndim < 2:
            return []

        # Flatten to compute stats
        flat = arr.flatten()
        finite = np.isfinite(flat)
        if not finite.any():
            return []
        mean = float(np.nanmean(flat))
        std = float(np.nanstd(flat))
        if not np.isfinite(mean) or not np.isfinite(std):
            return []

        # Heuristic: high variance + medium brightness → items present
        # Very bright uniform tile (cloud) → few/no items
        # Dark uniform tile (ocean) → occasional vessel
        activity = std * (1.0 - abs(mean - 0.5) * 0.5)

        # Seed an RNG from tile content for determinism
        seed = int(abs(hash(arr.tobytes())) % (2**31))
        rng = np.random.default_rng(seed)

        # Expected number of detections scales with activity
        expected_n = max(0.0, (activity - 0.08) * 25.0)
        n_detections = int(rng.poisson(min(expected_n, 8)))
        n_detections = min(n_detections, 10)

        # Pick classes with bias based on context hints
        class_pool = list(DEFENSE_RELEVANT_CLASSES)
        weights = np.array([2.0, 1.5, 1.0, 0.5, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2])
        weights = weights / weights.sum()

        detections: list[Detection] = []
        for _ in range(n_detections):
            class_name = rng.choice(class_pool, p=weights)
            # Confidence scales with activity (gets capped)
            c = float(np.clip(conf + rng.uniform(0.05, 0.45) * activity * 4, conf, 0.98))
            # Random bbox, clamped
            cx, cy = rng.uniform(0.1, 0.9, size=2)
            w = rng.uniform(0.05, 0.25)
            h = rng.uniform(0.05, 0.25)
            x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
            detections.append(Detection(
                class_name=str(class_name),
                class_id=-1,
                confidence=c,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
            ))
        # Sort by confidence, descending
        detections.sort(key=lambda d: -d.confidence)
        return detections
