"""Edge AI Satellite Data Triage — onboard filtering for bandwidth-constrained downlinks."""

__version__ = "0.1.0"

from .audit import AuditLogger
from .config import config
from .data_ingest import ImageIngestor
from .detection import Detection, DetectionResult, ObjectDetector
from .inference import QuantizedInferencer
from .live_data import (
    AOI_PRESETS,
    FIRMSFireSource,
    LiveFeedItem,
    NOAAGOESSource,
    Sentinel1Source,
    Sentinel2Source,
)
from .metrics import MetricsCollector
from .model_registry import ModelRegistry
from .secrets_store import SecretsStore, secrets
from .ship_detector import (
    AISCrossReference,
    MaritimeShipDetector,
    ShipDetection,
    ShipDetectionResult,
)
from .triage import EdgeTriageEngine, TriageResult
from .utils import PowerMonitor

__all__ = [
    "AISCrossReference",
    "AOI_PRESETS",
    "AuditLogger",
    "config",
    "Detection",
    "DetectionResult",
    "EdgeTriageEngine",
    "FIRMSFireSource",
    "ImageIngestor",
    "LiveFeedItem",
    "MaritimeShipDetector",
    "MetricsCollector",
    "ModelRegistry",
    "NOAAGOESSource",
    "ObjectDetector",
    "PowerMonitor",
    "QuantizedInferencer",
    "secrets",
    "SecretsStore",
    "Sentinel1Source",
    "Sentinel2Source",
    "ShipDetection",
    "ShipDetectionResult",
    "TriageResult",
    "__version__",
]
