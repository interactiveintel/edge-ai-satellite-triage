"""Edge AI Satellite Data Triage — onboard filtering for bandwidth-constrained downlinks."""

__version__ = "0.1.0"

from .audit import AuditLogger
from .config import config
from .data_ingest import ImageIngestor
from .detection import Detection, DetectionResult, ObjectDetector
from .inference import QuantizedInferencer
from .metrics import MetricsCollector
from .model_registry import ModelRegistry
from .triage import EdgeTriageEngine, TriageResult
from .utils import PowerMonitor

__all__ = [
    "AuditLogger",
    "config",
    "Detection",
    "DetectionResult",
    "EdgeTriageEngine",
    "ImageIngestor",
    "MetricsCollector",
    "ModelRegistry",
    "ObjectDetector",
    "PowerMonitor",
    "QuantizedInferencer",
    "TriageResult",
    "__version__",
]
