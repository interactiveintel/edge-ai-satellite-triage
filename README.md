<div align="center">

# Edge AI Satellite Data Triage

**Onboard agentic filtering for bandwidth-constrained satellites &amp; drones.**
Production-ready edge AI for NVIDIA Jetson Orin at under 20 watts — CNN inference, ReAct agentic reasoning, SAR ship detection, and an HMAC-authenticated audit trail. Built for US government and defense testing.

[![Live Demo](https://img.shields.io/badge/Live-Landing_Page-4a90e2?style=for-the-badge&logo=streamlit&logoColor=white)](https://interactiveintel.github.io/edge-ai-satellite-triage/)
[![Dashboard](https://img.shields.io/badge/Try-Dashboard-00d68f?style=for-the-badge&logo=react&logoColor=white)](https://interactiveintel-edge-ai-satellite-triage.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/interactiveintel/edge-ai-satellite-triage)

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-119_passing-success.svg)
![Power](https://img.shields.io/badge/power-%3C20W-brightgreen.svg)
![Bandwidth](https://img.shields.io/badge/bandwidth_saved-%3E85%25-brightgreen.svg)

</div>

---

## Live links

| | URL |
|---|---|
| 🛰️ **Landing page** | https://interactiveintel.github.io/edge-ai-satellite-triage/ |
| 🎛️ **Live dashboard** | https://interactiveintel-edge-ai-satellite-triage.streamlit.app/ |
| 📦 **Source** | https://github.com/interactiveintel/edge-ai-satellite-triage |

---

## Architecture

```
Image Tile ─► ImageIngestor ─► QuantizedInferencer ─► ObjectDetector ─► EdgeAgent ─► TriageResult
                │                       │                     │              │              │
            normalize/tile      MobileNetV3-Small      YOLOv8 / SAR CFAR   ReAct loop   audit + provenance
            any format →        INT8 via TRT/ONNX     "items of interest"  optional SLM HMAC-SHA256 log
            (C,H,W) f32         auto-build pipeline   dark-ship detection  <2W, 3 steps
```

### Pipeline stages

1. **Ingest** — load TIFF / JPEG / numpy, tile-split, normalize to `(C, H, W)` float32
2. **CNN Inference** — MobileNetV3-Small backbone, INT8 quantized via ONNX / TensorRT. Outputs `cloud_fraction`, `anomaly_score`, `value_score`
3. **Object Detection** — YOLOv8-nano (optical) or classical CFAR (SAR). Detects vessels, vehicles, aircraft, fires
4. **Agentic Reasoning** — ReAct loop (pure Python, <2 W) activated on high-value tiles or when items of interest are present
5. **Triage Decision** — KEEP or FILTER with human-readable explanation, bandwidth savings estimate, and full provenance

### Safety & hardening

- NaN/Inf guard at both inference and triage layers — bad sensor data defaults to safe FILTER
- SIGALRM-based inference timeout ceiling (`MAX_INFERENCE_MS`)
- Error recovery with safe fallback results at every pipeline stage
- Power budget enforcement via `AgentPowerGuard`
- HMAC-SHA256 authenticated audit log — tamper-evident

---

## Quick Start

```bash
# Install (lightweight — no GPU required for dev/CI)
pip install -e ".[dev]"

# CLI demo (uses simulation stub)
PYTHONPATH=src python -m edge_triage

# Streamlit dashboard (recommended)
pip install -e ".[all,dev]"
PYTHONPATH=src streamlit run app.py

# Run tests (119 tests)
make test
```

---

## Jetson Deployment

```bash
# Build Docker image (uses NVIDIA L4T base)
docker build -t edge-triage:latest .

# Or on Jetson directly
pip install -e ".[ml,jetson]"

# The model auto-downloads on first run:
# 1. MobileNetV3-Small pretrained weights from torchvision
# 2. ONNX export with dynamic axes
# 3. TensorRT INT8 engine cached via trtexec (Jetson only)

# Set power mode
sudo nvpmodel -m 1          # 15W mode
sudo jetson_clocks           # lock clocks for benchmarks
```

---

## Project Structure

```
src/edge_triage/
  __init__.py            — package exports
  __main__.py            — 5-tab Streamlit dashboard + CLI entry point
  config.py              — power budgets, thresholds, model paths
  data_ingest.py         — image loading, tiling, normalization
  inference.py           — quantized CNN inference (TensorRT > ONNX > PyTorch INT8 > stub)
  detection.py           — YOLOv8 object detection (items of interest)
  ship_detector.py       — SAR ship detection (classical CFAR + AIS cross-reference)
  live_data.py           — Sentinel-1, Sentinel-2, NOAA GOES, NASA FIRMS feeds
  secrets_store.py       — local API-key vault with env-var fallback
  reasoning_loop.py      — pure-Python ReAct agent loop
  agent.py               — EdgeAgent with optional SLM support
  triage.py              — main pipeline engine (EdgeTriageEngine)
  audit.py               — HMAC-SHA256 authenticated JSON Lines audit log
  metrics.py             — bandwidth / power / TOPS-per-Watt tracking
  model_registry.py      — versioned model management with SHA-256 integrity
  utils.py               — PowerMonitor, AgentPowerGuard

scripts/
  train_cloud_mask.py    — EuroSAT + BigEarthNet training (MobileNetV3-Small)
  retrain_pipeline.py    — continuous retraining from analyst feedback or audit logs

docs/
  index.html             — public landing page (GitHub Pages)
  og-image.png           — social preview card
  favicon.png            — site favicon

tests/
  test_triage.py         — 119 tests across 22 test classes

app.py                   — Streamlit Cloud entry point
.streamlit/config.toml   — dark mission-control theme
```

---

## Key Modules

### Live satellite data (`live_data.py`)
Four free public sources wired in — no paid API required:
- **Sentinel-1 SAR** — global all-weather radar, vessel detection over open water
- **Sentinel-2 L2A** — global 10 m optical, 2-5 day latency
- **NOAA GOES-18** — Americas near-real-time, ~10 min latency
- **NASA FIRMS** — global active-fire detections (needs free MAP_KEY)

### SAR ship detection (`ship_detector.py`)
Classical CFAR + connected components — no ML training needed. Ships appear as bright scatterers on dark water in Sentinel-1 VV polarization. Cross-references with simulated AIS to flag "dark ships" (vessels with no transponder broadcast — the actual intel signal for maritime ISR).

### Object detection (`detection.py`)
Pluggable backend: TensorRT/ONNX YOLOv8n → Ultralytics → heuristic stub. Detects vessels, vehicles, aircraft, smoke, fires. Detection results feed the agent's ReAct reasoning.

### Audit trail (`audit.py`)
Every triage decision is logged as HMAC-SHA256 authenticated JSON Lines. Tamper detection is built in — verify any log file with `AuditLogger.verify_log(path)`. Key sourced from `EDGE_TRIAGE_AUDIT_KEY` env var or machine-derived fallback.

### Model registry (`model_registry.py`)
File-based version registry (`models/registry.json`) with SHA-256 checksums, activation/rollback, and training metadata. Every deployed model is traceable to a training run and dataset.

```bash
make registry-list      # List registered models
make registry-verify    # Verify all model checksums
```

### Continuous retraining (`scripts/retrain_pipeline.py`)
Ground-station feedback loop: load analyst corrections or audit-log pseudo-labels, fine-tune, validate against hold-out set, auto-register if improved.

```bash
python scripts/retrain_pipeline.py --feedback corrections.csv
python scripts/retrain_pipeline.py --from-audit logs/triage_audit.jsonl
```

---

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

1. **Lint** — ruff check + format
2. **Test** — pytest with coverage (JUnit XML + coverage XML artifacts)
3. **Security** — bandit source scan + pip-audit dependency scan
4. **SBOM** — CycloneDX Software Bill of Materials
5. **Docker** — build + Trivy vulnerability scan (main branch only)

---

## Dashboard

The Streamlit dashboard provides five tabs:

| Tab | Contents |
|-----|----------|
| **Triage Pipeline** | Mission scenarios, file upload, or synthetic tiles → per-tile result cards |
| **Live Feed** | Real Sentinel-1 / Sentinel-2 / GOES / FIRMS imagery with on-the-fly triage |
| **Analytics** | Score bar charts, decision distribution, power budget gauge, detail table |
| **Audit Trail** | Browse audit log entries, HMAC integrity verification |
| **System Status** | Backend detection, model registry, configuration dump |

---

## Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| core | `numpy`, `Pillow` | Always installed — enough for stub inference and CI |
| `[ml]` | `torch`, `torchvision`, `onnxruntime`, `opencv` | Production inference |
| `[dashboard]` | `streamlit`, `pandas`, `pydeck` | Web UI |
| `[jetson]` | `jetson-stats`, `tensorrt` | NVIDIA Jetson hardware |
| `[dev]` | `pytest`, `ruff`, `bandit`, `pip-audit` | Development and security |

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Total inference power | <15 W | ✓ design |
| Agent reasoning power | <2 W | ✓ pure Python |
| TOPS/Watt (INT8 DLA) | >40 | requires Jetson hardware |
| Bandwidth reduction | >85% | ✓ verified on 24-tile scenarios |
| Per-tile latency | <50 ms | ✓ benchmarked |
| Test coverage | 119 tests, 22 classes | ✓ all passing |

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Built for NVIDIA Jetson Orin Nano / AGX / Thor**

<sub>Dual-use platform · Open architecture · Defense-aligned audit trail</sub>

</div>
