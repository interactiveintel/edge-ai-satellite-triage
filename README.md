# Edge AI Satellite Data Triage

Onboard data triage for bandwidth-constrained satellite and drone downlinks, running on NVIDIA Jetson Orin at <20 W. Production-ready for US government and defense testing with full audit trail, provenance tracking, and cryptographic integrity.

## Architecture

```
Image Tile ─► ImageIngestor ─► QuantizedInferencer (INT8 CNN) ─► EdgeAgent (ReAct) ─► TriageResult
                │                       │                              │                    │
            normalize/tile      MobileNetV3-Small             optional SLM            audit + provenance
            any format →        TRT > ONNX > PyTorch          (Phi-3 / Gemma-2B)     HMAC-SHA256 log
            (C,H,W) f32        auto-build from pretrained      <2W, max 3 steps
```

### Pipeline stages

1. **Ingest** — load TIFF / JPEG / numpy, tile-split, normalize to `(C, H, W)` float32
2. **CNN Inference** — MobileNetV3-Small backbone, INT8 quantized via ONNX / TensorRT. Outputs `cloud_fraction`, `anomaly_score`, `value_score`
3. **Agentic Reasoning** — ReAct loop (pure Python, <2 W) activated only when `value_score > 0.6`. Adds urgency assessment, priority scoring, follow-up suggestions
4. **Triage Decision** — KEEP or FILTER with human-readable explanation, bandwidth savings estimate, and full provenance

### Safety & hardening

- NaN/Inf guard at both inference and triage layers — bad sensor data defaults to safe FILTER
- SIGALRM-based inference timeout ceiling (`MAX_INFERENCE_MS`)
- Error recovery with safe fallback results at every pipeline stage
- Power budget enforcement via `AgentPowerGuard`

## Quick Start

```bash
# Install (lightweight — no GPU required for dev/CI)
pip install -e ".[dev]"

# CLI demo (uses simulation stub)
PYTHONPATH=src python -m edge_triage

# Streamlit dashboard
pip install -e ".[all,dev]"
PYTHONPATH=src streamlit run src/edge_triage/__main__.py

# Run tests (76 tests)
make test
```

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

## Project Structure

```
src/edge_triage/
  __init__.py         — package exports
  __main__.py         — Streamlit dashboard + CLI entry point
  config.py           — power budgets, thresholds, model paths (mutable singleton)
  data_ingest.py      — image loading, tiling, normalization
  inference.py        — quantized CNN inference (TensorRT > ONNX > PyTorch INT8 > stub)
  reasoning_loop.py   — pure-Python ReAct agent loop
  agent.py            — EdgeAgent with optional SLM support
  triage.py           — main pipeline engine (EdgeTriageEngine)
  audit.py            — HMAC-SHA256 authenticated JSON Lines audit log
  metrics.py          — bandwidth / power / TOPS-per-Watt tracking
  model_registry.py   — versioned model management with SHA-256 integrity
  utils.py            — PowerMonitor, AgentPowerGuard

scripts/
  train_cloud_mask.py   — EuroSAT + BigEarthNet training (MobileNetV3-Small)
  retrain_pipeline.py   — continuous retraining from analyst feedback or audit logs

tests/
  test_triage.py        — 76 tests across 14 test classes
```

## Key Modules

### Audit Trail (`audit.py`)
Every triage decision is logged as HMAC-SHA256 authenticated JSON Lines. Tamper detection is built in — verify any log file with `AuditLogger.verify_log(path)`. Key sourced from `EDGE_TRIAGE_AUDIT_KEY` env var or machine-derived fallback.

### Model Registry (`model_registry.py`)
File-based version registry (`models/registry.json`) with SHA-256 checksums, activation/rollback, and training metadata. Every deployed model is traceable to a training run and dataset.

```bash
# List registered models
make registry-list

# Verify all model checksums
make registry-verify
```

### Continuous Retraining (`scripts/retrain_pipeline.py`)
Ground-station feedback loop: load analyst corrections or audit-log pseudo-labels, fine-tune, validate against hold-out set, auto-register if improved.

```bash
# From analyst feedback CSV
python scripts/retrain_pipeline.py --feedback corrections.csv

# From audit log pseudo-labels
python scripts/retrain_pipeline.py --from-audit logs/triage_audit.jsonl
```

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

1. **Lint** — ruff check + format
2. **Test** — pytest with coverage (JUnit XML + coverage XML artifacts)
3. **Security** — bandit source scan + pip-audit dependency scan
4. **SBOM** — CycloneDX Software Bill of Materials
5. **Docker** — build + Trivy vulnerability scan (main branch only)

## Dashboard

The Streamlit dashboard provides four tabs:

| Tab | Contents |
|-----|----------|
| **Triage Pipeline** | Upload/generate tiles, run triage with mission context presets, per-tile result cards |
| **Analytics** | Score bar charts, decision distribution, power budget gauge, detail table |
| **Audit Trail** | Browse audit log entries, HMAC integrity verification |
| **System Status** | Backend detection, model registry, configuration dump |

## Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| core | `numpy`, `Pillow` | Always installed — enough for stub inference and CI |
| `[ml]` | `torch`, `torchvision`, `onnxruntime`, `opencv` | Production inference |
| `[dashboard]` | `streamlit` | Web UI |
| `[jetson]` | `jetson-stats`, `tensorrt` | NVIDIA Jetson hardware |
| `[dev]` | `pytest`, `ruff`, `bandit`, `pip-audit` | Development and security |

## Key Targets

| Metric | Target |
|--------|--------|
| Total inference power | <15 W |
| Agent reasoning power | <2 W |
| TOPS/Watt (INT8 DLA) | >40 |
| Bandwidth reduction | >85% |
| Per-tile latency | <50 ms |
| Test coverage | 76 tests, 14 classes |

## License

MIT — see [LICENSE](LICENSE).
