# ===========================================================================
# Edge AI Satellite Triage — Hardened Production Dockerfile for Jetson Orin
# Base: NVIDIA L4T ML (JetPack 6.x) with PyTorch + TensorRT pre-installed
#
# Build on Jetson:
#   docker build -t edge-triage:latest .
#
# Run CLI demo:
#   docker run --runtime nvidia --rm edge-triage:latest
#
# Run Streamlit dashboard:
#   docker run --runtime nvidia --rm -p 8501:8501 edge-triage:latest --dashboard
#
# Run with power monitoring (requires host device access):
#   docker run --runtime nvidia --rm --privileged edge-triage:latest
# ===========================================================================

# ── Base: JetPack 6.x L4T with ML stack ──────────────────────────────────
# nvcr.io/nvidia/l4t-ml includes: PyTorch, torchvision, ONNX Runtime, TensorRT,
# CUDA 12.x, cuDNN — everything needed for Jetson Orin inference.
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r36.4.0-py3
FROM ${BASE_IMAGE} AS base

LABEL maintainer="Paul Pereira"
LABEL description="Edge AI onboard satellite data triage for Jetson Orin"
LABEL jetpack.version="6.x"
LABEL org.opencontainers.image.source="https://github.com/paulpereira/edge-triage"
LABEL org.opencontainers.image.licenses="MIT"

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# ── Security: create non-root application user ───────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /usr/sbin/nologin appuser

# ── System dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# jetson-stats for jtop / tegrastats power monitoring
RUN pip3 install --no-cache-dir jetson-stats>=4.2

# ── Application install ─────────────────────────────────────────────────
WORKDIR /app

# Copy dependency files first for layer caching
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY pyproject.toml setup.cfg setup.py ./
COPY src/ src/

# Install the package (heavy ML deps already in L4T base image)
RUN pip3 install --no-cache-dir -e .

# Copy remaining project files
COPY scripts/ scripts/
COPY tests/ tests/
COPY README.md CLAUDE.md LICENSE ./

# ── Model + log directories (writable by non-root user) ─────────────────
RUN mkdir -p models logs \
    && chown -R appuser:appuser /app/models /app/logs /app/src

# ── Runtime config ───────────────────────────────────────────────────────
ENV EDGE_TRIAGE_POWER_MODE=15W
ENV EDGE_TRIAGE_MODE=space
ENV EDGE_TRIAGE_AUDIT_KEY=""
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

# ── Switch to non-root user ─────────────────────────────────────────────
USER appuser

# Health check — verify the package imports cleanly
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "from edge_triage import EdgeTriageEngine; print('OK')"

# Default: run CLI demo. Override with --dashboard for Streamlit.
ENTRYPOINT ["python3", "-m", "edge_triage"]

# ===========================================================================
# MULTI-ARCH: For development on x86 (no Jetson), use this alternate base:
#
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime \
#     -t edge-triage:dev .
#
# Note: TensorRT + tegrastats won't be available on x86.
# ===========================================================================
