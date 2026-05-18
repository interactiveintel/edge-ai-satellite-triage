"""Edge AI Satellite Triage — Polished Streamlit Dashboard + CLI.

Dashboard (recommended)::

    streamlit run src/edge_triage/__main__.py

CLI demo::

    python -m edge_triage
    python -m edge_triage --dashboard
"""

from __future__ import annotations

import sys


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS — tile preview and detection overlay
# ══════════════════════════════════════════════════════════════════════════


def _tile_to_rgb_uint8(tile):
    """Convert any tile shape to an HxWx3 uint8 RGB preview, or None if unsupported.

    Handles (H,W), (H,W,C), and (C,H,W). Picks the Sentinel-2 true-colour
    bands (B4,B3,B2) from 13-channel tiles.
    """
    import numpy as _np

    from .config import config

    arr = _np.asarray(tile)
    if arr.ndim == 2:
        arr = _np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4, 13) and arr.shape[0] <= arr.shape[-1]:
        # (C, H, W) — transpose
        arr = _np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        return None

    # Collapse multi-spectral → 3 channels (true-colour bands)
    if arr.shape[-1] > 3:
        rgb_idx = [i for i in config.RGB_CHANNELS if i < arr.shape[-1]][:3]
        if len(rgb_idx) < 3:
            rgb_idx = list(range(3))
        arr = arr[:, :, rgb_idx]
    elif arr.shape[-1] == 1:
        arr = _np.repeat(arr, 3, axis=-1)

    # Normalize to [0, 1]
    arr = arr.astype(_np.float32)
    if _np.nanmax(arr) > 1.5:
        arr = arr / 255.0
    arr = _np.clip(_np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)

    return (arr * 255).astype(_np.uint8)


def _draw_bboxes(rgb_uint8, detections, color=(255, 40, 40), thickness=2):
    """Draw bounding boxes + labels onto a uint8 RGB image. Returns modified array.

    BBoxes are normalized [0,1] — scaled to image dimensions.
    """
    import numpy as _np

    from PIL import Image as PILImage, ImageDraw, ImageFont

    img = PILImage.fromarray(rgb_uint8).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in detections:
        x1 = int(d.bbox[0] * w)
        y1 = int(d.bbox[1] * h)
        x2 = int(d.bbox[2] * w)
        y2 = int(d.bbox[3] * h)
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        label = f"{d.class_name} {d.confidence:.2f}"
        # Label background
        if font is not None:
            try:
                tb = draw.textbbox((x1, y1 - 12), label, font=font)
                draw.rectangle(tb, fill=color)
                draw.text((x1 + 2, y1 - 12), label, fill="white", font=font)
            except Exception:
                draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)

    return _np.asarray(img)


# ══════════════════════════════════════════════════════════════════════════
#  STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════


def _run_streamlit_dashboard() -> None:
    import json
    import time
    from pathlib import Path

    import numpy as np
    import streamlit as st

    from .config import config
    from .metrics import MetricsCollector
    from .model_registry import ModelRegistry
    from .triage import EdgeTriageEngine, TriageResult
    from . import __version__

    # ── Page config ────────────────────────────────────────────
    st.set_page_config(
        page_title="Edge AI Triage — Mission Control",
        page_icon="🛰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS — defense / mission-control aesthetic ───────
    st.markdown("""
    <style>
    /* ============================================================
       DESIGN TOKENS — single source of truth for the dark ISR theme
       ============================================================ */
    :root {
        --bg-deep:        #0a0e1a;
        --bg-surface:     #131826;
        --bg-elevated:    #1a2238;
        --bg-card:        #1d2742;
        --border:         #2a3548;
        --border-strong:  #3a4860;
        --text-primary:   #e8edf5;
        --text-secondary: #8a94a6;
        --text-muted:     #5a6478;
        --accent:         #4a90e2;
        --accent-bright:  #6ec1e4;
        --success:        #00d68f;
        --success-dim:    #1c4d3a;
        --warning:        #ffaa00;
        --danger:         #ff5252;
        --danger-dim:     #4d1c1c;
        --info:           #82b1ff;
        --gradient-hdr:   linear-gradient(135deg, #0a0e1a 0%, #131c33 35%, #1a2944 70%, #1e3a5f 100%);
        --shadow-soft:    0 1px 3px rgba(0,0,0,0.4);
        --shadow-elev:    0 4px 16px rgba(0,0,0,0.5), 0 1px 3px rgba(0,0,0,0.3);
    }

    /* ============================================================
       GLOBAL
       ============================================================ */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-deep) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, "Inter", "Segoe UI", Roboto, sans-serif !important;
    }
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1500px !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: var(--text-primary);
    }
    .stMarkdown p { color: var(--text-primary); }

    /* ============================================================
       HEADER BANNER — mission control look
       ============================================================ */
    .mc-header {
        background: var(--gradient-hdr);
        border: 1px solid var(--border-strong);
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-elev);
        position: relative;
        overflow: hidden;
    }
    .mc-header::before {
        content: "";
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-bright) 25%, var(--success) 50%, var(--accent-bright) 75%, transparent);
        opacity: 0.6;
    }
    .mc-header-grid {
        display: flex; justify-content: space-between; align-items: center; gap: 2rem;
    }
    .mc-title-block { flex: 1; min-width: 0; }
    .mc-classification {
        display: inline-block;
        background: rgba(0, 214, 143, 0.12);
        color: var(--success);
        border: 1px solid var(--success);
        padding: 2px 10px;
        border-radius: 3px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .mc-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin: 0;
        color: #fff;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .mc-title .accent { color: var(--accent-bright); }
    .mc-subtitle {
        margin: 0.4rem 0 0;
        color: var(--text-secondary);
        font-size: 0.85rem;
        letter-spacing: 0.04em;
    }
    .mc-status-block { display: flex; gap: 1.5rem; align-items: center; }
    .mc-stat {
        text-align: right;
        min-width: 80px;
        padding-left: 1.2rem;
        border-left: 1px solid var(--border);
    }
    .mc-stat-value {
        font-family: "SF Mono", "Menlo", "Consolas", monospace;
        font-size: 1.15rem;
        font-weight: 700;
        color: #fff;
        line-height: 1;
    }
    .mc-stat-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .mc-live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: var(--success);
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px var(--success);
        animation: mc-pulse 2s infinite;
    }
    @keyframes mc-pulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.4; }
    }

    /* ============================================================
       TABS — pill style
       ============================================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-surface);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 18px;
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 6px !important;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(74, 144, 226, 0.08) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #fff !important;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }

    /* ============================================================
       SIDEBAR
       ============================================================ */
    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    .sb-section-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent-bright);
        margin: 1.2rem 0 0.5rem;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sb-section-header .icon {
        display: inline-block; width: 6px; height: 6px;
        background: var(--accent-bright); border-radius: 50%;
        box-shadow: 0 0 4px var(--accent-bright);
    }
    .sb-foot {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-align: center;
        padding: 1rem 0.5rem;
        border-top: 1px solid var(--border);
        margin-top: 1.5rem;
        line-height: 1.6;
    }
    .sb-foot strong { color: var(--text-secondary); }

    /* ============================================================
       KPI METRICS — bigger, cleaner
       ============================================================ */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        transition: all 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent), var(--shadow-elev);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: "SF Mono", "Menlo", "Consolas", monospace !important;
        font-size: 1.6rem !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    /* ============================================================
       DECISION BADGES
       ============================================================ */
    .decision-keep, .decision-filter {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-family: "SF Mono", "Menlo", monospace;
    }
    .decision-keep {
        background: rgba(0, 214, 143, 0.15);
        color: var(--success);
        border: 1px solid var(--success);
        box-shadow: 0 0 12px rgba(0, 214, 143, 0.25);
    }
    .decision-filter {
        background: rgba(255, 82, 82, 0.12);
        color: var(--danger);
        border: 1px solid var(--danger);
    }

    /* ============================================================
       TILE RESULT CARDS
       ============================================================ */
    .tile-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.7rem 0;
        box-shadow: var(--shadow-soft);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .tile-card:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-elev);
        border-color: var(--border-strong);
    }
    .tile-card.keep::before, .tile-card.filter::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0;
        width: 3px;
    }
    .tile-card.keep::before   { background: var(--success); box-shadow: 0 0 12px var(--success); }
    .tile-card.filter::before { background: var(--danger); }
    .tile-card * { color: var(--text-primary); }

    /* ============================================================
       PROGRESS BARS
       ============================================================ */
    .stProgress > div > div > div {
        background: var(--bg-elevated) !important;
        border-radius: 3px !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-bright)) !important;
        border-radius: 3px !important;
    }

    /* ============================================================
       SELECTS / INPUTS / TOGGLES
       ============================================================ */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stMultiSelect > div > div {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--text-muted); }
    .stRadio label, .stCheckbox label { color: var(--text-secondary) !important; }

    /* ============================================================
       BUTTONS
       ============================================================ */
    .stButton > button {
        background: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.04em;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: var(--bg-card);
        border-color: var(--accent);
        color: var(--accent-bright);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), #2962ff);
        border: none;
        color: #fff;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5ba0f2, #3a72ff);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.5);
        color: #fff;
    }

    /* ============================================================
       DATAFRAMES
       ============================================================ */
    [data-testid="stDataFrame"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* ============================================================
       EXPANDERS
       ============================================================ */
    [data-testid="stExpander"] {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-top: 6px;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.85rem;
    }
    [data-testid="stExpander"]:hover { border-color: var(--border-strong); }

    /* ============================================================
       INFO / WARNING / SUCCESS BOXES
       ============================================================ */
    [data-baseweb="notification"] { border-radius: 8px !important; }

    /* ============================================================
       SCROLLBARS
       ============================================================ */
    ::-webkit-scrollbar       { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ============================================================
       HIDE STREAMLIT CHROME
       ============================================================ */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # ── Mission-control header ─────────────────────────────────
    import time as _time
    _utc_now = _time.strftime("%Y-%m-%d  %H:%M UTC", _time.gmtime())
    _session_tiles = len(st.session_state.get("results", [])) if "results" in st.session_state else 0

    st.markdown(f"""
    <div class="mc-header">
      <div class="mc-header-grid">
        <div class="mc-title-block">
          <div class="mc-classification">
            <span class="mc-live-dot"></span>System Operational · Dual-Use ISR
          </div>
          <h1 class="mc-title">Edge AI <span class="accent">Satellite Triage</span></h1>
          <p class="mc-subtitle">
            Onboard agentic filtering · Real-time SAR &amp; optical pipelines · NVIDIA Jetson Orin · Under {config.POWER_BUDGET_WATTS:.0f} W
          </p>
        </div>
        <div class="mc-status-block">
          <div class="mc-stat">
            <div class="mc-stat-value">{_utc_now.split('  ')[1]}</div>
            <div class="mc-stat-label">UTC TIME</div>
          </div>
          <div class="mc-stat">
            <div class="mc-stat-value">{config.POWER_BUDGET_WATTS:.0f}<span style="font-size:0.8rem; color:var(--text-muted);"> W</span></div>
            <div class="mc-stat-label">Power Budget</div>
          </div>
          <div class="mc-stat">
            <div class="mc-stat-value">{_session_tiles}</div>
            <div class="mc-stat-label">Session Tiles</div>
          </div>
          <div class="mc-stat">
            <div class="mc-stat-value" style="color: var(--success);">●</div>
            <div class="mc-stat-label">v{__version__}</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ─────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.collector = MetricsCollector()
        st.session_state.processing_log = []

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="sb-section-header"><span class="icon"></span>Mission Profile</div>',
            unsafe_allow_html=True,
        )

        config.MODE = st.radio(
            "Operating Mode",
            ["space", "ground"],
            horizontal=True,
            help="Space: satellite EO passes | Ground: drone/IoT/pipeline",
        )

        context_presets = {
            "Wildfire Monitoring": "Wildfire monitoring pass — prioritise hotspots and burn perimeters",
            "Defense ISR": "Defense surveillance — high priority on infrastructure and movement",
            "Maritime Domain": "Maritime domain awareness — vessel detection and sea state",
            "Disaster Response": "Disaster response — damage assessment and survivor detection",
            "Pipeline Inspection": "Oil/gas pipeline inspection — leak and anomaly detection",
            "Routine Observation": "Standard Earth observation pass",
            "Custom": "",
        }
        preset = st.selectbox("Mission Context", list(context_presets.keys()))
        if preset == "Custom":
            mission_context = st.text_input("Enter context", "")
        else:
            mission_context = context_presets[preset]

        st.markdown(
            '<div class="sb-section-header"><span class="icon"></span>System Controls</div>',
            unsafe_allow_html=True,
        )

        config.POWER_BUDGET_WATTS = st.slider(
            "Power Budget (W)", 7.0, 30.0, config.POWER_BUDGET_WATTS, 0.5,
            help="Total power envelope for inference + agent",
        )
        config.AGENT_ENABLED = st.toggle(
            "Agentic Reasoning", value=config.AGENT_ENABLED,
            help="ReAct reasoning loop on high-value tiles",
        )
        config.AGENT_SLM_ENABLED = st.toggle(
            "SLM Enhancement", value=config.AGENT_SLM_ENABLED,
            help="Phi-3/Gemma SLM for deep analysis (Orin AGX/Thor only)",
        )
        config.DETECTION_ENABLED = st.toggle(
            "Object Detection", value=config.DETECTION_ENABLED,
            help="YOLOv8-nano onboard detection of items of interest",
        )
        config.DETECTION_CONFIDENCE_THRESHOLD = st.slider(
            "Detection confidence", 0.05, 0.90, config.DETECTION_CONFIDENCE_THRESHOLD, 0.05,
            help="Minimum confidence for a detection to be counted",
        )

        st.markdown(
            '<div class="sb-section-header"><span class="icon"></span>Data Source</div>',
            unsafe_allow_html=True,
        )

        input_mode = st.radio(
            "Tile Source",
            ["Mission Scenarios", "Upload Files", "Synthetic"],
            horizontal=True,
            help="Mission Scenarios: 3 pre-built ops with 8 tiles each",
        )

        uploaded_files = None
        n_synthetic = 0
        selected_scenarios: list[str] = []

        if input_mode == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload tiles (TIFF / JPG / PNG)",
                type=["tif", "tiff", "jpg", "jpeg", "png"],
                accept_multiple_files=True,
            )
        elif input_mode == "Synthetic":
            n_synthetic = st.slider(
                "Synthetic test tiles", 1, 20, 3,
                help="Generate random tiles for testing",
            )
        else:
            selected_scenarios = st.multiselect(
                "Select scenarios",
                ["Wildfire Detection (California)", "Defense ISR (Maritime)", "Disaster Response (Earthquake)"],
                default=["Wildfire Detection (California)", "Defense ISR (Maritime)", "Disaster Response (Earthquake)"],
            )

        st.markdown(f"""
        <div class="sb-foot">
          <strong>EDGE TRIAGE</strong> v{__version__}<br>
          MobileNetV3 · YOLOv8 · ReAct<br>
          Jetson Orin Nano / AGX / Thor<br>
          <span style="color: var(--success);">●</span> Online
        </div>
        """, unsafe_allow_html=True)

    # ── Main tabs ──────────────────────────────────────────────
    tab_triage, tab_live, tab_analytics, tab_audit, tab_system = st.tabs([
        "◆  Triage Pipeline",
        "◉  Live Feed",
        "▤  Analytics",
        "⛨  Audit Trail",
        "◈  System Status",
    ])

    # ════════════════════════════════════════════════════════════
    #  TAB 1: TRIAGE PIPELINE
    # ════════════════════════════════════════════════════════════
    with tab_triage:
        # ── Scenario definitions ──────────────────────────────────
        MISSION_SCENARIOS = {
            "Wildfire Detection (California)": {
                "context": "Active wildfire monitoring — smoke plumes and thermal anomalies expected",
                "scene_id": "FIRE-CA-2026-041",
                "tiles": [
                    ("Hot spot — bright IR",     0.7, 1.0, 42),
                    ("Smoke plume edge",         0.5, 0.9, 43),
                    ("Clear forest canopy",      0.2, 0.5, 44),
                    ("Heavy cloud cover",        0.85, 1.0, 45),
                    ("Burn scar boundary",       0.3, 0.8, 46),
                    ("Urban edge near fire",     0.4, 0.7, 47),
                    ("Night thermal IR",         0.6, 0.95, 48),
                    ("Water body (lake)",        0.05, 0.2, 49),
                ],
            },
            "Defense ISR (Maritime)": {
                "context": "Defense maritime ISR — vessel detection in contested waters, high priority",
                "scene_id": "ISR-PAC-2026-088",
                "tiles": [
                    ("Open ocean — no targets",  0.1, 0.3, 50),
                    ("Vessel wake signature",    0.4, 0.75, 51),
                    ("Port infrastructure",      0.35, 0.65, 52),
                    ("Cloud-obscured zone",      0.88, 1.0, 53),
                    ("Coastal radar shadow",     0.15, 0.45, 54),
                    ("Fleet formation",          0.5, 0.85, 55),
                    ("Oil spill anomaly",        0.25, 0.6, 56),
                    ("Island coastline",         0.2, 0.55, 57),
                ],
            },
            "Disaster Response (Earthquake)": {
                "context": "Post-earthquake damage assessment — collapsed structures, road blockages, survivor detection",
                "scene_id": "HADR-TUR-2026-003",
                "tiles": [
                    ("Collapsed building zone",  0.45, 0.8, 60),
                    ("Intact neighborhood",      0.2, 0.45, 61),
                    ("Blocked highway",          0.35, 0.7, 62),
                    ("Dust cloud / debris",      0.7, 0.95, 63),
                    ("Refugee camp forming",     0.3, 0.6, 64),
                    ("Bridge damage",            0.5, 0.85, 65),
                    ("Agricultural field",       0.15, 0.35, 66),
                    ("Hospital / triage area",   0.4, 0.75, 67),
                ],
            },
        }

        def _make_tile(lo: float, hi: float, seed: int) -> np.ndarray:
            return np.random.default_rng(seed).uniform(lo, hi, (13, 256, 256)).astype(np.float32)

        # ── Build tile list ───────────────────────────────────────
        tiles: list[np.ndarray] = []
        tile_names: list[str] = []
        tile_metadata: list[dict] = []

        if input_mode == "Mission Scenarios" and selected_scenarios:
            for scenario_name in selected_scenarios:
                scenario = MISSION_SCENARIOS[scenario_name]
                for i, (label, lo, hi, seed) in enumerate(scenario["tiles"], 1):
                    tiles.append(_make_tile(lo, hi, seed))
                    tile_names.append(f"[{scenario_name.split('(')[0].strip()}] {label}")
                    tile_metadata.append({
                        "context": scenario["context"],
                        "tile_id": f"{scenario['scene_id']}-T{i:03d}",
                        "scene_id": scenario["scene_id"],
                    })

        elif input_mode == "Upload Files" and uploaded_files:
            from PIL import Image as PILImage
            for uf in uploaded_files:
                pil = PILImage.open(uf).convert("RGB")
                arr = np.array(pil, dtype=np.float32) / 255.0
                tiles.append(arr)
                tile_names.append(uf.name)
                tile_metadata.append({
                    "context": mission_context,
                    "tile_id": f"upload_{uf.name}",
                    "scene_id": f"session_{int(time.time())}",
                })

        elif input_mode == "Synthetic" and n_synthetic > 0:
            synthetic_configs = [
                ("Clear sky (high value)", lambda i: np.random.default_rng(i).random((256, 256, 3), dtype=np.float32) * 0.35),
                ("Heavy cloud", lambda i: np.ones((256, 256, 3), dtype=np.float32) * (0.82 + np.random.default_rng(i).random() * 0.1)),
                ("Mixed scene", lambda i: np.random.default_rng(i + 100).random((256, 256, 3), dtype=np.float32) * 0.55),
                ("Urban / industrial", lambda i: np.random.default_rng(i + 200).random((256, 256, 3), dtype=np.float32) * 0.25 + 0.05),
                ("Water body", lambda i: np.random.default_rng(i + 300).random((256, 256, 3), dtype=np.float32) * 0.15 + 0.1),
            ]
            for i in range(n_synthetic):
                idx = i % len(synthetic_configs)
                name, gen = synthetic_configs[idx]
                tiles.append(gen(i))
                tile_names.append(f"Synthetic: {name}")
                tile_metadata.append({
                    "context": mission_context,
                    "tile_id": f"tile_{i:04d}",
                    "scene_id": f"session_{int(time.time())}",
                })

        if tiles:
            n_scenarios = len(selected_scenarios) if input_mode == "Mission Scenarios" else 0
            label = f"**{len(tiles)} tile(s)** across **{n_scenarios} scenario(s)**" if n_scenarios else f"**{len(tiles)} tile(s) ready**"
            st.markdown(label)
        else:
            st.info("Select scenarios, upload images, or enable synthetic tiles in the sidebar to begin.")

        # ── Process button ────────────────────────────────────────
        if tiles and st.button("Run Triage on All Tiles", type="primary", use_container_width=True):
            engine = EdgeTriageEngine(audit=False)
            st.session_state.results = []
            st.session_state.collector = MetricsCollector()
            st.session_state.processing_log = []

            progress = st.progress(0, text="Initializing pipeline...")
            for i, tile in enumerate(tiles):
                progress.progress(
                    (i + 1) / len(tiles),
                    text=f"Processing tile {i + 1}/{len(tiles)}: {tile_names[i]}",
                )
                meta = tile_metadata[i] if i < len(tile_metadata) else {
                    "context": mission_context,
                    "tile_id": f"tile_{i:04d}",
                    "scene_id": f"session_{int(time.time())}",
                }
                result = engine.process_tile(tile, meta)
                st.session_state.collector.record(tile, result)
                st.session_state.results.append((tile_names[i], tile, result))
                st.session_state.processing_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "tile": tile_names[i],
                    "scenario": meta.get("scene_id", ""),
                    "decision": "KEEP" if result.keep else "FILTER",
                    "score": round(result.final_score, 3),
                })

            progress.empty()

        # Display results
        results = st.session_state.results
        if results:
            # ── Overall KPI row ────────────────────────────────────
            summary = st.session_state.collector.metrics.summary()
            total_items = sum(
                (r.detection_result.count if r.detection_result else 0)
                for _, _, r in results
            )
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Tiles", summary["tiles_processed"])
            k2.metric("Keep Rate", f"{summary['keep_rate'] * 100:.0f}%")
            k3.metric("Items Detected", total_items)
            k4.metric("BW Saved", f"{summary['bandwidth_saved_percent']:.1f}%")
            k5.metric("Avg Power", f"{summary['avg_power_watts']:.1f} W")
            k6.metric("Avg Latency", f"{summary['avg_inference_ms']:.1f} ms")

            st.markdown("---")

            # ── Group results by scenario ──────────────────────────
            # Extract scenario from tile name: "[Scenario] tile label"
            from collections import OrderedDict
            grouped: OrderedDict[str, list[tuple[int, str, np.ndarray, TriageResult]]] = OrderedDict()
            for i, (name, tile, result) in enumerate(results):
                if name.startswith("["):
                    scenario_key = name.split("]")[0].strip("[").strip()
                    tile_label = name.split("]")[1].strip()
                else:
                    scenario_key = "Results"
                    tile_label = name
                grouped.setdefault(scenario_key, []).append((i, tile_label, tile, result))

            for scenario_key, group in grouped.items():
                # Scenario header with stats
                group_kept = sum(1 for _, _, _, r in group if r.keep)
                group_filtered = len(group) - group_kept
                group_bw = np.mean([r.bandwidth_saved_percent for _, _, _, r in group])
                group_items = sum(
                    (r.detection_result.count if r.detection_result else 0)
                    for _, _, _, r in group
                )

                scenario_colors = {
                    "Wildfire Detection": ("#ff7043", "🔥"),
                    "Defense ISR":        ("#42a5f5", "◉"),
                    "Disaster Response":  ("#ef5350", "⚠"),
                }
                bar_color, icon = scenario_colors.get(scenario_key, ("#78909c", "◇"))

                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {bar_color}1a 0%, transparent 100%);
                            border-left: 3px solid {bar_color};
                            color: var(--text-primary); padding: 0.85rem 1.2rem;
                            border-radius: 8px; margin: 1.4rem 0 0.6rem;
                            border-top: 1px solid var(--border);
                            border-right: 1px solid var(--border);
                            border-bottom: 1px solid var(--border);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: {bar_color}; font-size: 1.1rem; margin-right: 8px;">{icon}</span>
                            <strong style="font-size: 1.05rem; letter-spacing: 0.04em;">{scenario_key}</strong>
                            <span style="color: var(--text-muted); margin-left: 0.8rem; font-size: 0.8rem;
                                         font-family: 'SF Mono', monospace;">
                                {len(group)} TILES
                            </span>
                        </div>
                        <div style="font-size: 0.78rem; font-family: 'SF Mono', monospace; font-weight: 600;">
                            <span style="background: rgba(0,214,143,0.15); color: var(--success);
                                         border: 1px solid var(--success);
                                         padding: 3px 10px; border-radius: 4px; margin-right: 6px;">
                                KEEP {group_kept}
                            </span>
                            <span style="background: rgba(255,82,82,0.10); color: var(--danger);
                                         border: 1px solid var(--danger);
                                         padding: 3px 10px; border-radius: 4px; margin-right: 6px;">
                                FILTER {group_filtered}
                            </span>
                            <span style="background: rgba(74,144,226,0.12); color: var(--accent-bright);
                                         border: 1px solid var(--accent);
                                         padding: 3px 10px; border-radius: 4px; margin-right: 6px;">
                                ★ {group_items} ITEMS
                            </span>
                            <span style="color: var(--text-muted);">BW SAVED · {group_bw:.0f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Per-tile result cards within scenario
                for idx, (i, tile_label, tile, result) in enumerate(group):
                    decision_class = "keep" if result.keep else "filter"
                    decision_badge = "decision-keep" if result.keep else "decision-filter"
                    decision_text = "KEEP" if result.keep else "FILTER"

                    with st.container():
                        st.markdown(f'<div class="tile-card {decision_class}">', unsafe_allow_html=True)
                        cols = st.columns([1.2, 1, 1, 1, 2])

                        with cols[0]:
                            st.markdown(f"**Tile {idx+1}:** {tile_label}")
                            st.markdown(
                                f'<span class="{decision_badge}">{decision_text}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Score: {result.final_score:.3f}")
                            # Items-of-interest chip
                            if result.detection_result and result.detection_result.count > 0:
                                st.markdown(
                                    f'<div style="margin-top:6px; '
                                    f'background:rgba(74,144,226,0.12); '
                                    f'color:var(--accent-bright); '
                                    f'border:1px solid var(--accent); '
                                    f'padding:3px 12px; border-radius:4px; font-size:0.78rem; '
                                    f'display:inline-block; font-weight:600; '
                                    f'font-family:\'SF Mono\', monospace; letter-spacing:0.04em;">'
                                    f'★ {result.detection_result.summary()}</div>',
                                    unsafe_allow_html=True,
                                )

                        with cols[1]:
                            cloud = result.cnn_results.get("cloud_fraction", 0)
                            st.markdown("**Cloud**")
                            st.progress(min(cloud, 1.0))
                            st.caption(f"{cloud:.1%}")

                        with cols[2]:
                            anomaly = result.cnn_results.get("anomaly_score", 0)
                            st.markdown("**Anomaly**")
                            st.progress(min(anomaly, 1.0))
                            st.caption(f"{anomaly:.1%}")

                        with cols[3]:
                            value = result.cnn_results.get("value_score", 0)
                            st.markdown("**Value**")
                            st.progress(min(value, 1.0))
                            st.caption(f"{value:.1%}")

                        with cols[4]:
                            st.markdown("**Explanation**")
                            explanation_lines = result.explanation.split("\n")
                            st.caption("\n".join(explanation_lines[:4]))
                            if result.actions:
                                st.caption("Actions: " + " | ".join(result.actions[:3]))

                        # Expandable details
                        with st.expander(f"Full details — {tile_label}"):
                            d1, d2, d3 = st.columns(3)
                            d1.markdown(f"""
                            - **Backend:** `{result.cnn_results.get('backend', '?')}`
                            - **Inference:** {result.cnn_results.get('inference_ms', 0):.2f} ms
                            - **Power:** {result.power_used_watts:.2f} W
                            """)
                            d2.markdown(f"""
                            - **Tile ID:** `{result.tile_id}`
                            - **Scene ID:** `{result.scene_id}`
                            - **BW Saved:** {result.bandwidth_saved_percent:.1f}%
                            """)
                            d3.markdown(f"""
                            - **Input Hash:** `{result.input_hash[:16]}...`
                            - **Timestamp:** {result.processing_timestamp_utc}
                            - **Timed Out:** {result.inference_timed_out}
                            """)

                            if result.cnn_results.get("backend") == "stub":
                                st.warning(
                                    "Stub backend active — scores are simulated. "
                                    "Deploy a trained ONNX model for production inference."
                                )

                            st.markdown("**Full explanation:**")
                            st.code(result.explanation, language=None)

                            # ── Detection details ──────────────────
                            det = result.detection_result
                            if det and det.count > 0:
                                st.markdown(f"**Items of interest ({det.count}):**")
                                det_rows = []
                                for d in det.detections:
                                    det_rows.append({
                                        "Class": d.class_name,
                                        "Confidence": f"{d.confidence:.2%}",
                                        "BBox (x1,y1,x2,y2)": f"({d.bbox[0]:.2f}, {d.bbox[1]:.2f}, {d.bbox[2]:.2f}, {d.bbox[3]:.2f})",
                                    })
                                import pandas as pd
                                st.dataframe(
                                    pd.DataFrame(det_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.caption(
                                    f"Detection backend: `{det.backend}` | "
                                    f"Inference: {det.inference_ms:.1f} ms"
                                )

                            # ── Tile thumbnail with bounding-box overlay ────
                            thumb = _tile_to_rgb_uint8(tile)
                            if thumb is not None:
                                if det and det.count > 0:
                                    thumb = _draw_bboxes(thumb, det.detections)
                                st.image(
                                    thumb,
                                    caption=f"Input tile with overlays ({thumb.shape[1]}x{thumb.shape[0]})",
                                    width=280,
                                )

                        st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  TAB 2: LIVE FEED — real satellite data
    # ════════════════════════════════════════════════════════════
    with tab_live:
        from .live_data import (
            AOI_PRESETS, GOES_SECTOR_PRESETS, SOURCES,
            FIRMSFireSource, NOAAGOESSource, Sentinel1Source, Sentinel2Source,
        )
        from .secrets_store import KNOWN_KEYS, secrets as _secrets_store
        from .ship_detector import AISCrossReference, MaritimeShipDetector

        st.markdown("### Live Satellite Feed")
        st.caption(
            "Pulls real imagery from public APIs. No authentication required for "
            "Sentinel-2 or NOAA GOES. FIRMS needs a free MAP_KEY — set it below."
        )

        # ── API Key setup expander ──────────────────────────────
        firms_status = _secrets_store.all_status().get("firms", {})
        key_set = firms_status.get("set", False)
        key_source = firms_status.get("source", "unset")
        key_redacted = firms_status.get("redacted", "")

        status_dot = "&#9679;"
        if key_set and key_source == "env":
            status_html = f'<span style="color:#2e7d32;">{status_dot}</span> Set via env var'
        elif key_set and key_source == "file":
            status_html = f'<span style="color:#2e7d32;">{status_dot}</span> Saved locally'
        else:
            status_html = f'<span style="color:#c62828;">{status_dot}</span> Not set'

        with st.expander(
            "NASA FIRMS MAP_KEY setup  " + ("✓" if key_set else "— required for fires"),
            expanded=not key_set,
        ):
            st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)
            if key_set:
                st.caption(f"Key value (redacted): `{key_redacted}`  |  Source: `{key_source}`")

            st.markdown("""
            **Setup steps:**
            1. Visit [firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/)
            2. Enter your email; NASA emails you a MAP_KEY within a minute
            3. Paste it below and click **Save**, or export it in your shell as
               `export EDGE_TRIAGE_FIRMS_KEY=your_key_here`
            4. Click **Test** to verify the key works
            """)

            setup_cols = st.columns([3, 1, 1, 1])
            with setup_cols[0]:
                entered_key = st.text_input(
                    "Paste MAP_KEY here",
                    value="",
                    type="password",
                    help="The key is stored locally at ~/.edge_triage/secrets.json with 0600 perms. "
                         "It's gitignored and never logged in plain text.",
                    placeholder="e.g. 1a2b3c4d5e6f7g8h9i0j",
                    label_visibility="collapsed",
                    key="firms_key_input",
                )
            with setup_cols[1]:
                save_clicked = st.button("Save", type="primary", use_container_width=True)
            with setup_cols[2]:
                test_clicked = st.button("Test", use_container_width=True)
            with setup_cols[3]:
                clear_clicked = st.button("Clear", use_container_width=True,
                                          disabled=key_source != "file")

            if save_clicked and entered_key.strip():
                _secrets_store.set("firms", entered_key.strip())
                st.success("MAP_KEY saved to local secrets file.")
                st.rerun()
            elif save_clicked:
                st.warning("Paste a key before clicking Save.")

            if test_clicked:
                with st.spinner("Pinging FIRMS status endpoint..."):
                    ok, msg = _secrets_store.test("firms")
                if ok:
                    st.success(f"✓ {msg}")
                else:
                    st.error(f"✗ {msg}")

            if clear_clicked:
                _secrets_store.delete("firms")
                st.info("Cleared saved MAP_KEY from local secrets file.")
                st.rerun()

            if key_source == "env":
                st.info(
                    "Key is coming from the `EDGE_TRIAGE_FIRMS_KEY` environment variable. "
                    "Unset it to use the local file instead, or set it to override the file."
                )

        live_col1, live_col2 = st.columns([1, 2])

        with live_col1:
            source_name = st.selectbox(
                "Data source",
                list(SOURCES.keys()),
                help=(
                    "Sentinel-2: global 10 m, ~2-5 day latency. "
                    "GOES-18: Americas, ~10 min latency. "
                    "FIRMS: global fires, ~3 h latency."
                ),
            )

            if source_name.startswith("Sentinel-2"):
                aoi_name = st.selectbox("Area of Interest", list(AOI_PRESETS.keys()))
                max_cloud = st.slider("Max cloud cover (%)", 0, 100, 40, 5)
                max_age = st.slider("Max age (days)", 1, 60, 14, 1)
                max_items = st.slider("Max scenes", 1, 12, 6, 1)
            elif source_name.startswith("Sentinel-1"):
                aoi_name = st.selectbox(
                    "Area of Interest",
                    list(AOI_PRESETS.keys()),
                    index=list(AOI_PRESETS.keys()).index("Strait of Hormuz")
                    if "Strait of Hormuz" in AOI_PRESETS else 0,
                )
                s1_max_age = st.slider("Max age (days)", 1, 30, 7, 1)
                s1_max_items = st.slider("Max scenes", 1, 8, 4, 1)
                polarization = st.selectbox(
                    "Polarization",
                    ["VV", "VH", "HH", "HV"],
                    index=0,
                    help="VV = best for ships on calm water (defense ISR standard)",
                )
                run_ship_detect = st.toggle(
                    "Run SAR ship detection", value=True,
                    help="Classical CFAR + connected-components vessel detector",
                )
                run_ais = st.toggle(
                    "Simulate AIS cross-reference (flag dark ships)", value=True,
                    help="Real AIS integration is a paid feed — this demos the pattern",
                )
            elif source_name.startswith("NOAA GOES"):
                sector_name = st.selectbox("GOES sector", list(GOES_SECTOR_PRESETS.keys()))
            else:  # FIRMS
                aoi_name = st.selectbox("Area of Interest", list(AOI_PRESETS.keys()))
                firms_days = st.slider("Days of fires", 1, 10, 1, 1)

            fetch_clicked = st.button("Fetch & Triage", type="primary", use_container_width=True)

        with live_col2:
            if fetch_clicked:
                engine = EdgeTriageEngine(audit=False)

                ship_results: list[tuple[object, object]] = []  # (item, ShipDetectionResult)
                with st.spinner(f"Fetching from {source_name}..."):
                    if source_name.startswith("Sentinel-2"):
                        src = Sentinel2Source(
                            max_cloud_cover=float(max_cloud),
                            max_age_days=int(max_age),
                            limit=int(max_items),
                        )
                        items = src.fetch(AOI_PRESETS[aoi_name])
                    elif source_name.startswith("Sentinel-1"):
                        s1src = Sentinel1Source(
                            max_age_days=int(s1_max_age),
                            limit=int(s1_max_items),
                            polarization=polarization,
                        )
                        items = s1src.fetch(AOI_PRESETS[aoi_name])

                        if run_ship_detect and items:
                            detector = MaritimeShipDetector()
                            ais = AISCrossReference(dark_ship_rate=0.25, seed=7) if run_ais else None
                            for it in items:
                                sr = detector.detect(it.image, scene_bbox=it.bbox)
                                if ais is not None:
                                    sr = ais.correlate(sr)
                                ship_results.append((it, sr))
                    elif source_name.startswith("NOAA GOES"):
                        gsrc = NOAAGOESSource()
                        items = gsrc.fetch(GOES_SECTOR_PRESETS[sector_name])
                    else:
                        fsrc = FIRMSFireSource(days=int(firms_days))
                        if not fsrc.map_key:
                            st.error(
                                "No FIRMS MAP_KEY found. Open the "
                                "**NASA FIRMS MAP_KEY setup** expander above to add one."
                            )
                            items = []
                        else:
                            items = fsrc.fetch(AOI_PRESETS[aoi_name])

                if not items:
                    st.warning(
                        "No scenes returned. Possible causes: no recent imagery for this AOI, "
                        "or a network issue."
                    )
                else:
                    st.success(f"Fetched {len(items)} scene(s) — running triage pipeline...")

                    # Index ship results by scene_id for quick lookup
                    ship_by_scene = {
                        it.scene_id: sr for it, sr in ship_results
                    } if ship_results else {}

                    live_results = []
                    progress = st.progress(0)
                    for i, item in enumerate(items):
                        progress.progress((i + 1) / len(items), text=f"Triaging {item.name}...")
                        meta = {
                            "context": mission_context,
                            "tile_id": item.scene_id or f"live_{i}",
                            "scene_id": item.scene_id,
                            "acquisition_time": item.acquired_utc,
                        }
                        # If we have SAR ship detection, inject it into metadata
                        # so the agent reasons about ship counts specifically
                        ship_result = ship_by_scene.get(item.scene_id)
                        if ship_result is not None:
                            meta["detection_count"] = ship_result.count
                            meta["detection_summary"] = ship_result.summary()
                            meta["detection_classes"] = (
                                ["ship", "dark-ship"] if ship_result.dark_ship_count
                                else ["ship"]
                            )
                        result = engine.process_tile(item.image, meta)
                        st.session_state.collector.record(item.image, result)
                        # Replace the generic detection_result with our ship result
                        if ship_result is not None:
                            result.detection_result = ship_result.to_detection_result()
                        live_results.append((item, result, ship_result))
                        st.session_state.results.append(
                            (f"[LIVE] {item.name}", item.image, result),
                        )
                        st.session_state.processing_log.append({
                            "time": time.strftime("%H:%M:%S"),
                            "tile": item.name,
                            "scenario": item.source,
                            "decision": "KEEP" if result.keep else "FILTER",
                            "score": round(result.final_score, 3),
                        })

                    progress.empty()

                    # Render live results
                    for item, r, ship_result in live_results:
                        decision_class = "keep" if r.keep else "filter"
                        decision_badge = "decision-keep" if r.keep else "decision-filter"
                        decision_text = "KEEP" if r.keep else "FILTER"

                        st.markdown(f'<div class="tile-card {decision_class}" style="margin: 0.8rem 0;">', unsafe_allow_html=True)
                        hcols = st.columns([1.5, 1, 1.5])

                        with hcols[0]:
                            st.markdown(f"**{item.name}**")
                            st.caption(
                                f"Source: `{item.source}` | "
                                f"Acquired: {item.acquired_utc[:19]} | "
                                f"Cloud: {(item.cloud_cover_pct or 0):.0f}%"
                            )
                            st.markdown(
                                f'<span class="{decision_badge}">{decision_text}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Score: {r.final_score:.3f}")
                            if r.detection_result and r.detection_result.count > 0:
                                st.markdown(
                                    f'<div style="margin-top:4px; background:#0d47a1; color:#fff; '
                                    f'padding:3px 10px; border-radius:12px; font-size:0.78rem; '
                                    f'display:inline-block; font-weight:600;">'
                                    f'&#9733; {r.detection_result.summary()}</div>',
                                    unsafe_allow_html=True,
                                )

                        with hcols[1]:
                            # Thumbnail with bbox overlay
                            thumb = _tile_to_rgb_uint8(item.image)
                            if thumb is not None:
                                if r.detection_result and r.detection_result.count > 0:
                                    thumb = _draw_bboxes(thumb, r.detection_result.detections)
                                st.image(thumb, width=220)

                        with hcols[2]:
                            st.markdown("**Triage Result**")
                            cnn = r.cnn_results
                            st.caption(
                                f"Cloud: {cnn.get('cloud_fraction', 0):.1%} | "
                                f"Anomaly: {cnn.get('anomaly_score', 0):.1%} | "
                                f"Value: {cnn.get('value_score', 0):.1%}"
                            )
                            st.caption(f"Backend: `{cnn.get('backend', '?')}` | "
                                       f"BW saved: {r.bandwidth_saved_percent:.1f}%")
                            if item.preview_url:
                                st.caption(f"[Source preview]({item.preview_url})")
                            if r.explanation:
                                with st.expander("Agent explanation"):
                                    st.code(r.explanation, language=None)

                        # ── SAR Ship Detail Panel (only for Sentinel-1 scenes) ──
                        if ship_result is not None and ship_result.count > 0:
                            dark = ship_result.dark_ship_count
                            dark_banner = (
                                f'<span style="background:rgba(255,82,82,0.18); color:var(--danger); '
                                f'border:1px solid var(--danger); padding:3px 12px; border-radius:4px; '
                                f'font-weight:700; font-family:\'SF Mono\',monospace; letter-spacing:0.06em;">'
                                f'⚠ {dark} DARK SHIPS</span>'
                                if dark else ""
                            )
                            st.markdown(
                                f'<div style="background: linear-gradient(90deg, rgba(74,144,226,0.18) 0%, transparent 100%); '
                                f'border-left:3px solid var(--accent); '
                                f'border-top:1px solid var(--border); border-right:1px solid var(--border); '
                                f'border-bottom:1px solid var(--border); '
                                f'padding:0.7rem 1.1rem; border-radius:8px; margin-top:0.5rem; '
                                f'display:flex; justify-content:space-between; align-items:center;">'
                                f'<div>'
                                f'<span style="color: var(--accent-bright); font-size:0.7rem; '
                                f'font-weight:700; letter-spacing:0.15em; text-transform:uppercase;">◉ Maritime ISR</span>'
                                f'<div style="margin-top:3px; font-size:1.0rem; font-weight:600;">'
                                f'{ship_result.count} vessel{"s" if ship_result.count != 1 else ""} detected '
                                f'<span style="color: var(--text-muted); font-size:0.8rem; font-family:\'SF Mono\',monospace;">'
                                f'· {ship_result.inference_ms:.0f} ms</span></div>'
                                f'</div>'
                                f'<div>{dark_banner}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            with st.expander(
                                f"Vessel detection details — {ship_result.summary()}",
                                expanded=bool(dark),
                            ):
                                import pandas as pd
                                rows = []
                                for j, s in enumerate(ship_result.ships, 1):
                                    ais_str = (
                                        "✓ AIS match" if s.ais_match
                                        else "⚠ DARK (no AIS)" if s.ais_match is False
                                        else "—"
                                    )
                                    rows.append({
                                        "#": j,
                                        "Lat": f"{s.lat:.3f}" if s.lat is not None else "—",
                                        "Lon": f"{s.lon:.3f}" if s.lon is not None else "—",
                                        "Est. length (m)": f"{s.length_m_est:.0f}",
                                        "Confidence": f"{s.confidence:.2f}",
                                        "Area (px)": s.area_pixels,
                                        "Status": ais_str,
                                    })
                                st.dataframe(
                                    pd.DataFrame(rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                # Plot detections on a small map if coordinates exist
                                map_rows = [
                                    {"lat": s.lat, "lon": s.lon,
                                     "color": [255, 60, 60] if s.ais_match is False else [60, 180, 255]}
                                    for s in ship_result.ships
                                    if s.lat is not None and s.lon is not None
                                ]
                                if map_rows:
                                    import pandas as pd
                                    map_df = pd.DataFrame(map_rows)
                                    try:
                                        import pydeck as pdk
                                        layer = pdk.Layer(
                                            "ScatterplotLayer",
                                            data=map_df,
                                            get_position="[lon, lat]",
                                            get_fill_color="color",
                                            get_radius=800,
                                            pickable=True,
                                        )
                                        lat_c = float(map_df["lat"].mean())
                                        lon_c = float(map_df["lon"].mean())
                                        view = pdk.ViewState(
                                            longitude=lon_c, latitude=lat_c,
                                            zoom=7, pitch=0,
                                        )
                                        st.pydeck_chart(pdk.Deck(
                                            layers=[layer],
                                            initial_view_state=view,
                                            map_style="light",
                                        ))
                                        st.caption("Blue = AIS match (known commercial traffic). "
                                                   "Red = dark ship (no transponder) — high interest.")
                                    except Exception:
                                        # Fallback to st.map if pydeck isn't available
                                        st.map(map_df[["lat", "lon"]])

                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(
                    "Pick a data source and AOI in the left panel, then click **Fetch & Triage** "
                    "to pull real imagery and run it through the edge AI pipeline."
                )
                st.markdown("""
                **What each source provides:**
                - **Sentinel-2 L2A** — Global 10 m optical multispectral imagery via Element84's
                  free Earth Search STAC API. Latency 2-5 days. Best for land / change detection.
                - **Sentinel-1 SAR** — Global all-weather radar (day/night, cloud-penetrating).
                  Defense / maritime ISR standard. Ships appear as bright scatterers on dark
                  water → classical CFAR detection. Paired with a simulated AIS overlay to
                  flag **dark ships** (no transponder broadcast — the real intel signal).
                - **NOAA GOES-18** — Geostationary weather imagery over the Americas, updated
                  every ~10 minutes. Best for near-real-time fire smoke plumes, storm tracking.
                - **NASA FIRMS** — Active fire hotspot detections from VIIRS/MODIS, ~3 h latency.
                  Requires a free MAP_KEY — use the setup panel above to add one.
                """)

    # ════════════════════════════════════════════════════════════
    #  TAB 3: ANALYTICS
    # ════════════════════════════════════════════════════════════
    with tab_analytics:
        results = st.session_state.results
        if not results:
            st.info("Run the triage pipeline to see analytics.")
        else:
            summary = st.session_state.collector.metrics.summary()

            # Top-level summary
            st.markdown("### Session Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Input", f"{summary['total_input_MB']:.2f} MB")
            m2.metric("Total Output", f"{summary['total_output_MB']:.2f} MB")
            m3.metric("Data Reduced", f"{summary['bandwidth_saved_percent']:.1f}%")
            m4.metric("Tiles Filtered", f"{summary['tiles_filtered']}/{summary['tiles_processed']}")

            st.markdown("---")

            # Power budget gauge
            st.markdown("### Power Budget Utilization")
            power_ratio = min(summary["avg_power_watts"] / config.POWER_BUDGET_WATTS, 1.0) if config.POWER_BUDGET_WATTS > 0 else 0
            pcol1, pcol2 = st.columns([3, 1])
            with pcol1:
                st.progress(power_ratio)
            with pcol2:
                st.markdown(f"**{summary['avg_power_watts']:.1f}W** / {config.POWER_BUDGET_WATTS:.0f}W")

            st.markdown("---")

            # Per-tile breakdown chart
            st.markdown("### Per-Tile Score Breakdown")
            per_tile = st.session_state.collector.metrics.per_tile
            if per_tile:
                import pandas as pd
                df = pd.DataFrame(per_tile)
                df.index = [f"Tile {i+1}" for i in range(len(df))]
                df_scores = df[["cloud", "score"]].rename(columns={
                    "cloud": "Cloud Fraction",
                    "score": "Triage Score",
                })
                st.bar_chart(df_scores, height=300)

                # Decision distribution
                st.markdown("### Decision Distribution")
                c1, c2 = st.columns(2)
                kept = sum(1 for r in per_tile if r["keep"])
                filtered = len(per_tile) - kept
                c1.metric("KEEP", kept)
                c2.metric("FILTER", filtered)

                # Detailed table
                st.markdown("### Tile Detail Table")
                display_df = df.copy()
                display_df["decision"] = display_df["keep"].map({True: "KEEP", False: "FILTER"})
                display_df["bw_saved"] = display_df["bw_saved"].map("{:.1f}%".format)
                display_df["score"] = display_df["score"].map("{:.3f}".format)
                display_df["cloud"] = display_df["cloud"].map("{:.1%}".format)
                display_df["power_w"] = display_df["power_w"].map("{:.2f}".format)
                display_df["agent"] = display_df["agent"].map({True: "Yes", False: "No"})
                st.dataframe(
                    display_df[["decision", "score", "cloud", "power_w", "bw_saved", "agent"]],
                    use_container_width=True,
                )

    # ════════════════════════════════════════════════════════════
    #  TAB 3: AUDIT TRAIL
    # ════════════════════════════════════════════════════════════
    with tab_audit:
        st.markdown("### Audit Log")
        log_path = config.AUDIT_LOG_PATH

        if log_path.exists() and log_path.stat().st_size > 0:
            st.caption(f"Log file: `{log_path}` ({log_path.stat().st_size / 1024:.1f} KB)")

            # Load and display records
            records = []
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        json_part = line.rsplit("\t", 1)[0]
                    else:
                        json_part = line
                    try:
                        records.append(json.loads(json_part))
                    except json.JSONDecodeError:
                        continue

            if records:
                st.markdown(f"**{len(records)} audit records**")

                # Summary table
                import pandas as pd
                audit_df = pd.DataFrame([{
                    "Timestamp": r.get("timestamp_utc", ""),
                    "Scene": r.get("scene_id", ""),
                    "Tile": r.get("tile_id", ""),
                    "Decision": "KEEP" if r.get("keep") else "FILTER",
                    "Score": round(r.get("final_score", 0), 3),
                    "Cloud": f"{r.get('cnn_cloud_fraction', 0):.1%}",
                    "Backend": r.get("cnn_backend", ""),
                    "HMAC": "Yes" if "\t" in open(log_path).readline() else "Unknown",
                } for r in records[-50:]])  # Show last 50

                st.dataframe(audit_df, use_container_width=True, height=400)

                # HMAC verification
                st.markdown("---")
                st.markdown("### Integrity Verification")
                if st.button("Verify HMAC Integrity"):
                    from .audit import AuditLogger
                    results = AuditLogger.verify_log(log_path)
                    passed = sum(1 for _, ok, _ in results if ok)
                    failed = sum(1 for _, ok, _ in results if not ok)
                    total = len(results)

                    if failed == 0:
                        st.success(f"All {total} records passed HMAC verification.")
                    else:
                        st.error(f"{failed}/{total} records FAILED verification.")
                        for line_num, ok, detail in results:
                            if not ok:
                                st.warning(f"Line {line_num}: {detail}")
            else:
                st.info("Audit log exists but contains no parseable records.")
        else:
            st.info(
                "No audit log found. Records are written automatically when "
                "the engine runs with `audit=True` (default in production)."
            )

        # Processing log from this session
        if st.session_state.processing_log:
            st.markdown("---")
            st.markdown("### This Session")
            import pandas as pd
            session_df = pd.DataFrame(st.session_state.processing_log)
            st.dataframe(session_df, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    #  TAB 4: SYSTEM STATUS
    # ════════════════════════════════════════════════════════════
    with tab_system:
        s1, s2 = st.columns(2)

        with s1:
            st.markdown("### Inference Backend")

            # Detect backend
            try:
                from .inference import QuantizedInferencer, TORCH_AVAILABLE, ORT_AVAILABLE
                inf = QuantizedInferencer()
                backend = inf.backend
            except Exception:
                backend = "unknown"
                TORCH_AVAILABLE = False
                ORT_AVAILABLE = False

            if backend == "stub":
                st.markdown('<span class="status-dot amber"></span> **Stub** (simulated)', unsafe_allow_html=True)
                st.caption("Deploy a trained ONNX model to models/ for real inference.")
            elif backend.startswith("onnx"):
                st.markdown('<span class="status-dot green"></span> **ONNX Runtime**', unsafe_allow_html=True)
                st.caption(f"Provider: {backend.split(':')[-1]}")
            elif backend == "torch_int8":
                st.markdown('<span class="status-dot green"></span> **PyTorch INT8**', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-dot red"></span> **Unknown**', unsafe_allow_html=True)

            st.markdown(f"""
            | Component | Status |
            |-----------|--------|
            | PyTorch | {'Available' if TORCH_AVAILABLE else 'Not installed'} |
            | ONNX Runtime | {'Available' if ORT_AVAILABLE else 'Not installed'} |
            | Agent Enabled | {'Yes' if config.AGENT_ENABLED else 'No'} |
            | SLM Enabled | {'Yes' if config.AGENT_SLM_ENABLED else 'No'} |
            """)

        with s2:
            st.markdown("### Model Registry")
            try:
                registry = ModelRegistry()
                versions = registry.list_versions()
                if versions:
                    import pandas as pd
                    reg_df = pd.DataFrame([{
                        "Version": v["version"],
                        "Active": ">> Active" if v.get("active") else "",
                        "Dataset": v.get("training_dataset", ""),
                        "Val Loss": round(v.get("val_loss", 0), 4),
                        "SHA-256": v.get("sha256", "")[:16] + "...",
                        "Created": v.get("created_utc", "")[:10],
                    } for v in versions])
                    st.dataframe(reg_df, use_container_width=True, hide_index=True)

                    # Integrity check
                    if st.button("Verify Model Integrity"):
                        checks = registry.verify_all()
                        all_ok = all(ok for _, ok, _ in checks)
                        if all_ok:
                            st.success(f"All {len(checks)} model(s) passed SHA-256 verification.")
                        else:
                            for ver, ok, detail in checks:
                                if not ok:
                                    st.error(f"v{ver}: {detail}")
                else:
                    st.info("No models registered. Use `scripts/train_cloud_mask.py` to train and register.")
            except Exception as e:
                st.warning(f"Registry unavailable: {e}")

        st.markdown("---")

        # Configuration dump
        st.markdown("### Active Configuration")
        config_data = {
            "POWER_BUDGET_WATTS": config.POWER_BUDGET_WATTS,
            "TARGET_TOPS_WATT": config.TARGET_TOPS_WATT,
            "MAX_INFERENCE_MS": config.MAX_INFERENCE_MS,
            "KEEP_SCORE_THRESHOLD": config.KEEP_SCORE_THRESHOLD,
            "MAX_CLOUD_FRACTION": config.MAX_CLOUD_FRACTION,
            "AGENT_ENABLED": config.AGENT_ENABLED,
            "AGENT_ACTIVATION_THRESHOLD": config.AGENT_ACTIVATION_THRESHOLD,
            "AGENT_MAX_STEPS": config.AGENT_MAX_STEPS,
            "AGENT_SLM_ENABLED": config.AGENT_SLM_ENABLED,
            "MODE": config.MODE,
            "TILE_SIZE": config.TILE_SIZE,
            "INPUT_CHANNELS": config.INPUT_CHANNELS,
            "MODEL_DIR": str(config.MODEL_DIR),
            "DEFAULT_MODEL": config.DEFAULT_MODEL,
            "AUDIT_LOG_PATH": str(config.AUDIT_LOG_PATH),
        }
        st.json(config_data)

        st.markdown("---")
        st.caption(
            f"Edge AI Satellite Data Triage v{__version__} | "
            "MobileNetV3-Small + ReAct Agent + Optional SLM | "
            f"Target: <{config.POWER_BUDGET_WATTS:.0f}W, >{config.TARGET_TOPS_WATT:.0f} TOPS/W"
        )


# ══════════════════════════════════════════════════════════════════════════
#  CLI DEMO
# ══════════════════════════════════════════════════════════════════════════


def _run_cli_demo() -> None:
    """Quick CLI demo without Streamlit."""
    import numpy as np

    from .config import config
    from .metrics import MetricsCollector
    from .triage import EdgeTriageEngine
    from . import __version__

    width = 62
    print("=" * width)
    print(f"  Edge AI Satellite Data Triage v{__version__}")
    print(f"  Mode: {config.MODE} | Power budget: {config.POWER_BUDGET_WATTS}W")
    print(f"  Agent: {'ON' if config.AGENT_ENABLED else 'OFF'} | "
          f"SLM: {'ON' if config.AGENT_SLM_ENABLED else 'OFF'}")
    print("=" * width)

    engine = EdgeTriageEngine(audit=False)
    collector = MetricsCollector()

    backend = "unknown"
    test_tiles = {
        "Clear sky (high value)": np.random.default_rng(42).random((256, 256, 3), dtype=np.float32) * 0.35,
        "Heavy cloud": np.ones((256, 256, 3), dtype=np.float32) * 0.85,
        "Mixed / anomaly": np.random.default_rng(7).random((256, 256, 3), dtype=np.float32) * 0.3,
    }

    for name, tile in test_tiles.items():
        print(f"\n--- {name} ---")
        result = engine.process_tile(tile, {"context": "Wildfire monitoring pass"})
        collector.record(tile, result)
        backend = result.cnn_results.get("backend", backend)

        decision = "KEEP " if result.keep else "FILTER"
        print(f"  Decision  : {decision}")
        print(f"  Score     : {result.final_score:.3f}")
        print(f"  Cloud     : {result.cnn_results.get('cloud_fraction', 0):.1%}")
        print(f"  Anomaly   : {result.cnn_results.get('anomaly_score', 0):.1%}")
        print(f"  Value     : {result.cnn_results.get('value_score', 0):.1%}")
        print(f"  BW Saved  : {result.bandwidth_saved_percent:.1f}%")
        print(f"  Power     : {result.power_used_watts:.1f} W")
        print(f"  Backend   : {result.cnn_results.get('backend', '?')}")
        if result.explanation:
            for line in result.explanation.split("\n")[:3]:
                print(f"  {line}")

    if backend == "stub":
        print(f"\n{'!' * width}")
        print("  WARNING: Stub backend (simulated scores).")
        print("  Deploy ONNX model to models/ for production.")
        print(f"{'!' * width}")

    print(f"\n{'=' * width}")
    print("  Session Summary")
    print(f"{'=' * width}")
    for k, v in collector.metrics.summary().items():
        print(f"  {k:30s}: {v}")


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point for ``python -m edge_triage``."""
    if "--dashboard" in sys.argv:
        try:
            _run_streamlit_dashboard()
        except ImportError as e:
            print(f"Streamlit not available ({e}) — running CLI demo.")
            _run_cli_demo()
    else:
        _run_cli_demo()


if __name__ == "__main__":
    main()
