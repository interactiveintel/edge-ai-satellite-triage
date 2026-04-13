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
        page_title="Edge AI Triage Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ─────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 40%, #1a3a5c 100%);
        color: #e0e8f0;
        padding: 1.6rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        border: 1px solid #2a4a6a;
    }
    .header-banner h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: 1px; color: #ffffff; }
    .header-banner .subtitle { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.75; }
    .header-banner .version { float: right; font-size: 0.8rem; opacity: 0.5; margin-top: 0.3rem; }

    /* Decision badges */
    .decision-keep {
        display: inline-block;
        background: #1b5e20; color: #ffffff;
        padding: 4px 16px; border-radius: 6px;
        font-weight: 700; font-size: 1.1rem; letter-spacing: 1px;
    }
    .decision-filter {
        display: inline-block;
        background: #b71c1c; color: #ffffff;
        padding: 4px 16px; border-radius: 6px;
        font-weight: 700; font-size: 1.1rem; letter-spacing: 1px;
    }

    /* Score bars */
    .score-bar-container { margin: 4px 0; }
    .score-bar-bg {
        background: #e0e0e0; border-radius: 4px; height: 20px;
        overflow: hidden; position: relative;
    }
    .score-bar-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.3s ease;
    }
    .score-bar-label {
        position: absolute; top: 0; left: 8px; line-height: 20px;
        font-size: 0.75rem; font-weight: 600; color: #333;
    }

    /* Tile result cards */
    .tile-card {
        border: 1px solid #ddd; border-radius: 8px;
        padding: 1rem; margin: 0.5rem 0;
        background: #fafbfc;
    }
    .tile-card.keep { border-left: 4px solid #2e7d32; }
    .tile-card.filter { border-left: 4px solid #c62828; }

    /* Status indicators */
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .status-dot.green { background: #2e7d32; }
    .status-dot.amber { background: #f57c00; }
    .status-dot.red { background: #c62828; }

    /* KPI row */
    .kpi-row { display: flex; gap: 1rem; margin: 0.5rem 0 1rem; }

    /* Sidebar polish */
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="header-banner">
        <span class="version">v{__version__}</span>
        <h1>EDGE AI SATELLITE DATA TRIAGE</h1>
        <p class="subtitle">Agentic Onboard Filtering &mdash; Dual-Use Space &amp; Ground &mdash; NVIDIA Jetson Orin &mdash; &lt;{config.POWER_BUDGET_WATTS:.0f}W Power Budget</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ─────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.collector = MetricsCollector()
        st.session_state.processing_log = []

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Mission Configuration")

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

        st.markdown("---")
        st.markdown("### System Controls")

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

        st.markdown("---")
        st.markdown("### Input Source")

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

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size: 0.75rem; opacity: 0.6; text-align: center;">
        Edge Triage v{__version__}<br>
        MobileNetV3 + ReAct Agent<br>
        Target: Jetson Orin Nano/AGX
        </div>
        """, unsafe_allow_html=True)

    # ── Main tabs ──────────────────────────────────────────────
    tab_triage, tab_analytics, tab_audit, tab_system = st.tabs([
        "Triage Pipeline", "Analytics", "Audit Trail", "System Status",
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
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Tiles", summary["tiles_processed"])
            k2.metric("Keep Rate", f"{summary['keep_rate'] * 100:.0f}%")
            k3.metric("BW Saved", f"{summary['bandwidth_saved_percent']:.1f}%")
            k4.metric("Avg Power", f"{summary['avg_power_watts']:.1f} W")
            k5.metric("Avg Latency", f"{summary['avg_inference_ms']:.1f} ms")

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

                scenario_colors = {
                    "Wildfire Detection": "#e65100",
                    "Defense ISR": "#1565c0",
                    "Disaster Response": "#c62828",
                }
                bar_color = scenario_colors.get(scenario_key, "#37474f")

                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {bar_color} 0%, {bar_color}33 100%);
                            color: #fff; padding: 0.8rem 1.2rem; border-radius: 8px; margin: 1rem 0 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.1rem;">{scenario_key}</strong>
                            <span style="opacity: 0.8; margin-left: 0.8rem; font-size: 0.85rem;">
                                {len(group)} tiles
                            </span>
                        </div>
                        <div style="font-size: 0.85rem;">
                            <span style="background: #1b5e20; padding: 2px 10px; border-radius: 4px; margin-right: 6px;">
                                KEEP {group_kept}
                            </span>
                            <span style="background: #b71c1c; padding: 2px 10px; border-radius: 4px; margin-right: 6px;">
                                FILTER {group_filtered}
                            </span>
                            <span style="opacity: 0.8;">BW saved: {group_bw:.0f}%</span>
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

                            # Show tile thumbnail for RGB tiles
                            if tile.ndim == 3 and tile.shape[-1] == 3:
                                st.image(
                                    np.clip(tile, 0, 1),
                                    caption=f"Input tile ({tile.shape[1]}x{tile.shape[0]})",
                                    width=200,
                                )

                        st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  TAB 2: ANALYTICS
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
