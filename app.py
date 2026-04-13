"""Streamlit Cloud entry point — thin wrapper around the dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from edge_triage.__main__ import _run_streamlit_dashboard  # noqa: E402

_run_streamlit_dashboard()
