"""
Smoke tests for real-time monitoring dashboard page.

Tests that the page loads without errors and has expected components.
"""

from pathlib import Path


def test_realtime_page_file_exists():
    """Test that the realtime page file exists."""
    dashboard_path = Path(__file__).parent.parent.parent / "hrp" / "dashboard" / "pages" / "13_Realtime_Data.py"
    assert dashboard_path.exists(), f"Expected dashboard file to exist at {dashboard_path}"


def test_realtime_page_has_render_function():
    """Test that the realtime page has render function."""
    dashboard_path = Path(__file__).parent.parent.parent / "hrp" / "dashboard" / "pages" / "13_Realtime_Data.py"
    content = dashboard_path.read_text()

    # Verify key functions exist
    assert "def render()" in content
    assert "def _get_connection_status" in content
    assert "def _get_intraday_bars" in content
    assert "def _get_latest_prices" in content
    assert "def _render_connection_status" in content


def test_realtime_page_imports():
    """Test that the page has correct imports."""
    dashboard_path = Path(__file__).parent.parent.parent / "hrp" / "dashboard" / "pages" / "13_Realtime_Data.py"
    content = dashboard_path.read_text()

    # Verify key imports
    assert "import streamlit as st" in content
    assert "import plotly.graph_objects as go" in content
    assert "from hrp.api.platform import PlatformAPI" in content
