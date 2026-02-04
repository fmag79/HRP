"""Ops Dashboard - System health and metrics monitoring."""

from datetime import datetime

import streamlit as st

st.set_page_config(page_title="HRP Ops", page_icon="[O]", layout="wide")

st.title("[O] Ops Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def get_system_metrics() -> dict:
    """Get system metrics from MetricsCollector."""
    try:
        from hrp.ops.metrics import MetricsCollector

        collector = MetricsCollector()
        return collector.collect_system_metrics()
    except Exception as e:
        return {"error": str(e)}


def get_data_health() -> dict:
    """Get data pipeline health."""
    try:
        from hrp.ops.metrics import MetricsCollector

        collector = MetricsCollector()
        return collector.collect_data_metrics()
    except Exception as e:
        return {"error": str(e)}


def get_ready_status() -> dict:
    """Get system readiness status."""
    try:
        from hrp.ops.server import check_system_ready

        is_ready, details = check_system_ready()
        return {"ready": is_ready, "details": details}
    except Exception as e:
        return {"ready": False, "error": str(e)}


# System Status
st.header("System Status")

col1, col2, col3 = st.columns(3)

ready_status = get_ready_status()
with col1:
    if ready_status.get("ready"):
        st.success("[OK] System Ready")
    else:
        st.error("[X] System Not Ready")
        if "details" in ready_status:
            st.json(ready_status["details"])

system_metrics = get_system_metrics()
with col2:
    st.metric("CPU Usage", f"{system_metrics.get('cpu_percent', 'N/A')}%")

with col3:
    st.metric("Memory Usage", f"{system_metrics.get('memory_percent', 'N/A')}%")


# Data Health
st.header("Data Pipeline Health")

data_health = get_data_health()
if data_health.get("status") == "ok":
    st.success(f"[OK] Data pipeline healthy (score: {data_health.get('health_score', 'N/A')})")
elif data_health.get("status") == "degraded":
    st.warning(f"[!] Data pipeline degraded (score: {data_health.get('health_score', 'N/A')})")
else:
    st.error(f"[X] Data pipeline: {data_health.get('status', 'unknown')}")


# Thresholds Info
st.header("Alert Thresholds")

try:
    from hrp.ops.thresholds import load_thresholds

    thresholds = load_thresholds()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Health Thresholds")
        st.write(f"- Warning: {thresholds.health_score_warning}%")
        st.write(f"- Critical: {thresholds.health_score_critical}%")

    with col2:
        st.subheader("Freshness Thresholds")
        st.write(f"- Warning: {thresholds.freshness_warning_days} days")
        st.write(f"- Critical: {thresholds.freshness_critical_days} days")
except Exception as e:
    st.error(f"Failed to load thresholds: {e}")


# Refresh button
if st.button("[R] Refresh"):
    st.rerun()
