"""Agents Monitor - Real-time agent status and historical timeline."""

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from hrp.dashboard.agents_monitor import get_all_agent_status, AgentStatus
from hrp.api.platform import PlatformAPI


st.title("🤖 Agents Monitor")

# Page controls
col1, col2 = st.columns([3, 1])
with col1:
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with col2:
    if st.button("Refresh Now"):
        st.rerun()

# Real-Time Monitor Section
st.subheader("Real-Time Monitor")

# Initialize API and get agent status
api = PlatformAPI()
agents = get_all_agent_status(api)

# 4-column grid of agent cards
cols = st.columns(4)
for idx, agent in enumerate(agents):
    with cols[idx % 4]:
        # Status colors
        status_colors = {
            "running": "🟦",
            "completed": "🟢",
            "failed": "🔴",
            "idle": "⚪",
        }
        status_icon = status_colors.get(agent.status, "⚪")

        st.markdown(f"### {status_icon} {agent.name}")
        st.markdown(f"**Status:** `{agent.status.upper()}`")

        if agent.status == "running" and agent.elapsed_seconds:
            st.caption(f"⏱ Elapsed: {agent.elapsed_seconds}s")

        if agent.current_hypothesis:
            st.caption(f"📋 `{agent.current_hypothesis}`")

        if agent.last_event:
            ts = datetime.fromisoformat(agent.last_event["timestamp"])
            st.caption(f"🕐 {ts.strftime('%H:%M:%S')}")

        st.markdown("---")

# Historical Timeline Section (placeholder)
st.markdown("---")
st.subheader("Historical Timeline")
st.info("Timeline view coming soon...")

# Auto-refresh logic
if auto_refresh:
    # Check if any agents are running
    active_agents = [a for a in agents if a.status == "running"]

    # Initialize session state for refresh interval
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 5  # Default 5 seconds
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = None

    # Adjust refresh interval based on activity
    now = time.time()

    if active_agents:
        st.session_state.last_activity = now
        st.session_state.refresh_interval = 2  # Fast refresh when active
    elif st.session_state.last_activity:
        idle_time = now - st.session_state.last_activity
        if idle_time > 30:
            st.session_state.refresh_interval = 10  # Slow refresh when idle

    # Sleep and rerun
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
