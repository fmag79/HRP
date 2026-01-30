"""Agents Monitor - Real-time agent status and historical timeline."""

from __future__ import annotations

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
