"""Agents Monitor - Real-time agent status and historical timeline."""

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from hrp.dashboard.agents_monitor import get_all_agent_status, get_timeline, AgentStatus
from hrp.api.platform import PlatformAPI


st.title("🤖 Agents Monitor")

# Page controls
col1, col2 = st.columns([3, 1])
with col1:
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with col2:
    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# Real-Time Monitor Section
st.subheader("Real-Time Monitor")

# Get agent status with error handling
try:
    agents = get_all_agent_status(PlatformAPI())
except Exception as e:
    st.error(f"Failed to load agent status: {e}")
    st.info("Make sure the database is accessible (scheduler might be holding a lock)")
    agents = []

# 4-column grid of agent cards
if agents:
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

    # Auto-refresh logic (only if enabled and there are active agents)
    if auto_refresh:
        active_agents = [a for a in agents if a.status == "running"]

        if active_agents:
            # Show auto-refresh indicator
            st.caption(f"🔄 Auto-refreshing (2s interval - {len(active_agents)} active agent(s))")
            time.sleep(2)
            st.rerun()
        else:
            st.caption("⏸ No active agents - auto-refresh paused")
else:
    st.warning("No agent data available. Click 'Refresh Now' to retry.")

# Historical Timeline Section
st.markdown("---")
st.subheader("Historical Timeline")

# Timeline filters
if agents:
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        agent_filter = st.multiselect(
            "Filter by Agent",
            options=[a.name for a in agents],
            default=[a.name for a in agents],
        )
    with col2:
        status_filter = st.multiselect(
            "Filter by Status",
            options=["Running", "Completed", "Failed"],
            default=["Running", "Completed", "Failed"],
        )
    with col3:
        limit = st.slider("Events to show", min_value=10, max_value=500, value=100)
    with col4:
        st.write("")  # Spacer
        if st.button("Apply Filters"):
            st.cache_data.clear()
            st.rerun()

    # Get and display timeline
    try:
        timeline = get_timeline(PlatformAPI(), limit=limit)
    except Exception as e:
        st.error(f"Failed to load timeline: {e}")
        timeline = []

    if not timeline:
        st.info("No events found.")
    else:
        for event in timeline:
            # Skip if not in agent filter
            if event.get("agent_name") not in agent_filter:
                continue

            # Determine status icon
            if "failed" in event["event_type"].lower():
                status_icon = "❌"
            elif "start" in event["event_type"].lower():
                status_icon = "🔄"
            else:
                status_icon = "✅"

            # Create expandable event
            with st.expander(
                f"{status_icon} **{event.get('agent_name', 'Unknown')}** — "
                f"{event['event_type'].replace('_', ' ').title()} • "
                f"{datetime.fromisoformat(event['timestamp']).strftime('%Y-%m-%d %H:%M')} "
                f"• `{event.get('hypothesis_id', 'N/A')}`"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Details**")
                    st.caption(f"Event ID: {event['lineage_id']}")
                    st.caption(f"Actor: `{event['actor']}`")
                    if event.get("experiment_id"):
                        st.caption(f"Experiment: `{event['experiment_id']}`")

                with col2:
                    st.markdown("**Info**")
                    if event.get("details"):
                        for key, value in event["details"].items():
                            st.caption(f"{key}: {value}")

                with col3:
                    st.markdown("**Actions**")
                    if event.get("experiment_id"):
                        st.link_button(
                            "View in MLflow",
                            f"http://localhost:5000/experiments/{event.get('experiment_id')}"
                        )
else:
    st.info("Timeline unavailable - no agent data loaded.")
