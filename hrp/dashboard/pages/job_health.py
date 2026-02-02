"""
Job Health Monitoring Dashboard

Monitor ingestion job health, track failures, and visualize system performance.
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from hrp.api.platform import PlatformAPI

def _get_api():
    return PlatformAPI()


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_job_stats() -> pd.DataFrame:
    """Get aggregated job statistics."""
    api = _get_api()
    stats = api.query_readonly(
        """
        SELECT
            source_id,
            COUNT(*) as total_runs,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
            ROUND(
                100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*),
                1
            ) as success_rate,
            MAX(started_at) as last_run,
            MIN(started_at) as first_run
        FROM ingestion_log
        GROUP BY source_id
        ORDER BY total_runs DESC
        """
    )
    return stats


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_recent_jobs(limit: int = 100) -> pd.DataFrame:
    """Get recent job executions with duration calculation."""
    api = _get_api()
    jobs = api.query_readonly(
        """
        SELECT
            log_id,
            source_id,
            started_at,
            completed_at,
            status,
            error_message,
            records_fetched,
            records_inserted,
            CASE
                WHEN completed_at IS NOT NULL
                THEN EXTRACT(EPOCH FROM (completed_at - started_at))
                ELSE NULL
            END as duration_seconds
        FROM ingestion_log
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return jobs


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_job_timeline(days: int = 7) -> pd.DataFrame:
    """Get job execution timeline for the last N days."""
    api = _get_api()
    timeline = api.query_readonly(
        f"""
        SELECT
            source_id,
            DATE(started_at) as date,
            COUNT(*) as runs,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM ingestion_log
        WHERE started_at >= CURRENT_DATE - INTERVAL '{days}' day
        GROUP BY source_id, DATE(started_at)
        ORDER BY date DESC, source_id
        """
    )
    return timeline


def get_status_emoji(status: str) -> str:
    """Get emoji for job status."""
    status_map = {
        "completed": "✅",
        "failed": "❌",
        "running": "🔄",
    }
    return status_map.get(status.lower(), "❓")


def get_success_rate_color(rate: float) -> str:
    """Get color for success rate."""
    if rate >= 90:
        return "🟢"
    elif rate >= 70:
        return "🟡"
    else:
        return "🔴"


def render() -> None:
    """Render the Job Health page."""
    st.title("📊 Ingestion Job Health Monitor")

    st.caption(
        """
        Monitor data pipeline job health, success rates, and failure patterns.
        Data refreshes automatically every 60 seconds.
        """
    )

    # Load data
    stats = get_job_stats()
    recent = get_recent_jobs(100)
    timeline = get_job_timeline(7)

    # KPI Cards
    st.subheader("Overview Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_jobs = len(stats)
        st.metric("Job Types", total_jobs)

    with col2:
        avg_success = stats["success_rate"].mean() if len(stats) > 0 else 0
        st.metric("Avg Success Rate", f"{avg_success:.1f}%")

    with col3:
        recent_completed = recent[recent["status"] == "completed"].shape[0]
        st.metric("Recent Success", recent_completed)

    with col4:
        recent_failed = recent[recent["status"] == "failed"].shape[0]
        st.metric("Recent Failures", recent_failed)

    with col5:
        if len(recent) > 0:
            last_activity = recent["started_at"].max()
            # Calculate time ago
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
            time_ago = datetime.now() - last_activity
            if time_ago.seconds < 3600:
                time_str = f"{time_ago.seconds // 60}m ago"
            else:
                time_str = f"{time_ago.seconds // 3600}h ago"
            st.metric("Last Activity", time_str)
        else:
            st.metric("Last Activity", "N/A")

    st.divider()

    # Job success rates table
    st.subheader("Job Success Rates")

    # Display stats table with color coding
    stats_display = stats.copy()
    stats_display["status"] = stats_display["success_rate"].apply(get_success_rate_color)
    stats_display["last_run_formatted"] = pd.to_datetime(stats_display["last_run"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    # Reorder columns for display
    stats_display = stats_display[
        ["source_id", "status", "success_rate", "total_runs", "completed", "failed", "last_run_formatted"]
    ]
    stats_display.columns = ["Job", "Health", "Success %", "Total", "✅", "❌", "Last Run"]

    st.dataframe(
        stats_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Success rate visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Success Rate by Job")
        fig = px.bar(
            stats,
            x="source_id",
            y="success_rate",
            title="Job Success Rate (%)",
            labels={"source_id": "Job Type", "success_rate": "Success %"},
            color="success_rate",
            color_continuous_scale=["#EF553B", "#FFC300", "#00CC96"],
            range_color=[0, 100],
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Execution Counts")
        fig_counts = px.bar(
            stats,
            x="source_id",
            y=["completed", "failed"],
            title="Job Execution Counts by Status",
            labels={"source_id": "Job Type", "value": "Count"},
            color_discrete_map={"completed": "#00CC96", "failed": "#EF553B"},
        )
        fig_counts.update_layout(xaxis_tickangle=-45, barmode="stack")
        st.plotly_chart(fig_counts, use_container_width=True)

    st.divider()

    # Recent job history
    st.subheader("Recent Job History")

    # Add status emoji
    recent_display = recent.copy()
    recent_display["status_emoji"] = recent_display["status"].apply(get_status_emoji)
    recent_display["started_at_formatted"] = pd.to_datetime(recent_display["started_at"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Select columns for display
    recent_display = recent_display[
        [
            "status_emoji",
            "source_id",
            "started_at_formatted",
            "records_inserted",
            "duration_seconds",
            "error_message",
        ]
    ]
    recent_display.columns = ["Status", "Job", "Started", "Records", "Duration (s)", "Error"]

    st.dataframe(
        recent_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Error analysis
    failed_jobs = recent[recent["status"] == "failed"]
    if len(failed_jobs) > 0:
        st.subheader("❌ Recent Failures Analysis")

        for idx, row in failed_jobs.iterrows():
            with st.expander(
                f"**{row['source_id']}** - {row['started_at']} - {row.get('error_message', 'Unknown error')[:50]}..."
            ):
                st.write(f"**Job ID:** {row['log_id']}")
                st.write(f"**Started:** {row['started_at']}")
                st.write(f"**Error:** {row['error_message']}")
                st.write(f"**Records:** {row['records_fetched']} fetched, {row['records_inserted']} inserted")
                if pd.notna(row["completed_at"]):
                    duration = row.get("duration_seconds")
                    if duration:
                        st.write(f"**Duration:** {duration:.2f} seconds")
    else:
        st.subheader("✅ No Recent Failures")
        st.write("All jobs in the recent history completed successfully!")

    st.divider()

    # 7-day timeline
    st.subheader("7-Day Execution Timeline")

    timeline_pivot = timeline.pivot(
        index="date", columns="source_id", values="runs"
    ).fillna(0)

    if not timeline_pivot.empty:
        fig_timeline = px.imshow(
            timeline_pivot.T,
            labels=dict(x="Date", y="Job Type", color="Runs"),
            title="Job Execution Frequency (Last 7 Days)",
            color_continuous_scale="Viridis",
            aspect="auto",
        )
        fig_timeline.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No timeline data available for the last 7 days.")

    # Auto-refresh
    st.divider()
    st.caption("📊 Dashboard auto-refreshes every 60 seconds | Last refresh: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
