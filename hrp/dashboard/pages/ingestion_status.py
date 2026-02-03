"""
Data Ingestion Status page for HRP Dashboard.

Displays detailed ingestion job status, source configurations, and scheduling information.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from hrp.api.platform import PlatformAPI

def _get_api():
    return PlatformAPI()


@st.cache_data(ttl=300)
def get_ingestion_logs(limit: int = 20) -> pd.DataFrame:
    """Get recent ingestion log entries."""
    api = _get_api()
    query = """
        SELECT
            log_id,
            source_id,
            started_at,
            completed_at,
            records_fetched,
            records_inserted,
            status,
            error_message
        FROM ingestion_log
        ORDER BY started_at DESC
        LIMIT ?
    """
    try:
        df = api.query_readonly(query, (limit,))
        return df
    except Exception:
        # Return empty DataFrame if table doesn't exist or query fails
        return pd.DataFrame(columns=[
            "log_id", "source_id", "started_at", "completed_at",
            "records_fetched", "records_inserted", "status", "error_message"
        ])


@st.cache_data(ttl=300)
def get_data_sources() -> pd.DataFrame:
    """Get configured data sources."""
    api = _get_api()
    query = """
        SELECT
            source_id,
            source_type,
            api_name,
            status,
            last_fetch
        FROM data_sources
        ORDER BY source_id
    """
    try:
        df = api.query_readonly(query)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "source_id", "source_type", "api_name", "status",
            "last_fetch"
        ])


@st.cache_data(ttl=300)
def get_ingestion_statistics() -> dict[str, Any]:
    """Get summary statistics for ingestion jobs."""
    api = _get_api()

    # Total runs
    total_result = api.fetchone_readonly("SELECT COUNT(*) FROM ingestion_log")
    total_runs = total_result[0] if total_result else 0

    # Success count
    success_result = api.fetchone_readonly(
        "SELECT COUNT(*) FROM ingestion_log WHERE status = 'completed'"
    )
    success_count = success_result[0] if success_result else 0

    # Failed count
    failed_result = api.fetchone_readonly(
        "SELECT COUNT(*) FROM ingestion_log WHERE status = 'failed'"
    )
    failed_count = failed_result[0] if failed_result else 0

    # Total records inserted
    records_result = api.fetchone_readonly(
        "SELECT SUM(records_inserted) FROM ingestion_log WHERE status = 'completed'"
    )
    total_records = records_result[0] if records_result and records_result[0] else 0

    # Last successful run
    last_success_result = api.fetchone_readonly(
        """
        SELECT MAX(completed_at)
        FROM ingestion_log
        WHERE status = 'completed'
        """
    )
    last_success = last_success_result[0] if last_success_result and last_success_result[0] else None

    return {
        "total_runs": total_runs,
        "success_count": success_count,
        "failed_count": failed_count,
        "total_records": total_records,
        "last_success": last_success,
        "success_rate": (success_count / total_runs * 100) if total_runs > 0 else 0,
    }


@st.cache_data(ttl=300)
def get_recent_errors(limit: int = 5) -> pd.DataFrame:
    """Get recent ingestion errors."""
    api = _get_api()
    query = """
        SELECT
            log_id,
            source_id,
            started_at,
            error_message
        FROM ingestion_log
        WHERE status = 'failed' AND error_message IS NOT NULL
        ORDER BY started_at DESC
        LIMIT ?
    """
    try:
        df = api.query_readonly(query, (limit,))
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "log_id", "source_id", "started_at", "error_message"
        ])


@st.cache_data(ttl=300)
def get_source_statistics() -> pd.DataFrame:
    """Get per-source statistics."""
    api = _get_api()
    query = """
        SELECT
            source_id,
            COUNT(*) as total_runs,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_runs,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
            SUM(records_inserted) as total_records,
            MAX(completed_at) as last_run
        FROM ingestion_log
        GROUP BY source_id
        ORDER BY source_id
    """
    try:
        df = api.query_readonly(query)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "source_id", "total_runs", "success_runs", "failed_runs",
            "total_records", "last_run"
        ])


def render() -> None:
    """Render the Data Ingestion Status page."""
    # Load custom CSS
    try:
        with open("hrp/dashboard/static/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # Page header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;">
            Data Ingestion Status
        </h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">
            Monitor data ingestion pipelines, job status, and scheduling
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
    st.markdown("""
    <p style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 1rem;">
        Overview
    </p>
    """, unsafe_allow_html=True)

    stats = get_ingestion_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1.25rem;">
            <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Total Runs
            </div>
            <div style="color: #f1f5f9; font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {stats['total_runs']:,}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        success_color = "#10b981" if stats['success_rate'] >= 90 else "#f59e0b" if stats['success_rate'] >= 70 else "#ef4444"
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1.25rem;">
            <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Success Rate
            </div>
            <div style="color: {success_color}; font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {stats['success_rate']:.1f}%
            </div>
            <div style="color: #6b7280; font-size: 0.75rem; margin-top: 0.25rem;">
                {stats['success_count']} / {stats['total_runs']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1.25rem;">
            <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Failed Runs
            </div>
            <div style="color: #f1f5f9; font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {stats['failed_runs']:,}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.metric(
            label="Total Records",
            value=f"{stats['total_records']:,}"
        )

    # Last successful run
    if stats["last_success"]:
        last_success_str = stats["last_success"]
        if isinstance(last_success_str, datetime):
            last_success_str = last_success_str.strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Last successful run: {last_success_str}")
    else:
        st.caption("No successful runs yet")

    st.divider()

    # -------------------------------------------------------------------------
    # Data Sources
    # -------------------------------------------------------------------------
    st.subheader("Data Sources")

    sources_df = get_data_sources()

    if sources_df.empty:
        st.info("No data sources configured. Add sources to begin data ingestion.")
    else:
        st.dataframe(
            sources_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_id": st.column_config.TextColumn("Source ID", width="medium"),
                "source_type": st.column_config.TextColumn("Type", width="small"),
                "api_name": st.column_config.TextColumn("API Name", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "last_fetch": st.column_config.DatetimeColumn("Last Fetch", width="medium"),
            }
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Per-Source Statistics
    # -------------------------------------------------------------------------
    st.subheader("Source Statistics")

    source_stats = get_source_statistics()

    if source_stats.empty:
        st.info("No ingestion runs yet. Start ingestion to see source statistics.")
    else:
        # Add success rate column
        source_stats["success_rate"] = (
            source_stats["success_runs"] / source_stats["total_runs"] * 100
        ).round(1)

        st.dataframe(
            source_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_id": st.column_config.TextColumn("Source", width="medium"),
                "total_runs": st.column_config.NumberColumn("Total Runs", width="small"),
                "success_runs": st.column_config.NumberColumn("Success", width="small"),
                "failed_runs": st.column_config.NumberColumn("Failed", width="small"),
                "success_rate": st.column_config.NumberColumn("Success Rate", width="small", format="%.1f%%"),
                "total_records": st.column_config.NumberColumn("Records", width="medium"),
                "last_run": st.column_config.DatetimeColumn("Last Run", width="medium"),
            }
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Recent Ingestion Jobs
    # -------------------------------------------------------------------------
    st.subheader("Recent Ingestion Jobs")

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl2:
        limit = st.selectbox("Show", options=[10, 20, 50, 100], index=1, key="job_limit")

    ingestion_logs = get_ingestion_logs(limit=limit)

    if ingestion_logs.empty:
        st.info("No ingestion jobs found. Run an ingestion pipeline to populate this table.")
    else:
        # Format the dataframe for display
        display_df = ingestion_logs.copy()

        # Add duration column if both started_at and completed_at exist
        if "started_at" in display_df.columns and "completed_at" in display_df.columns:
            display_df["duration"] = (
                pd.to_datetime(display_df["completed_at"]) - pd.to_datetime(display_df["started_at"])
            ).dt.total_seconds()

        # Format status with emoji indicators
        def format_status(status: str | None) -> str:
            if status is None:
                return "❓ Unknown"
            status_lower = str(status).lower()
            if status_lower == "completed":
                return "✅ Completed"
            elif status_lower == "failed":
                return "❌ Failed"
            elif status_lower == "running":
                return "🔄 Running"
            elif status_lower == "partial":
                return "⚠️ Partial"
            return str(status)

        if "status" in display_df.columns:
            display_df["status"] = display_df["status"].apply(format_status)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "log_id": st.column_config.NumberColumn("ID", width="small"),
                "source_id": st.column_config.TextColumn("Source", width="medium"),
                "started_at": st.column_config.DatetimeColumn("Started", width="medium"),
                "completed_at": st.column_config.DatetimeColumn("Completed", width="medium"),
                "duration": st.column_config.NumberColumn("Duration (s)", width="small", format="%.1f"),
                "records_fetched": st.column_config.NumberColumn("Fetched", width="small"),
                "records_inserted": st.column_config.NumberColumn("Inserted", width="small"),
                "status": st.column_config.TextColumn("Status", width="medium"),
                "error_message": st.column_config.TextColumn("Error", width="large"),
            }
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Recent Errors
    # -------------------------------------------------------------------------
    st.subheader("Recent Errors")

    errors_df = get_recent_errors(limit=5)

    if errors_df.empty:
        st.success("No recent errors! All ingestion jobs are running smoothly.")
    else:
        st.warning(f"Found {len(errors_df)} recent errors")

        for _, error in errors_df.iterrows():
            with st.expander(
                f"🔴 {error['source_id']} - {error['started_at']}"
            ):
                st.markdown(f"**Log ID:** {error['log_id']}")
                st.markdown(f"**Source:** {error['source_id']}")
                st.markdown(f"**Time:** {error['started_at']}")
                st.markdown("**Error:**")
                st.code(error['error_message'], language=None)


# Entry point for Streamlit multi-page apps
if __name__ == "__main__":
    render()
