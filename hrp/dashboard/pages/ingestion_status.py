"""
Data Ingestion Status page for HRP Dashboard.

Displays detailed ingestion job status, source configurations, and scheduling information.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from hrp.data.db import get_db


@st.cache_data(ttl=300)
def get_ingestion_logs(limit: int = 20) -> pd.DataFrame:
    """Get recent ingestion log entries."""
    db = get_db()
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
        df = db.fetchdf(query, (limit,))
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
    db = get_db()
    query = """
        SELECT
            source_id,
            source_type,
            provider,
            is_active,
            last_fetch_date,
            created_at
        FROM data_sources
        ORDER BY source_id
    """
    try:
        df = db.fetchdf(query)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "source_id", "source_type", "provider", "is_active",
            "last_fetch_date", "created_at"
        ])


@st.cache_data(ttl=300)
def get_ingestion_statistics() -> dict[str, Any]:
    """Get summary statistics for ingestion jobs."""
    db = get_db()

    # Total runs
    total_result = db.fetchone("SELECT COUNT(*) FROM ingestion_log")
    total_runs = total_result[0] if total_result else 0

    # Success count
    success_result = db.fetchone(
        "SELECT COUNT(*) FROM ingestion_log WHERE status = 'success'"
    )
    success_count = success_result[0] if success_result else 0

    # Failed count
    failed_result = db.fetchone(
        "SELECT COUNT(*) FROM ingestion_log WHERE status = 'failed'"
    )
    failed_count = failed_result[0] if failed_result else 0

    # Total records inserted
    records_result = db.fetchone(
        "SELECT SUM(records_inserted) FROM ingestion_log WHERE status = 'success'"
    )
    total_records = records_result[0] if records_result and records_result[0] else 0

    # Last successful run
    last_success_result = db.fetchone(
        """
        SELECT MAX(completed_at)
        FROM ingestion_log
        WHERE status = 'success'
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
    db = get_db()
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
        df = db.fetchdf(query, (limit,))
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "log_id", "source_id", "started_at", "error_message"
        ])


@st.cache_data(ttl=300)
def get_source_statistics() -> pd.DataFrame:
    """Get per-source statistics."""
    db = get_db()
    query = """
        SELECT
            source_id,
            COUNT(*) as total_runs,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_runs,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
            SUM(records_inserted) as total_records,
            MAX(completed_at) as last_run
        FROM ingestion_log
        GROUP BY source_id
        ORDER BY source_id
    """
    try:
        df = db.fetchdf(query)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "source_id", "total_runs", "success_runs", "failed_runs",
            "total_records", "last_run"
        ])


def render() -> None:
    """Render the Data Ingestion Status page."""
    st.title("Data Ingestion Status")
    st.markdown("Monitor data ingestion pipelines, job status, and scheduling.")

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
    st.subheader("Overview")

    stats = get_ingestion_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Runs",
            value=f"{stats['total_runs']:,}"
        )

    with col2:
        st.metric(
            label="Success Rate",
            value=f"{stats['success_rate']:.1f}%",
            delta=f"{stats['success_count']} / {stats['total_runs']}"
        )

    with col3:
        st.metric(
            label="Failed Runs",
            value=f"{stats['failed_count']:,}"
        )

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
                "provider": st.column_config.TextColumn("Provider", width="medium"),
                "is_active": st.column_config.CheckboxColumn("Active", width="small"),
                "last_fetch_date": st.column_config.DateColumn("Last Fetch", width="medium"),
                "created_at": st.column_config.DatetimeColumn("Created", width="medium"),
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
            if status_lower == "success":
                return "✅ Success"
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
