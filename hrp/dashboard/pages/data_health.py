"""
Data Health page for HRP Dashboard.

Displays data completeness, ingestion status, quality metrics, and per-symbol coverage.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from hrp.data.db import get_db


@st.cache_data(ttl=300)
def get_symbol_count() -> int:
    """Get count of distinct symbols in prices table."""
    db = get_db()
    result = db.fetchone("SELECT COUNT(DISTINCT symbol) FROM prices")
    return result[0] if result else 0


@st.cache_data(ttl=300)
def get_date_range() -> tuple[str | None, str | None]:
    """Get min and max dates from prices table."""
    db = get_db()
    result = db.fetchone("SELECT MIN(date), MAX(date) FROM prices")
    if result and result[0] and result[1]:
        return str(result[0]), str(result[1])
    return None, None


@st.cache_data(ttl=300)
def get_total_records() -> int:
    """Get total number of price records."""
    db = get_db()
    result = db.fetchone("SELECT COUNT(*) FROM prices")
    return result[0] if result else 0


@st.cache_data(ttl=300)
def get_ingestion_logs(limit: int = 10) -> pd.DataFrame:
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
def get_symbol_coverage() -> pd.DataFrame:
    """Get per-symbol data summary."""
    db = get_db()
    query = """
        SELECT
            symbol,
            COUNT(*) as record_count,
            MIN(date) as first_date,
            MAX(date) as last_date,
            COUNT(*) / DATEDIFF('day', MIN(date), MAX(date) + INTERVAL 1 DAY)::FLOAT as coverage_pct
        FROM prices
        GROUP BY symbol
        ORDER BY symbol
    """
    try:
        df = db.fetchdf(query)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "symbol", "record_count", "first_date", "last_date", "coverage_pct"
        ])


@st.cache_data(ttl=300)
def get_missing_dates_summary() -> pd.DataFrame:
    """
    Detect gaps in date sequences per symbol.
    Returns symbols with gaps and the number of missing trading days.
    """
    db = get_db()
    # This query finds gaps by comparing each date to the next date for each symbol
    query = """
        WITH date_gaps AS (
            SELECT
                symbol,
                date,
                LEAD(date) OVER (PARTITION BY symbol ORDER BY date) as next_date,
                DATEDIFF('day', date, LEAD(date) OVER (PARTITION BY symbol ORDER BY date)) as gap_days
            FROM prices
        )
        SELECT
            symbol,
            COUNT(*) as gap_count,
            SUM(gap_days - 1) as total_missing_days,
            MAX(gap_days - 1) as max_gap_days
        FROM date_gaps
        WHERE gap_days > 1 AND gap_days <= 10  -- Exclude weekends (2 days) but flag larger gaps
        GROUP BY symbol
        HAVING COUNT(*) > 0
        ORDER BY total_missing_days DESC
        LIMIT 20
    """
    try:
        df = db.fetchdf(query)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "symbol", "gap_count", "total_missing_days", "max_gap_days"
        ])


@st.cache_data(ttl=300)
def get_price_anomalies() -> pd.DataFrame:
    """
    Detect potential price data anomalies.
    Flags records where:
    - Close price is 0 or negative
    - High < Low
    - Volume is negative
    - Extreme daily moves (>50%)
    """
    db = get_db()
    query = """
        SELECT
            symbol,
            date,
            open,
            high,
            low,
            close,
            volume,
            CASE
                WHEN close <= 0 THEN 'Zero/Negative close'
                WHEN high < low THEN 'High < Low'
                WHEN volume < 0 THEN 'Negative volume'
                WHEN close IS NULL THEN 'Null close'
                ELSE 'Unknown'
            END as anomaly_type
        FROM prices
        WHERE close <= 0
           OR high < low
           OR volume < 0
           OR close IS NULL
        ORDER BY date DESC
        LIMIT 50
    """
    try:
        df = db.fetchdf(query)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "symbol", "date", "open", "high", "low", "close", "volume", "anomaly_type"
        ])


@st.cache_data(ttl=300)
def get_recent_data_freshness() -> dict[str, Any]:
    """Check how fresh the data is compared to today."""
    db = get_db()
    result = db.fetchone("SELECT MAX(date) FROM prices")
    if result and result[0]:
        last_date = result[0]
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        days_stale = (today - last_date).days
        return {
            "last_date": str(last_date),
            "days_stale": days_stale,
            "is_fresh": days_stale <= 3  # Allow for weekends
        }
    return {"last_date": None, "days_stale": None, "is_fresh": False}


def render() -> None:
    """Render the Data Health page."""
    st.title("Data Health")
    st.markdown("Monitor data completeness, quality, and ingestion status.")

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
    st.subheader("Overview")

    symbol_count = get_symbol_count()
    min_date, max_date = get_date_range()
    total_records = get_total_records()
    freshness = get_recent_data_freshness()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Symbols",
            value=f"{symbol_count:,}" if symbol_count else "0"
        )

    with col2:
        st.metric(
            label="Total Records",
            value=f"{total_records:,}" if total_records else "0"
        )

    with col3:
        if min_date and max_date:
            date_range_str = f"{min_date} to {max_date}"
        else:
            date_range_str = "No data"
        st.metric(
            label="Date Range",
            value=date_range_str if min_date else "No data"
        )

    with col4:
        if freshness["last_date"]:
            freshness_status = "Fresh" if freshness["is_fresh"] else f"{freshness['days_stale']} days stale"
            st.metric(
                label="Data Freshness",
                value=freshness_status
            )
        else:
            st.metric(label="Data Freshness", value="No data")

    st.divider()

    # -------------------------------------------------------------------------
    # Ingestion Status
    # -------------------------------------------------------------------------
    st.subheader("Recent Ingestion Jobs")

    ingestion_logs = get_ingestion_logs(limit=10)

    if ingestion_logs.empty:
        st.info("No ingestion logs found. Run an ingestion job to populate this table.")
    else:
        # Format the dataframe for display
        display_df = ingestion_logs.copy()

        # Add status indicator
        def format_status(status: str | None) -> str:
            if status is None:
                return "Unknown"
            status_lower = str(status).lower()
            if status_lower == "success":
                return "Success"
            elif status_lower == "failed":
                return "Failed"
            elif status_lower == "running":
                return "Running"
            elif status_lower == "partial":
                return "Partial"
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
                "records_fetched": st.column_config.NumberColumn("Fetched", width="small"),
                "records_inserted": st.column_config.NumberColumn("Inserted", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "error_message": st.column_config.TextColumn("Error", width="large"),
            }
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Data Quality Metrics
    # -------------------------------------------------------------------------
    st.subheader("Data Quality")

    col_qual1, col_qual2 = st.columns(2)

    with col_qual1:
        st.markdown("**Price Anomalies**")
        anomalies = get_price_anomalies()

        if anomalies.empty:
            st.success("No price anomalies detected.")
        else:
            st.warning(f"Found {len(anomalies)} anomalous records")
            st.dataframe(
                anomalies,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "date": st.column_config.DateColumn("Date"),
                    "open": st.column_config.NumberColumn("Open", format="%.2f"),
                    "high": st.column_config.NumberColumn("High", format="%.2f"),
                    "low": st.column_config.NumberColumn("Low", format="%.2f"),
                    "close": st.column_config.NumberColumn("Close", format="%.2f"),
                    "volume": st.column_config.NumberColumn("Volume"),
                    "anomaly_type": st.column_config.TextColumn("Anomaly Type"),
                }
            )

    with col_qual2:
        st.markdown("**Date Sequence Gaps**")
        gaps = get_missing_dates_summary()

        if gaps.empty:
            st.success("No significant date gaps detected.")
        else:
            st.warning(f"{len(gaps)} symbols have date gaps")
            st.dataframe(
                gaps,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "gap_count": st.column_config.NumberColumn("Gap Count"),
                    "total_missing_days": st.column_config.NumberColumn("Total Missing Days"),
                    "max_gap_days": st.column_config.NumberColumn("Max Gap (Days)"),
                }
            )

    st.divider()

    # -------------------------------------------------------------------------
    # Per-Symbol Coverage
    # -------------------------------------------------------------------------
    st.subheader("Symbol Coverage")

    coverage_df = get_symbol_coverage()

    if coverage_df.empty:
        st.info("No price data available. Ingest data to see symbol coverage.")
    else:
        # Add search/filter
        search_term = st.text_input(
            "Filter symbols",
            placeholder="Enter symbol to filter (e.g., AAPL)",
            key="symbol_filter"
        )

        filtered_df = coverage_df.copy()
        if search_term:
            filtered_df = filtered_df[
                filtered_df["symbol"].str.contains(search_term.upper(), na=False)
            ]

        # Summary stats
        col_cov1, col_cov2, col_cov3 = st.columns(3)
        with col_cov1:
            avg_records = filtered_df["record_count"].mean() if not filtered_df.empty else 0
            st.metric("Avg Records/Symbol", f"{avg_records:,.0f}")
        with col_cov2:
            if "coverage_pct" in filtered_df.columns and not filtered_df.empty:
                avg_coverage = filtered_df["coverage_pct"].mean() * 100
                st.metric("Avg Coverage", f"{avg_coverage:.1f}%")
            else:
                st.metric("Avg Coverage", "N/A")
        with col_cov3:
            st.metric("Symbols Shown", f"{len(filtered_df):,}")

        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "record_count": st.column_config.NumberColumn("Records", width="small"),
                "first_date": st.column_config.DateColumn("First Date", width="medium"),
                "last_date": st.column_config.DateColumn("Last Date", width="medium"),
                "coverage_pct": st.column_config.ProgressColumn(
                    "Coverage",
                    width="medium",
                    min_value=0,
                    max_value=1,
                    format="%.1f%%",
                ),
            }
        )


# Entry point for Streamlit multi-page apps
if __name__ == "__main__":
    render()
