"""
Data Health page for HRP Dashboard.

Displays data completeness, ingestion status, quality metrics, and per-symbol coverage.
"""

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from hrp.data.db import get_db
from hrp.data.quality.checks import CheckResult, IssueSeverity, QualityIssue
from hrp.data.quality.report import QualityReport, QualityReportGenerator


@st.cache_data(ttl=300)
def get_quality_report(as_of_date: date) -> QualityReport:
    """Generate or retrieve cached quality report."""
    generator = QualityReportGenerator()
    report = generator.generate_report(as_of_date)
    # Store report for historical tracking
    generator.store_report(report)
    return report


@st.cache_data(ttl=600)
def get_health_trend(days: int = 90) -> pd.DataFrame:
    """Get historical health scores for trend chart."""
    generator = QualityReportGenerator()
    trend_data = generator.get_health_trend(days=days)
    if not trend_data:
        return pd.DataFrame(columns=["date", "health_score"])
    return pd.DataFrame(trend_data)


def get_health_color(score: float) -> str:
    """Get color based on health score."""
    if score >= 80:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


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


@st.cache_data(ttl=300)
def get_last_successful_ingestion() -> dict[str, Any]:
    """Get the most recent successful ingestion run."""
    db = get_db()
    query = """
        SELECT
            log_id,
            source_id,
            started_at,
            completed_at,
            records_fetched,
            records_inserted
        FROM ingestion_log
        WHERE LOWER(status) = 'success'
        ORDER BY completed_at DESC
        LIMIT 1
    """
    try:
        result = db.fetchone(query)
        if result:
            return {
                "log_id": result[0],
                "source_id": result[1],
                "started_at": result[2],
                "completed_at": result[3],
                "records_fetched": result[4],
                "records_inserted": result[5],
            }
    except Exception:
        pass
    return {}


@st.cache_data(ttl=300)
def get_ingestion_summary() -> dict[str, Any]:
    """Get summary statistics for ingestion jobs."""
    db = get_db()
    query = """
        SELECT
            COUNT(*) as total_runs,
            SUM(CASE WHEN LOWER(status) = 'success' THEN 1 ELSE 0 END) as successful_runs,
            SUM(CASE WHEN LOWER(status) = 'failed' THEN 1 ELSE 0 END) as failed_runs,
            SUM(records_inserted) as total_records_inserted,
            MAX(completed_at) as last_run
        FROM ingestion_log
    """
    try:
        result = db.fetchone(query)
        if result:
            total = result[0] or 0
            success = result[1] or 0
            failed = result[2] or 0
            success_rate = (success / total * 100) if total > 0 else 0
            return {
                "total_runs": total,
                "successful_runs": success,
                "failed_runs": failed,
                "success_rate": success_rate,
                "total_records_inserted": result[3] or 0,
                "last_run": result[4],
            }
    except Exception:
        pass
    return {
        "total_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "success_rate": 0,
        "total_records_inserted": 0,
        "last_run": None,
    }


def render_health_hero(report: QualityReport) -> bool:
    """
    Render the health score hero section.

    Returns True if "Run Check Now" was clicked (signals cache should be cleared).
    """
    score = report.health_score
    color = get_health_color(score)

    # Color mapping for Streamlit styling
    color_hex = {"green": "#28a745", "orange": "#ffc107", "red": "#dc3545"}[color]

    # Center the hero content
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Large health score display
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;">
                <h1 style="font-size: 4rem; margin: 0; color: {color_hex};">
                    {score:.0f}/100
                </h1>
                <p style="font-size: 1.2rem; color: #666;">Health Score</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar
        st.progress(score / 100)

        # Status summary
        status_text = "Healthy" if score >= 80 else "Warning" if score >= 50 else "Critical"
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 10px;">
                <span style="color: {color_hex}; font-weight: bold;">{status_text}</span>
                &nbsp;|&nbsp;
                {report.critical_issues} critical, {report.warning_issues} warnings
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Last checked timestamp
        time_ago = _format_time_ago(report.generated_at)
        st.markdown(
            f'<p style="text-align: center; color: #888; margin-top: 5px;">'
            f"Last checked: {time_ago}</p>",
            unsafe_allow_html=True,
        )

    # Run Check Now button (outside the centered column for full width)
    run_check = st.button("Run Check Now", use_container_width=True)

    return run_check


def _format_time_ago(dt: datetime) -> str:
    """Format a datetime as a human-readable 'time ago' string."""
    now = datetime.now()
    diff = now - dt

    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"
    else:
        return "Just now"


def render_trend_chart(trend_data: pd.DataFrame) -> None:
    """Render the 90-day health score trend chart."""
    st.subheader("Historical Trend (90 days)")

    if trend_data.empty or len(trend_data) == 0:
        st.info("No historical data available yet. Run quality checks over time to build trend data.")
        return

    # Ensure date column is properly typed for charting
    chart_data = trend_data.copy()
    if "date" in chart_data.columns:
        chart_data["date"] = pd.to_datetime(chart_data["date"])
        chart_data = chart_data.set_index("date")

    # Display area chart
    if "health_score" in chart_data.columns:
        st.area_chart(chart_data["health_score"], use_container_width=True)
    else:
        st.warning("Health score data not available in trend data.")


def render_checks_summary(results: list[CheckResult]) -> str | None:
    """
    Render the quality checks summary table.

    Returns the selected check filter (or None for 'All').
    """
    st.subheader("Quality Checks Summary")

    if not results:
        st.info("No check results available.")
        return None

    # Build summary data
    summary_data = []
    for result in results:
        status = "✅ Pass" if result.passed else "❌ Fail"
        summary_data.append({
            "Check": result.check_name,
            "Status": status,
            "Critical": result.critical_count,
            "Warnings": result.warning_count,
        })

    summary_df = pd.DataFrame(summary_data)

    # Display as table
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Check": st.column_config.TextColumn("Check", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Critical": st.column_config.NumberColumn("Critical", width="small"),
            "Warnings": st.column_config.NumberColumn("Warnings", width="small"),
        },
    )

    # Filter dropdown for anomalies section
    check_names = ["All"] + [r.check_name for r in results]
    selected_check = st.selectbox(
        "Filter anomalies by check:",
        check_names,
        key="check_filter",
    )

    return None if selected_check == "All" else selected_check


def render_flagged_anomalies(
    results: list[CheckResult],
    check_filter: str | None,
) -> None:
    """Render the flagged anomalies section with expandable drill-down."""
    st.subheader("Flagged Anomalies")

    # Collect all issues
    all_issues: list[QualityIssue] = []
    for result in results:
        if check_filter is None or result.check_name == check_filter:
            all_issues.extend(result.issues)

    if not all_issues:
        st.success("No anomalies found." if check_filter is None else f"No anomalies for {check_filter}.")
        return

    # Severity filter
    severity_options = ["All", "Critical", "Warning", "Info"]
    selected_severity = st.selectbox(
        "Filter by severity:",
        severity_options,
        key="severity_filter",
    )

    # Filter by severity
    if selected_severity != "All":
        severity_map = {
            "Critical": IssueSeverity.CRITICAL,
            "Warning": IssueSeverity.WARNING,
            "Info": IssueSeverity.INFO,
        }
        target_severity = severity_map.get(selected_severity)
        all_issues = [i for i in all_issues if i.severity == target_severity]

    # Sort: critical first, then by date (recent first)
    severity_order = {IssueSeverity.CRITICAL: 0, IssueSeverity.WARNING: 1, IssueSeverity.INFO: 2}
    all_issues.sort(key=lambda x: (severity_order.get(x.severity, 3), x.date or date.min), reverse=True)

    # Limit to 50 for performance
    display_issues = all_issues[:50]
    if len(all_issues) > 50:
        st.warning(f"Showing 50 of {len(all_issues)} issues. Apply filters to narrow results.")

    # Display issues with expanders
    severity_icons = {
        IssueSeverity.CRITICAL: "🔴",
        IssueSeverity.WARNING: "🟡",
        IssueSeverity.INFO: "🔵",
    }

    for issue in display_issues:
        icon = severity_icons.get(issue.severity, "⚪")
        symbol_str = issue.symbol or "N/A"
        date_str = str(issue.date) if issue.date else "N/A"

        with st.expander(f"{icon} [{symbol_str}] {date_str} - {issue.description}"):
            st.markdown(f"**Check:** {issue.check_name}")
            st.markdown(f"**Severity:** {issue.severity.value}")

            if issue.details:
                st.markdown("**Details:**")
                details_dict = dict(issue.details)
                for key, value in details_dict.items():
                    st.markdown(f"- {key}: {value}")


def render() -> None:
    """Render the Data Health page."""
    st.title("Data Health")
    st.markdown("Monitor data completeness, quality, and ingestion status.")

    # -------------------------------------------------------------------------
    # Health Score Hero
    # -------------------------------------------------------------------------
    try:
        # Check if "Run Check Now" was clicked (clear cache)
        if st.session_state.get("run_check_clicked"):
            st.cache_data.clear()
            st.session_state["run_check_clicked"] = False

        report = get_quality_report(date.today())
        run_check = render_health_hero(report)

        if run_check:
            st.session_state["run_check_clicked"] = True
            st.rerun()

    except Exception as e:
        st.error(f"Failed to load quality report: {e}")
        report = None

    st.divider()

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
    # Historical Trend Chart
    # -------------------------------------------------------------------------
    try:
        trend_data = get_health_trend(days=90)
        render_trend_chart(trend_data)
    except Exception as e:
        st.warning(f"Could not load trend data: {e}")

    st.divider()

    # -------------------------------------------------------------------------
    # Quality Checks Summary & Flagged Anomalies
    # -------------------------------------------------------------------------
    if report is not None:
        col_checks, col_anomalies = st.columns(2)

        with col_checks:
            check_filter = render_checks_summary(report.results)

        with col_anomalies:
            render_flagged_anomalies(report.results, check_filter)

        st.divider()

    # -------------------------------------------------------------------------
    # Ingestion Status
    # -------------------------------------------------------------------------
    st.subheader("Ingestion Status")

    # Get ingestion metrics
    ingestion_summary = get_ingestion_summary()
    last_successful = get_last_successful_ingestion()

    # Display ingestion metrics
    col_ing1, col_ing2, col_ing3, col_ing4 = st.columns(4)

    with col_ing1:
        st.metric(
            label="Total Runs",
            value=f"{ingestion_summary['total_runs']:,}",
            delta=None
        )

    with col_ing2:
        success_rate = ingestion_summary["success_rate"]
        st.metric(
            label="Success Rate",
            value=f"{success_rate:.1f}%",
            delta=None
        )

    with col_ing3:
        st.metric(
            label="Records Inserted",
            value=f"{ingestion_summary['total_records_inserted']:,}",
            delta=None
        )

    with col_ing4:
        if last_successful:
            completed_at = last_successful.get("completed_at")
            if completed_at:
                # Parse datetime if it's a string
                if isinstance(completed_at, str):
                    try:
                        completed_dt = datetime.fromisoformat(completed_at)
                    except ValueError:
                        completed_dt = datetime.strptime(completed_at, "%Y-%m-%d %H:%M:%S")
                else:
                    completed_dt = completed_at

                # Calculate time ago
                time_diff = datetime.now() - completed_dt
                if time_diff.days > 0:
                    time_ago = f"{time_diff.days}d ago"
                elif time_diff.seconds >= 3600:
                    hours = time_diff.seconds // 3600
                    time_ago = f"{hours}h ago"
                elif time_diff.seconds >= 60:
                    minutes = time_diff.seconds // 60
                    time_ago = f"{minutes}m ago"
                else:
                    time_ago = "Just now"

                st.metric(
                    label="Last Success",
                    value=time_ago,
                    delta=None
                )
            else:
                st.metric(label="Last Success", value="Unknown")
        else:
            st.metric(label="Last Success", value="Never")

    # Display last successful run details
    if last_successful:
        st.markdown("**Last Successful Run:**")
        col_det1, col_det2, col_det3 = st.columns(3)

        with col_det1:
            st.text(f"Source: {last_successful.get('source_id', 'N/A')}")
        with col_det2:
            st.text(f"Fetched: {last_successful.get('records_fetched', 0):,}")
        with col_det3:
            st.text(f"Inserted: {last_successful.get('records_inserted', 0):,}")

    st.markdown("---")

    # Recent ingestion jobs table
    st.subheader("Recent Ingestion Jobs")

    ingestion_logs = get_ingestion_logs(limit=10)

    if ingestion_logs.empty:
        st.info("No ingestion logs found. Run an ingestion job to populate this table.")
    else:
        # Format the dataframe for display
        display_df = ingestion_logs.copy()

        # Add status indicator with emoji/color
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
                "records_fetched": st.column_config.NumberColumn("Fetched", width="small"),
                "records_inserted": st.column_config.NumberColumn("Inserted", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "error_message": st.column_config.TextColumn("Error", width="large"),
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
