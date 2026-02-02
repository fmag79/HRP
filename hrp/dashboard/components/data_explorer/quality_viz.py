"""
Quality Data Visualization - Anomalies, Gaps, Freshness

Visual representation of data quality issues with interactive drill-down.
"""

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hrp.api.platform import PlatformAPI

def _get_api():
    return PlatformAPI()

from hrp.dashboard.components.data_explorer.styles import (
    CHART_DEFAULTS,
    COLORS,
    apply_chart_theme,
    FONT_FAMILY,
    get_status_badge_style,
)


@st.cache_data(ttl=300)
def get_missing_dates() -> pd.DataFrame:
    """Get symbols with missing date gaps."""
    api = _get_api()

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
        WHERE gap_days > 1 AND gap_days <= 10
        GROUP BY symbol
        HAVING COUNT(*) > 0
        ORDER BY total_missing_days DESC
        LIMIT 50
    """

    return api.query_readonly(query)


@st.cache_data(ttl=300)
def get_price_anomalies() -> pd.DataFrame:
    """Get price data anomalies."""
    api = _get_api()

    query = """
        SELECT
            symbol,
            date,
            close,
            volume,
            CASE
                WHEN close <= 0 THEN 'Zero/Negative close'
                WHEN close IS NULL THEN 'Null close'
                ELSE 'Unknown'
            END as anomaly_type
        FROM prices
        WHERE close <= 0 OR close IS NULL
        ORDER BY date DESC
        LIMIT 100
    """

    return api.query_readonly(query)


@st.cache_data(ttl=60)
def get_symbol_freshness() -> pd.DataFrame:
    """Get data freshness per symbol."""
    api = _get_api()

    query = """
        SELECT
            symbol,
            MAX(date) as last_date,
            COUNT(*) as total_records,
            MIN(date) as first_date
        FROM prices
        GROUP BY symbol
        ORDER BY last_date DESC
    """

    df = api.query_readonly(query)

    # Calculate days stale
    today = datetime.now().date()
    df["last_date"] = pd.to_datetime(df["last_date"]).dt.date
    df["days_stale"] = (today - df["last_date"]).dt.days
    df["is_fresh"] = df["days_stale"] <= 3

    return df


def render_quality_viz() -> None:
    """Render quality data visualization interface."""
    st.subheader("Data Quality Visualization")

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
    freshness_df = get_symbol_freshness()

    if not freshness_df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Symbols", len(freshness_df))

        with col2:
            fresh_count = freshness_df["is_fresh"].sum()
            st.metric("Fresh Data", fresh_count)

        with col3:
            stale_count = (~freshness_df["is_fresh"]).sum()
            st.metric("Stale Data", stale_count)

        with col4:
            avg_staleness = freshness_df["days_stale"].mean()
            st.metric("Avg Days Stale", f"{avg_staleness:.1f}")

    st.divider()

    # -------------------------------------------------------------------------
    # View Selection
    # -------------------------------------------------------------------------
    view = st.selectbox(
        "Quality View",
        options=["Data Freshness", "Missing Date Gaps", "Price Anomalies", "Coverage Heatmap"],
        key="quality_view",
    )

    # -------------------------------------------------------------------------
    # Data Freshness View
    # -------------------------------------------------------------------------
    if view == "Data Freshness":
        st.subheader("Data Freshness by Symbol")

        if freshness_df.empty:
            st.info("No freshness data available")
            return

        # Color by freshness status
        freshness_df["status"] = freshness_df["is_fresh"].map({True: "Fresh", False: "Stale"})

        # Sort by days stale
        display_df = freshness_df.sort_values("days_stale", ascending=False).head(50)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=display_df["symbol"],
                y=display_df["days_stale"],
                marker_color=[
                    COLORS["success"] if is_fresh else COLORS["error"] for is_fresh in display_df["is_fresh"]
                ],
                text=[f"{d}d" for d in display_df["days_stale"]],
                textposition="outside",
            )
        )

        # Add reference line at 3 days
        fig.add_hline(
            y=3,
            line_dash="dash",
            line_color=COLORS["warning"],
            annotation_text="Fresh threshold (3 days)",
        )

        fig.update_layout(
            title={
                "text": "Days Since Last Update (Top 50)",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            xaxis_title="Symbol",
            yaxis_title="Days Stale",
            height=450,
            xaxis={"tickangle": -45},
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Freshness table
        with st.expander("📋 All Symbols Freshness", expanded=False):
            display_table = freshness_df[["symbol", "last_date", "days_stale", "is_fresh", "total_records"]].copy()
            display_table.columns = ["Symbol", "Last Date", "Days Stale", "Fresh", "Records"]
            display_table["Fresh"] = display_table["Fresh"].map({True: "✅", False: "❌"})

            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True,
            )

    # -------------------------------------------------------------------------
    # Missing Date Gaps View
    # -------------------------------------------------------------------------
    elif view == "Missing Date Gaps":
        st.subheader("Missing Trading Days by Symbol")

        gaps_df = get_missing_dates()

        if gaps_df.empty:
            st.success("✅ No significant date gaps detected!")
            return

        # Visualization
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=gaps_df["symbol"],
                y=gaps_df["total_missing_days"],
                marker_color=COLORS["warning"],
                text=[f"{d} days" for d in gaps_df["total_missing_days"]],
                textposition="outside",
            )
        )

        fig.update_layout(
            title={
                "text": "Total Missing Trading Days by Symbol",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            xaxis_title="Symbol",
            yaxis_title="Total Missing Days",
            height=450,
            xaxis={"tickangle": -45},
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Gaps table
        with st.expander("📋 Gap Details", expanded=True):
            display_df = gaps_df.copy()
            display_df.columns = ["Symbol", "Gap Count", "Total Missing", "Max Gap"]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

    # -------------------------------------------------------------------------
    # Price Anomalies View
    # -------------------------------------------------------------------------
    elif view == "Price Anomalies":
        st.subheader("Price Data Anomalies")

        anomalies_df = get_price_anomalies()

        if anomalies_df.empty:
            st.success("✅ No price anomalies detected!")
            return

        # Group by anomaly type
        type_counts = anomalies_df["anomaly_type"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Anomalies", len(anomalies_df))

        with col2:
            st.metric("Unique Symbols", anomalies_df["symbol"].nunique())

        # Anomaly type distribution
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=type_counts.index,
                y=type_counts.values,
                marker_color=COLORS["error"],
                text=[f"{c}" for c in type_counts.values],
                textposition="outside",
            )
        )

        fig.update_layout(
            title={
                "text": "Anomalies by Type",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            xaxis_title="Anomaly Type",
            yaxis_title="Count",
            height=400,
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Anomalies table
        with st.expander("📋 Anomaly Details", expanded=True):
            display_df = anomalies_df.copy()
            display_df["date"] = display_df["date"].astype(str)

            st.dataframe(
                display_df[["symbol", "date", "anomaly_type", "close"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "date": st.column_config.TextColumn("Date", width="medium"),
                    "anomaly_type": st.column_config.TextColumn("Type", width="medium"),
                    "close": st.column_config.NumberColumn("Close", width="small", format="%.2f"),
                },
            )

    # -------------------------------------------------------------------------
    # Coverage Heatmap View
    # -------------------------------------------------------------------------
    elif view == "Coverage Heatmap":
        st.subheader("Data Coverage Heatmap")

        if freshness_df.empty:
            st.info("No data available")
            return

        # Create coverage metrics
        freshness_df["coverage_days"] = (freshness_df["last_date"] - freshness_df["first_date"]).dt.days + 1
        freshness_df["coverage_pct"] = (freshness_df["total_records"] / freshness_df["coverage_days"] * 100).clip(
            0, 100
        )

        # Top 50 symbols by records
        top_symbols = freshness_df.nlargest(50, "total_records")

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=top_symbols["coverage_pct"].values.reshape(1, -1),
                x=top_symbols["symbol"],
                y=["Coverage"],
                colorscale="RdYlGn",
                colorbar=dict(title="Coverage %"),
                text=[f"{p:.0f}%" for p in top_symbols["coverage_pct"]],
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(
            title={
                "text": "Data Coverage Percentage (Top 50 Symbols)",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            xaxis_title="Symbol",
            height=200,
            xaxis={"tickangle": -45},
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig, "heatmap")
        st.plotly_chart(fig, use_container_width=True)

        # Coverage stats
        with st.expander("📊 Coverage Statistics", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_coverage = freshness_df["coverage_pct"].mean()
                st.metric("Avg Coverage", f"{avg_coverage:.1f}%")

            with col2:
                min_coverage = freshness_df["coverage_pct"].min()
                st.metric("Min Coverage", f"{min_coverage:.1f}%")

            with col3:
                max_coverage = freshness_df["coverage_pct"].max()
                st.metric("Max Coverage", f"{max_coverage:.1f}%")
