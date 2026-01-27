"""
Fundamentals Data Explorer - Company Metrics Visualization

P/E, P/B, market cap, dividend yield history and peer comparison.
"""

from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hrp.dashboard.components.data_explorer.query_engine import QueryEngine
from hrp.dashboard.components.data_explorer.styles import (
    CHART_DEFAULTS,
    COLORS,
    apply_chart_theme,
    FONT_FAMILY,
)

# Available fundamental metrics
FUNDAMENTAL_METRICS = [
    "pe_ratio",
    "pb_ratio",
    "market_cap",
    "dividend_yield",
    "ev_ebitda",
    "revenue",
    "eps",
    "book_value",
]


def render_fundamentals_explorer() -> None:
    """Render the fundamentals data explorer interface."""
    # Get available symbols
    all_symbols = QueryEngine.get_universe_symbols(_active_only=True)

    # -------------------------------------------------------------------------
    # Filters Section
    # -------------------------------------------------------------------------
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

        with col1:
            selected_symbols = st.multiselect(
                "Symbols",
                options=all_symbols,
                default=["AAPL", "MSFT"] if len(all_symbols) >= 2 else all_symbols[:2],
                key="fund_symbols",
                help="Select up to 5 symbols",
            )

        with col2:
            viz_type = st.selectbox(
                "View",
                options=["Metric History", "Peer Comparison", "Data Table"],
                key="fund_viz_type",
            )

        with col3:
            if viz_type == "Metric History":
                metric = st.selectbox(
                    "Metric",
                    options=FUNDAMENTAL_METRICS,
                    index=0,
                    key="fund_metric",
                )

        with col4:
            date_preset = st.selectbox(
                "Period",
                options=["1M", "3M", "6M", "YTD", "1Y", "ALL"],
                index=2,
                key="fund_period",
            )

    # Validation
    if not selected_symbols:
        st.info("👈 Select symbols to view fundamentals")
        return

    if len(selected_symbols) > 5:
        st.warning("⚠️ Max 5 symbols for comparison")
        selected_symbols = selected_symbols[:5]

    # Calculate date range
    presets = QueryEngine.get_date_presets()
    start_date = presets[date_preset]
    end_date = datetime.now().date()

    # -------------------------------------------------------------------------
    # Metric History View
    # -------------------------------------------------------------------------
    if viz_type == "Metric History":
        st.subheader(f"{metric.replace('_', ' ').title()} History")

        with st.spinner(f"Loading {metric} data..."):
            df = QueryEngine.get_fundamentals_history(
                _symbols=tuple(selected_symbols),
                _metrics=(metric,),
                _start_date=start_date,
                _end_date=end_date,
            )

        if df.empty or metric not in df.columns:
            st.warning(f"No {metric} data found for selected symbols")
            return

        # Create line chart
        fig = go.Figure()

        for symbol in selected_symbols:
            sym_df = df[df["symbol"] == symbol].sort_values("as_of_date")
            sym_df = sym_df.dropna(subset=[metric])

            if not sym_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sym_df["as_of_date"],
                        y=sym_df[metric],
                        mode="lines+markers",
                        name=symbol,
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )

        fig.update_layout(
            title={
                "text": f"{metric.replace('_', ' ').title()} | {date_preset}",
                "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            },
            xaxis_title="Date",
            yaxis_title=metric.replace("_", " ").title(),
            height=450,
            hovermode="x unified",
            **CHART_DEFAULTS,
        )

        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Current values table
        with st.expander("📋 Current Values", expanded=True):
            current_values = []

            for symbol in selected_symbols:
                sym_df = df[df["symbol"] == symbol].sort_values("as_of_date", ascending=False)
                if not sym_df.empty and metric in sym_df.columns:
                    latest = sym_df.iloc[0]
                    current_values.append(
                        {
                            "Symbol": symbol,
                            "Value": f"{latest[metric]:.2f}" if pd.notna(latest[metric]) else "N/A",
                            "Date": str(latest["as_of_date"]),
                        }
                    )

            if current_values:
                st.dataframe(
                    pd.DataFrame(current_values),
                    use_container_width=True,
                    hide_index=True,
                )

    # -------------------------------------------------------------------------
    # Peer Comparison View
    # -------------------------------------------------------------------------
    elif viz_type == "Peer Comparison":
        st.subheader("Peer Comparison")

        # Multi-metric comparison
        metrics_to_compare = st.multiselect(
            "Metrics to Compare",
            options=FUNDAMENTAL_METRICS,
            default=["pe_ratio", "pb_ratio", "market_cap", "dividend_yield"],
            key="fund_compare_metrics",
        )

        if not metrics_to_compare:
            st.warning("Select at least one metric")
            return

        with st.spinner("Loading fundamentals data..."):
            df = QueryEngine.get_fundamentals_history(
                _symbols=tuple(selected_symbols),
                _metrics=tuple(metrics_to_compare),
                _start_date=start_date,
                _end_date=end_date,
            )

        if df.empty:
            st.warning("No fundamentals data found")
            return

        # Get latest values per symbol
        latest_data = []
        for symbol in selected_symbols:
            sym_df = df[df["symbol"] == symbol].sort_values("as_of_date", ascending=False)
            if not sym_df.empty:
                latest = sym_df.iloc[0].to_dict()
                latest_data.append(latest)

        if not latest_data:
            st.warning("No data to display")
            return

        latest_df = pd.DataFrame(latest_data)

        # Create comparison chart for first selected metric
        primary_metric = metrics_to_compare[0]
        if primary_metric in latest_df.columns:
            fig = go.Figure()

            values = latest_df[primary_metric].values
            symbols = latest_df["symbol"].values

            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=values,
                    marker_color=COLORS["accent"],
                    text=[f"{v:.2f}" if pd.notna(v) else "N/A" for v in values],
                    textposition="outside",
                )
            )

            fig.update_layout(
                title={
                    "text": f"{primary_metric.replace('_', ' ').title()} Comparison",
                    "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
                },
                xaxis_title="Symbol",
                yaxis_title=primary_metric.replace("_", " ").title(),
                height=400,
                **CHART_DEFAULTS,
            )

            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Multi-metric table
        st.subheader("Metrics Comparison Table")

        display_df = latest_df[["symbol"] + metrics_to_compare].copy()
        display_df.columns = ["Symbol"] + [m.replace("_ " " ").title() for m in metrics_to_compare]

        # Format numeric columns
        for col in display_df.columns:
            if col != "Symbol":
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------------------------------------------------------
    # Data Table View
    # -------------------------------------------------------------------------
    elif viz_type == "Data Table":
        st.subheader("Fundamentals Data Table")

        metrics_to_show = st.multiselect(
            "Metrics",
            options=FUNDAMENTAL_METRICS,
            default=FUNDAMENTAL_METRICS[:5],
            key="fund_table_metrics",
        )

        if not metrics_to_show:
            st.warning("Select at least one metric")
            return

        with st.spinner("Loading fundamentals data..."):
            df = QueryEngine.get_fundamentals_history(
                _symbols=tuple(selected_symbols),
                _metrics=tuple(metrics_to_show),
                _start_date=start_date,
                _end_date=end_date,
            )

        if df.empty:
            st.warning("No data found")
            return

        # Flatten and display
        display_df = df.copy()
        display_df["as_of_date"] = display_df["as_of_date"].astype(str)

        # Rename columns
        column_map = {"symbol": "Symbol", "as_of_date": "Date"}
        column_map.update({m: m.replace("_", " ").title() for m in metrics_to_show})
        display_df = display_df.rename(columns=column_map)

        # Format numeric columns
        for col in metrics_to_show:
            formatted_col = col.replace("_", " ").title()
            if formatted_col in display_df.columns:
                display_df[formatted_col] = display_df[formatted_col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )

        st.dataframe(
            display_df[["Symbol", "Date"] + [m.replace("_", " ").title() for m in metrics_to_show]],
            use_container_width=True,
            hide_index=True,
        )
