"""
Prices Data Explorer - Interactive OHLCV Visualization

Candlestick charts with indicators, multi-symbol comparison,
and drill-down capabilities.
"""

from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hrp.dashboard.components.data_explorer.query_engine import QueryEngine
from hrp.dashboard.components.data_explorer.styles import (
    CANDLESTICK_COLORS,
    CHART_DEFAULTS,
    COLORS,
    INDICATOR_COLORS,
    apply_chart_theme,
    FONT_FAMILY,
)


def render_prices_explorer() -> None:
    """Render the prices data explorer interface."""
    # Get available symbols
    all_symbols = QueryEngine.get_universe_symbols(_active_only=True)

    # -------------------------------------------------------------------------
    # Filters Section
    # -------------------------------------------------------------------------
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

        with col1:
            selected_symbols = st.multiselect(
                "Symbols",
                options=all_symbols,
                default=["AAPL", "MSFT", "GOOGL"] if len(all_symbols) >= 3 else all_symbols[:3],
                key="price_symbols",
                help="Select 1-5 symbols to compare",
            )

        with col2:
            date_preset = st.selectbox(
                "Range",
                options=["1M", "3M", "6M", "YTD", "1Y", "5Y", "ALL"],
                index=3,
                key="price_date_preset",
            )

        with col3:
            end_date = st.date_input("End Date", value=datetime.now().date(), key="price_end_date")

        with col4:
            show_indicators = st.multiselect(
                "Indicators",
                options=["SMA 20", "SMA 50", "SMA 200", "Bollinger Bands", "Volume"],
                default=["Volume"],
                key="price_indicators",
            )

        with col5:
            compare_mode = st.checkbox(
                "Compare Mode",
                value=False,
                key="price_compare_mode",
                help="Normalize prices to compare relative performance",
            )

    # Validation
    if not selected_symbols:
        st.info("👈 Select symbols to view price data")
        return

    if len(selected_symbols) > 5:
        st.warning("⚠️ Max 5 symbols for comparison")
        selected_symbols = selected_symbols[:5]

    # Calculate date range
    presets = QueryEngine.get_date_presets()
    start_date = presets[date_preset]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Fetch Data
    # -------------------------------------------------------------------------
    with st.spinner(f"Loading {', '.join(selected_symbols)} price data..."):
        price_df = QueryEngine.get_price_ohlcv(
            symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

    if price_df.empty:
        st.warning(f"No price data found for {', '.join(selected_symbols)}")
        return

    # -------------------------------------------------------------------------
    # Calculate Indicators
    # -------------------------------------------------------------------------
    indicator_data = {}

    if any(x in show_indicators for x in ["SMA 20", "SMA 50", "SMA 200"]):
        for symbol in selected_symbols:
            sym_df = price_df[price_df["symbol"] == symbol].copy()
            sym_df = sym_df.sort_values("date")

            if "SMA 20" in show_indicators:
                sym_df["sma_20"] = sym_df["close"].rolling(window=20).mean()
            if "SMA 50" in show_indicators:
                sym_df["sma_50"] = sym_df["close"].rolling(window=50).mean()
            if "SMA 200" in show_indicators:
                sym_df["sma_200"] = sym_df["close"].rolling(window=200).mean()

            indicator_data[symbol] = sym_df

    # -------------------------------------------------------------------------
    # Main Chart - Candlestick or Comparison Line
    # -------------------------------------------------------------------------
    fig = go.Figure()

    if len(selected_symbols) == 1 or not compare_mode:
        # Single symbol or overlay mode - Candlestick
        for symbol in selected_symbols:
            sym_df = price_df[price_df["symbol"] == symbol]

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=sym_df["date"],
                    open=sym_df["open"],
                    high=sym_df["high"],
                    low=sym_df["low"],
                    close=sym_df["close"],
                    name=symbol,
                    increasing_line_color=CANDLESTICK_COLORS["up"],
                    decreasing_line_color=CANDLESTICK_COLORS["down"],
                )
            )

            # Add SMAs
            if symbol in indicator_data and "SMA 20" in show_indicators:
                sma_df = indicator_data[symbol].dropna(subset=["sma_20"])
                fig.add_trace(
                    go.Scatter(
                        x=sma_df["date"],
                        y=sma_df["sma_20"],
                        mode="lines",
                        name=f"{symbol} SMA 20",
                        line=dict(color=INDICATOR_COLORS["sma_20"], width=1),
                    )
                )

            if symbol in indicator_data and "SMA 50" in show_indicators:
                sma_df = indicator_data[symbol].dropna(subset=["sma_50"])
                fig.add_trace(
                    go.Scatter(
                        x=sma_df["date"],
                        y=sma_df["sma_50"],
                        mode="lines",
                        name=f"{symbol} SMA 50",
                        line=dict(color=INDICATOR_COLORS["sma_50"], width=1),
                    )
                )

            if symbol in indicator_data and "SMA 200" in show_indicators:
                sma_df = indicator_data[symbol].dropna(subset=["sma_200"])
                fig.add_trace(
                    go.Scatter(
                        x=sma_df["date"],
                        y=sma_df["sma_200"],
                        mode="lines",
                        name=f"{symbol} SMA 200",
                        line=dict(color=INDICATOR_COLORS["sma_200"], width=1),
                    )
                )

    else:
        # Compare mode - Normalize to 100
        for symbol in selected_symbols:
            sym_df = price_df[price_df["symbol"] == symbol].copy()
            sym_df = sym_df.sort_values("date")
            normalized = (sym_df["close"] / sym_df["close"].iloc[0]) * 100

            fig.add_trace(
                go.Scatter(
                    x=sym_df["date"],
                    y=normalized,
                    mode="lines",
                    name=symbol,
                    line=dict(width=2),
                )
            )

    # Volume
    if "Volume" in show_indicators:
        for symbol in selected_symbols:
            sym_df = price_df[price_df["symbol"] == symbol]
            fig.add_trace(
                go.Bar(
                    x=sym_df["date"],
                    y=sym_df["volume"],
                    name=f"{symbol} Volume",
                    yaxis="y2",
                    opacity=0.3,
                    marker_color=INDICATOR_COLORS["volume"],
                )
            )

    # Layout
    fig.update_layout(
        title={
            "text": f"{' vs '.join(selected_symbols)} | {date_preset}",
            "font": {"family": FONT_FAMILY, "size": 14, "color": COLORS["text"]},
            "x": 0,
            "xanchor": "left",
        },
        xaxis_title="Date",
        yaxis_title="Price" if not compare_mode else "Normalized (100 = base)",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(family=FONT_FAMILY, size=10),
        ),
        height=500,
        **{k: v for k, v in CHART_DEFAULTS.items() if k not in ["height"]},
    )

    fig = apply_chart_theme(fig, "candlestick")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Summary Statistics Table
    # -------------------------------------------------------------------------
    with st.expander("📊 Summary Statistics", expanded=False):
        stats_df = QueryEngine.get_price_summary_stats(_symbols=tuple(selected_symbols))

        if not stats_df.empty:
            display_stats = stats_df.copy()
            display_stats["avg_close"] = display_stats["avg_close"].round(2)
            display_stats["std_close"] = display_stats["std_close"].round(2)

            display_stats = display_stats.rename(
                columns={
                    "symbol": "Symbol",
                    "total_records": "Records",
                    "first_date": "First",
                    "last_date": "Last",
                    "avg_close": "Avg Close",
                    "max_close": "Max Close",
                    "min_close": "Min Close",
                    "std_close": "Std Dev",
                    "avg_volume": "Avg Volume",
                }
            )

            st.dataframe(
                display_stats,
                use_container_width=True,
                hide_index=True,
            )

    # -------------------------------------------------------------------------
    # Recent Price Data Table
    # -------------------------------------------------------------------------
    with st.expander("📋 Recent Price Data", expanded=False):
        recent_df = price_df.groupby("symbol").head(10).sort_values(["symbol", "date"], ascending=[True, False])
        display_df = recent_df[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
        display_df.columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]

        # Format numeric columns
        for col in ["Open", "High", "Low", "Close"]:
            display_df[col] = display_df[col].round(2)
        display_df["Volume"] = (display_df["Volume"] / 1_000_000).round(2).astype(str) + "M"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )
