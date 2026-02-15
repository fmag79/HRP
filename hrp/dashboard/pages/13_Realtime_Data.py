"""Real-time Data Monitoring page for HRP.

Displays WebSocket connection status, intraday bars, features, and session stats.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from hrp.api.platform import PlatformAPI


def _format_timestamp(ts: datetime | str | None) -> str:
    """Format a timestamp for display."""
    if ts is None:
        return "N/A"

    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return ts

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    delta = now - ts

    if delta < timedelta(minutes=1):
        return "just now"
    elif delta < timedelta(hours=1):
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes}m ago"
    elif delta < timedelta(days=1):
        hours = int(delta.total_seconds() / 3600)
        return f"{hours}h ago"
    else:
        return ts.strftime("%Y-%m-%d %H:%M")


def _get_connection_status(api: PlatformAPI) -> dict[str, Any]:
    """Get real-time ingestion connection status from ingestion_log."""
    try:
        # Query most recent intraday session from ingestion_log
        query = """
            SELECT source_id, status, start_time, end_time, records_count,
                   error_message, metadata
            FROM ingestion_log
            WHERE source_id = 'intraday_ingestion'
            ORDER BY start_time DESC
            LIMIT 1
        """
        result = api.fetchone_readonly(query)

        if not result:
            return {
                "status": "never_started",
                "last_activity": None,
                "session_start": None,
                "bars_received": 0,
                "is_connected": False,
            }

        status_str = result[1]
        start_time = result[2]
        end_time = result[3]
        records_count = result[4] or 0
        metadata = result[6] or {}

        # Determine if currently active
        is_active = status_str == "running" and end_time is None

        return {
            "status": "connected" if is_active else "disconnected",
            "last_activity": end_time if end_time else start_time,
            "session_start": start_time,
            "bars_received": records_count,
            "is_connected": is_active,
            "metadata": metadata,
        }

    except Exception as e:
        logger.debug(f"Could not get connection status: {e}")
        return {
            "status": "error",
            "last_activity": None,
            "session_start": None,
            "bars_received": 0,
            "is_connected": False,
            "error": str(e),
        }


def _get_intraday_bars(
    api: PlatformAPI, symbol: str | None = None, limit: int = 100
) -> pd.DataFrame:
    """Get recent intraday bars."""
    try:
        if symbol:
            query = """
                SELECT symbol, timestamp, open, high, low, close, volume, vwap
                FROM intraday_bars
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return api.query_readonly(query, (symbol, limit))
        else:
            query = """
                SELECT symbol, timestamp, open, high, low, close, volume, vwap
                FROM intraday_bars
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return api.query_readonly(query, (limit,))

    except Exception as e:
        logger.debug(f"Could not load intraday bars: {e}")
        return pd.DataFrame()


def _get_latest_prices(api: PlatformAPI) -> pd.DataFrame:
    """Get the most recent intraday bar for each symbol (live prices)."""
    try:
        query = """
            WITH ranked AS (
                SELECT symbol, timestamp, close, volume,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                FROM intraday_bars
            )
            SELECT symbol, timestamp, close, volume
            FROM ranked
            WHERE rn = 1
            ORDER BY symbol
        """
        return api.query_readonly(query)

    except Exception as e:
        logger.debug(f"Could not load latest prices: {e}")
        return pd.DataFrame()


def _get_intraday_features(
    api: PlatformAPI, symbol: str, limit: int = 50
) -> pd.DataFrame:
    """Get recent intraday features for a symbol."""
    try:
        query = """
            SELECT timestamp, feature_name, value
            FROM intraday_features
            WHERE symbol = ?
            ORDER BY timestamp DESC, feature_name
            LIMIT ?
        """
        return api.query_readonly(query, (symbol, limit))

    except Exception as e:
        logger.debug(f"Could not load intraday features: {e}")
        return pd.DataFrame()


def _render_connection_status(status: dict[str, Any]) -> None:
    """Render connection status indicator."""
    st.subheader("Connection Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if status["is_connected"]:
            st.success("✓ Connected")
        elif status["status"] == "never_started":
            st.info("○ Not Started")
        else:
            st.error("✗ Disconnected")

    with col2:
        last_activity = status.get("last_activity")
        if last_activity:
            st.metric("Last Activity", _format_timestamp(last_activity))
        else:
            st.metric("Last Activity", "N/A")

    with col3:
        st.metric("Bars Received", f"{status['bars_received']:,}")

    with col4:
        session_start = status.get("session_start")
        if session_start and status["is_connected"]:
            if isinstance(session_start, str):
                session_start = datetime.fromisoformat(session_start.replace("Z", "+00:00"))
            uptime_seconds = (datetime.now(UTC) - session_start).total_seconds()
            uptime_minutes = int(uptime_seconds / 60)
            st.metric("Uptime", f"{uptime_minutes} min")
        else:
            st.metric("Uptime", "N/A")


def _render_live_prices(api: PlatformAPI) -> None:
    """Render live prices table."""
    st.subheader("Live Prices (Latest Intraday Bars)")

    prices_df = _get_latest_prices(api)

    if prices_df.empty:
        st.info("No intraday data available. Start real-time ingestion to see live prices.")
        return

    # Format for display
    display_df = prices_df.copy()
    display_df["close"] = display_df["close"].apply(lambda x: f"${x:.2f}")
    display_df["volume"] = display_df["volume"].apply(lambda x: f"{x:,}")

    if "timestamp" in display_df.columns:
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime(
            "%H:%M:%S"
        )

    display_df = display_df.rename(columns={
        "symbol": "Symbol",
        "timestamp": "Time",
        "close": "Price",
        "volume": "Volume",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )


def _render_candlestick_chart(api: PlatformAPI, symbol: str) -> None:
    """Render intraday candlestick chart for a symbol."""
    st.subheader(f"Intraday Chart: {symbol}")

    bars = _get_intraday_bars(api, symbol=symbol, limit=390)  # Full trading day

    if bars.empty:
        st.info(f"No intraday data for {symbol}")
        return

    # Sort by timestamp ascending for chart
    bars = bars.sort_values("timestamp")

    # Create candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=bars["timestamp"],
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name=symbol,
        )
    ])

    fig.update_layout(
        title=f"{symbol} - Intraday (Minute Bars)",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show volume bars below
    vol_fig = go.Figure(data=[
        go.Bar(
            x=bars["timestamp"],
            y=bars["volume"],
            name="Volume",
            marker_color="rgba(100, 149, 237, 0.6)",
        )
    ])

    vol_fig.update_layout(
        title=f"{symbol} - Volume",
        xaxis_title="Time",
        yaxis_title="Volume",
        height=200,
    )

    st.plotly_chart(vol_fig, use_container_width=True)


def _render_intraday_features(api: PlatformAPI, symbol: str) -> None:
    """Render intraday features for a symbol."""
    st.subheader(f"Intraday Features: {symbol}")

    features = _get_intraday_features(api, symbol, limit=200)

    if features.empty:
        st.info(f"No intraday features for {symbol}")
        return

    # Pivot to show features as columns
    pivot_df = features.pivot(index="timestamp", columns="feature_name", values="value")
    pivot_df = pivot_df.sort_index(ascending=False)

    # Format timestamps
    pivot_df.index = pd.to_datetime(pivot_df.index).strftime("%H:%M:%S")

    # Display as table
    st.dataframe(
        pivot_df.head(20),
        use_container_width=True,
        height=400,
    )


def _render_session_stats(api: PlatformAPI) -> None:
    """Render session statistics."""
    st.subheader("Session Statistics")

    try:
        # Get stats from ingestion_log metadata
        query = """
            SELECT start_time, end_time, records_count, metadata
            FROM ingestion_log
            WHERE source_id = 'intraday_ingestion'
            ORDER BY start_time DESC
            LIMIT 10
        """
        sessions = api.query_readonly(query)

        if sessions.empty:
            st.info("No session history available")
            return

        # Format for display
        display_df = sessions.copy()

        if "start_time" in display_df.columns:
            display_df["start_time"] = pd.to_datetime(display_df["start_time"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        if "end_time" in display_df.columns:
            display_df["end_time"] = pd.to_datetime(display_df["end_time"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        display_df["records_count"] = display_df["records_count"].apply(
            lambda x: f"{x:,}" if pd.notna(x) else "0"
        )

        display_df = display_df.rename(columns={
            "start_time": "Start",
            "end_time": "End",
            "records_count": "Bars",
        })

        st.dataframe(
            display_df[["Start", "End", "Bars"]],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        logger.debug(f"Could not load session stats: {e}")
        st.info("Session statistics unavailable")


def render() -> None:
    """Render the real-time data monitoring page."""
    st.title("Real-time Data Monitoring")
    st.caption("Live intraday data ingestion via Polygon.io WebSocket")

    # Initialize API (read-only for dashboard queries)
    try:
        api = PlatformAPI(read_only=True)
    except Exception as e:
        st.error(f"Failed to initialize Platform API: {e}")
        return

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.toggle("Auto-refresh (10s)", value=False)

    if auto_refresh:
        st.info("Auto-refresh enabled. Page will update every 10 seconds.")

    # =========================================================================
    # Connection Status
    # =========================================================================
    status = _get_connection_status(api)
    _render_connection_status(status)

    st.divider()

    # =========================================================================
    # Live Prices Table
    # =========================================================================
    _render_live_prices(api)

    st.divider()

    # =========================================================================
    # Symbol Selection for Detailed View
    # =========================================================================
    st.header("Detailed View")

    # Get available symbols from intraday_bars
    try:
        symbols_query = "SELECT DISTINCT symbol FROM intraday_bars ORDER BY symbol"
        available_symbols_df = api.query_readonly(symbols_query)
        available_symbols = (
            available_symbols_df["symbol"].tolist()
            if not available_symbols_df.empty
            else []
        )
    except Exception:
        available_symbols = []

    if available_symbols:
        selected_symbol = st.selectbox(
            "Select symbol for detailed view",
            options=available_symbols,
            index=0,
        )

        if selected_symbol:
            # Candlestick chart
            _render_candlestick_chart(api, selected_symbol)

            st.divider()

            # Intraday features
            _render_intraday_features(api, selected_symbol)
    else:
        st.info("No symbols with intraday data. Start real-time ingestion to see charts.")

    st.divider()

    # =========================================================================
    # Session Statistics
    # =========================================================================
    _render_session_stats(api)

    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()


# Run the page
if __name__ == "__main__" or True:  # Always run when loaded as a Streamlit page
    render()
