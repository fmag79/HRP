"""Trading dashboard page for HRP.

Displays portfolio overview, positions, trades, and model performance.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger

from hrp.api.platform import PlatformAPI


def render() -> None:
    """Render the trading dashboard page."""
    st.title("Trading")
    st.caption("Portfolio overview, positions, and trade execution monitoring")

    api = PlatformAPI()

    # Portfolio Overview Section
    _render_portfolio_overview(api)

    st.divider()

    # Positions and Trades in columns
    col1, col2 = st.columns(2)

    with col1:
        _render_positions(api)

    with col2:
        _render_recent_trades(api)

    st.divider()

    # Model Performance Section
    _render_model_performance(api)


def _render_portfolio_overview(api: PlatformAPI) -> None:
    """Render portfolio overview metrics."""
    st.subheader("Portfolio Overview")

    # Get portfolio data
    try:
        positions = api.query_readonly("SELECT * FROM live_positions")
        trades = api.query_readonly(
            "SELECT * FROM executed_trades WHERE filled_at >= ?",
            (date.today() - timedelta(days=30),),
        )
    except Exception as e:
        logger.debug(f"Could not load trading data: {e}")
        positions = pd.DataFrame()
        trades = pd.DataFrame()

    # Calculate metrics
    if not positions.empty:
        portfolio_value = positions["market_value"].sum()
        total_cost = positions["cost_basis"].sum()
        unrealized_pnl = positions["unrealized_pnl"].sum()
        position_count = len(positions)
    else:
        portfolio_value = Decimal("0")
        total_cost = Decimal("0")
        unrealized_pnl = Decimal("0")
        position_count = 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${portfolio_value:,.2f}" if portfolio_value else "$0.00",
        )

    with col2:
        pnl_delta = f"{float(unrealized_pnl):+,.2f}" if unrealized_pnl else None
        st.metric(
            "Unrealized P&L",
            f"${unrealized_pnl:,.2f}" if unrealized_pnl else "$0.00",
            delta=pnl_delta,
            delta_color="normal" if unrealized_pnl >= 0 else "inverse",
        )

    with col3:
        st.metric("Positions", position_count)

    with col4:
        trades_today = 0
        if not trades.empty and "filled_at" in trades.columns:
            today = date.today()
            trades_today = len(trades[trades["filled_at"].dt.date == today])
        st.metric("Trades Today", trades_today)


def _render_positions(api: PlatformAPI) -> None:
    """Render current positions table."""
    st.subheader("Current Positions")

    try:
        positions = api.query_readonly(
            """
            SELECT symbol, quantity, entry_price, current_price,
                   market_value, unrealized_pnl, unrealized_pnl_pct,
                   hypothesis_id, as_of_date
            FROM live_positions
            ORDER BY unrealized_pnl DESC
            """
        )
    except Exception as e:
        logger.debug(f"Could not load positions: {e}")
        positions = pd.DataFrame()

    if positions.empty:
        st.info("No positions to display")
        return

    # Format for display
    display_df = positions.copy()
    display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
    display_df["current_price"] = display_df["current_price"].apply(lambda x: f"${x:.2f}")
    display_df["market_value"] = display_df["market_value"].apply(lambda x: f"${x:,.2f}")
    display_df["unrealized_pnl"] = display_df["unrealized_pnl"].apply(
        lambda x: f"${x:+,.2f}"
    )
    display_df["unrealized_pnl_pct"] = display_df["unrealized_pnl_pct"].apply(
        lambda x: f"{x:+.2%}"
    )

    # Rename columns for display
    display_df = display_df.rename(columns={
        "symbol": "Symbol",
        "quantity": "Qty",
        "entry_price": "Entry",
        "current_price": "Current",
        "market_value": "Value",
        "unrealized_pnl": "P&L",
        "unrealized_pnl_pct": "P&L %",
    })

    # Select columns to display
    st.dataframe(
        display_df[["Symbol", "Qty", "Entry", "Current", "Value", "P&L", "P&L %"]],
        use_container_width=True,
        hide_index=True,
    )


def _render_recent_trades(api: PlatformAPI) -> None:
    """Render recent trades table."""
    st.subheader("Recent Trades")

    try:
        trades = api.query_readonly(
            """
            SELECT symbol, side, quantity, filled_price, commission,
                   status, filled_at
            FROM executed_trades
            ORDER BY filled_at DESC
            LIMIT 10
            """
        )
    except Exception as e:
        logger.debug(f"Could not load trades: {e}")
        trades = pd.DataFrame()

    if trades.empty:
        st.info("No recent trades to display")
        return

    # Format for display
    display_df = trades.copy()
    display_df["filled_price"] = display_df["filled_price"].apply(
        lambda x: f"${x:.2f}" if x else "N/A"
    )
    display_df["commission"] = display_df["commission"].apply(
        lambda x: f"${x:.2f}" if x else "$0.00"
    )
    display_df["side"] = display_df["side"].str.upper()

    if "filled_at" in display_df.columns:
        display_df["filled_at"] = pd.to_datetime(display_df["filled_at"]).dt.strftime(
            "%m/%d %H:%M"
        )

    # Rename columns
    display_df = display_df.rename(columns={
        "symbol": "Symbol",
        "side": "Side",
        "quantity": "Qty",
        "filled_price": "Price",
        "commission": "Comm",
        "status": "Status",
        "filled_at": "Time",
    })

    st.dataframe(
        display_df[["Symbol", "Side", "Qty", "Price", "Comm", "Time"]],
        use_container_width=True,
        hide_index=True,
    )


def _render_model_performance(api: PlatformAPI) -> None:
    """Render deployed model performance section."""
    st.subheader("Model Performance")

    # Get deployed strategies
    deployed = api.get_deployed_strategies()

    if not deployed:
        st.info("No deployed strategies")
        return

    for strategy in deployed:
        hypothesis_id = strategy.get("hypothesis_id") or getattr(
            strategy, "hypothesis_id", None
        )
        metadata = strategy.get("metadata") or getattr(strategy, "metadata", {})
        model_name = metadata.get("model_name") if isinstance(metadata, dict) else None
        title = strategy.get("title", hypothesis_id)

        with st.expander(f"{title} ({model_name or 'N/A'})", expanded=False):
            # Get drift checks
            try:
                drift_checks = api.query_readonly(
                    """
                    SELECT check_timestamp, drift_type, metric_value,
                           is_drift_detected, threshold_value
                    FROM model_drift_checks
                    WHERE model_name = ?
                    ORDER BY check_timestamp DESC
                    LIMIT 5
                    """,
                    (model_name,),
                )
            except Exception as e:
                logger.debug(f"Could not load drift checks: {e}")
                drift_checks = pd.DataFrame()

            if drift_checks.empty:
                st.info("No drift checks recorded")
            else:
                latest = drift_checks.iloc[0]
                is_drift = latest.get("is_drift_detected", False)

                col1, col2, col3 = st.columns(3)

                with col1:
                    status = "Drift Detected" if is_drift else "Stable"
                    st.metric("Status", status)

                with col2:
                    drift_val = latest.get("metric_value", 0)
                    st.metric("Drift Score", f"{drift_val:.4f}")

                with col3:
                    threshold = latest.get("threshold_value", 0.2)
                    st.metric("Threshold", f"{threshold:.2f}")

                # Show recent drift history
                st.caption("Recent Drift Checks")
                st.dataframe(
                    drift_checks[["check_timestamp", "drift_type", "metric_value", "is_drift_detected"]],
                    use_container_width=True,
                    hide_index=True,
                )


def _render_actions(api: PlatformAPI) -> None:
    """Render action buttons."""
    st.subheader("Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sync Positions", use_container_width=True):
            st.info("Position sync would be triggered here")
            # In production, would call position sync

    with col2:
        if st.button("Check Drift", use_container_width=True):
            st.info("Drift check would be triggered here")
            # In production, would run drift check


# Run the page
if __name__ == "__main__" or True:  # Always run when loaded as a Streamlit page
    render()
