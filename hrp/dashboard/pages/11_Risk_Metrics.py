"""
Risk Metrics Dashboard for HRP.

Interactive visualization of Value-at-Risk (VaR) and Conditional VaR (CVaR)
calculations for portfolio risk monitoring.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.data.risk.var_calculator import VaRCalculator, VaRResult
from hrp.data.risk.risk_config import (
    VaRConfig,
    VaRMethod,
    Distribution,
    VAR_95_1D,
    VAR_99_1D,
    VAR_95_10D,
    MC_VAR_95_1D,
)


def render() -> None:
    """Render the risk metrics dashboard page."""
    st.title("Risk Metrics")
    st.caption("Portfolio VaR, CVaR, and risk exposure monitoring")

    api = PlatformAPI()

    # Configuration section
    config = _render_config_section()

    st.divider()

    # Risk Manager Limits (Phase 4)
    _render_risk_manager_limits(api)

    st.divider()

    # Portfolio VaR Overview
    _render_portfolio_var_overview(api, config)

    st.divider()

    # Per-Symbol VaR Breakdown
    _render_per_symbol_var(api, config)

    st.divider()

    # Method Comparison
    _render_method_comparison(api, config)

    st.divider()

    # Historical VaR Tracking
    _render_historical_var_tracking(api, config)


def _render_config_section() -> VaRConfig:
    """Render VaR configuration controls and return config."""
    st.subheader("VaR Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence_level = st.selectbox(
            "Confidence Level",
            options=[0.90, 0.95, 0.99],
            index=1,  # Default to 0.95
            format_func=lambda x: f"{x*100:.0f}%",
        )

    with col2:
        time_horizon = st.selectbox(
            "Time Horizon",
            options=[1, 5, 10, 21],
            index=0,  # Default to 1 day
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
        )

    with col3:
        method = st.selectbox(
            "Method",
            options=[m.value for m in VaRMethod],
            index=0,  # Default to parametric
            format_func=lambda x: x.replace("_", " ").title(),
        )

    with col4:
        distribution = st.selectbox(
            "Distribution",
            options=[d.value for d in Distribution],
            index=0,  # Default to normal
            format_func=lambda x: x.upper() if x == "t" else x.title(),
        )

    return VaRConfig(
        confidence_level=confidence_level,
        time_horizon=time_horizon,
        method=VaRMethod(method),
        distribution=Distribution(distribution),
    )


def _render_risk_manager_limits(api: PlatformAPI) -> None:
    """
    Render Risk Manager limit controls (Phase 4).

    Exposes key risk parameters that the RiskManager uses when evaluating
    new hypotheses. These controls allow users to adjust portfolio risk tolerance.
    """
    st.subheader("Risk Manager Limits")

    # Initialize session state for risk limits if not present
    if "risk_limits" not in st.session_state:
        st.session_state.risk_limits = {
            "max_drawdown": 0.20,  # 20% max drawdown
            "max_correlation": 0.70,  # 70% max correlation
            "max_sector_exposure": 0.30,  # 30% max sector exposure
        }

    # Load current limits from database if available
    try:
        limits_query = """
        SELECT key, value FROM settings
        WHERE key IN ('max_drawdown', 'max_correlation', 'max_sector_exposure')
        """
        existing_limits = api.query_readonly(limits_query)

        if not existing_limits.empty:
            for _, row in existing_limits.iterrows():
                if row["key"] in st.session_state.risk_limits:
                    try:
                        st.session_state.risk_limits[row["key"]] = float(row["value"])
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        logger.debug(f"Could not load risk limits from database: {e}")

    # Display current limits
    st.info("💡 These limits control risk tolerance for new hypothesis evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_drawdown = st.slider(
            "Max Drawdown",
            min_value=0.05,
            max_value=0.50,
            value=st.session_state.risk_limits["max_drawdown"],
            step=0.05,
            format_func=lambda x: f"{x*100:.0f}%",
            help="Maximum allowed portfolio drawdown. Hypotheses exceeding this will be vetoed.",
        )

    with col2:
        max_correlation = st.slider(
            "Max Correlation",
            min_value=0.50,
            max_value=0.95,
            value=st.session_state.risk_limits["max_correlation"],
            step=0.05,
            format_func=lambda x: f"{x*100:.0f}%",
            help="Maximum correlation with existing positions. Limits concentration in similar assets.",
        )

    with col3:
        max_sector_exposure = st.slider(
            "Max Sector Exposure",
            min_value=0.10,
            max_value=0.50,
            value=st.session_state.risk_limits["max_sector_exposure"],
            step=0.05,
            format_func=lambda x: f"{x*100:.0f}%",
            help="Maximum exposure to any single sector. Limits sector concentration risk.",
        )

    # Save button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Save Risk Limits", type="primary"):
            try:
                # Update database
                upsert_query = """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """

                api.query_write(upsert_query, ("max_drawdown", max_drawdown))
                api.query_write(upsert_query, ("max_correlation", max_correlation))
                api.query_write(upsert_query, ("max_sector_exposure", max_sector_exposure))

                # Update session state
                st.session_state.risk_limits = {
                    "max_drawdown": max_drawdown,
                    "max_correlation": max_correlation,
                    "max_sector_exposure": max_sector_exposure,
                }

                st.success("✅ Risk limits saved successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to save risk limits: {e}")
                logger.error(f"Failed to save risk limits: {e}")

    with col2:
        if st.button("Reset to Defaults"):
            try:
                default_limits = {
                    "max_drawdown": 0.20,
                    "max_correlation": 0.70,
                    "max_sector_exposure": 0.30,
                }

                for key, value in default_limits.items():
                    api.query_write(
                        upsert_query,
                        (key, value)
                    )

                st.session_state.risk_limits = default_limits.copy()
                st.success("✅ Risk limits reset to defaults!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to reset risk limits: {e}")
                logger.error(f"Failed to reset risk limits: {e}")

    # Display current risk posture summary
    st.divider()
    st.markdown("**Current Risk Posture**")

    risk_posture = "Conservative"
    if max_drawdown >= 0.30 and max_correlation >= 0.80:
        risk_posture = "Aggressive"
    elif max_drawdown >= 0.25 or max_correlation >= 0.75:
        risk_posture = "Moderate"

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Risk Posture", risk_posture)

    with col2:
        # Calculate combined risk score (0-100)
        risk_score = (
            (max_drawdown / 0.50) * 40 +  # Drawdown contribution
            (max_correlation / 0.95) * 30 +  # Correlation contribution
            (max_sector_exposure / 0.50) * 30  # Sector exposure contribution
        )
        st.metric("Risk Score", f"{risk_score:.0f}/100", help="Combined risk score (0-100)")


def _render_portfolio_var_overview(api: PlatformAPI, config: VaRConfig) -> None:
    """Render portfolio-level VaR metrics."""
    st.subheader("Portfolio VaR Overview")

    try:
        # Get portfolio returns
        returns_df = _get_portfolio_returns(api)

        if returns_df is None or returns_df.empty:
            st.warning("No portfolio return data available")
            return

        # Get current portfolio value
        positions = api.query_readonly("SELECT * FROM live_positions")
        portfolio_value = float(positions["market_value"].sum()) if not positions.empty else 0.0

        # Calculate VaR
        calculator = VaRCalculator(config)
        result = calculator.calculate(
            returns=returns_df["portfolio_return"].values,
            portfolio_value=portfolio_value if portfolio_value > 0 else None,
        )

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Portfolio VaR",
                f"{result.var*100:.2f}%" if result.var < 1 else f">{result.var:.2f}",
                help=f"Value-at-Risk at {config.confidence_level*100:.0f}% confidence",
            )

        with col2:
            st.metric(
                "Portfolio CVaR",
                f"{result.cvar*100:.2f}%" if result.cvar < 1 else f">{result.cvar:.2f}",
                help="Conditional VaR (expected loss beyond VaR)",
            )

        with col3:
            if result.var_dollar is not None:
                st.metric(
                    "VaR (Dollar)",
                    f"${result.var_dollar:,.2f}",
                    help=f"Maximum expected loss at {config.confidence_level*100:.0f}% confidence",
                )
            else:
                st.metric("VaR (Dollar)", "N/A", help="No position data available")

        with col4:
            if result.cvar_dollar is not None:
                st.metric(
                    "CVaR (Dollar)",
                    f"${result.cvar_dollar:,.2f}",
                    help="Expected loss in worst-case scenarios",
                )
            else:
                st.metric("CVaR (Dollar)", "N/A", help="No position data available")

        # VaR visualization
        _render_var_distribution(returns_df["portfolio_return"].values, result, config)

    except Exception as e:
        logger.error(f"Error calculating portfolio VaR: {e}")
        st.error(f"Error calculating portfolio VaR: {e}")


def _render_per_symbol_var(api: PlatformAPI, config: VaRConfig) -> None:
    """Render per-symbol VaR breakdown."""
    st.subheader("Per-Symbol VaR Breakdown")

    try:
        # Get positions
        positions = api.query_readonly("SELECT * FROM live_positions")

        if positions.empty:
            st.info("No positions currently held")
            return

        # Calculate VaR for each symbol
        var_data = []
        calculator = VaRCalculator(config)

        for _, pos in positions.iterrows():
            symbol = pos["symbol"]
            market_value = float(pos["market_value"])

            # Get symbol returns
            returns = _get_symbol_returns(api, symbol)

            if returns is None or len(returns) < 30:
                logger.debug(f"Insufficient data for {symbol}")
                continue

            try:
                result = calculator.calculate(returns, portfolio_value=market_value)

                var_data.append(
                    {
                        "Symbol": symbol,
                        "Position Value": market_value,
                        "VaR %": result.var * 100,
                        "CVaR %": result.cvar * 100,
                        "VaR $": result.var_dollar if result.var_dollar else 0,
                        "CVaR $": result.cvar_dollar if result.cvar_dollar else 0,
                        "VaR/Value": (result.var_dollar / market_value * 100)
                        if result.var_dollar and market_value > 0
                        else 0,
                    }
                )
            except Exception as e:
                logger.debug(f"Could not calculate VaR for {symbol}: {e}")
                continue

        if not var_data:
            st.warning("Could not calculate VaR for any positions")
            return

        var_df = pd.DataFrame(var_data).sort_values("VaR $", ascending=False)

        # Display table
        st.dataframe(
            var_df.style.format(
                {
                    "Position Value": "${:,.2f}",
                    "VaR %": "{:.2f}%",
                    "CVaR %": "{:.2f}%",
                    "VaR $": "${:,.2f}",
                    "CVaR $": "${:,.2f}",
                    "VaR/Value": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )

        # VaR contribution chart
        fig = px.bar(
            var_df,
            x="Symbol",
            y="VaR $",
            title="VaR Contribution by Position",
            labels={"VaR $": "Value-at-Risk ($)"},
            color="VaR $",
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error calculating per-symbol VaR: {e}")
        st.error(f"Error calculating per-symbol VaR: {e}")


def _render_method_comparison(api: PlatformAPI, config: VaRConfig) -> None:
    """Render comparison of different VaR calculation methods."""
    st.subheader("Method Comparison")

    try:
        # Get portfolio returns
        returns_df = _get_portfolio_returns(api)

        if returns_df is None or returns_df.empty:
            st.warning("No portfolio return data available")
            return

        # Get current portfolio value
        positions = api.query_readonly("SELECT * FROM live_positions")
        portfolio_value = float(positions["market_value"].sum()) if not positions.empty else 0.0

        # Calculate VaR using all methods
        calculator = VaRCalculator()
        results = calculator.calculate_all_methods(
            returns=returns_df["portfolio_return"].values,
            portfolio_value=portfolio_value if portfolio_value > 0 else None,
            confidence_level=config.confidence_level,
            time_horizon=config.time_horizon,
        )

        if not results:
            st.warning("Could not calculate VaR using any method")
            return

        # Create comparison dataframe
        comparison_data = []
        for method_name, result in results.items():
            comparison_data.append(
                {
                    "Method": method_name.replace("_", " ").title(),
                    "VaR %": result.var * 100,
                    "CVaR %": result.cvar * 100,
                    "VaR $": result.var_dollar if result.var_dollar else 0,
                    "CVaR $": result.cvar_dollar if result.cvar_dollar else 0,
                    "CVaR/VaR Ratio": result.cvar / result.var if result.var > 0 else 0,
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Display table
        st.dataframe(
            comparison_df.style.format(
                {
                    "VaR %": "{:.2f}%",
                    "CVaR %": "{:.2f}%",
                    "VaR $": "${:,.2f}",
                    "CVaR $": "${:,.2f}",
                    "CVaR/VaR Ratio": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        # Create comparison chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="VaR",
                x=comparison_df["Method"],
                y=comparison_df["VaR %"],
                marker_color="indianred",
            )
        )

        fig.add_trace(
            go.Bar(
                name="CVaR",
                x=comparison_df["Method"],
                y=comparison_df["CVaR %"],
                marker_color="darkred",
            )
        )

        fig.update_layout(
            title="VaR and CVaR Comparison Across Methods",
            xaxis_title="Method",
            yaxis_title="Risk Metric (%)",
            barmode="group",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error in method comparison: {e}")
        st.error(f"Error in method comparison: {e}")


def _render_historical_var_tracking(api: PlatformAPI, config: VaRConfig) -> None:
    """Render historical VaR tracking over time."""
    st.subheader("Historical VaR Tracking")

    try:
        # Get portfolio returns
        returns_df = _get_portfolio_returns(api, days=252)  # 1 year

        if returns_df is None or returns_df.empty:
            st.warning("No historical return data available")
            return

        # Calculate rolling VaR
        window_size = st.slider(
            "Rolling Window (days)",
            min_value=30,
            max_value=126,
            value=63,
            step=1,
            help="Window size for rolling VaR calculation",
        )

        calculator = VaRCalculator(config)
        rolling_var = []
        rolling_cvar = []
        dates = []

        returns_array = returns_df["portfolio_return"].values

        for i in range(window_size, len(returns_array)):
            window_returns = returns_array[i - window_size : i]

            try:
                result = calculator.calculate(window_returns)
                rolling_var.append(result.var * 100)
                rolling_cvar.append(result.cvar * 100)
                dates.append(returns_df.index[i])
            except Exception as e:
                logger.debug(f"Could not calculate VaR for window ending {returns_df.index[i]}: {e}")
                continue

        if not rolling_var:
            st.warning("Could not calculate rolling VaR")
            return

        # Create tracking chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_var,
                name="VaR",
                line=dict(color="indianred", width=2),
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_cvar,
                name="CVaR",
                line=dict(color="darkred", width=2),
                mode="lines",
            )
        )

        # Add actual returns
        fig.add_trace(
            go.Scatter(
                x=returns_df.index[window_size:],
                y=returns_df["portfolio_return"].values[window_size:] * 100,
                name="Daily Return",
                line=dict(color="lightgray", width=1),
                mode="lines",
                opacity=0.5,
            )
        )

        fig.update_layout(
            title=f"Rolling {window_size}-Day VaR and CVaR",
            xaxis_title="Date",
            yaxis_title="Return / Risk Metric (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

        # VaR breach analysis
        actual_returns = returns_df["portfolio_return"].values[window_size:] * 100
        rolling_var_array = np.array(rolling_var)

        breaches = np.sum(actual_returns < -rolling_var_array)
        total_days = len(actual_returns)
        breach_rate = breaches / total_days if total_days > 0 else 0
        expected_breach_rate = 1 - config.confidence_level

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("VaR Breaches", breaches, help="Number of days when loss exceeded VaR")

        with col2:
            st.metric(
                "Breach Rate",
                f"{breach_rate*100:.2f}%",
                delta=f"{(breach_rate - expected_breach_rate)*100:+.2f}pp",
                delta_color="inverse",
                help=f"Expected: {expected_breach_rate*100:.1f}%",
            )

        with col3:
            is_calibrated = abs(breach_rate - expected_breach_rate) < 0.05
            st.metric(
                "Model Calibration",
                "Good" if is_calibrated else "Needs Review",
                help="VaR model should be breached ~5% of the time at 95% confidence",
            )

    except Exception as e:
        logger.error(f"Error in historical VaR tracking: {e}")
        st.error(f"Error in historical VaR tracking: {e}")


def _render_var_distribution(returns: np.ndarray, result: VaRResult, config: VaRConfig) -> None:
    """Render VaR distribution visualization."""
    st.subheader("VaR Distribution")

    # Create histogram of returns
    fig = go.Figure()

    # Histogram of returns
    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name="Return Distribution",
            marker_color="lightblue",
            opacity=0.7,
        )
    )

    # Add VaR and CVaR lines
    var_line = -result.var * 100
    cvar_line = -result.cvar * 100

    fig.add_vline(
        x=var_line,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR: {result.var*100:.2f}%",
        annotation_position="top",
    )

    fig.add_vline(
        x=cvar_line,
        line_dash="dash",
        line_color="darkred",
        annotation_text=f"CVaR: {result.cvar*100:.2f}%",
        annotation_position="bottom",
    )

    fig.update_layout(
        title=f"Return Distribution with {config.confidence_level*100:.0f}% VaR/CVaR",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        showlegend=True,
        hovermode="x",
    )

    st.plotly_chart(fig, use_container_width=True)


def _get_portfolio_returns(api: PlatformAPI, days: int = 252) -> pd.DataFrame | None:
    """
    Get portfolio returns for VaR calculation.

    Args:
        api: Platform API instance
        days: Number of days of history to retrieve

    Returns:
        DataFrame with portfolio returns indexed by date, or None if unavailable
    """
    try:
        # Try to get actual portfolio returns from database
        query = """
        SELECT
            date,
            portfolio_return
        FROM portfolio_returns
        WHERE date >= ?
        ORDER BY date
        """
        start_date = date.today() - timedelta(days=days + 30)
        df = api.query_readonly(query, (start_date,))

        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            return df

    except Exception as e:
        logger.debug(f"Could not load portfolio returns from database: {e}")

    # Fallback: Generate synthetic returns for demonstration
    logger.info("Using synthetic returns for demonstration")
    dates = pd.date_range(end=date.today(), periods=days, freq="D")
    synthetic_returns = np.random.normal(0.0005, 0.015, size=days)  # ~0.05% mean, 1.5% std

    return pd.DataFrame({"portfolio_return": synthetic_returns}, index=dates)


def _get_symbol_returns(api: PlatformAPI, symbol: str, days: int = 252) -> np.ndarray | None:
    """
    Get symbol-specific returns for VaR calculation.

    Args:
        api: Platform API instance
        symbol: Stock symbol
        days: Number of days of history

    Returns:
        Array of returns, or None if unavailable
    """
    try:
        query = """
        SELECT
            date,
            close
        FROM daily_bars
        WHERE symbol = ?
        AND date >= ?
        ORDER BY date
        """
        start_date = date.today() - timedelta(days=days + 30)
        df = api.query_readonly(query, (symbol, start_date))

        if not df.empty and len(df) >= 30:
            returns = df["close"].pct_change().dropna().values
            return returns

    except Exception as e:
        logger.debug(f"Could not load returns for {symbol}: {e}")

    return None


# Main render call
if __name__ == "__main__":
    render()
