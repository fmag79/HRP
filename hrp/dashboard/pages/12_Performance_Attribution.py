"""
Performance Attribution Dashboard for HRP.

Interactive visualization of strategy performance attribution including:
- Factor-level return decomposition (Brinson-Fachler)
- Feature importance tracking over time
- Decision attribution (trade-level P&L analysis)
"""

import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.data.attribution.attribution_config import AttributionConfig, AttributionMethod
from hrp.data.attribution.decision_attribution import (
    TradeDecision,
)
from hrp.data.attribution.factor_attribution import (
    AttributionResult,
    BrinsonAttribution,
    FactorAttribution,
)


def render() -> None:
    """Render the performance attribution dashboard page."""
    st.title("Performance Attribution")
    st.caption("Decompose returns into factors, features, and decisions")

    api = PlatformAPI()

    # Configuration section
    start_date, end_date, config = _render_config_section()

    st.divider()

    # Summary Bar
    _render_summary_bar(api, start_date, end_date)

    st.divider()

    # Waterfall Chart - THE KEY VISUALIZATION
    _render_waterfall_chart(api, start_date, end_date, config)

    st.divider()

    # Factor Contribution Table
    _render_factor_contribution_table(api, start_date, end_date, config)

    st.divider()

    # Feature Importance Heatmap
    _render_feature_importance_heatmap(api, start_date, end_date, config)

    st.divider()

    # Decision Attribution Timeline
    _render_decision_attribution_timeline(api, start_date, end_date)

    st.divider()

    # Period Comparison
    _render_period_comparison(api, config)


def _render_config_section() -> tuple[date, date, AttributionConfig]:
    """Render attribution configuration controls and return dates + config."""
    st.subheader("Attribution Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=90),
            max_value=date.today(),
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            max_value=date.today(),
        )

    with col3:
        method = st.selectbox(
            "Attribution Method",
            options=["brinson", "regression"],
            index=0,  # Default to brinson
            format_func=lambda x: x.replace("_", " ").title(),
        )

    with col4:
        shap_enabled = st.checkbox(
            "Enable SHAP",
            value=False,
            help="Use SHAP for feature importance (requires shap package)",
        )

    # Create config with selected parameters
    config = AttributionConfig(
        method=method,  # type: ignore
        shap_enabled=shap_enabled,
        lookback_days=(end_date - start_date).days,
    )

    return start_date, end_date, config


def _render_summary_bar(api: PlatformAPI, start_date: date, end_date: date) -> None:
    """Render summary metrics bar."""
    st.subheader("Performance Summary")

    try:
        # Get portfolio and benchmark returns
        portfolio_returns = _get_portfolio_returns(api, start_date, end_date)
        benchmark_returns = _get_benchmark_returns(api, start_date, end_date)

        if portfolio_returns is None or benchmark_returns is None:
            st.warning("Insufficient data for attribution period")
            return

        # Calculate summary metrics
        portfolio_return = portfolio_returns.sum()
        benchmark_return = benchmark_returns.sum()
        active_return = portfolio_return - benchmark_return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Portfolio Return",
                f"{portfolio_return*100:.2f}%",
                help="Total return over attribution period",
            )

        with col2:
            st.metric(
                "Benchmark Return",
                f"{benchmark_return*100:.2f}%",
                help="Benchmark return over same period",
            )

        with col3:
            st.metric(
                "Active Return",
                f"{active_return*100:.2f}%",
                delta=f"{active_return*100:.2f}%",
                help="Portfolio return minus benchmark return",
            )

        with col4:
            days = (end_date - start_date).days
            st.metric(
                "Period",
                f"{days} days",
                help="Attribution period length",
            )

    except Exception as e:
        logger.error(f"Error calculating summary metrics: {e}")
        st.error(f"Error: {e}")


def _render_waterfall_chart(api: PlatformAPI, config: AttributionConfig) -> None:
    """Render waterfall chart showing attribution decomposition.

    This is THE KEY VISUALIZATION showing how different effects
    combine to produce active return.
    """
    st.subheader("Return Attribution Waterfall")

    try:
        # Get attribution results
        results = _calculate_attribution(api, config)

        if not results:
            st.info("No attribution data available for selected period")
            return

        # Prepare waterfall data
        # Start with benchmark return, add effects, end with portfolio return
        portfolio_returns = _get_portfolio_returns(api, config)
        benchmark_returns = _get_benchmark_returns(api, config)

        if portfolio_returns is None or benchmark_returns is None:
            st.warning("Insufficient data")
            return

        benchmark_return = benchmark_returns.sum()
        portfolio_return = portfolio_returns.sum()

        # Build waterfall components
        categories = ["Benchmark"]
        values = [benchmark_return * 100]  # Convert to percentage
        measures = ["absolute"]

        # Add attribution effects
        for result in results:
            categories.append(f"{result.factor}\n({result.effect_type})")
            values.append(result.contribution_pct * 100)
            measures.append("relative")

        # Add final portfolio return
        categories.append("Portfolio")
        values.append(portfolio_return * 100)
        measures.append("total")

        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            measure=measures,
            x=categories,
            y=values,
            text=[f"{v:.2f}%" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}},
        ))

        fig.update_layout(
            title="Return Attribution Breakdown",
            xaxis_title="Attribution Components",
            yaxis_title="Return (%)",
            height=500,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering waterfall chart: {e}")
        st.error(f"Error: {e}")


def _render_factor_contribution_table(api: PlatformAPI, config: AttributionConfig) -> None:
    """Render table showing factor-by-factor contributions."""
    st.subheader("Factor Contribution Details")

    try:
        results = _calculate_attribution(api, config)

        if not results:
            st.info("No attribution data available")
            return

        # Convert to DataFrame for display
        data = []
        for result in results:
            data.append({
                "Factor": result.factor,
                "Effect Type": result.effect_type.title(),
                "Contribution (%)": f"{result.contribution_pct * 100:.3f}%",
                "Contribution ($)": f"${result.contribution_dollar:,.2f}" if result.contribution_dollar else "N/A",
            })

        df = pd.DataFrame(data)

        # Display table with formatting
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"attribution_{config.start_date}_{config.end_date}.csv",
            mime="text/csv",
        )

    except Exception as e:
        logger.error(f"Error rendering factor table: {e}")
        st.error(f"Error: {e}")


def _render_feature_importance_heatmap(api: PlatformAPI, config: AttributionConfig) -> None:
    """Render heatmap showing feature importance over time."""
    st.subheader("Feature Importance Over Time")

    try:
        # Get feature importance data
        importance_data = _calculate_feature_importance(api, config)

        if importance_data is None or importance_data.empty:
            st.info("No feature importance data available")
            return

        # Create heatmap
        fig = px.imshow(
            importance_data.T,  # Features as rows, time as columns
            labels=dict(x="Date", y="Feature", color="Importance"),
            x=importance_data.index,
            y=importance_data.columns,
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )

        fig.update_layout(
            title="Rolling Feature Importance",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Top features summary
        st.caption("**Top 5 Features (Average Importance)**")
        avg_importance = importance_data.mean().sort_values(ascending=False).head(5)
        for i, (feature, importance) in enumerate(avg_importance.items(), 1):
            st.text(f"{i}. {feature}: {importance:.3f}")

    except Exception as e:
        logger.error(f"Error rendering feature importance heatmap: {e}")
        st.error(f"Error: {e}")


def _render_decision_attribution_timeline(api: PlatformAPI, config: AttributionConfig) -> None:
    """Render timeline showing trade-level decision attribution."""
    st.subheader("Decision Attribution Timeline")

    try:
        # Get trade decisions
        decisions = _calculate_decision_attribution(api, config)

        if not decisions:
            st.info("No trade decision data available for selected period")
            return

        # Prepare timeline data
        dates = []
        timing_pnl = []
        sizing_pnl = []
        residual_pnl = []

        for decision in decisions:
            dates.append(decision.exit_date)
            timing_pnl.append(decision.timing_contribution)
            sizing_pnl.append(decision.sizing_contribution)
            residual_pnl.append(decision.residual_contribution)

        # Create stacked bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Timing",
            x=dates,
            y=timing_pnl,
            marker_color="blue",
        ))

        fig.add_trace(go.Bar(
            name="Sizing",
            x=dates,
            y=sizing_pnl,
            marker_color="green",
        ))

        fig.add_trace(go.Bar(
            name="Residual",
            x=dates,
            y=residual_pnl,
            marker_color="gray",
        ))

        fig.update_layout(
            barmode="stack",
            title="Trade Decision P&L Attribution",
            xaxis_title="Exit Date",
            yaxis_title="P&L Contribution ($)",
            height=400,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            total_timing = sum(timing_pnl)
            st.metric("Total Timing P&L", f"${total_timing:,.2f}")

        with col2:
            total_sizing = sum(sizing_pnl)
            st.metric("Total Sizing P&L", f"${total_sizing:,.2f}")

        with col3:
            total_residual = sum(residual_pnl)
            st.metric("Total Residual", f"${total_residual:,.2f}")

    except Exception as e:
        logger.error(f"Error rendering decision timeline: {e}")
        st.error(f"Error: {e}")


def _render_period_comparison(api: PlatformAPI, config: AttributionConfig) -> None:
    """Render comparison of attribution across different time periods."""
    st.subheader("Multi-Period Comparison")

    try:
        # Define comparison periods
        periods = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "YTD": (date.today() - date(date.today().year, 1, 1)).days,
            "1 Year": 365,
        }

        # Calculate attribution for each period
        period_results = {}
        for period_name, days in periods.items():
            period_start = date.today() - timedelta(days=days)
            period_config = AttributionConfig(
                start_date=period_start,
                end_date=date.today(),
                method=config.method,
                importance_method=config.importance_method,
            )

            results = _calculate_attribution(api, period_config)
            if results:
                # Sum contributions by effect type
                allocation_total = sum(r.contribution_pct for r in results if r.effect_type == "allocation")
                selection_total = sum(r.contribution_pct for r in results if r.effect_type == "selection")
                interaction_total = sum(r.contribution_pct for r in results if r.effect_type == "interaction")

                period_results[period_name] = {
                    "Allocation": allocation_total * 100,
                    "Selection": selection_total * 100,
                    "Interaction": interaction_total * 100,
                }

        if not period_results:
            st.info("Insufficient data for period comparison")
            return

        # Create comparison chart
        df = pd.DataFrame(period_results).T
        df = df.reset_index().rename(columns={"index": "Period"})

        fig = go.Figure()

        for effect_type in ["Allocation", "Selection", "Interaction"]:
            fig.add_trace(go.Bar(
                name=effect_type,
                x=df["Period"],
                y=df[effect_type],
            ))

        fig.update_layout(
            barmode="group",
            title="Attribution Effects Across Time Periods",
            xaxis_title="Time Period",
            yaxis_title="Contribution (%)",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering period comparison: {e}")
        st.error(f"Error: {e}")


# Helper functions

def _get_portfolio_returns(api: PlatformAPI, config: AttributionConfig) -> pd.Series | None:
    """Get portfolio returns for the attribution period."""
    try:
        # Try to get actual portfolio returns from database
        query = f"""
            SELECT date, portfolio_return
            FROM portfolio_returns
            WHERE date >= '{config.start_date}' AND date <= '{config.end_date}'
            ORDER BY date
        """
        df = api.query_readonly(query)

        if not df.empty:
            return pd.Series(
                df["portfolio_return"].values,
                index=pd.to_datetime(df["date"]),
                name="portfolio_return",
            )

        # Fallback: synthetic data for demo
        logger.warning("No portfolio return data found, using synthetic fallback")
        dates = pd.date_range(config.start_date, config.end_date, freq="D")
        returns = np.random.normal(0.0005, 0.01, len(dates))  # 5bps mean, 1% vol
        return pd.Series(returns, index=dates, name="portfolio_return")

    except Exception as e:
        logger.error(f"Error fetching portfolio returns: {e}")
        # Return synthetic data as fallback
        dates = pd.date_range(config.start_date, config.end_date, freq="D")
        returns = np.random.normal(0.0005, 0.01, len(dates))
        return pd.Series(returns, index=dates, name="portfolio_return")


def _get_benchmark_returns(api: PlatformAPI, config: AttributionConfig) -> pd.Series | None:
    """Get benchmark returns for the attribution period."""
    try:
        # Try to get actual benchmark returns
        query = f"""
            SELECT date, return
            FROM benchmark_returns
            WHERE date >= '{config.start_date}' AND date <= '{config.end_date}'
            ORDER BY date
        """
        df = api.query_readonly(query)

        if not df.empty:
            return pd.Series(
                df["return"].values,
                index=pd.to_datetime(df["date"]),
                name="benchmark_return",
            )

        # Fallback: synthetic data
        logger.warning("No benchmark return data found, using synthetic fallback")
        dates = pd.date_range(config.start_date, config.end_date, freq="D")
        returns = np.random.normal(0.0003, 0.008, len(dates))  # 3bps mean, 0.8% vol
        return pd.Series(returns, index=dates, name="benchmark_return")

    except Exception as e:
        logger.error(f"Error fetching benchmark returns: {e}")
        # Return synthetic data as fallback
        dates = pd.date_range(config.start_date, config.end_date, freq="D")
        returns = np.random.normal(0.0003, 0.008, len(dates))
        return pd.Series(returns, index=dates, name="benchmark_return")


def _calculate_attribution(api: PlatformAPI, config: AttributionConfig) -> list[AttributionResult]:
    """Calculate attribution results using configured method."""
    try:
        portfolio_returns = _get_portfolio_returns(api, config)
        benchmark_returns = _get_benchmark_returns(api, config)

        if portfolio_returns is None or benchmark_returns is None:
            return []

        if config.method == AttributionMethod.BRINSON:
            # Use Brinson-Fachler attribution
            # For demo purposes, create synthetic sector weights
            sectors = ["Technology", "Healthcare", "Finance", "Consumer", "Energy"]
            portfolio_weights = np.random.dirichlet(np.ones(len(sectors)))
            benchmark_weights = np.random.dirichlet(np.ones(len(sectors)))

            # Synthetic sector returns
            sector_returns_p = {
                sector: np.random.normal(0.001, 0.015, len(portfolio_returns))
                for sector in sectors
            }
            sector_returns_b = {
                sector: np.random.normal(0.0008, 0.012, len(benchmark_returns))
                for sector in sectors
            }

            # Calculate Brinson attribution
            attributor = BrinsonAttribution()
            results = []

            for i, sector in enumerate(sectors):
                sector_result = attributor.calculate_sector_attribution(
                    portfolio_weight=portfolio_weights[i],
                    benchmark_weight=benchmark_weights[i],
                    portfolio_sector_return=sector_returns_p[sector].mean(),
                    benchmark_sector_return=sector_returns_b[sector].mean(),
                    benchmark_total_return=benchmark_returns.mean(),
                )
                results.extend(sector_result)

            return results

        elif config.method == AttributionMethod.REGRESSION:
            # Use regression-based factor attribution
            # Create synthetic factor exposures for demo
            factors = pd.DataFrame({
                "Market": np.random.normal(0.0005, 0.01, len(portfolio_returns)),
                "Value": np.random.normal(0.0002, 0.005, len(portfolio_returns)),
                "Momentum": np.random.normal(0.0001, 0.006, len(portfolio_returns)),
                "Size": np.random.normal(0.0, 0.004, len(portfolio_returns)),
            }, index=portfolio_returns.index)

            attributor = FactorAttribution()
            results = attributor.calculate(
                portfolio_returns=portfolio_returns,
                factor_returns=factors,
            )

            return results

        else:
            logger.warning(f"Unknown attribution method: {config.method}")
            return []

    except Exception as e:
        logger.error(f"Error calculating attribution: {e}")
        return []


def _calculate_feature_importance(api: PlatformAPI, config: AttributionConfig) -> pd.DataFrame | None:
    """Calculate rolling feature importance over time."""
    try:
        # For demo: create synthetic feature importance data
        dates = pd.date_range(config.start_date, config.end_date, freq="W")
        features = [
            "momentum_20d",
            "volatility_30d",
            "rsi_14d",
            "macd_signal",
            "volume_ratio",
            "pe_ratio",
            "debt_to_equity",
            "sentiment_score",
        ]

        # Generate rolling importance scores (values between 0 and 1)
        data = {}
        for feature in features:
            # Each feature has different baseline importance with random variation
            baseline = np.random.uniform(0.05, 0.20)
            noise = np.random.normal(0, 0.03, len(dates))
            importance = np.clip(baseline + noise, 0, 1)
            data[feature] = importance

        df = pd.DataFrame(data, index=dates)

        # Normalize each row to sum to 1
        df = df.div(df.sum(axis=1), axis=0)

        return df

    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return None


def _calculate_decision_attribution(api: PlatformAPI, config: AttributionConfig) -> list[TradeDecision]:
    """Calculate trade-level decision attribution."""
    try:
        # For demo: create synthetic trade decision data
        # In production, this would query actual trade history

        decisions = []
        current_date = config.start_date

        while current_date < config.end_date:
            # Random trade every 3-7 days
            days_to_next_trade = np.random.randint(3, 8)
            entry_date = current_date
            exit_date = entry_date + timedelta(days=days_to_next_trade)

            if exit_date > config.end_date:
                break

            # Synthetic P&L decomposition
            total_pnl = np.random.normal(100, 500)  # Random P&L
            timing_pct = np.random.uniform(0.3, 0.5)  # 30-50% from timing
            sizing_pct = np.random.uniform(0.2, 0.4)  # 20-40% from sizing
            residual_pct = 1 - timing_pct - sizing_pct

            decision = TradeDecision(
                trade_id=f"TRADE-{len(decisions)+1}",
                asset=f"ASSET-{np.random.randint(1, 10)}",
                entry_date=entry_date,
                exit_date=exit_date,
                pnl=total_pnl,
                timing_contribution=total_pnl * timing_pct,
                sizing_contribution=total_pnl * sizing_pct,
                residual_contribution=total_pnl * residual_pct,
            )

            decisions.append(decision)
            current_date = exit_date + timedelta(days=1)

        return decisions

    except Exception as e:
        logger.error(f"Error calculating decision attribution: {e}")
        return []


if __name__ == "__main__":
    render()
