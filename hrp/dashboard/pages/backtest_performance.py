"""
Backtest Performance Dashboard page for HRP.

Interactive visualization of backtest results including equity curves,
drawdowns, strategy comparisons, and export functionality.
"""

import sys
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from loguru import logger
import xlsxwriter

from hrp.research.mlflow_utils import setup_mlflow, get_best_runs, get_or_create_experiment
from hrp.dashboard.components.tearsheet_viz import (
    render_drawdown_analysis,
    render_rolling_metrics,
    render_monthly_returns_heatmap,
    render_tail_risk_metrics,
)

# MLflow configuration
MLFLOW_DIR = Path.home() / "hrp-data" / "mlflow"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR}/mlflow.db"
MLFLOW_UI_URL = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "backtests"


@st.cache_data(ttl=300)
def get_experiment_runs(
    experiment_name: str = DEFAULT_EXPERIMENT,
    max_results: int = 200,
) -> pd.DataFrame:
    """
    Get recent experiment runs from MLflow.

    Args:
        experiment_name: Name of the MLflow experiment
        max_results: Maximum number of runs to retrieve

    Returns:
        DataFrame with run information and metrics
    """
    setup_mlflow()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    return runs


@st.cache_data(ttl=300)
def get_run_artifact(run_id: str, artifact_path: str) -> pd.DataFrame | None:
    """
    Load a DataFrame artifact from MLflow.

    Args:
        run_id: MLflow run ID
        artifact_path: Path to the artifact (e.g., "equity_curve.csv")

    Returns:
        DataFrame or None if not found
    """
    setup_mlflow()

    try:
        # Download artifact to temp location
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path
        )

        # Load based on file extension
        if artifact_path.endswith('.csv'):
            return pd.read_csv(local_path, index_col=0, parse_dates=True)
        elif artifact_path.endswith('.parquet'):
            return pd.read_parquet(local_path)
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not load artifact {artifact_path} from run {run_id}: {e}")
        return None


@st.cache_data(ttl=300)
def get_run_metrics_dataframe(run_id: str) -> dict[str, Any]:
    """
    Get metrics and artifacts for a specific run.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary with metrics, equity_curve, trades, etc.
    """
    setup_mlflow()

    try:
        run = mlflow.get_run(run_id)

        data = {
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
            "run_name": run.info.run_name,
            "start_time": run.info.start_time,
        }

        # Try to load equity curve
        equity_curve = get_run_artifact(run_id, "equity_curve.csv")
        if equity_curve is not None:
            data["equity_curve"] = equity_curve

        # Try to load trades
        trades = get_run_artifact(run_id, "trades.csv")
        if trades is not None:
            data["trades"] = trades

        # Try to load returns
        returns = get_run_artifact(run_id, "returns.csv")
        if returns is not None:
            if isinstance(returns, pd.DataFrame):
                data["returns"] = returns.iloc[:, 0]  # First column as series
            else:
                data["returns"] = returns

        # Try to load benchmark returns
        benchmark_returns = get_run_artifact(run_id, "benchmark_returns.csv")
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.DataFrame):
                data["benchmark_returns"] = benchmark_returns.iloc[:, 0]
            else:
                data["benchmark_returns"] = benchmark_returns

        return data

    except Exception as e:
        logger.warning(f"Could not get run data for {run_id}: {e}")
        return {}


def format_timestamp(ts: int | None) -> str:
    """Format Unix timestamp to readable string."""
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")


def format_metric(value: float | None, metric_name: str) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"

    pct_metrics = {
        "total_return", "cagr", "volatility", "max_drawdown",
        "downside_volatility", "alpha", "tracking_error"
    }
    ratio_metrics = {
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "information_ratio", "beta", "profit_factor"
    }

    if metric_name in pct_metrics:
        return f"{value * 100:.2f}%"
    elif metric_name in ratio_metrics:
        return f"{value:.2f}"
    elif metric_name == "win_rate":
        return f"{value * 100:.1f}%"
    else:
        return f"{value:.4f}"


def render_equity_curve(
    equity_curves: dict[str, pd.Series],
    title: str = "Equity Curves"
) -> None:
    """
    Render interactive equity curve chart with multiple strategies.

    Args:
        equity_curves: Dictionary mapping run names to equity curve Series
        title: Chart title
    """
    fig = go.Figure()

    # Normalize to starting value of 100 for comparison
    for name, curve in equity_curves.items():
        if curve is None or curve.empty:
            continue
        normalized = (curve / curve.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            mode="lines",
            name=name,
            line=dict(width=2),
            hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized Value (Start = 100)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
        ),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_strategy_comparison(runs_data: dict[str, dict[str, Any]]) -> None:
    """
    Render side-by-side comparison of selected strategies.

    Args:
        runs_data: Dictionary mapping run IDs to run data dictionaries
    """
    if not runs_data:
        st.info("No strategies selected for comparison")
        return

    st.markdown("### Strategy Comparison")

    # Extract key metrics for comparison
    comparison_data = []
    for run_id, data in runs_data.items():
        metrics = data.get("metrics", {})
        comparison_data.append({
            "Strategy": data.get("run_name", run_id[:8]),
            "Total Return": format_metric(metrics.get("total_return"), "total_return"),
            "CAGR": format_metric(metrics.get("cagr"), "cagr"),
            "Sharpe Ratio": format_metric(metrics.get("sharpe_ratio"), "sharpe_ratio"),
            "Max Drawdown": format_metric(metrics.get("max_drawdown"), "max_drawdown"),
            "Volatility": format_metric(metrics.get("volatility"), "volatility"),
            "Win Rate": format_metric(metrics.get("win_rate"), "win_rate"),
            "Sortino Ratio": format_metric(metrics.get("sortino_ratio"), "sortino_ratio"),
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index("Strategy")

    # Display as a styled table
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=False,
    )

    # Render equity curves together
    equity_curves = {
        data.get("run_name", run_id[:8]): data.get("equity_curve")
        for run_id, data in runs_data.items()
    }

    render_equity_curve(equity_curves, title="Equity Curves Comparison")

    # Render drawdowns together
    with st.expander("Drawdown Comparison", expanded=False):
        fig = go.Figure()

        for run_id, data in runs_data.items():
            equity_curve = data.get("equity_curve")
            if equity_curve is None or equity_curve.empty:
                continue

            # Calculate drawdown
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max * 100

            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=data.get("run_name", run_id[:8]),
                fill="tozeroy",
                opacity=0.3,
            ))

        fig.update_layout(
            title="Drawdown Comparison",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis_tickformat=".1f",
            hovermode="x unified",
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True)


def export_to_csv(runs_data: dict[str, dict[str, Any]]) -> BytesIO:
    """
    Export run data to CSV format.

    Args:
        runs_data: Dictionary mapping run IDs to run data

    Returns:
        BytesIO object with CSV data
    """
    output = BytesIO()

    # Create summary CSV
    summary_data = []
    for run_id, data in runs_data.items():
        metrics = data.get("metrics", {})
        summary_data.append({
            "run_id": run_id,
            "run_name": data.get("run_name", ""),
            "total_return": metrics.get("total_return"),
            "cagr": metrics.get("cagr"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "volatility": metrics.get("volatility"),
            "win_rate": metrics.get("win_rate"),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output, index=False)
    output.seek(0)
    return output


def export_to_excel(runs_data: dict[str, dict[str, Any]]) -> BytesIO:
    """
    Export run data to Excel format with multiple sheets.

    Args:
        runs_data: Dictionary mapping run IDs to run data

    Returns:
        BytesIO object with Excel data
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_data = []
        for run_id, data in runs_data.items():
            metrics = data.get("metrics", {})
            summary_data.append({
                "run_id": run_id,
                "run_name": data.get("run_name", ""),
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "volatility": metrics.get("volatility"),
                "win_rate": metrics.get("win_rate"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "calmar_ratio": metrics.get("calmar_ratio"),
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Individual sheets for each run
        for run_id, data in runs_data.items():
            run_name = data.get("run_name", run_id[:8])
            safe_name = "".join(c if c.isalnum() else "_" for c in run_name)[:31]  # Excel sheet name limit

            # Equity curve
            equity_curve = data.get("equity_curve")
            if equity_curve is not None and not equity_curve.empty:
                equity_curve.to_excel(writer, sheet_name=f"{safe_name}_Equity")

            # Trades
            trades = data.get("trades")
            if trades is not None and not trades.empty:
                trades.to_excel(writer, sheet_name=f"{safe_name}_Trades", index=False)

    output.seek(0)
    return output


def render_single_run_view(run_data: dict[str, Any]) -> None:
    """
    Render detailed view of a single backtest run.

    Args:
        run_data: Dictionary with run metrics, equity_curve, trades, etc.
    """
    metrics = run_data.get("metrics", {})
    run_name = run_data.get("run_name", "Unknown")

    st.markdown(f"### 📊 {run_name}")

    # Key metrics cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(
        "Total Return",
        format_metric(metrics.get("total_return"), "total_return"),
        delta=None
    )
    col2.metric(
        "Sharpe Ratio",
        format_metric(metrics.get("sharpe_ratio"), "sharpe_ratio"),
        delta=None
    )
    col3.metric(
        "Max Drawdown",
        format_metric(metrics.get("max_drawdown"), "max_drawdown"),
        delta=None
    )
    col4.metric(
        "Volatility",
        format_metric(metrics.get("volatility"), "volatility"),
        delta=None
    )
    col5.metric(
        "CAGR",
        format_metric(metrics.get("cagr"), "cagr"),
        delta=None
    )
    col6.metric(
        "Win Rate",
        format_metric(metrics.get("win_rate"), "win_rate"),
        delta=None
    )

    st.markdown("---")

    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Equity Curve",
        "📉 Drawdown",
        "📊 Metrics Detail",
        "📋 Trades",
        "🔄 Rolling Analysis"
    ])

    with tab1:
        # Equity curve
        equity_curve = run_data.get("equity_curve")
        if equity_curve is not None and not equity_curve.empty:
            # Also show benchmark if available
            equity_curves = {run_name: equity_curve}

            # Try to get benchmark from metrics or separate file
            benchmark_returns = run_data.get("benchmark_returns")
            if benchmark_returns is not None and not benchmark_returns.empty:
                benchmark_equity = (1 + benchmark_returns).cumprod() * 100
                equity_curves["Benchmark"] = benchmark_equity

            render_equity_curve(equity_curves, title="Equity Curve")

            # Cumulative returns by year
            st.markdown("#### Annual Returns")
            annual_returns = equity_curve.resample("YE").last().pct_change().dropna() * 100
            annual_returns_df = annual_returns.to_frame(name="Return (%)")
            annual_returns_df.index = annual_returns_df.index.year
            st.dataframe(annual_returns_df, use_container_width=True)
        else:
            st.warning("No equity curve data available")

    with tab2:
        returns = run_data.get("returns")
        if returns is not None and not returns.empty:
            render_drawdown_analysis(returns)
        else:
            st.warning("No returns data available for drawdown analysis")

    with tab3:
        st.markdown("#### All Metrics")

        # All metrics in a table
        all_metrics = pd.DataFrame([
            {"Metric": k.replace("_", " ").title(), "Value": v}
            for k, v in metrics.items()
        ])
        st.dataframe(all_metrics, use_container_width=True, hide_index=True)

        # Monthly returns heatmap
        if returns is not None and not returns.empty:
            render_monthly_returns_heatmap(returns)

        # Tail risk
        render_tail_risk_metrics(returns)

    with tab4:
        trades = run_data.get("trades")
        if trades is not None and not trades.empty:
            st.markdown(f"#### Trade History ({len(trades)} trades)")

            # Summary stats
            col1, col2, col3 = st.columns(3)
            if "pnl" in trades.columns:
                total_pnl = trades["pnl"].sum()
                winning_trades = (trades["pnl"] > 0).sum()
                win_rate = winning_trades / len(trades) * 100

                col1.metric("Total P&L", f"${total_pnl:,.2f}")
                col2.metric("Winning Trades", f"{winning_trades}")
                col3.metric("Win Rate", f"{win_rate:.1f}%")

            st.dataframe(trades, use_container_width=True)
        else:
            st.info("No trades data available")

    with tab5:
        if returns is not None and not returns.empty:
            col1, col2 = st.columns([3, 1])
            with col2:
                window = st.selectbox(
                    "Rolling Window",
                    options=[21, 63, 126, 252],
                    format_func=lambda x: f"{x} days (~{x // 21} months)",
                    index=1,
                    key="single_rolling_window"
                )
            render_rolling_metrics(returns, window=window)
        else:
            st.warning("No returns data available for rolling analysis")


def main():
    """Main function for the backtest performance page."""
    st.set_page_config(
        page_title="Backtest Performance",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Backtest Performance Dashboard")

    # Setup MLflow
    setup_mlflow()

    # Get all runs
    runs_df = get_experiment_runs(max_results=200)

    if runs_df.empty:
        st.info("No backtest runs found. Run a backtest first to see results here.")
        return

    # Sidebar filters
    st.sidebar.markdown("### Filters")

    # Date range filter
    if "start_time" in runs_df.columns:
        runs_df["date"] = pd.to_datetime(runs_df["start_time"], unit="ms")
        min_date = runs_df["date"].min().date()
        max_date = runs_df["date"].max().date()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range_filter"
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            runs_df = runs_df[
                (runs_df["date"].dt.date >= start_date) &
                (runs_df["date"].dt.date <= end_date)
            ]

    # Status filter
    if "status" in runs_df.columns:
        status_filter = st.sidebar.multiselect(
            "Status",
            options=runs_df["status"].unique().tolist(),
            default=["COMPLETED", "FINISHED"],
            key="status_filter"
        )
        if status_filter:
            runs_df = runs_df[runs_df["status"].isin(status_filter)]

    # Search by name
    search_term = st.sidebar.text_input(
        "Search by name...",
        key="search_backtest"
    )

    if search_term:
        # Search in run name or tags
        mask = runs_df["tags.mlflow.runName"].str.contains(
            search_term, case=False, na=False
        )
        runs_df = runs_df[mask]

    if runs_df.empty:
        st.warning("No runs match the selected filters.")
        return

    # Display runs list
    st.markdown("### Backtest Runs")

    # Create selectbox for run selection
    run_options = [
        f"{row['tags.mlflow.runName'] or row['run_id'][:8]} - {format_metric(row.get('metrics.sharpe_ratio'), 'sharpe_ratio')}"
        for idx, row in runs_df.iterrows()
    ]
    run_options.insert(0, "Select a run...")

    selected_run_idx = st.selectbox(
        "Select a backtest run to view details:",
        options=range(len(run_options)),
        format_func=lambda i: run_options[i],
        key="run_selector"
    )

    # Compare mode
    st.markdown("---")
    st.markdown("### Compare Strategies")

    compare_mode = st.checkbox("Enable comparison mode", key="compare_mode")

    if compare_mode:
        # Allow selecting multiple runs for comparison
        compare_runs = st.multiselect(
            "Select runs to compare:",
            options=runs_df["run_id"].tolist(),
            format_func=lambda rid: runs_df[runs_df["run_id"] == rid]["tags.mlflow.runName"].iloc[0] or rid[:8],
            key="compare_runs_selector"
        )

        if len(compare_runs) > 1:
            # Load data for comparison
            runs_data = {}
            for run_id in compare_runs:
                data = get_run_metrics_dataframe(run_id)
                if data:
                    runs_data[run_id] = data

            if runs_data:
                render_strategy_comparison(runs_data)

                # Export options
                st.markdown("---")
                st.markdown("### Export Data")

                col1, col2 = st.columns(2)

                with col1:
                    csv_data = export_to_csv(runs_data)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"backtest_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    excel_data = export_to_excel(runs_data)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"backtest_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("Could not load data for selected runs")
        elif len(compare_runs) == 1:
            st.info("Select at least 2 runs to compare")

    # Single run view
    if selected_run_idx > 0:
        st.markdown("---")

        # Get the actual run index (adjusting for the "Select a run..." placeholder)
        run_idx = selected_run_idx - 1
        run_id = runs_df.iloc[run_idx]["run_id"]

        # Load run data
        run_data = get_run_metrics_dataframe(run_id)

        if run_data:
            render_single_run_view(run_data)

            # Export single run
            st.markdown("---")
            st.markdown("### Export This Run")

            col1, col2 = st.columns(2)

            single_runs_data = {run_id: run_data}

            with col1:
                csv_data = export_to_csv(single_runs_data)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"backtest_{run_data.get('run_name', run_id[:8])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                excel_data = export_to_excel(single_runs_data)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"backtest_{run_data.get('run_name', run_id[:8])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning(f"Could not load data for run {run_id}")

    # Link to MLflow UI
    st.markdown("---")
    st.markdown(f"### 🔗 MLflow Tracking Server")
    st.markdown(
        f"View detailed logs and artifacts in the [MLflow UI]({MLFLOW_UI_URL})"
    )


if __name__ == "__main__":
    main()
