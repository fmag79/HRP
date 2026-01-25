"""
Experiments page for HRP Dashboard.

Browse, compare, and run backtests with MLflow integration.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to Python path so Streamlit can find the hrp module
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.research.backtest import get_price_data, generate_momentum_signals, run_backtest
from hrp.research.config import BacktestConfig, CostModel
from hrp.research.mlflow_utils import setup_mlflow, get_or_create_experiment, get_best_runs
from hrp.research.strategies import (
    generate_multifactor_signals,
    generate_ml_predicted_signals,
    STRATEGY_REGISTRY,
)
from hrp.dashboard.components.strategy_config import (
    render_multifactor_config,
    render_ml_predicted_config,
)


# MLflow configuration
MLFLOW_DIR = Path.home() / "hrp-data" / "mlflow"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR}/mlflow.db"
# Using 127.0.0.1 instead of localhost because MLflow blocks localhost
MLFLOW_UI_URL = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "backtests"


@st.cache_data(ttl=300)
def get_experiment_runs(
    experiment_name: str = DEFAULT_EXPERIMENT,
    max_results: int = 100,
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
def get_run_details(run_id: str) -> dict[str, Any] | None:
    """
    Get detailed information for a specific run.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary with run details or None if not found
    """
    setup_mlflow()

    try:
        run = mlflow.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "artifact_uri": run.info.artifact_uri,
        }
    except Exception as e:
        logger.warning(f"Could not fetch run {run_id}: {e}")
        return None


@st.cache_data(ttl=300)
def get_available_symbols() -> list[str]:
    """Get list of available symbols from database."""
    try:
        api = PlatformAPI()
        result = api._db.fetchall(
            "SELECT DISTINCT symbol FROM prices ORDER BY symbol"
        )
        return [r[0] for r in result]
    except Exception as e:
        logger.warning(f"Could not fetch symbols: {e}")
        return []


@st.cache_data(ttl=300)
def get_hypotheses_for_filter() -> list[dict[str, str]]:
    """Get hypotheses for filtering dropdown."""
    try:
        api = PlatformAPI()
        hypotheses = api.list_hypotheses(limit=50)
        return [{"id": h["hypothesis_id"], "title": h["title"]} for h in hypotheses]
    except Exception as e:
        logger.warning(f"Could not fetch hypotheses: {e}")
        return []


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


def render_browse_tab() -> None:
    """Render the Browse Experiments tab."""
    st.subheader("Recent Experiments")

    # Filters
    col1, col2 = st.columns([2, 2])

    with col1:
        hypotheses = get_hypotheses_for_filter()
        hypothesis_options = ["All"] + [f"{h['id']}: {h['title'][:40]}" for h in hypotheses]
        selected_hypothesis = st.selectbox(
            "Filter by Hypothesis",
            hypothesis_options,
            key="browse_hypothesis_filter"
        )

    with col2:
        max_results = st.slider(
            "Max Results",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="browse_max_results"
        )

    # Fetch runs
    runs_df = get_experiment_runs(max_results=max_results)

    if runs_df.empty:
        st.info("No experiments found. Run a backtest to see results here.")
        return

    # Filter by hypothesis if selected
    if selected_hypothesis != "All" and "tags.hypothesis_id" in runs_df.columns:
        hyp_id = selected_hypothesis.split(":")[0]
        runs_df = runs_df[runs_df["tags.hypothesis_id"] == hyp_id]

    if runs_df.empty:
        st.info("No experiments match the selected filter.")
        return

    # Display summary metrics
    metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
    key_metrics = ["metrics.sharpe_ratio", "metrics.total_return", "metrics.max_drawdown", "metrics.cagr"]
    display_metrics = [m for m in key_metrics if m in metric_cols]

    # Summary stats
    if display_metrics:
        st.markdown("### Summary Statistics")
        summary_cols = st.columns(len(display_metrics))
        for i, metric in enumerate(display_metrics):
            metric_name = metric.replace("metrics.", "")
            values = runs_df[metric].dropna()
            if not values.empty:
                summary_cols[i].metric(
                    metric_name.replace("_", " ").title(),
                    format_metric(values.mean(), metric_name),
                    delta=f"Best: {format_metric(values.max(), metric_name)}"
                    if metric_name != "max_drawdown"
                    else f"Best: {format_metric(values.max(), metric_name)}"
                )

    st.markdown("---")

    # List experiments
    st.markdown("### Experiment List")

    for idx, row in runs_df.iterrows():
        run_id = row.get("run_id", "unknown")
        run_name = row.get("tags.mlflow.runName", run_id[:8])
        start_time = row.get("start_time")

        # Build expander title
        sharpe = row.get("metrics.sharpe_ratio")
        total_return = row.get("metrics.total_return")

        title_parts = [f"**{run_name}**"]
        if start_time:
            title_parts.append(f" | {format_timestamp(int(start_time.timestamp() * 1000))}")
        if pd.notna(sharpe):
            title_parts.append(f" | Sharpe: {sharpe:.2f}")
        if pd.notna(total_return):
            title_parts.append(f" | Return: {total_return * 100:.1f}%")

        with st.expander("".join(title_parts)):
            render_run_details(run_id, row)


def render_run_details(run_id: str, row: pd.Series) -> None:
    """Render detailed view for a single run."""
    # Get full run details
    run_details = get_run_details(run_id)

    if run_details is None:
        st.error(f"Could not load details for run {run_id}")
        return

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    metrics = run_details.get("metrics", {})

    col1.metric(
        "Sharpe Ratio",
        format_metric(metrics.get("sharpe_ratio"), "sharpe_ratio")
    )
    col2.metric(
        "Total Return",
        format_metric(metrics.get("total_return"), "total_return")
    )
    col3.metric(
        "Max Drawdown",
        format_metric(metrics.get("max_drawdown"), "max_drawdown")
    )
    col4.metric(
        "CAGR",
        format_metric(metrics.get("cagr"), "cagr")
    )

    # Additional metrics in expandable section
    with st.expander("All Metrics"):
        metrics_data = []
        for name, value in sorted(metrics.items()):
            if not name.startswith("benchmark_"):
                metrics_data.append({
                    "Metric": name.replace("_", " ").title(),
                    "Value": format_metric(value, name)
                })
        if metrics_data:
            st.dataframe(
                pd.DataFrame(metrics_data),
                use_container_width=True,
                hide_index=True
            )

        # Benchmark comparison if available
        benchmark_metrics = {k: v for k, v in metrics.items() if k.startswith("benchmark_")}
        if benchmark_metrics:
            st.markdown("**Benchmark (SPY) Metrics:**")
            bench_data = []
            for name, value in sorted(benchmark_metrics.items()):
                clean_name = name.replace("benchmark_", "").replace("_", " ").title()
                bench_data.append({
                    "Metric": clean_name,
                    "Value": format_metric(value, name.replace("benchmark_", ""))
                })
            st.dataframe(
                pd.DataFrame(bench_data),
                use_container_width=True,
                hide_index=True
            )

    # Parameters
    with st.expander("Parameters"):
        params = run_details.get("params", {})
        if params:
            params_data = [{"Parameter": k, "Value": v} for k, v in sorted(params.items())]
            st.dataframe(
                pd.DataFrame(params_data),
                use_container_width=True,
                hide_index=True
            )

    # Links and actions
    col1, col2, col3 = st.columns(3)

    with col1:
        hypothesis_id = run_details.get("tags", {}).get("hypothesis_id")
        if hypothesis_id:
            st.markdown(f"**Linked Hypothesis:** `{hypothesis_id}`")

    with col2:
        mlflow_url = f"{MLFLOW_UI_URL}/#/experiments/0/runs/{run_id}"
        # Use styled anchor tag with JavaScript fallback to open in new tab
        st.markdown(
            f"""
            <a href="{mlflow_url}" target="_blank" rel="noopener noreferrer" 
               onclick="window.open('{mlflow_url}', '_blank'); return false;"
               style="
                   display: inline-block;
                   background-color: #FF4B4B;
                   color: white;
                   text-decoration: none;
                   padding: 0.5rem 1rem;
                   border-radius: 0.25rem;
                   font-size: 0.875rem;
                   text-align: center;
                   cursor: pointer;
               ">Open in MLflow UI</a>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        if st.button("Add to Comparison", key=f"add_compare_{run_id}"):
            if "compare_runs" not in st.session_state:
                st.session_state.compare_runs = []
            if run_id not in st.session_state.compare_runs:
                st.session_state.compare_runs.append(run_id)
                st.success(f"Added {run_id[:8]} to comparison")
            else:
                st.warning("Already in comparison list")


def render_compare_tab() -> None:
    """Render the Compare Experiments tab."""
    st.subheader("Compare Experiments")

    # Initialize comparison list
    if "compare_runs" not in st.session_state:
        st.session_state.compare_runs = []

    # Get all available runs for selection
    runs_df = get_experiment_runs(max_results=100)

    if runs_df.empty:
        st.info("No experiments available for comparison.")
        return

    # Build options for multiselect
    run_options = {}
    for _, row in runs_df.iterrows():
        run_id = row.get("run_id", "")
        run_name = row.get("tags.mlflow.runName", run_id[:8])
        sharpe = row.get("metrics.sharpe_ratio")
        label = f"{run_name}"
        if pd.notna(sharpe):
            label += f" (Sharpe: {sharpe:.2f})"
        run_options[label] = run_id

    # Multi-select for runs
    selected_labels = st.multiselect(
        "Select experiments to compare (minimum 2)",
        options=list(run_options.keys()),
        default=[
            label for label, rid in run_options.items()
            if rid in st.session_state.compare_runs
        ][:5],  # Limit default selection
        key="compare_multiselect"
    )

    # Update session state
    st.session_state.compare_runs = [run_options[label] for label in selected_labels]

    # Clear button
    if st.button("Clear Selection"):
        st.session_state.compare_runs = []
        st.rerun()

    if len(selected_labels) < 2:
        st.info("Select at least 2 experiments to compare.")
        return

    selected_run_ids = [run_options[label] for label in selected_labels]

    # Comparison metrics selection
    available_metrics = [
        "sharpe_ratio", "sortino_ratio", "total_return", "cagr",
        "max_drawdown", "volatility", "calmar_ratio", "win_rate",
        "profit_factor", "alpha", "beta", "information_ratio"
    ]

    selected_metrics = st.multiselect(
        "Select metrics to compare",
        available_metrics,
        default=["sharpe_ratio", "total_return", "max_drawdown", "cagr"],
        key="compare_metrics_select"
    )

    if not selected_metrics:
        st.warning("Select at least one metric to compare.")
        return

    # Build comparison dataframe
    api = PlatformAPI()
    comparison_df = api.compare_experiments(selected_run_ids, selected_metrics)

    if comparison_df.empty:
        st.warning("Could not load experiment data for comparison.")
        return

    # Format the comparison table
    st.markdown("### Side-by-Side Comparison")

    # Transpose for better readability (metrics as rows)
    display_df = comparison_df.T.copy()

    # Format values
    for col in display_df.columns:
        for metric in display_df.index:
            value = display_df.loc[metric, col]
            if pd.notna(value):
                display_df.loc[metric, col] = format_metric(value, metric)

    # Shorten column names (run IDs)
    display_df.columns = [col[:8] for col in display_df.columns]
    display_df.index = [idx.replace("_", " ").title() for idx in display_df.index]

    st.dataframe(display_df, use_container_width=True)

    # Visual comparison chart
    st.markdown("### Visual Comparison")

    chart_metric = st.selectbox(
        "Select metric for chart",
        selected_metrics,
        key="compare_chart_metric"
    )

    if chart_metric:
        chart_data = comparison_df[[chart_metric]].reset_index()
        chart_data.columns = ["Experiment", chart_metric.replace("_", " ").title()]
        chart_data["Experiment"] = chart_data["Experiment"].str[:8]

        fig = px.bar(
            chart_data,
            x="Experiment",
            y=chart_metric.replace("_", " ").title(),
            title=f"Comparison: {chart_metric.replace('_', ' ').title()}",
            color=chart_metric.replace("_", " ").title(),
            color_continuous_scale="RdYlGn" if chart_metric != "max_drawdown" else "RdYlGn_r"
        )

        fig.update_layout(
            xaxis_title="Experiment",
            yaxis_title=chart_metric.replace("_", " ").title(),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Radar chart for multiple metrics
    if len(selected_metrics) >= 3:
        st.markdown("### Multi-Metric Radar Chart")

        # Normalize metrics for radar chart (0-1 scale)
        radar_data = comparison_df[selected_metrics].copy()

        # Handle max_drawdown (lower is better, so invert)
        if "max_drawdown" in radar_data.columns:
            radar_data["max_drawdown"] = -radar_data["max_drawdown"]

        # Normalize each metric
        for col in radar_data.columns:
            col_min = radar_data[col].min()
            col_max = radar_data[col].max()
            if col_max != col_min:
                radar_data[col] = (radar_data[col] - col_min) / (col_max - col_min)
            else:
                radar_data[col] = 0.5

        fig = go.Figure()

        for run_id in radar_data.index:
            values = radar_data.loc[run_id].tolist()
            values.append(values[0])  # Close the polygon

            categories = [m.replace("_", " ").title() for m in selected_metrics]
            categories.append(categories[0])

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=run_id[:8]
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Normalized Metric Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)


def render_run_backtest_tab() -> None:
    """Render the Run Backtest tab."""
    st.subheader("Run New Backtest")

    # Get available symbols
    available_symbols = get_available_symbols()

    if not available_symbols:
        st.warning(
            "No symbols found in database. Please ingest price data first."
        )
        return

    # Get hypotheses for linking
    hypotheses = get_hypotheses_for_filter()

    with st.form("run_backtest_form"):
        st.markdown("### Configuration")

        # Symbol selection
        col1, col2 = st.columns(2)

        with col1:
            selected_symbols = st.multiselect(
                "Select Symbols",
                available_symbols,
                default=available_symbols[:10] if len(available_symbols) >= 10 else available_symbols,
                help="Select stocks to include in the backtest"
            )

        with col2:
            # Quick selection helpers
            if st.checkbox("Use all available symbols"):
                selected_symbols = available_symbols

        # Date range
        st.markdown("### Date Range")
        col1, col2 = st.columns(2)

        default_end = date.today()
        default_start = default_end - timedelta(days=365 * 3)  # 3 years

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=date(2000, 1, 1),
                max_value=default_end
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=start_date,
                max_value=date.today()
            )

        # Strategy configuration
        st.markdown("### Strategy Settings")

        strategy_type = st.selectbox(
            "Strategy Type",
            ["momentum", "multifactor", "ml_predicted"],
            format_func=lambda x: STRATEGY_REGISTRY.get(x, {}).get("name", x.title()),
            help="Strategy to backtest"
        )

        # Strategy-specific configuration
        strategy_config = {}

        if strategy_type == "momentum":
            col1, col2 = st.columns(2)
            with col1:
                lookback_period = st.number_input(
                    "Lookback Period (days)",
                    min_value=20,
                    max_value=504,
                    value=252,
                    step=21,
                    help="Period for signal calculation"
                )
            with col2:
                top_n = st.number_input(
                    "Number of Holdings",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Number of top stocks to hold"
                )
            strategy_config = {"lookback": lookback_period, "top_n": top_n}

        elif strategy_type == "multifactor":
            strategy_config = render_multifactor_config()

        elif strategy_type == "ml_predicted":
            strategy_config = render_ml_predicted_config()

        # Position sizing
        st.markdown("### Position Sizing")
        col1, col2, col3 = st.columns(3)

        with col1:
            sizing_method = st.selectbox(
                "Sizing Method",
                ["equal", "volatility", "signal_scaled"],
                help="How to size positions"
            )

        with col2:
            max_positions = st.number_input(
                "Max Positions",
                min_value=1,
                max_value=100,
                value=20,
                help="Maximum number of positions"
            )

        with col3:
            max_position_pct = st.slider(
                "Max Position %",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                help="Maximum position size as % of portfolio"
            )

        # Cost model
        st.markdown("### Cost Model (IBKR Realistic)")
        col1, col2 = st.columns(2)

        with col1:
            spread_bps = st.number_input(
                "Spread (bps)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                help="Bid-ask spread in basis points"
            )

        with col2:
            slippage_bps = st.number_input(
                "Slippage (bps)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                help="Execution slippage in basis points"
            )

        # Hypothesis linking
        st.markdown("### Experiment Tracking")
        col1, col2 = st.columns(2)

        with col1:
            backtest_name = st.text_input(
                "Backtest Name",
                value=f"{strategy_type}_backtest_{date.today().isoformat()}",
                help="Name for this backtest run"
            )

        with col2:
            hypothesis_options = ["None"] + [f"{h['id']}: {h['title'][:30]}" for h in hypotheses]
            linked_hypothesis = st.selectbox(
                "Link to Hypothesis",
                hypothesis_options,
                help="Optionally link this backtest to a hypothesis"
            )

        # Submit button
        submitted = st.form_submit_button("Run Backtest", type="primary")

    if submitted:
        if not selected_symbols:
            st.error("Please select at least one symbol.")
            return

        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return

        # Build config
        config = BacktestConfig(
            symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
            sizing_method=sizing_method,
            max_positions=max_positions,
            max_position_pct=max_position_pct,
            costs=CostModel(spread_bps=spread_bps, slippage_bps=slippage_bps),
            name=backtest_name,
        )

        # Parse hypothesis ID
        hypothesis_id = None
        if linked_hypothesis != "None":
            hypothesis_id = linked_hypothesis.split(":")[0]

        # Run backtest with progress
        with st.spinner("Running backtest..."):
            try:
                # Load prices
                st.text("Loading price data...")
                prices = get_price_data(selected_symbols, start_date, end_date)

                # Generate signals
                st.text("Generating signals...")
                if strategy_type == "momentum":
                    signals = generate_momentum_signals(
                        prices,
                        lookback=strategy_config.get("lookback", 252),
                        top_n=strategy_config.get("top_n", 10)
                    )
                elif strategy_type == "multifactor":
                    if not strategy_config.get("feature_weights"):
                        st.error("Please select at least one factor for multi-factor strategy.")
                        return
                    signals = generate_multifactor_signals(
                        prices,
                        feature_weights=strategy_config["feature_weights"],
                        top_n=strategy_config.get("top_n", 10),
                    )
                elif strategy_type == "ml_predicted":
                    if not strategy_config.get("features"):
                        st.error("Please select at least one feature for ML-predicted strategy.")
                        return
                    signals = generate_ml_predicted_signals(
                        prices,
                        model_type=strategy_config.get("model_type", "ridge"),
                        features=strategy_config.get("features"),
                        signal_method=strategy_config.get("signal_method", "rank"),
                        top_pct=strategy_config.get("top_pct", 0.1),
                        threshold=strategy_config.get("threshold", 0.0),
                        train_lookback=strategy_config.get("train_lookback", 252),
                        retrain_frequency=strategy_config.get("retrain_frequency", 21),
                    )
                else:
                    st.error(f"Unknown strategy type: {strategy_type}")
                    return

                # Run backtest
                st.text("Running simulation...")
                result = run_backtest(signals, config, prices)

                # Log to MLflow
                st.text("Logging to MLflow...")
                from hrp.research.mlflow_utils import log_backtest

                run_id = log_backtest(
                    result=result,
                    experiment_name=DEFAULT_EXPERIMENT,
                    run_name=backtest_name,
                    hypothesis_id=hypothesis_id,
                    tags={"actor": "user", "strategy": strategy_type},
                    strategy_config=strategy_config,
                )

                st.success(f"Backtest completed! Run ID: `{run_id}`")

                # Display results
                st.markdown("### Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sharpe Ratio", f"{result.sharpe:.2f}")
                col2.metric("Total Return", f"{result.total_return * 100:.1f}%")
                col3.metric("Max Drawdown", f"{result.max_drawdown * 100:.1f}%")
                col4.metric("CAGR", f"{result.metrics.get('cagr', 0) * 100:.1f}%")

                # Equity curve with benchmark comparison
                if result.equity_curve is not None and len(result.equity_curve) > 0:
                    st.markdown("### Equity Curve")

                    # Get benchmark data for comparison
                    try:
                        from hrp.research.benchmark import get_benchmark_prices
                        
                        benchmark_prices = get_benchmark_prices(
                            "SPY",
                            start=start_date,
                            end=end_date
                        )
                        
                        # Calculate benchmark equity curve (buy and hold)
                        benchmark_prices['date'] = pd.to_datetime(benchmark_prices['date'])
                        benchmark_prices = benchmark_prices.set_index('date')
                        
                        # Align with strategy dates
                        benchmark_prices = benchmark_prices.reindex(result.equity_curve.index, method='ffill')
                        
                        # Normalize to same starting value as strategy
                        initial_value = result.equity_curve.iloc[0]
                        benchmark_normalized = (benchmark_prices['adj_close'] / benchmark_prices['adj_close'].iloc[0]) * initial_value
                        
                        # Create comparison dataframe
                        eq_df = pd.DataFrame({
                            "Date": result.equity_curve.index,
                            "Strategy": result.equity_curve.values,
                            "SPY (Buy & Hold)": benchmark_normalized.values
                        })
                        
                        # Melt for plotly
                        eq_df_melted = eq_df.melt(
                            id_vars=["Date"],
                            value_vars=["Strategy", "SPY (Buy & Hold)"],
                            var_name="Portfolio",
                            value_name="Value"
                        )
                        
                        fig = px.line(
                            eq_df_melted,
                            x="Date",
                            y="Value",
                            color="Portfolio",
                            title="Equity Curve vs Benchmark",
                            color_discrete_map={
                                "Strategy": "#1f77b4",
                                "SPY (Buy & Hold)": "#ff7f0e"
                            }
                        )
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            hovermode="x unified",
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        logger.warning(f"Could not add benchmark comparison: {e}")
                        # Fallback to strategy only
                        eq_df = pd.DataFrame({
                            "Date": result.equity_curve.index,
                            "Portfolio Value": result.equity_curve.values
                        })

                        fig = px.line(
                            eq_df,
                            x="Date",
                            y="Portfolio Value",
                            title="Equity Curve"
                        )
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Trade summary
                if result.trades is not None and len(result.trades) > 0:
                    with st.expander("Trade Summary"):
                        st.dataframe(result.trades.head(50), use_container_width=True)
                        st.caption(f"Showing first 50 of {len(result.trades)} trades")

                # Benchmark comparison
                if result.benchmark_metrics:
                    with st.expander("Benchmark Comparison (SPY)"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Strategy**")
                            for key, value in sorted(result.metrics.items()):
                                if not key.startswith("benchmark"):
                                    st.text(f"{key}: {format_metric(value, key)}")

                        with col2:
                            st.markdown("**Benchmark (SPY)**")
                            for key, value in sorted(result.benchmark_metrics.items()):
                                st.text(f"{key}: {format_metric(value, key)}")

                # Clear cache to show new run in browse tab
                get_experiment_runs.clear()

            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                st.error(f"Backtest failed: {str(e)}")


def render() -> None:
    """Main render function for the Experiments page."""
    st.title("Experiments")

    st.markdown(
        """
        Browse, compare, and run backtests. All experiments are tracked in MLflow
        for reproducibility and analysis.
        """
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Browse", "Compare", "Run Backtest"])

    with tab1:
        render_browse_tab()

    with tab2:
        render_compare_tab()

    with tab3:
        render_run_backtest_tab()


# Entry point for Streamlit multipage apps
if __name__ == "__main__":
    render()
