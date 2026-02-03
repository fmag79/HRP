"""
HRP Research MCP Server.

Provides Claude with access to the hedge fund research platform via FastMCP tools.
Implements 22 tools covering hypothesis management, data access, backtesting,
ML training, quality checks, and lineage tracking.

Usage:
    python -m hrp.mcp.research_server
"""

from datetime import date
from typing import Any, Optional

from fastmcp import FastMCP
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.mcp.errors import handle_api_error
from hrp.mcp.formatters import (
    format_experiment,
    format_hypothesis,
    format_lineage_event,
    format_response,
    parse_date,
)

# Create FastMCP server
mcp = FastMCP(
    "HRP Research",
    instructions=(
        "Hedge Fund Research Platform - Tools for hypothesis-driven quantitative "
        "research including backtesting, ML training, and data quality monitoring."
    ),
)

# Actor identifier for all MCP tool calls
ACTOR = "agent:claude-interactive"


def get_api(read_only: bool = True) -> PlatformAPI:
    """Get or create a PlatformAPI instance with short-lived connections.

    IMPORTANT: We use use_singleton=False because DuckDB uses file-level locking.
    Even read-only connections hold locks that prevent other processes (agents)
    from acquiring write locks. Non-singleton instances release locks when closed.

    The caller should use the API as a context manager or call close() explicitly:

        # Option 1: Context manager (preferred)
        with get_api() as api:
            result = api.list_hypotheses()

        # Option 2: Explicit close
        api = get_api()
        try:
            result = api.list_hypotheses()
        finally:
            api.close()

    Args:
        read_only: If True, returns a read-only instance (default).
                   If False, returns a read-write instance for mutations.

    Returns:
        A PlatformAPI instance with use_singleton=False. Connections are
        released when close() is called or the context manager exits.
    """
    return PlatformAPI(read_only=read_only, use_singleton=False)


# =============================================================================
# Hypothesis Management Tools (5)
# =============================================================================


@mcp.tool()
@handle_api_error
def list_hypotheses(
    status: Optional[str] = None,
    limit: int = 100,
) -> dict[str, Any]:
    """
    List research hypotheses, optionally filtered by status.

    Args:
        status: Filter by status ('draft', 'testing', 'validated', 'rejected', 'deployed')
        limit: Maximum number of results (default 100)

    Returns:
        List of hypothesis summaries with id, title, status, and dates
    """
    with get_api() as api:
        hypotheses = api.list_hypotheses(status=status, limit=limit)

        # Format each hypothesis
        formatted = [format_hypothesis(h) for h in hypotheses]

        return format_response(
            success=True,
            data=formatted,
            message=f"Found {len(formatted)} hypotheses",
        )


@mcp.tool()
@handle_api_error
def get_hypothesis(hypothesis_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific hypothesis.

    Args:
        hypothesis_id: The hypothesis ID (e.g., 'HYP-2026-001')

    Returns:
        Full hypothesis details including thesis, prediction, falsification criteria
    """
    with get_api() as api:
        hypothesis = api.get_hypothesis(hypothesis_id)

        if hypothesis is None:
            return format_response(
                success=False,
                message=f"Hypothesis {hypothesis_id} not found",
                error=f"No hypothesis found with ID: {hypothesis_id}",
            )

        return format_response(
            success=True,
            data=format_hypothesis(hypothesis),
            message=f"Retrieved hypothesis {hypothesis_id}",
        )


@mcp.tool()
@handle_api_error
def create_hypothesis(
    title: str,
    thesis: str,
    prediction: str,
    falsification: str,
) -> dict[str, Any]:
    """
    Create a new research hypothesis for testing.

    Every strategy should start as a hypothesis with clear falsification criteria.

    Args:
        title: Short descriptive title (e.g., "Momentum predicts returns")
        thesis: The hypothesis being tested (what you believe to be true)
        prediction: Testable prediction (measurable outcome expected)
        falsification: Criteria for rejecting the hypothesis (e.g., "Sharpe < 0.5")

    Returns:
        The newly created hypothesis ID
    """
    with get_api(read_only=False) as api:
        hypothesis_id = api.create_hypothesis(
            title=title,
            thesis=thesis,
            prediction=prediction,
            falsification=falsification,
            actor=ACTOR,
        )

        return format_response(
            success=True,
            data={"hypothesis_id": hypothesis_id},
            message=f"Created hypothesis {hypothesis_id}",
        )


@mcp.tool()
@handle_api_error
def update_hypothesis(
    hypothesis_id: str,
    status: str,
    outcome: Optional[str] = None,
) -> dict[str, Any]:
    """
    Update a hypothesis status and optionally record the outcome.

    Args:
        hypothesis_id: The hypothesis ID to update
        status: New status ('draft', 'testing', 'validated', 'rejected')
        outcome: Optional outcome description (why validated/rejected)

    Returns:
        Confirmation of the update
    """
    with get_api(read_only=False) as api:
        api.update_hypothesis(
            hypothesis_id=hypothesis_id,
            status=status,
            outcome=outcome,
            actor=ACTOR,
        )

        return format_response(
            success=True,
            data={"hypothesis_id": hypothesis_id, "status": status},
            message=f"Updated hypothesis {hypothesis_id} to status '{status}'",
        )


@mcp.tool()
@handle_api_error
def get_experiments_for_hypothesis(hypothesis_id: str) -> dict[str, Any]:
    """
    Get all experiments (backtests) linked to a hypothesis.

    Args:
        hypothesis_id: The hypothesis ID

    Returns:
        List of experiment IDs associated with the hypothesis
    """
    with get_api() as api:
        experiment_ids = api.get_experiments_for_hypothesis(hypothesis_id)

        return format_response(
            success=True,
            data={"hypothesis_id": hypothesis_id, "experiment_ids": experiment_ids},
            message=f"Found {len(experiment_ids)} experiments for {hypothesis_id}",
        )


# =============================================================================
# Data Access Tools (5)
# =============================================================================


@mcp.tool()
@handle_api_error
def get_universe(as_of_date: Optional[str] = None) -> dict[str, Any]:
    """
    Get the list of tradeable symbols in the universe.

    The universe is the set of stocks available for trading, filtered by
    liquidity, market cap, and sector constraints (excludes financials, REITs).

    Args:
        as_of_date: Date to get universe for (ISO format: YYYY-MM-DD). Defaults to today.

    Returns:
        List of ticker symbols in the universe
    """
    with get_api() as api:
        target_date = parse_date(as_of_date) or date.today()
        symbols = api.get_universe(target_date)

        return format_response(
            success=True,
            data={"as_of_date": str(target_date), "symbols": symbols, "count": len(symbols)},
            message=f"Universe contains {len(symbols)} symbols as of {target_date}",
        )


@mcp.tool()
@handle_api_error
def get_features(
    symbols: list[str],
    features: list[str],
    as_of_date: str,
) -> dict[str, Any]:
    """
    Get feature values for symbols as of a specific date.

    Features are pre-computed signals like momentum, volatility, etc.
    Uses point-in-time data to prevent look-ahead bias.

    Args:
        symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        features: List of feature names (e.g., ['momentum_20d', 'volatility_60d'])
        as_of_date: Date to get features for (ISO format: YYYY-MM-DD)

    Returns:
        DataFrame with symbols as rows and features as columns
    """
    target_date = parse_date(as_of_date)

    if target_date is None:
        return format_response(
            success=False,
            message="as_of_date is required",
            error="Missing required parameter: as_of_date",
        )

    with get_api() as api:
        df = api.get_features(symbols, features, target_date)

        return format_response(
            success=True,
            data=df,
            message=f"Retrieved {len(features)} features for {len(symbols)} symbols",
        )


@mcp.tool()
@handle_api_error
def get_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """
    Get historical price data for symbols over a date range.

    Returns OHLCV data (open, high, low, close, adjusted close, volume).
    Data is split-adjusted but NOT dividend-adjusted by default.

    Args:
        symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)

    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        return format_response(
            success=False,
            message="Both start_date and end_date are required",
            error="Missing required date parameters",
        )

    with get_api() as api:
        df = api.get_prices(symbols, start, end)

        # Return summary for large datasets
        summary = {
            "symbols": symbols,
            "start_date": str(start),
            "end_date": str(end),
            "total_rows": len(df),
            "columns": list(df.columns) if not df.empty else [],
        }

        # Include sample data for small datasets, summary for large ones
        if len(df) <= 100:
            return format_response(
                success=True,
                data=df,
                message=f"Retrieved {len(df)} price records",
            )
        else:
            return format_response(
                success=True,
                data=summary,
                message=f"Retrieved {len(df)} price records (summary only - data too large)",
            )


@mcp.tool()
@handle_api_error
def get_available_features() -> dict[str, Any]:
    """
    List all available features in the feature store.

    Returns feature names and descriptions that can be used in backtests
    and ML models.

    Returns:
        List of feature definitions with names, descriptions, and versions
    """
    with get_api() as api:
        formatted = api.get_available_features()

        return format_response(
            success=True,
            data=formatted,
            message=f"Found {len(formatted)} available features",
        )


@mcp.tool()
@handle_api_error
def is_trading_day(check_date: str) -> dict[str, Any]:
    """
    Check if a date is a valid NYSE trading day.

    Weekends and NYSE holidays return False.

    Args:
        check_date: Date to check (ISO format: YYYY-MM-DD)

    Returns:
        Boolean indicating if the date is a trading day
    """
    target_date = parse_date(check_date)

    if target_date is None:
        return format_response(
            success=False,
            message="check_date is required",
            error="Missing required parameter: check_date",
        )

    with get_api() as api:
        is_trading = api.is_trading_day(target_date)

        return format_response(
            success=True,
            data={"date": str(target_date), "is_trading_day": is_trading},
            message=f"{target_date} {'is' if is_trading else 'is not'} a trading day",
        )


# =============================================================================
# Backtesting Tools (4)
# =============================================================================


@mcp.tool()
@handle_api_error
def run_backtest(
    hypothesis_id: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    max_positions: int = 20,
    max_position_pct: float = 0.10,
    sizing_method: str = "equal",
    name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run a backtest and log results to MLflow.

    Executes a systematic strategy backtest with realistic costs (IBKR model)
    and links results to the hypothesis for tracking.

    Args:
        hypothesis_id: Hypothesis to link this backtest to
        symbols: List of symbols to trade
        start_date: Backtest start date (ISO format)
        end_date: Backtest end date (ISO format)
        max_positions: Maximum number of positions (default 20)
        max_position_pct: Maximum position size as portfolio % (default 0.10)
        sizing_method: Position sizing method: 'equal', 'volatility', or 'signal_scaled'
        name: Optional name for this backtest run

    Returns:
        Experiment ID and summary metrics (Sharpe, return, max drawdown)
    """
    from hrp.research.config import BacktestConfig

    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        return format_response(
            success=False,
            message="Both start_date and end_date are required",
            error="Missing required date parameters",
        )

    # Create config
    config = BacktestConfig(
        symbols=symbols,
        start_date=start,
        end_date=end,
        max_positions=max_positions,
        max_position_pct=max_position_pct,
        sizing_method=sizing_method,
        name=name or f"backtest_{hypothesis_id}",
    )

    with get_api(read_only=False) as api:
        # Run backtest
        experiment_id = api.run_backtest(
            config=config,
            hypothesis_id=hypothesis_id,
            actor=ACTOR,
        )

        # Get results
        experiment = api.get_experiment(experiment_id)
        metrics = experiment.get("metrics", {}) if experiment else {}

        return format_response(
            success=True,
            data={
                "experiment_id": experiment_id,
                "hypothesis_id": hypothesis_id,
                "metrics": {
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "total_return": metrics.get("total_return"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "cagr": metrics.get("cagr"),
                    "volatility": metrics.get("volatility"),
                },
            },
            message=f"Backtest complete. Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.2f}",
        )


@mcp.tool()
@handle_api_error
def get_experiment(experiment_id: str) -> dict[str, Any]:
    """
    Get detailed information about a backtest experiment.

    Args:
        experiment_id: The MLflow run ID

    Returns:
        Full experiment details including params, metrics, and tags
    """
    with get_api() as api:
        experiment = api.get_experiment(experiment_id)

        if experiment is None:
            return format_response(
                success=False,
                message=f"Experiment {experiment_id} not found",
                error=f"No experiment found with ID: {experiment_id}",
            )

        return format_response(
            success=True,
            data=format_experiment(experiment),
            message=f"Retrieved experiment {experiment_id}",
        )


@mcp.tool()
@handle_api_error
def compare_experiments(
    experiment_ids: list[str],
    metrics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compare multiple experiments side by side.

    Useful for evaluating different strategy variations or parameter choices.

    Args:
        experiment_ids: List of experiment IDs to compare
        metrics: Optional list of metrics to compare (defaults to common set)

    Returns:
        DataFrame with experiments as rows and metrics as columns
    """
    with get_api() as api:
        comparison_df = api.compare_experiments(experiment_ids, metrics)

        return format_response(
            success=True,
            data=comparison_df,
            message=f"Compared {len(experiment_ids)} experiments",
        )


@mcp.tool()
@handle_api_error
def analyze_results(experiment_id: str) -> dict[str, Any]:
    """
    Get a formatted analysis summary for an experiment.

    Provides a human-readable interpretation of backtest results
    including risk-adjusted metrics and performance assessment.

    Args:
        experiment_id: The experiment to analyze

    Returns:
        Formatted analysis text and key insights
    """
    with get_api() as api:
        experiment = api.get_experiment(experiment_id)

        if experiment is None:
            return format_response(
                success=False,
                message=f"Experiment {experiment_id} not found",
                error=f"No experiment found with ID: {experiment_id}",
            )

        metrics = experiment.get("metrics", {})
        params = experiment.get("params", {})

        # Generate analysis text
        sharpe = metrics.get("sharpe_ratio", 0)
        total_return = metrics.get("total_return", 0)
        max_dd = metrics.get("max_drawdown", 0)
        cagr = metrics.get("cagr", 0)

        # Assess performance
        if sharpe >= 1.5:
            sharpe_assessment = "Excellent risk-adjusted returns"
        elif sharpe >= 1.0:
            sharpe_assessment = "Good risk-adjusted returns"
        elif sharpe >= 0.5:
            sharpe_assessment = "Moderate risk-adjusted returns"
        else:
            sharpe_assessment = "Poor risk-adjusted returns"

        analysis = {
            "experiment_id": experiment_id,
            "summary": {
                "sharpe_ratio": sharpe,
                "sharpe_assessment": sharpe_assessment,
                "total_return_pct": total_return * 100 if total_return else None,
                "cagr_pct": cagr * 100 if cagr else None,
                "max_drawdown_pct": max_dd * 100 if max_dd else None,
            },
            "parameters": params,
            "all_metrics": metrics,
            "interpretation": (
                f"The strategy achieved a Sharpe ratio of {sharpe:.2f} "
                f"({sharpe_assessment}). "
                f"Total return was {total_return*100:.1f}% with a maximum drawdown "
                f"of {abs(max_dd)*100:.1f}%."
                if sharpe and total_return and max_dd
                else "Insufficient data for full analysis."
            ),
        }

        return format_response(
            success=True,
            data=analysis,
            message=sharpe_assessment,
        )


# =============================================================================
# ML Training Tools (3)
# =============================================================================


@mcp.tool()
@handle_api_error
def run_walk_forward_validation(
    model_type: str,
    target: str,
    features: list[str],
    symbols: list[str],
    start_date: str,
    end_date: str,
    n_folds: int = 5,
    window_type: str = "expanding",
) -> dict[str, Any]:
    """
    Run walk-forward validation for an ML model.

    Walk-forward validation trains on historical data and tests on future data
    repeatedly, simulating real trading conditions. This prevents overfitting.

    Args:
        model_type: Model to train ('ridge', 'lasso', 'elastic_net', 'random_forest', 'lightgbm')
        target: Target variable (e.g., 'returns_20d')
        features: Feature names to use as inputs
        symbols: Symbols to train on
        start_date: Validation period start (ISO format)
        end_date: Validation period end (ISO format)
        n_folds: Number of walk-forward folds (default 5)
        window_type: 'expanding' or 'rolling' (default 'expanding')

    Returns:
        Validation results including stability score, mean IC, and per-fold metrics
    """
    from hrp.ml.validation import WalkForwardConfig, walk_forward_validate

    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        return format_response(
            success=False,
            message="Both start_date and end_date are required",
            error="Missing required date parameters",
        )

    config = WalkForwardConfig(
        model_type=model_type,
        target=target,
        features=features,
        start_date=start,
        end_date=end,
        n_folds=n_folds,
        window_type=window_type,
    )

    result = walk_forward_validate(
        config=config,
        symbols=symbols,
        log_to_mlflow=True,
    )

    # Format results
    fold_summaries = [
        {
            "fold": fr.fold_index,
            "train_period": f"{fr.train_start} to {fr.train_end}",
            "test_period": f"{fr.test_start} to {fr.test_end}",
            "ic": fr.metrics.get("ic"),
            "mse": fr.metrics.get("mse"),
        }
        for fr in result.fold_results
    ]

    return format_response(
        success=True,
        data={
            "model_type": model_type,
            "stability_score": result.stability_score,
            "is_stable": result.is_stable,
            "mean_ic": result.mean_ic,
            "aggregate_metrics": result.aggregate_metrics,
            "fold_results": fold_summaries,
        },
        message=(
            f"Walk-forward complete. Stability: {result.stability_score:.3f} "
            f"({'stable' if result.is_stable else 'unstable'})"
        ),
    )


@mcp.tool()
@handle_api_error
def get_supported_models() -> dict[str, Any]:
    """
    List available ML model types for training.

    Returns:
        List of model types with descriptions
    """
    from hrp.ml.models import SUPPORTED_MODELS

    models = [
        {"model_type": name, "class": cls.__name__}
        for name, cls in SUPPORTED_MODELS.items()
    ]

    return format_response(
        success=True,
        data=models,
        message=f"Found {len(models)} supported model types",
    )


@mcp.tool()
@handle_api_error
def train_ml_model(
    model_type: str,
    target: str,
    features: list[str],
    symbols: list[str],
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, Any]:
    """
    Train an ML model with proper date splits.

    Uses train/validation/test split to evaluate model performance without look-ahead bias.

    Args:
        model_type: Model to train ('ridge', 'lasso', 'elastic_net', 'random_forest')
        target: Target variable (e.g., 'returns_20d')
        features: Feature names to use as inputs
        symbols: Symbols to train on
        train_start: Training period start (ISO format)
        train_end: Training period end (ISO format)
        validation_start: Validation period start (ISO format)
        validation_end: Validation period end (ISO format)
        test_start: Test period start (ISO format)
        test_end: Test period end (ISO format)

    Returns:
        Model training results including test set metrics
    """
    from hrp.ml.models import MLConfig
    from hrp.ml.training import train_model, TrainingResult

    train_start_date = parse_date(train_start)
    train_end_date = parse_date(train_end)
    val_start_date = parse_date(validation_start)
    val_end_date = parse_date(validation_end)
    test_start_date = parse_date(test_start)
    test_end_date = parse_date(test_end)

    if None in [
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date,
    ]:
        return format_response(
            success=False,
            message="All date parameters are required",
            error="Missing required date parameters",
        )

    # Create MLConfig
    config = MLConfig(
        model_type=model_type,
        target=target,
        features=features,
        train_start=train_start_date,
        train_end=train_end_date,
        validation_start=val_start_date,
        validation_end=val_end_date,
        test_start=test_start_date,
        test_end=test_end_date,
    )

    result: TrainingResult = train_model(
        config=config,
        symbols=symbols,
        log_to_mlflow=True,
    )

    return format_response(
        success=True,
        data={
            "model_type": model_type,
            "metrics": result.metrics,
            "selected_features": result.selected_features,
            "feature_importance": result.feature_importance,
        },
        message=f"Model trained. Test MSE: {result.metrics.get('test_mse', 'N/A'):.6f}",
    )


# =============================================================================
# Quality & Health Tools (3)
# =============================================================================


@mcp.tool()
@handle_api_error
def run_quality_checks(as_of_date: Optional[str] = None) -> dict[str, Any]:
    """
    Run data quality checks and return a health report.

    Checks for missing data, stale prices, invalid values, and other
    data quality issues that could affect research accuracy.

    Args:
        as_of_date: Date to check (ISO format). Defaults to today.

    Returns:
        Quality report with health score, issues found, and recommendations
    """
    target_date = parse_date(as_of_date) or date.today()
    with get_api() as api:
        result = api.run_quality_checks(as_of_date=target_date)

        return format_response(
            success=True,
            data={
                "report_date": str(target_date),
                "health_score": result["health_score"],
                "passed": result["passed"],
                "checks_run": len(result.get("results", [])),
                "checks_passed": sum(1 for r in result.get("results", []) if r.get("passed")),
                "critical_issues": result["critical_issues"],
                "warning_issues": result["warning_issues"],
                "summary": f"Health score: {result['health_score']:.0f}/100, {result['critical_issues']} critical, {result['warning_issues']} warnings",
            },
            message=f"Health score: {result['health_score']:.0f}/100",
        )


@mcp.tool()
@handle_api_error
def get_health_status() -> dict[str, Any]:
    """
    Get platform health status.

    Quick check of API and database connectivity.

    Returns:
        Health status for API, database, and tables
    """
    with get_api() as api:
        health = api.health_check()

        return format_response(
            success=True,
            data=health,
            message=f"API: {health['api']}, Database: {health['database']}",
        )


@mcp.tool()
@handle_api_error
def get_data_coverage(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """
    Check data availability for symbols over a date range.

    Useful for verifying data exists before running backtests.

    Args:
        symbols: Symbols to check
        start_date: Start of date range (ISO format)
        end_date: End of date range (ISO format)

    Returns:
        Coverage statistics per symbol
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        return format_response(
            success=False,
            message="Both start_date and end_date are required",
            error="Missing required date parameters",
        )

    with get_api() as api:
        # Get trading days in range
        trading_days = api.get_trading_days(start, end)
        expected_days = len(trading_days)

        # Get actual data per symbol
        coverage = []
        for symbol in symbols:
            try:
                df = api.get_prices([symbol], start, end)
                actual_days = len(df) if not df.empty else 0
                pct = (actual_days / expected_days * 100) if expected_days > 0 else 0

                coverage.append({
                    "symbol": symbol,
                    "expected_days": expected_days,
                    "actual_days": actual_days,
                    "coverage_pct": round(pct, 1),
                    "has_full_coverage": actual_days >= expected_days * 0.95,
                })
            except Exception as e:
                coverage.append({
                    "symbol": symbol,
                    "expected_days": expected_days,
                    "actual_days": 0,
                    "coverage_pct": 0,
                    "has_full_coverage": False,
                    "error": str(e),
                })

        # Summary
        full_coverage_count = sum(1 for c in coverage if c["has_full_coverage"])

        return format_response(
            success=True,
            data={
                "date_range": {"start": str(start), "end": str(end)},
                "expected_trading_days": expected_days,
                "symbols_with_full_coverage": full_coverage_count,
                "total_symbols": len(symbols),
                "coverage_by_symbol": coverage,
            },
            message=f"{full_coverage_count}/{len(symbols)} symbols have full coverage",
        )


# =============================================================================
# Lineage Tools (2)
# =============================================================================


@mcp.tool()
@handle_api_error
def get_lineage(
    hypothesis_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get audit trail / lineage events.

    Track the history of hypotheses, experiments, and actions.

    Args:
        hypothesis_id: Filter by hypothesis
        experiment_id: Filter by experiment
        limit: Maximum events to return (default 50)

    Returns:
        List of lineage events with timestamps and actors
    """
    with get_api() as api:
        events = api.get_lineage(
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            limit=limit,
        )

        formatted = [format_lineage_event(e) for e in events]

        return format_response(
            success=True,
            data=formatted,
            message=f"Found {len(formatted)} lineage events",
        )


@mcp.tool()
@handle_api_error
def get_deployed_strategies() -> dict[str, Any]:
    """
    List all currently deployed strategies.

    Deployed strategies are hypotheses that have been validated and
    approved for live trading.

    Returns:
        List of deployed hypothesis details
    """
    with get_api() as api:
        strategies = api.get_deployed_strategies()

        formatted = [format_hypothesis(s) for s in strategies]

        return format_response(
            success=True,
            data=formatted,
            message=f"Found {len(formatted)} deployed strategies",
        )


# =============================================================================
# Agent Tools (1)
# =============================================================================


@mcp.tool()
@handle_api_error
def run_ml_quality_sentinel(
    experiment_ids: Optional[list[str]] = None,
    hypothesis_ids: Optional[list[str]] = None,
    audit_window_days: int = 1,
    include_monitoring: bool = True,
) -> dict[str, Any]:
    """
    Run ML Quality Sentinel to audit experiments for overfitting and quality issues.

    The Sentinel acts as an independent watchdog, checking ML experiments for:
    - Sharpe decay (train vs test) - flags if >50%
    - Target leakage - flags if correlation >0.95
    - Feature count - flags if >50 features or insufficient samples/feature
    - Fold stability - flags if CV >2.0 or sign flips across folds
    - Suspiciously good results - flags if IC >0.15 or Sharpe >3.0

    Can also monitor deployed models for IC degradation and loss streaks.

    Args:
        experiment_ids: Specific experiment IDs to audit (optional)
        hypothesis_ids: Audit all experiments for these hypothesis IDs (optional)
        audit_window_days: Days of recent experiments to audit if no IDs provided (default 1)
        include_monitoring: Whether to monitor deployed models (default True)

    Returns:
        Audit summary with counts of passed/flagged experiments and issues found
    """
    from hrp.agents.research_agents import MLQualitySentinel

    sentinel = MLQualitySentinel(
        experiment_ids=experiment_ids,
        hypothesis_ids=hypothesis_ids,
        audit_window_days=audit_window_days,
        include_monitoring=include_monitoring,
        fail_on_critical=False,  # Don't raise exception in MCP context
        send_alerts=False,  # Let Claude decide on notifications
    )

    result = sentinel.run()

    return format_response(
        success=result.get("status") == "success",
        data={
            "report_date": result.get("result", {}).get("report_date"),
            "experiments_audited": result.get("result", {}).get("experiments_audited", 0),
            "experiments_passed": result.get("result", {}).get("experiments_passed", 0),
            "experiments_flagged": result.get("result", {}).get("experiments_flagged", 0),
            "critical_issues_count": result.get("result", {}).get("critical_issues_count", 0),
            "warnings_count": result.get("result", {}).get("warnings_count", 0),
            "models_monitored": result.get("result", {}).get("models_monitored", 0),
            "model_alerts_count": result.get("result", {}).get("model_alerts_count", 0),
            "duration_seconds": result.get("result", {}).get("duration_seconds", 0),
        },
        message=(
            f"Audited {result.get('result', {}).get('experiments_audited', 0)} experiments, "
            f"flagged {result.get('result', {}).get('experiments_flagged', 0)} with issues"
        ),
    )


# =============================================================================
# Alpha Researcher Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_alpha_researcher(
    hypothesis_ids: Optional[list[str]] = None,
    write_research_note: bool = True,
) -> dict[str, Any]:
    """
    Run Alpha Researcher to review and refine draft hypotheses.

    The Alpha Researcher uses Claude to analyze hypotheses and add:
    - Economic rationale (why the signal might work)
    - Regime context (performance in different market conditions)
    - Related hypothesis search (novelty assessment)
    - Refined thesis and falsification criteria

    Hypotheses that pass review are promoted from "draft" to "testing" status.

    Args:
        hypothesis_ids: Specific hypothesis IDs to review (optional).
                       If not provided, reviews all hypotheses in "draft" status.
        write_research_note: Whether to write research note to ~/hrp-data/output/research/ (default True)

    Returns:
        Summary of reviewed hypotheses with recommendations
    """
    from hrp.agents.alpha_researcher import AlphaResearcher, AlphaResearcherConfig

    config = AlphaResearcherConfig(
        hypothesis_ids=hypothesis_ids,
        write_research_note=write_research_note,
    )

    researcher = AlphaResearcher(config=config)
    result = researcher.run()

    return format_response(
        success=True,
        data={
            "hypotheses_reviewed": result.get("hypotheses_reviewed", 0),
            "hypotheses_promoted": result.get("hypotheses_promoted", 0),
            "hypotheses_deferred": result.get("hypotheses_deferred", 0),
            "analyses": result.get("analyses", []),
            "research_note_path": result.get("research_note_path"),
            "token_usage": result.get("token_usage", {}),
        },
        message=(
            f"Reviewed {result.get('hypotheses_reviewed', 0)} hypotheses, "
            f"promoted {result.get('hypotheses_promoted', 0)} to testing"
        ),
    )


# =============================================================================
# Report Generator Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_report_generator(
    report_type: str = "daily",
) -> dict[str, Any]:
    """
    Generate a research report (daily or weekly) synthesizing platform activity.

    The Report Generator aggregates data from:
    - Hypothesis pipeline (draft, testing, validated, deployed counts)
    - MLflow experiments (top performers, model statistics)
    - Signal discoveries (recent findings from Signal Scientist)
    - Agent activity (recent runs of all research agents)
    - AI-powered insights (actionable recommendations)

    Reports are written to ~/hrp-data/output/reports/YYYY-MM-DD/HH-MM-{daily,weekly}.md

    Args:
        report_type: Type of report to generate ("daily" or "weekly", default "daily")

    Returns:
        Summary with report path, token usage, and key metrics
    """
    from hrp.agents.report_generator import ReportGenerator

    generator = ReportGenerator(report_type=report_type)
    result = generator.run()

    return format_response(
        success=True,
        data={
            "report_path": result.get("report_path"),
            "report_type": result.get("report_type"),
            "token_usage": result.get("token_usage", {}),
        },
        message=(
            f"Generated {result.get('report_type')} report: {result.get('report_path')}, "
            f"{result['token_usage']['total']} tokens (${result['token_usage']['estimated_cost_usd']:.4f})"
        ),
    )


# =============================================================================
# Signal Scientist Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_signal_scientist() -> dict[str, Any]:
    """
    Run Signal Scientist to scan for predictive signals and create draft hypotheses.

    The Signal Scientist performs systematic IC (Information Coefficient) analysis
    across all features to identify those with predictive power for forward returns.
    When promising signals are found, it creates draft hypotheses for review.

    Features analyzed:
    - Single-factor IC analysis across all 44+ features
    - Two-factor combination scanning (momentum + value, etc.)
    - Multi-horizon testing (5, 10, 20 day returns)
    - Rolling IC calculation for robust signal detection

    Returns:
        Summary of signals found and hypotheses created
    """
    from hrp.agents.signal_scientist import SignalScientist

    agent = SignalScientist()
    result = agent.run()

    return format_response(
        success=result.get("status") != "failed",
        data={
            "signals_found": result.get("signals_found", 0),
            "hypotheses_created": result.get("hypotheses_created", []),
            "features_scanned": result.get("features_scanned", 0),
            "scan_date": result.get("scan_date"),
        },
        message=(
            f"Scanned {result.get('features_scanned', 0)} features, "
            f"found {result.get('signals_found', 0)} signals, "
            f"created {len(result.get('hypotheses_created', []))} hypotheses"
        ),
    )


# =============================================================================
# ML Scientist Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_ml_scientist(
    hypothesis_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run ML Scientist to train and validate ML models for hypotheses in testing status.

    The ML Scientist takes hypotheses created by the Signal Scientist (or manually)
    and systematically trains ML models using walk-forward validation. It identifies
    the best model/feature combinations and updates hypothesis status based on
    statistical rigor.

    Features:
    - Multi-model type testing (ridge, lasso, lightgbm)
    - Walk-forward validation with stability scoring
    - Feature combination search
    - Hyperparameter optimization with trial budget
    - Automatic hypothesis status updates (testing -> validated/rejected)

    Args:
        hypothesis_ids: Specific hypothesis IDs to process (optional).
                       If not provided, processes all hypotheses in "testing" status.

    Returns:
        Summary of hypotheses validated and experiments run
    """
    from hrp.agents.ml_scientist import MLScientist

    agent = MLScientist(hypothesis_ids=hypothesis_ids)
    result = agent.run()

    return format_response(
        success=result.get("status") != "failed",
        data={
            "hypotheses_validated": result.get("hypotheses_validated", 0),
            "hypotheses_rejected": result.get("hypotheses_rejected", 0),
            "experiments_run": result.get("total_trials", 0),
            "hypotheses_processed": result.get("hypotheses_processed", 0),
        },
        message=(
            f"Processed {result.get('hypotheses_processed', 0)} hypotheses, "
            f"validated {result.get('hypotheses_validated', 0)}, "
            f"rejected {result.get('hypotheses_rejected', 0)}"
        ),
    )


# =============================================================================
# Quant Developer Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_quant_developer(
    hypothesis_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run Quant Developer to produce deployment-ready backtests for validated hypotheses.

    The Quant Developer takes hypotheses that have passed ML validation and produces
    comprehensive backtest packages including:
    - Model retraining on full historical data
    - ML-predicted signal generation (rank-based selection)
    - VectorBT backtest with realistic IBKR costs
    - Parameter variations (lookback, signal thresholds)
    - Time period and regime split analysis
    - Trade statistics for cost analysis

    Args:
        hypothesis_ids: Specific hypothesis IDs to backtest (optional).
                       If not provided, processes all hypotheses that passed ML audit.

    Returns:
        Summary of backtests completed
    """
    from hrp.agents.quant_developer import QuantDeveloper

    agent = QuantDeveloper(hypothesis_ids=hypothesis_ids)
    result = agent.run()

    return format_response(
        success=result.get("status") != "failed",
        data={
            "backtests_completed": len(result.get("backtests", [])),
            "backtests_failed": result.get("backtests_failed", 0),
            "hypotheses_processed": result.get("hypotheses_processed", 0),
        },
        message=(
            f"Completed {len(result.get('backtests', []))} backtests "
            f"for {result.get('hypotheses_processed', 0)} hypotheses"
        ),
    )


# =============================================================================
# Kill Gate Enforcer Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_kill_gate_enforcer(
    hypothesis_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run Kill Gate Enforcer to apply early termination gates and save compute.

    The Kill Gate Enforcer applies hard quality thresholds to terminate unpromising
    research early, saving compute resources for promising hypotheses.

    Kill Gates Applied:
    1. Baseline Sharpe - Must exceed equal-weight baseline (min 0.5)
    2. Train Sharpe - Flags suspiciously high training Sharpe (>3.0)
    3. Max Drawdown - Rejects if drawdown exceeds 30%
    4. Feature Count - Flags >50 features (overfitting risk)
    5. Instability - Flags high fold-to-fold variance

    Args:
        hypothesis_ids: Specific hypothesis IDs to evaluate (optional).
                       If not provided, evaluates all hypotheses ready for orchestration.

    Returns:
        Summary of hypotheses checked and gate pass/fail counts
    """
    from hrp.agents.kill_gate_enforcer import KillGateEnforcer, KillGateEnforcerConfig

    config = KillGateEnforcerConfig(hypothesis_ids=hypothesis_ids)
    agent = KillGateEnforcer(hypothesis_ids=hypothesis_ids, config=config)
    result = agent.run()

    report = result.get("report", {})
    return format_response(
        success=result.get("status") != "failed",
        data={
            "hypotheses_checked": report.get("hypotheses_processed", 0),
            "gates_passed": report.get("hypotheses_processed", 0) - report.get("hypotheses_killed", 0),
            "gates_failed": report.get("hypotheses_killed", 0),
            "experiments_run": report.get("experiments_run", 0),
            "time_saved_seconds": report.get("time_saved_seconds", 0),
        },
        message=(
            f"Checked {report.get('hypotheses_processed', 0)} hypotheses, "
            f"killed {report.get('hypotheses_killed', 0)} at gates"
        ),
    )


# =============================================================================
# Validation Analyst Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_validation_analyst(
    hypothesis_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run Validation Analyst to stress test hypotheses before deployment approval.

    The Validation Analyst performs pre-deployment stress testing including:
    - Parameter sensitivity - Tests stability under parameter changes
    - Time stability - Verifies consistent performance across periods
    - Regime analysis - Checks performance in bull/bear/sideways markets
    - Execution cost estimation - Calculates realistic transaction costs

    Hypotheses that fail validation are demoted back to "testing" status.

    Args:
        hypothesis_ids: Specific hypothesis IDs to validate (optional).
                       If not provided, validates all hypotheses ready for final review.

    Returns:
        Summary of stress tests run and pass/fail counts
    """
    from hrp.agents.validation_analyst import ValidationAnalyst

    agent = ValidationAnalyst(hypothesis_ids=hypothesis_ids, send_alerts=False)
    result = agent.run()

    return format_response(
        success=result.get("status") != "failed",
        data={
            "stress_tests_run": result.get("hypotheses_validated", 0),
            "passed": result.get("hypotheses_passed", 0),
            "failed": result.get("hypotheses_failed", 0),
            "validations": result.get("validations_summary", []),
        },
        message=(
            f"Validated {result.get('hypotheses_validated', 0)} hypotheses, "
            f"passed {result.get('hypotheses_passed', 0)}, "
            f"failed {result.get('hypotheses_failed', 0)}"
        ),
    )


# =============================================================================
# Risk Manager Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_risk_manager(
    hypothesis_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run Risk Manager to assess portfolio-level risk before deployment.

    The Risk Manager provides independent portfolio risk oversight:
    - Drawdown risk assessment - Max drawdown limits, duration
    - Concentration risk - Position sizes, sector exposure, correlation
    - Portfolio fit - Correlation with existing positions, diversification value
    - Risk limits validation - Position limits, turnover, leverage

    Can veto strategies but CANNOT approve deployment (only human CIO can approve).

    Args:
        hypothesis_ids: Specific hypothesis IDs to assess (optional).
                       If not provided, assesses all hypotheses awaiting risk review.

    Returns:
        Summary of assessments made and vetoes issued
    """
    from hrp.agents.risk_manager import RiskManager

    agent = RiskManager(hypothesis_ids=hypothesis_ids, send_alerts=False)
    result = agent.run()

    return format_response(
        success=result.get("status") != "failed",
        data={
            "assessments_made": result.get("hypotheses_assessed", 0),
            "vetoes_issued": result.get("hypotheses_vetoed", 0),
            "hypotheses_passed": result.get("hypotheses_passed", 0),
        },
        message=(
            f"Assessed {result.get('hypotheses_assessed', 0)} hypotheses, "
            f"vetoed {result.get('hypotheses_vetoed', 0)}"
        ),
    )


# =============================================================================
# CIO Agent
# =============================================================================


@mcp.tool()
@handle_api_error
def run_cio_agent(
    hypothesis_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run CIO Agent to score hypotheses across 4 dimensions and make strategic decisions.

    The CIO Agent acts as Chief Investment Officer, making strategic decisions about
    research lines. It scores each hypothesis across 4 dimensions:
    - Statistical quality (Sharpe, stability, IC)
    - Risk profile (drawdown, correlation, concentration)
    - Economic rationale (why the signal might work)
    - Cost realism (transaction costs, capacity)

    Decisions: CONTINUE (>=0.75), CONDITIONAL (0.50-0.74), KILL (<0.50), PIVOT (critical failure)

    NOTE: CIO Agent is advisory only - only human CIO can approve deployment.

    Args:
        hypothesis_ids: Specific hypothesis IDs to review (optional).
                       If not provided, reviews all validated hypotheses awaiting CIO review.

    Returns:
        Summary of decisions made and counts by decision type
    """
    from hrp.agents.cio import CIOAgent

    agent = CIOAgent(
        job_id="cio_agent_review",
        actor="agent:cio",
    )
    result = agent.run()

    decisions = result.get("decisions", [])
    continue_count = sum(1 for d in decisions if d.get("decision") == "CONTINUE")
    kill_count = sum(1 for d in decisions if d.get("decision") == "KILL")
    conditional_count = sum(1 for d in decisions if d.get("decision") == "CONDITIONAL")
    pivot_count = sum(1 for d in decisions if d.get("decision") == "PIVOT")

    return format_response(
        success=result.get("status") != "failed",
        data={
            "decisions_made": len(decisions),
            "continue_count": continue_count,
            "conditional_count": conditional_count,
            "kill_count": kill_count,
            "pivot_count": pivot_count,
            "report_path": result.get("report_path"),
        },
        message=(
            f"Made {len(decisions)} decisions: "
            f"{continue_count} CONTINUE, {conditional_count} CONDITIONAL, "
            f"{kill_count} KILL, {pivot_count} PIVOT"
        ),
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the MCP server."""
    import sys

    # Configure loguru to write to stderr (stdout must be clean for MCP JSON-RPC)
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("Starting HRP Research MCP Server")
    mcp.run()


if __name__ == "__main__":
    main()
