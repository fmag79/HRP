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

# Cached API instance
_api: Optional[PlatformAPI] = None


def get_api() -> PlatformAPI:
    """Get or create the PlatformAPI instance."""
    global _api
    if _api is None:
        _api = PlatformAPI()
        logger.info("PlatformAPI initialized for MCP server")
    return _api


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
    api = get_api()
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
    api = get_api()
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
    api = get_api()
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
    api = get_api()
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
    api = get_api()
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
    api = get_api()
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
    api = get_api()
    target_date = parse_date(as_of_date)

    if target_date is None:
        return format_response(
            success=False,
            message="as_of_date is required",
            error="Missing required parameter: as_of_date",
        )

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
    api = get_api()
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        return format_response(
            success=False,
            message="Both start_date and end_date are required",
            error="Missing required date parameters",
        )

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
    from hrp.data.features.registry import FeatureRegistry

    registry = FeatureRegistry()
    features = registry.list_all_features(active_only=True)

    # Format output
    formatted = [
        {
            "feature_name": f["feature_name"],
            "version": f["version"],
            "description": f.get("description", "No description"),
        }
        for f in features
    ]

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
    api = get_api()
    target_date = parse_date(check_date)

    if target_date is None:
        return format_response(
            success=False,
            message="check_date is required",
            error="Missing required parameter: check_date",
        )

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

    api = get_api()
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
    api = get_api()
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
    api = get_api()
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
    api = get_api()
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
    from hrp.data.quality.report import QualityReportGenerator

    target_date = parse_date(as_of_date) or date.today()

    generator = QualityReportGenerator()
    report = generator.generate_report(target_date)

    return format_response(
        success=True,
        data={
            "report_date": str(report.report_date),
            "health_score": report.health_score,
            "passed": report.passed,
            "checks_run": report.checks_run,
            "checks_passed": report.checks_passed,
            "critical_issues": report.critical_issues,
            "warning_issues": report.warning_issues,
            "summary": report.get_summary_text(),
        },
        message=f"Health score: {report.health_score:.0f}/100",
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
    api = get_api()
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
    api = get_api()
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        return format_response(
            success=False,
            message="Both start_date and end_date are required",
            error="Missing required date parameters",
        )

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
    api = get_api()
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
    api = get_api()
    strategies = api.get_deployed_strategies()

    formatted = [format_hypothesis(s) for s in strategies]

    return format_response(
        success=True,
        data=formatted,
        message=f"Found {len(formatted)} deployed strategies",
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
