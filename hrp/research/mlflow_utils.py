"""
MLflow integration for experiment tracking.
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

import mlflow
import pandas as pd
from loguru import logger

from hrp.research.config import BacktestConfig, BacktestResult


# Default MLflow directory
MLFLOW_DIR = Path.home() / "hrp-data" / "mlflow"


def setup_mlflow(tracking_uri: str = None) -> None:
    """
    Setup MLflow tracking.

    Args:
        tracking_uri: MLflow tracking URI (default: local sqlite)
    """
    if tracking_uri is None:
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{MLFLOW_DIR}/mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")


def get_or_create_experiment(name: str) -> str:
    """
    Get or create an MLflow experiment.

    Args:
        name: Experiment name

    Returns:
        Experiment ID
    """
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name,
            artifact_location=str(MLFLOW_DIR / "artifacts" / name),
        )
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.debug(f"Using existing experiment: {name} (ID: {experiment_id})")

    return experiment_id


def log_backtest(
    result: BacktestResult,
    experiment_name: str = "backtests",
    run_name: str = None,
    hypothesis_id: str = None,
    tags: dict = None,
    feature_versions: dict[str, str] = None,
) -> str:
    """
    Log a backtest result to MLflow.

    Args:
        result: BacktestResult to log
        experiment_name: MLflow experiment name
        run_name: Optional run name
        hypothesis_id: Optional linked hypothesis ID
        tags: Additional tags
        feature_versions: Dict mapping feature names to versions

    Returns:
        MLflow run ID
    """
    setup_mlflow()
    experiment_id = get_or_create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters from config
        config = result.config
        mlflow.log_param("symbols", ",".join(config.symbols[:10]))  # Limit length
        mlflow.log_param("num_symbols", len(config.symbols))
        mlflow.log_param("start_date", str(config.start_date))
        mlflow.log_param("end_date", str(config.end_date))
        mlflow.log_param("sizing_method", config.sizing_method)
        mlflow.log_param("max_positions", config.max_positions)
        mlflow.log_param("max_position_pct", config.max_position_pct)

        # Log cost model
        mlflow.log_param("spread_bps", config.costs.spread_bps)
        mlflow.log_param("slippage_bps", config.costs.slippage_bps)

        # Log feature versions
        if feature_versions:
            mlflow.log_param("feature_versions", json.dumps(feature_versions))
            for feature_name, version in feature_versions.items():
                mlflow.log_param(f"feature_version_{feature_name}", version)

        # Log metrics
        for metric_name, metric_value in result.metrics.items():
            if isinstance(metric_value, (int, float)) and not pd.isna(metric_value):
                mlflow.log_metric(metric_name, metric_value)

        # Log benchmark metrics with prefix
        if result.benchmark_metrics:
            for metric_name, metric_value in result.benchmark_metrics.items():
                if isinstance(metric_value, (int, float)) and not pd.isna(metric_value):
                    mlflow.log_metric(f"benchmark_{metric_name}", metric_value)

        # Log tags
        if hypothesis_id:
            mlflow.set_tag("hypothesis_id", hypothesis_id)
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, str(value))

        # Log equity curve as artifact
        if result.equity_curve is not None and len(result.equity_curve) > 0:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 6))
                result.equity_curve.plot(ax=ax, label='Strategy')
                ax.set_title('Equity Curve')
                ax.set_xlabel('Date')
                ax.set_ylabel('Portfolio Value')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Save and log
                equity_path = MLFLOW_DIR / "temp_equity.png"
                fig.savefig(equity_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                mlflow.log_artifact(str(equity_path), "plots")
                equity_path.unlink()  # Clean up

            except Exception as e:
                logger.warning(f"Could not save equity curve plot: {e}")

        # Log trades as CSV
        if result.trades is not None and len(result.trades) > 0:
            try:
                trades_path = MLFLOW_DIR / "temp_trades.csv"
                result.trades.to_csv(trades_path, index=False)
                mlflow.log_artifact(str(trades_path), "data")
                trades_path.unlink()
            except Exception as e:
                logger.warning(f"Could not save trades: {e}")

        run_id = run.info.run_id
        logger.info(f"Logged backtest to MLflow run: {run_id}")

        return run_id


def get_best_runs(
    experiment_name: str,
    metric: str = "sharpe_ratio",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get top runs by a metric.

    Args:
        experiment_name: Experiment name
        metric: Metric to sort by
        top_n: Number of runs to return

    Returns:
        DataFrame with run info and metrics
    """
    setup_mlflow()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
    )

    return runs
