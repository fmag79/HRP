"""
Cross-validated optimization for ML models.

Provides parameterized cross-validation optimization that integrates with
existing walk-forward validation and overfitting guards.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
import optuna
from optuna.distributions import (
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
from optuna.samplers import GridSampler, RandomSampler, TPESampler, CmaEsSampler

from hrp.ml.models import SUPPORTED_MODELS
from hrp.ml.validation import (
    FoldResult,
    WalkForwardConfig,
    generate_folds,
    compute_fold_metrics,
    aggregate_fold_metrics,
)
from hrp.ml.training import _fetch_features, select_features
from hrp.ml.models import get_model
from hrp.risk.overfitting import (
    HyperparameterTrialCounter,
    SharpeDecayMonitor,
    OverfittingError,
)


# Scoring metrics and their optimization direction (higher is better)
SCORING_METRICS = {
    "ic": True,  # Higher is better
    "r2": True,  # Higher is better
    "mse": False,  # Lower is better
    "mae": False,  # Lower is better
    "sharpe": True,  # Higher is better
}

# Valid Optuna samplers
VALID_SAMPLERS = {"grid", "random", "tpe", "cmaes"}


def _distributions_to_grid(
    param_space: dict[str, BaseDistribution],
) -> dict[str, list]:
    """
    Convert Optuna distributions to explicit grid for GridSampler.

    Args:
        param_space: Dict of param name to distribution

    Returns:
        Dict of param name to list of values

    Raises:
        ValueError: If FloatDistribution lacks step
    """
    grid = {}
    for name, dist in param_space.items():
        if isinstance(dist, CategoricalDistribution):
            grid[name] = list(dist.choices)
        elif isinstance(dist, IntDistribution):
            step = dist.step if dist.step else 1
            grid[name] = list(range(dist.low, dist.high + 1, step))
        elif isinstance(dist, FloatDistribution):
            if dist.step:
                values = []
                v = dist.low
                while v <= dist.high:
                    values.append(v)
                    v += dist.step
                grid[name] = values
            else:
                raise ValueError(
                    f"Grid sampler requires step for float param '{name}'. "
                    f"Use FloatDistribution({dist.low}, {dist.high}, step=X)"
                )
        else:
            raise ValueError(f"Unsupported distribution type for '{name}': {type(dist)}")
    return grid


def _get_sampler(
    sampler_name: str,
    param_space: dict[str, BaseDistribution],
) -> optuna.samplers.BaseSampler:
    """
    Create Optuna sampler from name.

    Args:
        sampler_name: One of 'grid', 'random', 'tpe', 'cmaes'
        param_space: Parameter space (needed for GridSampler)

    Returns:
        Configured Optuna sampler

    Raises:
        ValueError: If sampler_name invalid or grid requires step for floats
    """
    if sampler_name == "grid":
        search_space = _distributions_to_grid(param_space)
        return GridSampler(search_space)
    elif sampler_name == "random":
        return RandomSampler(seed=42)
    elif sampler_name == "tpe":
        return TPESampler(seed=42)
    elif sampler_name == "cmaes":
        return CmaEsSampler(seed=42)
    else:
        raise ValueError(
            f"Unknown sampler: '{sampler_name}'. "
            f"Options: grid, random, tpe, cmaes"
        )


@dataclass
class OptimizationConfig:
    """
    Configuration for cross-validated optimization using Optuna.

    Attributes:
        model_type: Type of model (must be in SUPPORTED_MODELS)
        target: Target variable name (e.g., 'returns_20d')
        features: List of feature names from feature store
        param_space: Optuna distributions for hyperparameters
        start_date: Start of the entire date range
        end_date: End of the entire date range
        n_folds: Number of cross-validation folds (default 5)
        window_type: 'expanding' or 'rolling' (default 'expanding')
        scoring_metric: Metric to optimize (default 'ic')
        n_trials: Number of optimization trials (default 50)
        hypothesis_id: Optional hypothesis ID for overfitting tracking
        sampler: Optuna sampler type ('grid', 'random', 'tpe', 'cmaes')
        enable_pruning: Enable Optuna pruning for early trial termination
        early_stop_decay_threshold: Sharpe decay threshold for early stopping
        min_train_periods: Minimum training periods for each fold
        feature_selection: Whether to perform feature selection
        max_features: Maximum features to select
    """

    model_type: str
    target: str
    features: list[str]
    param_space: dict[str, BaseDistribution]
    start_date: date
    end_date: date
    n_folds: int = 5
    window_type: str = "expanding"
    scoring_metric: str = "ic"
    n_trials: int = 50
    hypothesis_id: str | None = None
    sampler: str = "tpe"
    enable_pruning: bool = True
    early_stop_decay_threshold: float = 0.5
    min_train_periods: int = 252
    feature_selection: bool = True
    max_features: int = 20

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.model_type not in SUPPORTED_MODELS:
            available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
            raise ValueError(
                f"Unsupported model type: '{self.model_type}'. "
                f"Available: {available}"
            )
        if self.window_type not in ("expanding", "rolling"):
            raise ValueError(
                f"window_type must be 'expanding' or 'rolling', "
                f"got '{self.window_type}'"
            )
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.scoring_metric not in SCORING_METRICS:
            available = ", ".join(sorted(SCORING_METRICS.keys()))
            raise ValueError(
                f"Unsupported scoring_metric: '{self.scoring_metric}'. "
                f"Available: {available}"
            )
        if not self.param_space:
            raise ValueError("param_space cannot be empty")
        if self.sampler not in VALID_SAMPLERS:
            available = ", ".join(sorted(VALID_SAMPLERS))
            raise ValueError(
                f"Unknown sampler: '{self.sampler}'. "
                f"Available: {available}"
            )

        logger.debug(
            f"OptimizationConfig created: {self.model_type}, "
            f"{self.n_folds} folds, scoring={self.scoring_metric}, sampler={self.sampler}"
        )


@dataclass
class OptimizationResult:
    """
    Result of cross-validated optimization.

    Attributes:
        best_params: Best hyperparameters found
        best_score: Best score achieved
        cv_results: DataFrame with all trial results
        fold_results: List of FoldResult for best params
        all_trials: List of all trial dictionaries
        hypothesis_id: Hypothesis ID if provided
        n_trials_completed: Number of trials completed
        early_stopped: Whether optimization was early stopped
        early_stop_reason: Reason for early stopping if applicable
        execution_time_seconds: Total execution time
    """

    best_params: dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    fold_results: list[FoldResult]
    all_trials: list[dict]
    hypothesis_id: str | None
    n_trials_completed: int = 0
    early_stopped: bool = False
    early_stop_reason: str | None = None
    execution_time_seconds: float = 0.0


def _generate_param_combinations(
    param_grid: dict[str, list[Any]],
    search_type: str = "grid",
    n_random_samples: int = 20,
    max_trials: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate parameter combinations from grid.

    Args:
        param_grid: Dict mapping param name to list of values
        search_type: 'grid' for full grid, 'random' for random sampling
        n_random_samples: Number of samples for random search
        max_trials: Maximum trials to generate

    Returns:
        List of parameter dictionaries
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    if search_type == "grid":
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        # Random search
        np.random.seed(42)  # For reproducibility
        combinations = []
        for _ in range(n_random_samples):
            combo = {k: np.random.choice(v) for k, v in param_grid.items()}
            combinations.append(combo)

    # Limit by max_trials if specified
    if max_trials is not None and len(combinations) > max_trials:
        combinations = combinations[:max_trials]

    return combinations


def _evaluate_params(
    params: dict[str, Any],
    config: OptimizationConfig,
    all_data: pd.DataFrame,
    folds: list[tuple[date, date, date, date]],
) -> tuple[float, list[FoldResult], dict[str, float]]:
    """
    Evaluate a single parameter combination across all folds.

    Returns:
        Tuple of (mean_score, fold_results, per_fold_scores)
    """
    fold_results = []
    fold_scores = []
    train_sharpes = []
    test_sharpes = []

    higher_is_better = SCORING_METRICS[config.scoring_metric]

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        # Split data
        dates = all_data.index.get_level_values("date")
        if hasattr(dates[0], "date"):
            train_mask = (dates.date >= train_start) & (dates.date <= train_end)
            test_mask = (dates.date >= test_start) & (dates.date <= test_end)
        else:
            train_mask = (dates >= pd.Timestamp(train_start)) & (
                dates <= pd.Timestamp(train_end)
            )
            test_mask = (dates >= pd.Timestamp(test_start)) & (
                dates <= pd.Timestamp(test_end)
            )

        train_data = all_data.loc[train_mask]
        test_data = all_data.loc[test_mask]

        if len(train_data) == 0 or len(test_data) == 0:
            continue

        X_train = train_data[config.features]
        y_train = train_data[config.target]
        X_test = test_data[config.features]
        y_test = test_data[config.target]

        # Feature selection
        selected_features = list(config.features)
        if config.feature_selection and len(config.features) > config.max_features:
            selected_features = select_features(X_train, y_train, config.max_features)
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]

        # Train model with params
        model = get_model(config.model_type, params)
        model.fit(X_train, y_train)

        # Predict on train and test
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        train_metrics = compute_fold_metrics(y_train, y_train_pred)
        test_metrics = compute_fold_metrics(y_test, y_test_pred)

        # Track Sharpe proxy (using IC as proxy for factor investing)
        train_sharpes.append(train_metrics.get("ic", 0.0))
        test_sharpes.append(test_metrics.get("ic", 0.0))

        # Extract scoring metric
        score = test_metrics.get(config.scoring_metric, float("nan"))
        if not np.isnan(score):
            fold_scores.append(score)

        fold_result = FoldResult(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            metrics=test_metrics,
            model=model,
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
        )
        fold_results.append(fold_result)

    if not fold_scores:
        mean_score = float("-inf") if higher_is_better else float("inf")
    else:
        mean_score = float(np.mean(fold_scores))

    per_fold_scores = {
        "train_sharpes": train_sharpes,
        "test_sharpes": test_sharpes,
        "fold_scores": fold_scores,
    }

    return mean_score, fold_results, per_fold_scores


def _evaluate_with_pruning(
    trial: optuna.Trial,
    params: dict[str, Any],
    config: OptimizationConfig,
    all_data: pd.DataFrame,
    folds: list[tuple[date, date, date, date]],
) -> tuple[float, list[FoldResult]]:
    """
    Evaluate a parameter combination with Optuna pruning support.

    Reports intermediate values after each fold and checks for pruning signals.
    Also monitors Sharpe decay to catch overfitting early.

    Args:
        trial: Optuna trial object for reporting and pruning
        params: Hyperparameters to evaluate
        config: Optimization configuration
        all_data: Full dataset with features and target
        folds: List of (train_start, train_end, test_start, test_end) tuples

    Returns:
        Tuple of (mean_score, fold_results)

    Raises:
        optuna.TrialPruned: If trial should be pruned based on intermediate results
                           or Sharpe decay threshold exceeded
    """
    fold_results = []
    fold_scores = []
    train_sharpes = []
    test_sharpes = []

    higher_is_better = SCORING_METRICS[config.scoring_metric]
    decay_monitor = SharpeDecayMonitor(max_decay_ratio=config.early_stop_decay_threshold)

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        # Split data
        dates = all_data.index.get_level_values("date")
        if hasattr(dates[0], "date"):
            train_mask = (dates.date >= train_start) & (dates.date <= train_end)
            test_mask = (dates.date >= test_start) & (dates.date <= test_end)
        else:
            train_mask = (dates >= pd.Timestamp(train_start)) & (
                dates <= pd.Timestamp(train_end)
            )
            test_mask = (dates >= pd.Timestamp(test_start)) & (
                dates <= pd.Timestamp(test_end)
            )

        train_data = all_data.loc[train_mask]
        test_data = all_data.loc[test_mask]

        if len(train_data) == 0 or len(test_data) == 0:
            continue

        X_train = train_data[config.features]
        y_train = train_data[config.target]
        X_test = test_data[config.features]
        y_test = test_data[config.target]

        # Feature selection
        selected_features = list(config.features)
        if config.feature_selection and len(config.features) > config.max_features:
            selected_features = select_features(X_train, y_train, config.max_features)
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]

        # Train model with params
        model = get_model(config.model_type, params)
        model.fit(X_train, y_train)

        # Predict on train and test
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        train_metrics = compute_fold_metrics(y_train, y_train_pred)
        test_metrics = compute_fold_metrics(y_test, y_test_pred)

        # Track Sharpe proxy (using IC as proxy for factor investing)
        train_sharpes.append(train_metrics.get("ic", 0.0))
        test_sharpes.append(test_metrics.get("ic", 0.0))

        # Extract scoring metric
        score = test_metrics.get(config.scoring_metric, float("nan"))
        if not np.isnan(score):
            fold_scores.append(score)

        fold_result = FoldResult(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            metrics=test_metrics,
            model=model,
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
        )
        fold_results.append(fold_result)

        # Report intermediate value to Optuna
        current_mean = float(np.mean(fold_scores)) if fold_scores else 0.0
        trial.report(current_mean, fold_idx)

        # Check if trial should be pruned
        if trial.should_prune():
            logger.debug(f"Trial pruned at fold {fold_idx}")
            raise optuna.TrialPruned()

        # Check Sharpe decay
        if train_sharpes and test_sharpes:
            mean_train_sharpe = float(np.mean(train_sharpes))
            mean_test_sharpe = float(np.mean(test_sharpes))
            decay_result = decay_monitor.check(mean_train_sharpe, mean_test_sharpe)
            if not decay_result.passed:
                logger.debug(f"Trial pruned due to Sharpe decay: {decay_result.message}")
                raise optuna.TrialPruned()

    if not fold_scores:
        mean_score = float("-inf") if higher_is_better else float("inf")
    else:
        mean_score = float(np.mean(fold_scores))

    return mean_score, fold_results


def cross_validated_optimize(
    config: OptimizationConfig,
    symbols: list[str],
    log_to_mlflow: bool = True,
) -> OptimizationResult:
    """
    Run cross-validated parameter optimization.

    Combines grid/random search with walk-forward validation and
    integrates with overfitting guards.

    Args:
        config: Optimization configuration
        symbols: List of symbols to use
        log_to_mlflow: Whether to log results to MLflow

    Returns:
        OptimizationResult with best parameters and all trial results

    Raises:
        OverfittingError: If trial limit exceeded
        ValueError: If no data found
    """
    logger.info(
        f"Starting cross-validated optimization: {config.model_type}, "
        f"{config.n_folds} folds, metric={config.scoring_metric}"
    )

    start_time = time.perf_counter()

    # Initialize overfitting guards
    trial_counter = None
    if config.hypothesis_id:
        trial_counter = HyperparameterTrialCounter(
            hypothesis_id=config.hypothesis_id,
            max_trials=config.max_trials,
        )

    decay_monitor = SharpeDecayMonitor(max_decay_ratio=config.early_stop_decay_threshold)

    # Fetch all data once
    all_data = _fetch_features(
        symbols=symbols,
        features=config.features,
        start_date=config.start_date,
        end_date=config.end_date,
        target=config.target,
    )

    if all_data.empty:
        raise ValueError(f"No data found for symbols {symbols}")

    all_data = all_data.dropna()

    # Get available dates
    available_dates = sorted(all_data.index.get_level_values("date").unique())
    available_dates = [d.date() if hasattr(d, "date") else d for d in available_dates]

    # Generate fold date ranges using WalkForwardConfig
    wf_config = WalkForwardConfig(
        model_type=config.model_type,
        target=config.target,
        features=config.features,
        start_date=config.start_date,
        end_date=config.end_date,
        n_folds=config.n_folds,
        window_type=config.window_type,
        min_train_periods=config.min_train_periods,
    )
    folds = generate_folds(wf_config, available_dates)

    # Generate parameter combinations
    param_combinations = _generate_param_combinations(
        config.param_grid,
        config.search_type,
        config.n_random_samples,
        config.max_trials,
    )

    logger.info(f"Evaluating {len(param_combinations)} parameter combinations")

    # Track results
    all_trials = []
    best_params = None
    best_score = float("-inf") if SCORING_METRICS[config.scoring_metric] else float("inf")
    best_fold_results = []
    early_stopped = False
    early_stop_reason = None

    higher_is_better = SCORING_METRICS[config.scoring_metric]

    for trial_idx, params in enumerate(param_combinations):
        # Check trial counter
        if trial_counter and not trial_counter.can_try():
            early_stopped = True
            early_stop_reason = f"Trial limit ({config.max_trials}) exceeded"
            logger.warning(early_stop_reason)
            break

        logger.debug(f"Trial {trial_idx + 1}/{len(param_combinations)}: {params}")

        # Evaluate params
        mean_score, fold_results, per_fold_scores = _evaluate_params(
            params, config, all_data, folds
        )

        # Check for Sharpe decay (early stopping)
        if per_fold_scores["train_sharpes"] and per_fold_scores["test_sharpes"]:
            mean_train_sharpe = float(np.mean(per_fold_scores["train_sharpes"]))
            mean_test_sharpe = float(np.mean(per_fold_scores["test_sharpes"]))

            decay_result = decay_monitor.check(mean_train_sharpe, mean_test_sharpe)
            if not decay_result.passed:
                logger.warning(f"Trial {trial_idx + 1}: {decay_result.message}")

        # Log trial
        trial_record = {
            "trial_idx": trial_idx,
            "params": params,
            "mean_score": mean_score,
            "fold_scores": per_fold_scores["fold_scores"],
            "train_sharpes": per_fold_scores["train_sharpes"],
            "test_sharpes": per_fold_scores["test_sharpes"],
        }
        all_trials.append(trial_record)

        # Log to trial counter if available
        if trial_counter:
            try:
                trial_counter.log_trial(
                    model_type=config.model_type,
                    hyperparameters=params,
                    metric_name=config.scoring_metric,
                    metric_value=mean_score,
                )
            except OverfittingError:
                early_stopped = True
                early_stop_reason = "Trial limit exceeded via counter"
                break

        # Update best
        is_better = (
            (mean_score > best_score)
            if higher_is_better
            else (mean_score < best_score)
        )
        if is_better and not np.isnan(mean_score) and not np.isinf(mean_score):
            best_score = mean_score
            best_params = params
            best_fold_results = fold_results

    # Build CV results DataFrame
    cv_results_data = []
    for trial in all_trials:
        row = {"mean_score": trial["mean_score"]}
        row.update({f"param_{k}": v for k, v in trial["params"].items()})
        cv_results_data.append(row)

    cv_results = pd.DataFrame(cv_results_data)

    execution_time = time.perf_counter() - start_time

    result = OptimizationResult(
        best_params=best_params or {},
        best_score=best_score,
        cv_results=cv_results,
        fold_results=best_fold_results,
        all_trials=all_trials,
        hypothesis_id=config.hypothesis_id,
        n_trials_completed=len(all_trials),
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        execution_time_seconds=execution_time,
    )

    logger.info(
        f"Optimization complete: best_score={best_score:.4f}, "
        f"best_params={best_params}, trials={len(all_trials)}"
    )

    if log_to_mlflow:
        _log_optimization_to_mlflow(result, config)

    return result


def _log_optimization_to_mlflow(
    result: OptimizationResult,
    config: OptimizationConfig,
) -> None:
    """Log optimization results to MLflow."""
    try:
        import mlflow

        experiment_name = f"hrp_optimization_{config.model_type}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"opt_{config.scoring_metric}"):
            # Log config
            mlflow.log_param("model_type", config.model_type)
            mlflow.log_param("n_folds", config.n_folds)
            mlflow.log_param("window_type", config.window_type)
            mlflow.log_param("scoring_metric", config.scoring_metric)
            mlflow.log_param("search_type", config.search_type)
            mlflow.log_param("n_trials", result.n_trials_completed)

            # Log best params
            for k, v in result.best_params.items():
                mlflow.log_param(f"best_{k}", v)

            # Log metrics
            mlflow.log_metric("best_score", result.best_score)
            mlflow.log_metric("execution_time_seconds", result.execution_time_seconds)

            if result.early_stopped:
                mlflow.log_param("early_stopped", True)
                mlflow.log_param("early_stop_reason", result.early_stop_reason)

            # Log CV results as artifact
            mlflow.log_dict(
                {"all_trials": [t["params"] for t in result.all_trials]},
                "all_trials.json",
            )

        logger.info(f"Logged optimization results to MLflow: {experiment_name}")

    except ImportError:
        logger.warning("MLflow not installed, skipping logging")
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
