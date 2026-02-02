"""
Walk-forward validation for ML models.

Provides temporal cross-validation to assess model robustness
and detect overfitting across multiple time periods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

import time

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hrp.ml.models import SUPPORTED_MODELS, get_model
from hrp.ml.training import _fetch_features, select_features
from hrp.utils.timing import TimingMetrics, timed_section


# Module-level cache for feature selection results
# Key: "fold_{fold_idx}_{window_type}", Value: list of selected features
_feature_selection_cache: dict[str, list[str]] = {}


@lru_cache(maxsize=128)
def select_features_cached(
    X_hash: int,
    y_hash: int,
    max_features: int,
) -> tuple[str, ...]:
    """
    Feature selection with caching using lru_cache.

    This function provides a simplified caching mechanism for feature selection.
    It uses hashes of the input data as cache keys, making it suitable for use
    with functools.lru_cache.

    Args:
        X_hash: Hash of X DataFrame values (use hash(X.values.tobytes()))
        y_hash: Hash of y Series values (use hash(y.values.tobytes()))
        max_features: Maximum number of features to select

    Returns:
        Tuple of selected feature names (tuple is hashable for caching)

    Note:
        This is a simplified version that delegates to select_features.
        The caller is responsible for computing the hashes of X and y.
    """
    # Import select_features here to avoid issues with the caching decorator
    from hrp.ml.training import select_features as _select_features

    # We can't reconstruct X and y from hashes, so we return an empty tuple
    # This function is meant to be used differently - see the module-level cache
    # (_feature_selection_cache) for the actual caching implementation.
    logger.warning(
        "select_features_cached called but cannot reconstruct data from hashes. "
        "Use _feature_selection_cache dict directly instead."
    )
    return ()


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.

    Attributes:
        model_type: Type of model (must be in SUPPORTED_MODELS)
        target: Target variable name (e.g., 'returns_20d')
        features: List of feature names from feature store
        start_date: Start of the entire date range
        end_date: End of the entire date range
        n_folds: Number of walk-forward folds (default 5)
        window_type: 'expanding' or 'rolling' (default 'expanding')
        min_train_periods: Minimum training samples per fold (default 252)
        hyperparameters: Model-specific hyperparameters
        feature_selection: Whether to apply feature selection per fold
        max_features: Maximum features to select per fold
        purge_days: Days gap between train end and test start (prevents temporal leakage)
        embargo_days: Initial days of test period excluded from metric computation
    """

    model_type: str
    target: str
    features: list[str]
    start_date: date
    end_date: date
    n_folds: int = 5
    window_type: str = "expanding"
    min_train_periods: int = 252
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_selection: bool = True
    max_features: int = 20
    n_jobs: int = 1  # Number of parallel jobs (-1 = use all cores)
    purge_days: int = 0  # Days between train and test (gap after training)
    embargo_days: int = 0  # Days excluded from test metrics
    tags: dict[str, str] = field(default_factory=dict)

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
        if self.purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {self.purge_days}")
        if self.embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {self.embargo_days}")
        if self.purge_days > self.min_train_periods:
            raise ValueError(
                f"purge_days ({self.purge_days}) "
                f"cannot exceed min_train_periods ({self.min_train_periods})"
            )

        logger.debug(
            f"WalkForwardConfig created: {self.model_type}, "
            f"{self.n_folds} folds, {self.window_type} window, "
            f"purge={self.purge_days}d, embargo={self.embargo_days}d"
        )


@dataclass
class FoldResult:
    """
    Result from a single walk-forward fold.

    Attributes:
        fold_index: Zero-based index of this fold
        train_start: Training period start date
        train_end: Training period end date
        test_start: Test period start date
        test_end: Test period end date
        metrics: Dict of evaluation metrics (mse, mae, r2, ic)
        model: Trained model object for this fold
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
    """

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    metrics: dict[str, float]
    model: Any
    n_train_samples: int
    n_test_samples: int


@dataclass
class WalkForwardResult:
    """
    Aggregated result from walk-forward validation.

    Attributes:
        config: Configuration used for validation
        fold_results: List of results from each fold
        aggregate_metrics: Mean and std of metrics across folds
        stability_score: Coefficient of variation (std/mean) of MSE
        symbols: List of symbols used in validation
    """

    config: WalkForwardConfig
    fold_results: list[FoldResult]
    aggregate_metrics: dict[str, float]
    stability_score: float
    symbols: list[str]
    mlflow_run_id: str | None = None

    @property
    def is_stable(self) -> bool:
        """Return True if model is stable (stability_score <= 1.0)."""
        return self.stability_score <= 1.0

    @property
    def mean_ic(self) -> float:
        """Return mean information coefficient across folds."""
        return self.aggregate_metrics.get("mean_ic", float("nan"))


def generate_folds(
    config: WalkForwardConfig,
    available_dates: list[date],
) -> list[tuple[date, date, date, date]]:
    """
    Generate train/test date ranges for walk-forward validation.

    Respects purge_days and embargo_days to prevent temporal leakage:
    - Purge: Gap between end of training and start of test (execution lag)
    - Embargo: Initial test period excluded from metrics (implementation delay)

    Example with purge=5, embargo=10:
        Train: [2020-01-01 to 2020-12-31]
        Purge: [2021-01-01 to 2021-01-05] (excluded)
        Test:  [2021-01-06 to 2021-01-15] (embargo, excluded from metrics)
        Eval:  [2021-01-16 to ...] (included in metrics)

    Args:
        config: Walk-forward configuration
        available_dates: List of available dates in the data (sorted)

    Returns:
        List of tuples: (train_start, train_end, test_start, test_end)
        Note: test_start includes embargo period if specified
    """
    # Filter dates to config range
    dates = [d for d in available_dates if config.start_date <= d <= config.end_date]

    # Account for purge periods in validation
    # Note: embargo_days are *within* the test period (excluded from metrics, not from data)
    required_dates = (
        config.min_train_periods
        + config.purge_days
        + config.n_folds
    )
    if len(dates) < required_dates:
        raise ValueError(
            f"Insufficient data: {len(dates)} dates available, "
            f"need at least {required_dates} "
            f"(min_train={config.min_train_periods}, purge={config.purge_days}, "
            f"embargo={config.embargo_days}, min_folds={config.n_folds})"
        )

    n_dates = len(dates)
    n_folds = config.n_folds

    # Calculate test period size (divide remaining dates after min_train + purge)
    reserved_dates = config.min_train_periods + config.purge_days
    test_dates_total = n_dates - reserved_dates
    test_period_size = test_dates_total // n_folds

    if test_period_size < 1:
        raise ValueError(
            f"Test period too small: {test_period_size} dates. "
            f"Reduce n_folds, min_train_periods, purge_days, or embargo_days."
        )

    if config.embargo_days >= test_period_size:
        raise ValueError(
            f"embargo_days ({config.embargo_days}) must be less than "
            f"test_period_size ({test_period_size}). "
            f"No test dates would remain for metric evaluation."
        )

    folds = []

    for fold_idx in range(n_folds):
        # Test period starts after train + purge gap (embargo is within test)
        test_start_idx = (
            config.min_train_periods
            + config.purge_days
            + fold_idx * test_period_size
        )
        test_end_idx = test_start_idx + test_period_size - 1

        # Handle last fold: include remaining dates
        if fold_idx == n_folds - 1:
            test_end_idx = n_dates - 1

        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]

        # Train period depends on window type
        if config.window_type == "expanding":
            # Expanding: always start from the beginning
            train_start = dates[0]
        else:
            # Rolling: fixed window size ending just before purge period
            train_start_idx = max(0, test_start_idx - config.purge_days - config.min_train_periods)
            train_start = dates[train_start_idx]

        # Train ends before purge period
        # Purge creates gap between train_end and test_start
        train_end_idx = test_start_idx - config.purge_days - 1
        train_end = dates[train_end_idx]

        folds.append((train_start, train_end, test_start, test_end))

        embargo_note = ""
        if config.embargo_days > 0:
            embargo_end = dates[test_start_idx + config.embargo_days - 1]
            embargo_note = f", embargo [{test_start} to {embargo_end}] (excluded from metrics)"

        logger.debug(
            f"Fold {fold_idx}: train [{train_start} to {train_end}], "
            f"purge={config.purge_days}d, "
            f"test [{test_start} to {test_end}]{embargo_note}"
        )

    return folds


def compute_fold_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute evaluation metrics for a single fold.

    Args:
        y_true: Actual target values
        y_pred: Predicted values

    Returns:
        Dict with mse, mae, r2, ic metrics
    """
    # Handle potential NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        logger.warning("No valid predictions for metrics computation")
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "ic": float("nan"),
        }

    mse = float(mean_squared_error(y_true_clean, y_pred_clean))
    mae = float(mean_absolute_error(y_true_clean, y_pred_clean))

    # R² can be negative for poor models
    r2 = float(r2_score(y_true_clean, y_pred_clean))

    # Information Coefficient (Spearman rank correlation)
    if len(y_true_clean) > 1:
        ic, _ = spearmanr(y_true_clean, y_pred_clean)
        ic = float(ic) if not np.isnan(ic) else 0.0
    else:
        ic = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "ic": ic,
    }


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> tuple[dict[str, float], float]:
    """
    Aggregate metrics across folds.

    Args:
        fold_metrics: List of metric dicts from each fold

    Returns:
        Tuple of (aggregate_metrics dict, stability_score)
    """
    if not fold_metrics:
        return {}, float("inf")

    metric_names = ["mse", "mae", "r2", "ic"]
    aggregate = {}

    for name in metric_names:
        values = [fm[name] for fm in fold_metrics if not np.isnan(fm.get(name, float("nan")))]
        if values:
            aggregate[f"mean_{name}"] = float(np.mean(values))
            aggregate[f"std_{name}"] = float(np.std(values))
        else:
            aggregate[f"mean_{name}"] = float("nan")
            aggregate[f"std_{name}"] = float("nan")

    # Stability score: coefficient of variation of MSE
    mean_mse = aggregate.get("mean_mse", 0)
    std_mse = aggregate.get("std_mse", 0)

    if mean_mse > 0 and not np.isnan(mean_mse) and not np.isnan(std_mse):
        stability_score = std_mse / mean_mse
    else:
        stability_score = float("inf") if mean_mse == 0 else float("nan")

    logger.debug(
        f"Aggregated {len(fold_metrics)} folds: "
        f"mean_mse={aggregate.get('mean_mse', 'nan'):.6f}, "
        f"stability={stability_score:.4f}"
    )

    return aggregate, stability_score


def walk_forward_validate(
    config: WalkForwardConfig,
    symbols: list[str],
    log_to_mlflow: bool = False,
) -> WalkForwardResult:
    """
    Run walk-forward validation.

    Args:
        config: Walk-forward configuration
        symbols: List of symbols to validate on
        log_to_mlflow: Whether to log results to MLflow

    Returns:
        WalkForwardResult with per-fold and aggregate metrics
    """
    logger.info(
        f"Starting walk-forward validation: {config.model_type}, "
        f"{config.n_folds} folds, {config.window_type} window"
    )

    # Track timing
    timing = TimingMetrics(name="walk_forward_validate")
    total_start = time.perf_counter()

    # Fetch all data once
    fetch_start = time.perf_counter()
    all_data = _fetch_features(
        symbols=symbols,
        features=config.features,
        start_date=config.start_date,
        end_date=config.end_date,
        target=config.target,
    )
    timing.sub_timings["data_fetch"] = time.perf_counter() - fetch_start

    if all_data.empty:
        raise ValueError(f"No data found for symbols {symbols}")

    # Drop NaN rows
    all_data = all_data.dropna()

    # Get available dates
    available_dates = sorted(all_data.index.get_level_values("date").unique())
    available_dates = [d.date() if hasattr(d, "date") else d for d in available_dates]

    # Generate fold date ranges
    folds = generate_folds(config, available_dates)

    # Clear module-level feature selection cache for this validation run
    _feature_selection_cache.clear()

    # Process each fold (sequential or parallel based on n_jobs)
    fold_start = time.perf_counter()
    fold_results = []

    if config.n_jobs == 1:
        # Sequential processing with feature cache
        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{len(folds)}")

            try:
                fold_result = _process_fold(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    config=config,
                    all_data=all_data,
                )
                fold_results.append(fold_result)
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                continue
    else:
        # Parallel processing with joblib (cache not shared across processes)
        n_jobs = config.n_jobs if config.n_jobs != -1 else os.cpu_count() or 1
        logger.info(f"Processing {len(folds)} folds in parallel (n_jobs={n_jobs})")

        results = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
            joblib.delayed(_process_fold_safe)(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                config=config,
                all_data=all_data,
            )
            for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds)
        )

        # Filter out None results (failed folds)
        fold_results = [r for r in results if r is not None]

    timing.sub_timings["fold_processing"] = time.perf_counter() - fold_start

    if not fold_results:
        raise ValueError("All folds failed, cannot compute results")

    # Aggregate metrics
    fold_metrics = [fr.metrics for fr in fold_results]
    aggregate_metrics, stability_score = aggregate_fold_metrics(fold_metrics)

    # Finalize timing
    timing.elapsed_seconds = time.perf_counter() - total_start

    result = WalkForwardResult(
        config=config,
        fold_results=fold_results,
        aggregate_metrics=aggregate_metrics,
        stability_score=stability_score,
        symbols=symbols,
    )

    # Log timing metrics
    timing.log()

    logger.info(
        f"Walk-forward complete: {len(fold_results)} folds, "
        f"mean_mse={aggregate_metrics.get('mean_mse', 'nan'):.6f}, "
        f"stability={stability_score:.4f}"
    )

    if log_to_mlflow:
        result.mlflow_run_id = _log_to_mlflow(result)

    return result


def _process_fold_safe(
    fold_idx: int,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    config: WalkForwardConfig,
    all_data: pd.DataFrame,
) -> FoldResult | None:
    """
    Wrapper for _process_fold that catches exceptions for parallel execution.

    Returns None if the fold fails, matching sequential behavior where
    failed folds are skipped.
    """
    try:
        return _process_fold(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            config=config,
            all_data=all_data,
        )
    except Exception as e:
        logger.warning(f"Fold {fold_idx} failed: {e}")
        return None


def _process_fold(
    fold_idx: int,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    config: WalkForwardConfig,
    all_data: pd.DataFrame,
) -> FoldResult:
    """
    Process a single walk-forward fold.

    Args:
        fold_idx: Index of this fold
        train_start, train_end: Training period
        test_start, test_end: Test period
        config: Walk-forward configuration
        all_data: Full dataset

    Returns:
        FoldResult with trained model and metrics
    """
    # Split data by date
    dates = all_data.index.get_level_values("date")

    # Convert to comparable format
    if hasattr(dates[0], "date"):
        train_mask = (dates.date >= train_start) & (dates.date <= train_end)
        test_mask = (dates.date >= test_start) & (dates.date <= test_end)
    else:
        train_mask = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
        test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))

    train_data = all_data.loc[train_mask]
    test_data = all_data.loc[test_mask]

    # Separate features and target
    X_train = train_data[config.features]
    y_train = train_data[config.target]
    X_test = test_data[config.features]
    y_test = test_data[config.target]

    # Feature selection (on training data only)
    selected_features = list(config.features)
    if config.feature_selection and len(config.features) > config.max_features:
        cache_key = f"fold_{fold_idx}_{config.window_type}"
        if cache_key in _feature_selection_cache:
            # Use cached result
            selected_features = _feature_selection_cache[cache_key]
            logger.debug(f"Feature selection cache hit for {cache_key}")
        else:
            # Compute and cache
            selected_features = select_features(X_train, y_train, config.max_features)
            _feature_selection_cache[cache_key] = selected_features
            logger.debug(
                f"Feature selection cache miss for {cache_key}, "
                f"computed {len(selected_features)} features"
            )
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    # Train model
    model = get_model(config.model_type, config.hyperparameters)
    model.fit(X_train, y_train)

    # Predict on test
    y_pred = model.predict(X_test)

    # Exclude embargo period from metric computation.
    # Embargo = first N days of the test set, excluded from evaluation
    # (model still predicts on them, but they don't affect scores).
    if config.embargo_days > 0 and len(y_test) > config.embargo_days:
        y_test_eval = y_test.iloc[config.embargo_days:]
        y_pred_eval = y_pred[config.embargo_days:]
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred

    # Compute metrics (on post-embargo test data only)
    metrics = compute_fold_metrics(y_test_eval, y_pred_eval)

    return FoldResult(
        fold_index=fold_idx,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        metrics=metrics,
        model=model,
        n_train_samples=len(X_train),
        n_test_samples=len(X_test),
    )


def _log_to_mlflow(result: WalkForwardResult) -> str | None:
    """Log walk-forward results to MLflow.

    Returns:
        The MLflow run ID if logging succeeded, None otherwise.
    """
    try:
        import mlflow
        from hrp.research.mlflow_utils import setup_mlflow

        setup_mlflow()
        experiment_name = f"hrp_walkforward_{result.config.model_type}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"wf_{result.config.n_folds}folds") as run:
            run_id = run.info.run_id

            # Log user-provided tags (e.g. hypothesis_id)
            if result.config.tags:
                mlflow.set_tags(result.config.tags)

            # Log config
            mlflow.log_param("model_type", result.config.model_type)
            mlflow.log_param("n_folds", result.config.n_folds)
            mlflow.log_param("window_type", result.config.window_type)
            mlflow.log_param("n_features", len(result.config.features))

            # Log aggregate metrics
            for key, value in result.aggregate_metrics.items():
                if not np.isnan(value):
                    mlflow.log_metric(key, value)

            mlflow.log_metric("stability_score", result.stability_score)

            # Log per-fold metrics as nested runs
            for fold in result.fold_results:
                with mlflow.start_run(run_name=f"fold_{fold.fold_index}", nested=True):
                    for key, value in fold.metrics.items():
                        if not np.isnan(value):
                            mlflow.log_metric(key, value)
                    mlflow.log_param("n_train_samples", fold.n_train_samples)
                    mlflow.log_param("n_test_samples", fold.n_test_samples)

        logger.info(f"Logged walk-forward results to MLflow: {experiment_name} (run_id={run_id})")
        return run_id

    except ImportError:
        logger.warning("MLflow not installed, skipping logging")
        return None
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
        return None
