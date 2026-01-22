"""
Walk-forward validation for ML models.

Provides temporal cross-validation to assess model robustness
and detect overfitting across multiple time periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from loguru import logger

from hrp.ml.models import SUPPORTED_MODELS


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

        logger.debug(
            f"WalkForwardConfig created: {self.model_type}, "
            f"{self.n_folds} folds, {self.window_type} window"
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

    Args:
        config: Walk-forward configuration
        available_dates: List of available dates in the data (sorted)

    Returns:
        List of tuples: (train_start, train_end, test_start, test_end)
    """
    # Filter dates to config range
    dates = [d for d in available_dates if config.start_date <= d <= config.end_date]

    if len(dates) < config.min_train_periods + config.n_folds:
        raise ValueError(
            f"Insufficient data: {len(dates)} dates available, "
            f"need at least {config.min_train_periods + config.n_folds}"
        )

    n_dates = len(dates)
    n_folds = config.n_folds

    # Calculate test period size (divide remaining dates after min_train equally)
    # Reserve min_train_periods for the first fold's training
    test_dates_total = n_dates - config.min_train_periods
    test_period_size = test_dates_total // n_folds

    if test_period_size < 1:
        raise ValueError(
            f"Test period too small: {test_period_size} dates. "
            f"Reduce n_folds or min_train_periods."
        )

    folds = []

    for fold_idx in range(n_folds):
        # Test period: fixed size, non-overlapping
        test_start_idx = config.min_train_periods + fold_idx * test_period_size
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
            # Rolling: fixed window size ending just before test
            train_start_idx = max(0, test_start_idx - config.min_train_periods)
            train_start = dates[train_start_idx]

        # Train ends one day before test starts
        train_end_idx = test_start_idx - 1
        train_end = dates[train_end_idx]

        folds.append((train_start, train_end, test_start, test_end))

        logger.debug(
            f"Fold {fold_idx}: train [{train_start} to {train_end}], "
            f"test [{test_start} to {test_end}]"
        )

    return folds
