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
