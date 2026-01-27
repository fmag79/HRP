# Walk-Forward Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add walk-forward validation to the ML framework for robust model selection and overfitting detection.

**Architecture:** Single new module `hrp/ml/validation.py` with three dataclasses (`WalkForwardConfig`, `FoldResult`, `WalkForwardResult`) and two main functions (`generate_folds`, `walk_forward_validate`). Reuses existing `_fetch_features`, `select_features`, and `get_model` from the ML module.

**Tech Stack:** scikit-learn (metrics), scipy (spearmanr), pandas, numpy, loguru (logging), optional MLflow

**Design Doc:** `docs/plans/2025-01-22-walk-forward-validation-design.md`

---

## Task 1: WalkForwardConfig Dataclass

**Files:**
- Create: `hrp/ml/validation.py`
- Test: `tests/test_ml/test_validation.py`

**Step 1: Write the failing test for WalkForwardConfig**

```python
# tests/test_ml/test_validation.py
"""Tests for walk-forward validation."""

from datetime import date

import pytest

from hrp.ml.validation import WalkForwardConfig


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert config.model_type == "ridge"
        assert config.n_folds == 5
        assert config.window_type == "expanding"
        assert config.min_train_periods == 252
        assert config.feature_selection is True
        assert config.max_features == 20
        assert config.hyperparameters == {}

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = WalkForwardConfig(
            model_type="lightgbm",
            target="returns_5d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=10,
            window_type="rolling",
            min_train_periods=504,
            hyperparameters={"n_estimators": 100},
        )
        assert config.n_folds == 10
        assert config.window_type == "rolling"
        assert config.min_train_periods == 504
        assert config.hyperparameters == {"n_estimators": 100}

    def test_config_invalid_model_type(self):
        """Test config rejects invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            WalkForwardConfig(
                model_type="invalid_model",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
            )

    def test_config_invalid_window_type(self):
        """Test config rejects invalid window type."""
        with pytest.raises(ValueError, match="window_type must be"):
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                window_type="invalid",
            )

    def test_config_invalid_n_folds(self):
        """Test config rejects n_folds < 2."""
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                n_folds=1,
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_validation.py::TestWalkForwardConfig -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.ml.validation'"

**Step 3: Write minimal implementation**

```python
# hrp/ml/validation.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_validation.py::TestWalkForwardConfig -v`
Expected: All PASS (5 tests)

**Step 5: Commit**

```bash
git add hrp/ml/validation.py tests/test_ml/test_validation.py
git commit -m "feat(ml): add WalkForwardConfig dataclass for walk-forward validation"
```

---

## Task 2: FoldResult and WalkForwardResult Dataclasses

**Files:**
- Modify: `hrp/ml/validation.py`
- Test: `tests/test_ml/test_validation.py`

**Step 1: Write the failing test for result dataclasses**

```python
# Add to tests/test_ml/test_validation.py

from hrp.ml.validation import FoldResult, WalkForwardResult


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self):
        """Test creating FoldResult."""
        result = FoldResult(
            fold_index=0,
            train_start=date(2015, 1, 1),
            train_end=date(2018, 12, 31),
            test_start=date(2019, 1, 1),
            test_end=date(2019, 12, 31),
            metrics={"mse": 0.001, "mae": 0.02, "r2": 0.15, "ic": 0.05},
            model=None,  # Mock model
            n_train_samples=1000,
            n_test_samples=250,
        )
        assert result.fold_index == 0
        assert result.metrics["mse"] == 0.001
        assert result.n_train_samples == 1000


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_walk_forward_result_creation(self):
        """Test creating WalkForwardResult."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
        )
        fold_results = [
            FoldResult(
                fold_index=i,
                train_start=date(2015, 1, 1),
                train_end=date(2018, 12, 31),
                test_start=date(2019, 1, 1),
                test_end=date(2019, 12, 31),
                metrics={"mse": 0.001 + i * 0.0001},
                model=None,
                n_train_samples=1000,
                n_test_samples=250,
            )
            for i in range(3)
        ]
        result = WalkForwardResult(
            config=config,
            fold_results=fold_results,
            aggregate_metrics={"mean_mse": 0.0011, "std_mse": 0.0001},
            stability_score=0.09,
            symbols=["AAPL", "MSFT"],
        )
        assert len(result.fold_results) == 3
        assert result.stability_score == 0.09
        assert result.aggregate_metrics["mean_mse"] == 0.0011
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_validation.py::TestFoldResult -v`
Expected: FAIL with "cannot import name 'FoldResult'"

**Step 3: Add FoldResult and WalkForwardResult to implementation**

```python
# Add to hrp/ml/validation.py after WalkForwardConfig

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_validation.py::TestFoldResult tests/test_ml/test_validation.py::TestWalkForwardResult -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add hrp/ml/validation.py tests/test_ml/test_validation.py
git commit -m "feat(ml): add FoldResult and WalkForwardResult dataclasses"
```

---

## Task 3: Fold Generation Function

**Files:**
- Modify: `hrp/ml/validation.py`
- Test: `tests/test_ml/test_validation.py`

**Step 1: Write failing tests for generate_folds**

```python
# Add to tests/test_ml/test_validation.py

import pandas as pd

from hrp.ml.validation import generate_folds


class TestGenerateFolds:
    """Tests for generate_folds function."""

    @pytest.fixture
    def sample_dates(self):
        """Generate sample business dates."""
        return pd.date_range("2015-01-01", "2023-12-31", freq="B").date.tolist()

    def test_generate_folds_count(self, sample_dates):
        """Test that correct number of folds are generated."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=5,
        )
        folds = generate_folds(config, sample_dates)
        assert len(folds) == 5

    def test_generate_folds_no_overlap(self, sample_dates):
        """Test that test periods do not overlap."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=5,
        )
        folds = generate_folds(config, sample_dates)

        # Check test periods don't overlap
        for i in range(len(folds) - 1):
            _, _, _, test_end_i = folds[i]
            _, _, test_start_next, _ = folds[i + 1]
            assert test_end_i < test_start_next

    def test_generate_folds_expanding_window(self, sample_dates):
        """Test expanding window: train_start is always the same."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=3,
            window_type="expanding",
        )
        folds = generate_folds(config, sample_dates)

        # All folds should have the same train_start
        train_starts = [f[0] for f in folds]
        assert all(ts == train_starts[0] for ts in train_starts)

        # train_end should increase with each fold
        train_ends = [f[1] for f in folds]
        assert train_ends == sorted(train_ends)

    def test_generate_folds_rolling_window(self, sample_dates):
        """Test rolling window: train window size is constant."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=3,
            window_type="rolling",
        )
        folds = generate_folds(config, sample_dates)

        # Calculate training period lengths (in days)
        train_lengths = []
        for train_start, train_end, _, _ in folds:
            length = (train_end - train_start).days
            train_lengths.append(length)

        # All training periods should be approximately the same length
        # (allow 10% variance due to business days)
        avg_length = sum(train_lengths) / len(train_lengths)
        for length in train_lengths:
            assert abs(length - avg_length) / avg_length < 0.15

    def test_generate_folds_train_before_test(self, sample_dates):
        """Test that train_end < test_start for all folds."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=5,
        )
        folds = generate_folds(config, sample_dates)

        for train_start, train_end, test_start, test_end in folds:
            assert train_end < test_start
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_validation.py::TestGenerateFolds -v`
Expected: FAIL with "cannot import name 'generate_folds'"

**Step 3: Implement generate_folds function**

```python
# Add to hrp/ml/validation.py after WalkForwardResult

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_validation.py::TestGenerateFolds -v`
Expected: All PASS (5 tests)

**Step 5: Commit**

```bash
git add hrp/ml/validation.py tests/test_ml/test_validation.py
git commit -m "feat(ml): add generate_folds function for walk-forward splits"
```

---

## Task 4: Metrics Computation Function

**Files:**
- Modify: `hrp/ml/validation.py`
- Test: `tests/test_ml/test_validation.py`

**Step 1: Write failing tests for compute_fold_metrics**

```python
# Add to tests/test_ml/test_validation.py

import numpy as np

from hrp.ml.validation import compute_fold_metrics


class TestComputeFoldMetrics:
    """Tests for compute_fold_metrics function."""

    def test_compute_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_fold_metrics(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["ic"] == 1.0

    def test_compute_metrics_known_values(self):
        """Test metrics with known values."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])

        metrics = compute_fold_metrics(y_true, y_pred)

        # MSE = mean((0.5)^2 + (0.5)^2 + (0.5)^2) = 0.25
        assert abs(metrics["mse"] - 0.25) < 0.001
        # MAE = mean(0.5 + 0.5 + 0.5) = 0.5
        assert abs(metrics["mae"] - 0.5) < 0.001

    def test_compute_metrics_returns_all_keys(self):
        """Test that all expected metrics are returned."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        metrics = compute_fold_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "ic" in metrics

    def test_compute_metrics_ic_range(self):
        """Test that IC is in valid range [-1, 1]."""
        y_true = pd.Series(np.random.randn(100))
        y_pred = np.random.randn(100)

        metrics = compute_fold_metrics(y_true, y_pred)

        assert -1.0 <= metrics["ic"] <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_validation.py::TestComputeFoldMetrics -v`
Expected: FAIL with "cannot import name 'compute_fold_metrics'"

**Step 3: Implement compute_fold_metrics function**

```python
# Add imports at top of hrp/ml/validation.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add function after generate_folds

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_validation.py::TestComputeFoldMetrics -v`
Expected: All PASS (4 tests)

**Step 5: Commit**

```bash
git add hrp/ml/validation.py tests/test_ml/test_validation.py
git commit -m "feat(ml): add compute_fold_metrics function"
```

---

## Task 5: Aggregate Metrics Function

**Files:**
- Modify: `hrp/ml/validation.py`
- Test: `tests/test_ml/test_validation.py`

**Step 1: Write failing tests for aggregate_fold_metrics**

```python
# Add to tests/test_ml/test_validation.py

from hrp.ml.validation import aggregate_fold_metrics


class TestAggregateFoldMetrics:
    """Tests for aggregate_fold_metrics function."""

    def test_aggregate_metrics_mean_std(self):
        """Test aggregation computes mean and std."""
        fold_metrics = [
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.002, "mae": 0.03, "r2": 0.2, "ic": 0.06},
            {"mse": 0.003, "mae": 0.04, "r2": 0.3, "ic": 0.07},
        ]

        agg, stability = aggregate_fold_metrics(fold_metrics)

        assert "mean_mse" in agg
        assert "std_mse" in agg
        assert "mean_mae" in agg
        assert "std_mae" in agg
        assert "mean_r2" in agg
        assert "std_r2" in agg
        assert "mean_ic" in agg
        assert "std_ic" in agg

        # Check mean_mse = mean([0.001, 0.002, 0.003]) = 0.002
        assert abs(agg["mean_mse"] - 0.002) < 0.0001

    def test_aggregate_metrics_stability_score(self):
        """Test stability score calculation."""
        # High variance case
        fold_metrics = [
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.010, "mae": 0.03, "r2": 0.2, "ic": 0.06},
            {"mse": 0.001, "mae": 0.04, "r2": 0.3, "ic": 0.07},
        ]

        agg, stability = aggregate_fold_metrics(fold_metrics)

        # stability = std_mse / mean_mse
        expected_stability = agg["std_mse"] / agg["mean_mse"]
        assert abs(stability - expected_stability) < 0.0001

    def test_aggregate_metrics_stable_model(self):
        """Test stability score for consistent model."""
        # Low variance case - stable
        fold_metrics = [
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
        ]

        agg, stability = aggregate_fold_metrics(fold_metrics)

        # Zero variance = zero stability score
        assert stability == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_validation.py::TestAggregateFoldMetrics -v`
Expected: FAIL with "cannot import name 'aggregate_fold_metrics'"

**Step 3: Implement aggregate_fold_metrics function**

```python
# Add to hrp/ml/validation.py after compute_fold_metrics

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_validation.py::TestAggregateFoldMetrics -v`
Expected: All PASS (3 tests)

**Step 5: Commit**

```bash
git add hrp/ml/validation.py tests/test_ml/test_validation.py
git commit -m "feat(ml): add aggregate_fold_metrics function with stability score"
```

---

## Task 6: Main walk_forward_validate Function

**Files:**
- Modify: `hrp/ml/validation.py`
- Test: `tests/test_ml/test_validation.py`

**Step 1: Write failing tests for walk_forward_validate**

```python
# Add to tests/test_ml/test_validation.py

from unittest.mock import patch

from hrp.ml.validation import walk_forward_validate


class TestWalkForwardValidate:
    """Tests for walk_forward_validate function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample WalkForwardConfig."""
        return WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            feature_selection=False,
        )

    @pytest.fixture
    def mock_features_df(self):
        """Create mock features DataFrame."""
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

        np.random.seed(42)
        n = len(index)

        # Create features with weak signal
        momentum = np.random.randn(n) * 0.1
        volatility = np.abs(np.random.randn(n)) * 0.2
        target = 0.1 * momentum + np.random.randn(n) * 0.05

        return pd.DataFrame(
            {
                "momentum_20d": momentum,
                "volatility_20d": volatility,
                "returns_20d": target,
            },
            index=index,
        )

    def test_walk_forward_validate_returns_result(self, sample_config, mock_features_df):
        """Test walk_forward_validate returns WalkForwardResult."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            result = walk_forward_validate(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
            )

        assert isinstance(result, WalkForwardResult)
        assert len(result.fold_results) == 3
        assert result.config == sample_config

    def test_walk_forward_validate_fold_metrics(self, sample_config, mock_features_df):
        """Test that each fold has valid metrics."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            result = walk_forward_validate(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
            )

        for fold in result.fold_results:
            assert "mse" in fold.metrics
            assert "ic" in fold.metrics
            assert fold.n_train_samples > 0
            assert fold.n_test_samples > 0

    def test_walk_forward_validate_aggregate_metrics(self, sample_config, mock_features_df):
        """Test aggregate metrics are computed."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            result = walk_forward_validate(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
            )

        assert "mean_mse" in result.aggregate_metrics
        assert "std_mse" in result.aggregate_metrics
        assert result.stability_score >= 0

    def test_walk_forward_validate_expanding_vs_rolling(self, mock_features_df):
        """Test both window types produce results."""
        for window_type in ["expanding", "rolling"]:
            config = WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d", "volatility_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2020, 12, 31),
                n_folds=3,
                window_type=window_type,
                feature_selection=False,
            )

            with patch("hrp.ml.validation._fetch_features") as mock_fetch:
                mock_fetch.return_value = mock_features_df

                result = walk_forward_validate(config, symbols=["AAPL", "MSFT"])

            assert len(result.fold_results) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_validation.py::TestWalkForwardValidate -v`
Expected: FAIL with "cannot import name 'walk_forward_validate'"

**Step 3: Implement walk_forward_validate function**

```python
# Add imports at top of hrp/ml/validation.py (if not already present)
from hrp.ml.models import get_model
from hrp.ml.training import _fetch_features, select_features

# Add main function after aggregate_fold_metrics

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

    # Drop NaN rows
    all_data = all_data.dropna()

    # Get available dates
    available_dates = sorted(all_data.index.get_level_values("date").unique())
    available_dates = [d.date() if hasattr(d, "date") else d for d in available_dates]

    # Generate fold date ranges
    folds = generate_folds(config, available_dates)

    # Process each fold
    fold_results = []

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

    if not fold_results:
        raise ValueError("All folds failed, cannot compute results")

    # Aggregate metrics
    fold_metrics = [fr.metrics for fr in fold_results]
    aggregate_metrics, stability_score = aggregate_fold_metrics(fold_metrics)

    result = WalkForwardResult(
        config=config,
        fold_results=fold_results,
        aggregate_metrics=aggregate_metrics,
        stability_score=stability_score,
        symbols=symbols,
    )

    logger.info(
        f"Walk-forward complete: {len(fold_results)} folds, "
        f"mean_mse={aggregate_metrics.get('mean_mse', 'nan'):.6f}, "
        f"stability={stability_score:.4f}"
    )

    if log_to_mlflow:
        _log_to_mlflow(result)

    return result


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
        selected_features = select_features(X_train, y_train, config.max_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    # Train model
    model = get_model(config.model_type, config.hyperparameters)
    model.fit(X_train, y_train)

    # Predict on test
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = compute_fold_metrics(y_test, y_pred)

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


def _log_to_mlflow(result: WalkForwardResult) -> None:
    """Log walk-forward results to MLflow."""
    try:
        import mlflow

        experiment_name = f"hrp_walkforward_{result.config.model_type}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"wf_{result.config.n_folds}folds"):
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

        logger.info(f"Logged walk-forward results to MLflow: {experiment_name}")

    except ImportError:
        logger.warning("MLflow not installed, skipping logging")
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_validation.py::TestWalkForwardValidate -v`
Expected: All PASS (4 tests)

**Step 5: Commit**

```bash
git add hrp/ml/validation.py tests/test_ml/test_validation.py
git commit -m "feat(ml): add walk_forward_validate main function"
```

---

## Task 7: Update Module Exports

**Files:**
- Modify: `hrp/ml/__init__.py`

**Step 1: Write test for module exports**

```python
# Add to tests/test_ml/test_validation.py

class TestModuleExports:
    """Test that validation module is properly exported."""

    def test_import_from_ml_module(self):
        """Test importing from hrp.ml."""
        from hrp.ml import (
            WalkForwardConfig,
            WalkForwardResult,
            FoldResult,
            walk_forward_validate,
        )

        assert WalkForwardConfig is not None
        assert WalkForwardResult is not None
        assert FoldResult is not None
        assert walk_forward_validate is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_validation.py::TestModuleExports -v`
Expected: FAIL with "cannot import name 'WalkForwardConfig'"

**Step 3: Update hrp/ml/__init__.py**

```python
# hrp/ml/__init__.py
"""
ML Framework for HRP.

Provides model training, signal generation, and validation.
"""

from hrp.ml.models import MLConfig, SUPPORTED_MODELS, get_model, HAS_LIGHTGBM, HAS_XGBOOST
from hrp.ml.signals import predictions_to_signals
from hrp.ml.training import TrainingResult, train_model, load_training_data, select_features
from hrp.ml.validation import (
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
    walk_forward_validate,
    generate_folds,
    compute_fold_metrics,
    aggregate_fold_metrics,
)

__all__ = [
    # Models
    "MLConfig",
    "SUPPORTED_MODELS",
    "get_model",
    "HAS_LIGHTGBM",
    "HAS_XGBOOST",
    # Signals
    "predictions_to_signals",
    # Training
    "TrainingResult",
    "train_model",
    "load_training_data",
    "select_features",
    # Validation
    "WalkForwardConfig",
    "WalkForwardResult",
    "FoldResult",
    "walk_forward_validate",
    "generate_folds",
    "compute_fold_metrics",
    "aggregate_fold_metrics",
]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_validation.py::TestModuleExports -v`
Expected: PASS

**Step 5: Run all validation tests**

Run: `pytest tests/test_ml/test_validation.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add hrp/ml/__init__.py tests/test_ml/test_validation.py
git commit -m "feat(ml): export walk-forward validation from ml module"
```

---

## Task 8: Update Spec to Mark Walk-Forward Complete

**Files:**
- Modify: `docs/plans/2025-01-19-hrp-spec.md`

**Step 1: Update Phase 5 section**

Find the Phase 5 deliverables section and update:

```markdown
### Phase 5: ML Framework ⏳ MVP COMPLETE

**Goal:** ML training pipeline with proper validation.

#### Deliverables (MVP)

- [x] ML model registry (`hrp/ml/models.py`) - Ridge, Lasso, ElasticNet, RandomForest, MLP + optional LightGBM/XGBoost
- [x] Training pipeline (`hrp/ml/training.py`) - Data loading, train/val/test splits, metrics
- [x] Walk-forward validation (`hrp/ml/validation.py`) - Expanding/rolling windows, stability score
- [x] Feature selection (`hrp/ml/training.py:select_features`) - Mutual information based
- [x] Signal generation from predictions (`hrp/ml/signals.py`) - rank, threshold, zscore methods
- [ ] Overfitting guards (test set discipline) - Future
```

**Step 2: Update verification checklist**

Find the verification checklist and update:

```markdown
#### Verification Checklist

- [x] Linear models (Ridge, Lasso) working
- [x] Tree-based models (RandomForest) working
- [x] Walk-forward validation implemented
- [ ] Test set guard enforced - Future
- [ ] ML experiments logged to MLflow - Future (framework ready)
```

**Step 3: Commit**

```bash
git add docs/plans/2025-01-19-hrp-spec.md
git commit -m "docs: mark walk-forward validation complete in spec"
```

---

## Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | `hrp/ml/validation.py` | WalkForwardConfig dataclass |
| 2 | `hrp/ml/validation.py` | FoldResult, WalkForwardResult dataclasses |
| 3 | `hrp/ml/validation.py` | generate_folds function |
| 4 | `hrp/ml/validation.py` | compute_fold_metrics function |
| 5 | `hrp/ml/validation.py` | aggregate_fold_metrics function |
| 6 | `hrp/ml/validation.py` | walk_forward_validate main function |
| 7 | `hrp/ml/__init__.py` | Module exports |
| 8 | `docs/plans/2025-01-19-hrp-spec.md` | Documentation update |

**Total:** ~350 lines of implementation, ~250 lines of tests, 8 commits.

**Test command:** `pytest tests/test_ml/test_validation.py -v`
