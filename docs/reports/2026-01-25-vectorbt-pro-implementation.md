# VectorBT PRO-Inspired Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add cross-validated optimization, parallel parameter sweeps with Sharpe decay analysis, ATR trailing stops, walk-forward visualization, and HMM regime detection to HRP.

**Architecture:** Build on existing `hrp/ml/validation.py` patterns (dataclass configs, fold generation, overfitting guards). New modules integrate with `HyperparameterTrialCounter`, `SharpeDecayMonitor`, and MLflow logging. Dashboard components follow Streamlit patterns from `hrp/dashboard/components/`.

**Tech Stack:** Python 3.11+, scikit-learn, joblib (parallel), hmmlearn (Phase 5), plotly (visualization), pytest

**Source Design Doc:** `docs/plans/2025-01-25-vectorbt-pro-patterns.md`

---

## Phase 1: Cross-Validated Optimization Framework

### Task 1.1: Create OptimizationConfig dataclass

**Files:**
- Create: `hrp/ml/optimization.py`
- Test: `tests/test_ml/test_optimization.py`

**Step 1: Write the failing test**

```python
# tests/test_ml/test_optimization.py
"""Tests for cross-validated optimization."""

import pytest
from datetime import date


def test_optimization_config_creation_with_defaults():
    """Test OptimizationConfig can be created with required fields."""
    from hrp.ml.optimization import OptimizationConfig

    config = OptimizationConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d", "volatility_20d"],
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
    )

    assert config.model_type == "ridge"
    assert config.n_folds == 5  # default
    assert config.window_type == "expanding"  # default
    assert config.scoring_metric == "ic"  # default
    assert config.max_trials == 50  # default
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::test_optimization_config_creation_with_defaults -v`
Expected: FAIL with "cannot import name 'OptimizationConfig'"

**Step 3: Write minimal implementation**

```python
# hrp/ml/optimization.py
"""
Cross-validated optimization framework.

Provides parameterized cross-validation optimization that integrates
with walk-forward validation and overfitting guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from hrp.ml.models import SUPPORTED_MODELS


@dataclass
class OptimizationConfig:
    """
    Configuration for cross-validated optimization.

    Attributes:
        model_type: Type of model (must be in SUPPORTED_MODELS)
        target: Target variable name (e.g., 'returns_20d')
        features: List of feature names from feature store
        param_grid: Dict mapping param names to lists of values
        start_date: Start of the entire date range
        end_date: End of the entire date range
        n_folds: Number of walk-forward folds (default 5)
        window_type: 'expanding' or 'rolling' (default 'expanding')
        scoring_metric: Metric to optimize ('ic', 'mse', 'sharpe')
        constraints: Optional constraints on param combinations
        max_trials: Maximum trials (integrates with HyperparameterTrialCounter)
        hypothesis_id: Optional hypothesis ID for audit trail
    """

    model_type: str
    target: str
    features: list[str]
    param_grid: dict[str, list[Any]]
    start_date: date
    end_date: date
    n_folds: int = 5
    window_type: str = "expanding"
    scoring_metric: str = "ic"
    constraints: dict[str, Any] | None = None
    max_trials: int = 50
    hypothesis_id: str | None = None

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
        if self.scoring_metric not in ("ic", "mse", "sharpe"):
            raise ValueError(
                f"scoring_metric must be 'ic', 'mse', or 'sharpe', "
                f"got '{self.scoring_metric}'"
            )
        if not self.param_grid:
            raise ValueError("param_grid cannot be empty")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::test_optimization_config_creation_with_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(ml): add OptimizationConfig dataclass for CV optimization"
```

---

### Task 1.2: Add OptimizationConfig validation tests

**Files:**
- Modify: `tests/test_ml/test_optimization.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_ml/test_optimization.py

def test_config_validates_model_type():
    """Test OptimizationConfig rejects invalid model types."""
    from hrp.ml.optimization import OptimizationConfig

    with pytest.raises(ValueError, match="Unsupported model type"):
        OptimizationConfig(
            model_type="invalid_model",
            target="returns_20d",
            features=["momentum_20d"],
            param_grid={"alpha": [1.0]},
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )


def test_config_validates_param_grid_not_empty():
    """Test OptimizationConfig rejects empty param_grid."""
    from hrp.ml.optimization import OptimizationConfig

    with pytest.raises(ValueError, match="param_grid cannot be empty"):
        OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            param_grid={},
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )


def test_config_validates_scoring_metric():
    """Test OptimizationConfig rejects invalid scoring metrics."""
    from hrp.ml.optimization import OptimizationConfig

    with pytest.raises(ValueError, match="scoring_metric must be"):
        OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            param_grid={"alpha": [1.0]},
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            scoring_metric="invalid",
        )
```

**Step 2: Run tests to verify they pass (validation already implemented)**

Run: `pytest tests/test_ml/test_optimization.py -v`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
git add tests/test_ml/test_optimization.py
git commit -m "test(ml): add validation tests for OptimizationConfig"
```

---

### Task 1.3: Create OptimizationResult dataclass

**Files:**
- Modify: `hrp/ml/optimization.py`
- Modify: `tests/test_ml/test_optimization.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_ml/test_optimization.py
import pandas as pd


def test_optimization_result_creation():
    """Test OptimizationResult can be created with all fields."""
    from hrp.ml.optimization import OptimizationResult, FoldOptResult

    fold_results = [
        FoldOptResult(
            fold_index=0,
            params={"alpha": 1.0},
            train_score=0.10,
            test_score=0.08,
        ),
    ]

    result = OptimizationResult(
        best_params={"alpha": 1.0},
        best_score=0.08,
        cv_results=pd.DataFrame({"alpha": [1.0], "mean_test_score": [0.08]}),
        fold_results=fold_results,
        all_trials=[{"params": {"alpha": 1.0}, "score": 0.08}],
        hypothesis_id="HYP-2025-001",
    )

    assert result.best_params == {"alpha": 1.0}
    assert result.best_score == 0.08
    assert len(result.fold_results) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::test_optimization_result_creation -v`
Expected: FAIL with "cannot import name 'OptimizationResult'"

**Step 3: Write minimal implementation**

```python
# Add to hrp/ml/optimization.py after OptimizationConfig

import pandas as pd


@dataclass
class FoldOptResult:
    """Result from optimization on a single fold."""

    fold_index: int
    params: dict[str, Any]
    train_score: float
    test_score: float


@dataclass
class OptimizationResult:
    """
    Result of cross-validated optimization.

    Attributes:
        best_params: Best hyperparameters found
        best_score: Best score achieved (mean across folds)
        cv_results: DataFrame with all parameter combinations and scores
        fold_results: Per-fold results for the best parameters
        all_trials: List of all trials with params and scores
        hypothesis_id: Optional hypothesis ID for audit trail
    """

    best_params: dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    fold_results: list[FoldOptResult]
    all_trials: list[dict]
    hypothesis_id: str | None = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::test_optimization_result_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(ml): add OptimizationResult and FoldOptResult dataclasses"
```

---

### Task 1.4: Implement grid generation helper

**Files:**
- Modify: `hrp/ml/optimization.py`
- Modify: `tests/test_ml/test_optimization.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_ml/test_optimization.py

def test_generate_param_combinations():
    """Test generating all parameter combinations from grid."""
    from hrp.ml.optimization import _generate_param_combinations

    grid = {"alpha": [0.1, 1.0], "l1_ratio": [0.0, 0.5]}
    combos = _generate_param_combinations(grid)

    assert len(combos) == 4
    assert {"alpha": 0.1, "l1_ratio": 0.0} in combos
    assert {"alpha": 1.0, "l1_ratio": 0.5} in combos


def test_generate_param_combinations_single_param():
    """Test grid generation with single parameter."""
    from hrp.ml.optimization import _generate_param_combinations

    grid = {"alpha": [0.1, 1.0, 10.0]}
    combos = _generate_param_combinations(grid)

    assert len(combos) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::test_generate_param_combinations -v`
Expected: FAIL with "cannot import name '_generate_param_combinations'"

**Step 3: Write minimal implementation**

```python
# Add to hrp/ml/optimization.py
import itertools


def _generate_param_combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Generate all combinations of parameters from a grid.

    Args:
        param_grid: Dict mapping param names to lists of values

    Returns:
        List of dicts, each with one value per parameter
    """
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_optimization.py::test_generate_param_combinations -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(ml): add _generate_param_combinations helper"
```

---

### Task 1.5: Implement cross_validated_optimize function (core logic)

**Files:**
- Modify: `hrp/ml/optimization.py`
- Modify: `tests/test_ml/test_optimization.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_ml/test_optimization.py
from unittest.mock import patch, MagicMock
import numpy as np


def test_cross_validated_optimize_returns_best_params(test_db):
    """Test cross_validated_optimize returns best parameters."""
    from hrp.ml.optimization import OptimizationConfig, cross_validated_optimize

    # Create synthetic data in the database
    from hrp.data.db import get_db
    from datetime import timedelta

    db = get_db(test_db)

    # Insert test symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for sym in symbols:
        db.execute(
            "INSERT INTO symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
            (sym,),
        )

    # Insert prices and features
    start = date(2020, 1, 1)
    for i in range(500):  # ~2 years of daily data
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for sym in symbols:
            db.execute(
                """INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, 100, 101, 99, 100.5, 100.5, 1000000, 'test')""",
                (sym, d),
            )
            db.execute(
                """INSERT INTO features (symbol, date, feature_name, value)
                VALUES (?, ?, 'momentum_20d', ?), (?, ?, 'returns_20d', ?)""",
                (sym, d, np.random.randn(), sym, d, np.random.randn() * 0.01),
            )

    config = OptimizationConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d"],
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        start_date=date(2020, 1, 1),
        end_date=date(2021, 6, 30),
        n_folds=3,
        max_trials=10,
    )

    result = cross_validated_optimize(
        config=config,
        symbols=symbols,
        log_to_mlflow=False,
    )

    assert result.best_params is not None
    assert "alpha" in result.best_params
    assert result.best_params["alpha"] in [0.1, 1.0, 10.0]
    assert len(result.all_trials) <= config.max_trials
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::test_cross_validated_optimize_returns_best_params -v`
Expected: FAIL with "cannot import name 'cross_validated_optimize'"

**Step 3: Write minimal implementation**

```python
# Add to hrp/ml/optimization.py
from loguru import logger
import numpy as np

from hrp.ml.validation import generate_folds, WalkForwardConfig, compute_fold_metrics
from hrp.ml.training import _fetch_features
from hrp.ml.models import get_model
from hrp.risk.overfitting import HyperparameterTrialCounter


def cross_validated_optimize(
    config: OptimizationConfig,
    symbols: list[str],
    log_to_mlflow: bool = True,
) -> OptimizationResult:
    """
    Run cross-validated parameter optimization.

    Combines grid search with HRP's walk-forward validation
    and overfitting guards.

    Args:
        config: Optimization configuration
        symbols: List of symbols to include
        log_to_mlflow: Whether to log results to MLflow

    Returns:
        OptimizationResult with best params and all trial results
    """
    logger.info(f"Starting cross-validated optimization for {config.model_type}")

    # Initialize trial counter if hypothesis provided
    trial_counter = None
    if config.hypothesis_id:
        trial_counter = HyperparameterTrialCounter(
            hypothesis_id=config.hypothesis_id,
            max_trials=config.max_trials,
        )

    # Fetch all data once
    all_features = config.features + [config.target]
    data = _fetch_features(symbols, all_features, config.start_date, config.end_date)

    if data.empty:
        raise ValueError("No data available for the specified date range")

    # Get available dates
    available_dates = sorted(data.index.get_level_values("date").unique())

    # Generate folds using WalkForwardConfig
    wf_config = WalkForwardConfig(
        model_type=config.model_type,
        target=config.target,
        features=config.features,
        start_date=config.start_date,
        end_date=config.end_date,
        n_folds=config.n_folds,
        window_type=config.window_type,
    )
    folds = generate_folds(wf_config, available_dates)

    # Generate parameter combinations
    param_combos = _generate_param_combinations(config.param_grid)
    logger.info(f"Testing {len(param_combos)} parameter combinations across {len(folds)} folds")

    # Limit by max_trials
    if len(param_combos) > config.max_trials:
        logger.warning(
            f"Limiting to {config.max_trials} trials (from {len(param_combos)} combos)"
        )
        param_combos = param_combos[: config.max_trials]

    all_trials = []
    best_score = float("-inf") if config.scoring_metric == "ic" else float("inf")
    best_params = None
    best_fold_results = []

    for params in param_combos:
        # Check trial counter
        if trial_counter and not trial_counter.can_try():
            logger.warning("Max trials reached, stopping optimization")
            break

        fold_results = []
        scores = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            # Split data
            train_mask = (
                (data.index.get_level_values("date") >= train_start)
                & (data.index.get_level_values("date") <= train_end)
            )
            test_mask = (
                (data.index.get_level_values("date") >= test_start)
                & (data.index.get_level_values("date") <= test_end)
            )

            train_data = data[train_mask]
            test_data = data[test_mask]

            if train_data.empty or test_data.empty:
                continue

            X_train = train_data[config.features]
            y_train = train_data[config.target]
            X_test = test_data[config.features]
            y_test = test_data[config.target]

            # Drop NaN
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())

            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]

            if len(X_train) < 10 or len(X_test) < 5:
                continue

            # Train model
            model = get_model(config.model_type, params)
            model.fit(X_train, y_train)

            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_metrics = compute_fold_metrics(y_train, train_pred)
            test_metrics = compute_fold_metrics(y_test, test_pred)

            fold_results.append(
                FoldOptResult(
                    fold_index=fold_idx,
                    params=params,
                    train_score=train_metrics[config.scoring_metric],
                    test_score=test_metrics[config.scoring_metric],
                )
            )
            scores.append(test_metrics[config.scoring_metric])

        if not scores:
            continue

        mean_score = np.nanmean(scores)

        # Log trial
        if trial_counter:
            trial_counter.log_trial(
                model_type=config.model_type,
                hyperparameters=params,
                metric_name=config.scoring_metric,
                metric_value=mean_score,
            )

        all_trials.append({"params": params, "score": mean_score, "fold_scores": scores})

        # Check if best (IC: higher is better, MSE: lower is better)
        is_better = (
            mean_score > best_score
            if config.scoring_metric == "ic"
            else mean_score < best_score
        )
        if is_better:
            best_score = mean_score
            best_params = params
            best_fold_results = fold_results

    if best_params is None:
        raise ValueError("No valid parameter combinations found")

    # Build cv_results DataFrame
    cv_results = pd.DataFrame(
        [
            {**t["params"], "mean_test_score": t["score"], "std_test_score": np.std(t["fold_scores"])}
            for t in all_trials
        ]
    )

    logger.info(f"Optimization complete. Best params: {best_params}, score: {best_score:.4f}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        cv_results=cv_results,
        fold_results=best_fold_results,
        all_trials=all_trials,
        hypothesis_id=config.hypothesis_id,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::test_cross_validated_optimize_returns_best_params -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(ml): implement cross_validated_optimize core function"
```

---

### Task 1.6: Add integration tests for trial counter and early stopping

**Files:**
- Modify: `tests/test_ml/test_optimization.py`

**Step 1: Write tests**

```python
# Add to tests/test_ml/test_optimization.py

def test_respects_max_trials_limit(test_db):
    """Test optimization respects max_trials limit."""
    from hrp.ml.optimization import OptimizationConfig, cross_validated_optimize
    from hrp.data.db import get_db
    from datetime import timedelta

    db = get_db(test_db)

    # Setup minimal test data
    symbols = ["AAPL"]
    for sym in symbols:
        db.execute(
            "INSERT INTO symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
            (sym,),
        )

    start = date(2020, 1, 1)
    for i in range(400):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for sym in symbols:
            db.execute(
                """INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, 100, 101, 99, 100.5, 100.5, 1000000, 'test')""",
                (sym, d),
            )
            db.execute(
                """INSERT INTO features (symbol, date, feature_name, value)
                VALUES (?, ?, 'momentum_20d', ?), (?, ?, 'returns_20d', ?)""",
                (sym, d, np.random.randn(), sym, d, np.random.randn() * 0.01),
            )

    config = OptimizationConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d"],
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},  # 5 combos
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 31),
        n_folds=2,
        max_trials=3,  # Limit to 3
    )

    result = cross_validated_optimize(config, symbols, log_to_mlflow=False)

    # Should only have tested 3 combinations
    assert len(result.all_trials) <= 3


def test_integrates_with_trial_counter(test_db):
    """Test optimization integrates with HyperparameterTrialCounter."""
    from hrp.ml.optimization import OptimizationConfig, cross_validated_optimize
    from hrp.risk.overfitting import HyperparameterTrialCounter
    from hrp.data.db import get_db
    from datetime import timedelta

    db = get_db(test_db)

    symbols = ["AAPL"]
    for sym in symbols:
        db.execute(
            "INSERT INTO symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
            (sym,),
        )

    start = date(2020, 1, 1)
    for i in range(400):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for sym in symbols:
            db.execute(
                """INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, 100, 101, 99, 100.5, 100.5, 1000000, 'test')""",
                (sym, d),
            )
            db.execute(
                """INSERT INTO features (symbol, date, feature_name, value)
                VALUES (?, ?, 'momentum_20d', ?), (?, ?, 'returns_20d', ?)""",
                (sym, d, np.random.randn(), sym, d, np.random.randn() * 0.01),
            )

    hypothesis_id = "HYP-TEST-001"
    config = OptimizationConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d"],
        param_grid={"alpha": [1.0, 10.0]},
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 31),
        n_folds=2,
        max_trials=50,
        hypothesis_id=hypothesis_id,
    )

    result = cross_validated_optimize(config, symbols, log_to_mlflow=False)

    # Verify trials were logged
    counter = HyperparameterTrialCounter(hypothesis_id, max_trials=50)
    assert counter.trial_count >= len(result.all_trials)
```

**Step 2: Run tests**

Run: `pytest tests/test_ml/test_optimization.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_ml/test_optimization.py
git commit -m "test(ml): add integration tests for optimization trial limits"
```

---

### Task 1.7: Add MLflow logging to optimization

**Files:**
- Modify: `hrp/ml/optimization.py`
- Modify: `tests/test_ml/test_optimization.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_ml/test_optimization.py
from unittest.mock import patch


def test_logs_to_mlflow_when_enabled(test_db):
    """Test optimization logs to MLflow when enabled."""
    from hrp.ml.optimization import OptimizationConfig, cross_validated_optimize
    from hrp.data.db import get_db
    from datetime import timedelta

    db = get_db(test_db)

    symbols = ["AAPL"]
    for sym in symbols:
        db.execute(
            "INSERT INTO symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
            (sym,),
        )

    start = date(2020, 1, 1)
    for i in range(400):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for sym in symbols:
            db.execute(
                """INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, 100, 101, 99, 100.5, 100.5, 1000000, 'test')""",
                (sym, d),
            )
            db.execute(
                """INSERT INTO features (symbol, date, feature_name, value)
                VALUES (?, ?, 'momentum_20d', ?), (?, ?, 'returns_20d', ?)""",
                (sym, d, np.random.randn(), sym, d, np.random.randn() * 0.01),
            )

    config = OptimizationConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d"],
        param_grid={"alpha": [1.0]},
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 31),
        n_folds=2,
    )

    with patch("hrp.ml.optimization.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = lambda x: None
        mock_mlflow.start_run.return_value.__exit__ = lambda x, *args: None

        result = cross_validated_optimize(config, symbols, log_to_mlflow=True)

        # Verify MLflow was called
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_metric.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::test_logs_to_mlflow_when_enabled -v`
Expected: May pass or fail depending on implementation state

**Step 3: Add MLflow logging to implementation**

```python
# At top of hrp/ml/optimization.py, add import:
import mlflow

# In cross_validated_optimize, wrap the main logic:
def cross_validated_optimize(
    config: OptimizationConfig,
    symbols: list[str],
    log_to_mlflow: bool = True,
) -> OptimizationResult:
    """..."""
    logger.info(f"Starting cross-validated optimization for {config.model_type}")

    # MLflow context
    mlflow_run = None
    if log_to_mlflow:
        mlflow_run = mlflow.start_run(run_name=f"cv_opt_{config.model_type}")

    try:
        # ... existing implementation ...

        # After finding best_params, log to MLflow
        if log_to_mlflow and mlflow_run:
            mlflow.log_param("model_type", config.model_type)
            mlflow.log_param("n_folds", config.n_folds)
            mlflow.log_param("scoring_metric", config.scoring_metric)
            mlflow.log_param("n_trials", len(all_trials))
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            mlflow.log_metric("best_score", best_score)

        return OptimizationResult(...)

    finally:
        if log_to_mlflow and mlflow_run:
            mlflow.end_run()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::test_logs_to_mlflow_when_enabled -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(ml): add MLflow logging to cross_validated_optimize"
```

---

### Task 1.8: Export optimization module from hrp.ml

**Files:**
- Modify: `hrp/ml/__init__.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_ml/test_optimization.py

def test_optimization_importable_from_hrp_ml():
    """Test optimization classes are importable from hrp.ml."""
    from hrp.ml import OptimizationConfig, OptimizationResult, cross_validated_optimize

    assert OptimizationConfig is not None
    assert OptimizationResult is not None
    assert cross_validated_optimize is not None
```

**Step 2: Run test**

Run: `pytest tests/test_ml/test_optimization.py::test_optimization_importable_from_hrp_ml -v`
Expected: FAIL with "cannot import name 'OptimizationConfig' from 'hrp.ml'"

**Step 3: Update __init__.py**

```python
# hrp/ml/__init__.py - add to existing exports
from hrp.ml.optimization import (
    OptimizationConfig,
    OptimizationResult,
    FoldOptResult,
    cross_validated_optimize,
)
```

**Step 4: Run test**

Run: `pytest tests/test_ml/test_optimization.py::test_optimization_importable_from_hrp_ml -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/__init__.py tests/test_ml/test_optimization.py
git commit -m "feat(ml): export optimization module from hrp.ml"
```

---

## Phase 2: Parallel Parameter Sweeps with Sharpe Decay Analysis

### Task 2.1: Create SweepConstraint and SweepConfig dataclasses

**Files:**
- Create: `hrp/research/parameter_sweep.py`
- Create: `tests/test_research/test_parameter_sweep.py`

**Step 1: Write the failing test**

```python
# tests/test_research/test_parameter_sweep.py
"""Tests for parallel parameter sweeps."""

import pytest
from datetime import date


def test_sweep_constraint_creation():
    """Test SweepConstraint dataclass creation."""
    from hrp.research.parameter_sweep import SweepConstraint

    constraint = SweepConstraint(
        constraint_type="difference_min",
        params=["slow_period", "fast_period"],
        value=5.0,
    )

    assert constraint.constraint_type == "difference_min"
    assert constraint.params == ["slow_period", "fast_period"]
    assert constraint.value == 5.0


def test_sweep_config_creation_with_defaults():
    """Test SweepConfig dataclass with default values."""
    from hrp.research.parameter_sweep import SweepConfig, SweepConstraint

    config = SweepConfig(
        strategy_type="momentum",
        param_ranges={"fast_period": [10, 20, 30], "slow_period": [20, 30, 40]},
        constraints=[SweepConstraint("difference_min", ["slow_period", "fast_period"], 5)],
        symbols=["AAPL", "MSFT"],
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
    )

    assert config.n_folds == 5  # default
    assert config.n_jobs == -1  # default
    assert config.scoring == "sharpe_ratio"  # default
    assert config.aggregation == "median"  # default
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_research/test_parameter_sweep.py -v`
Expected: FAIL with "cannot import name 'SweepConstraint'"

**Step 3: Write minimal implementation**

```python
# hrp/research/parameter_sweep.py
"""
Parallel parameter sweeps with constraint validation and Sharpe decay analysis.

Provides efficient parallel parameter exploration with constraint validation
for multi-factor strategies, including Sharpe decay analysis to identify
parameter combinations that generalize well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class SweepConstraint:
    """
    Constraint on parameter combinations.

    Supported constraint_types:
    - "sum_equals": params must sum to value
    - "max_total": sum of params <= value
    - "ratio_bound": ratio between params bounded by value
    - "difference_min": first param - second param >= value
    - "exclusion": params cannot all be non-zero

    Example:
        SweepConstraint("difference_min", ["slow_period", "fast_period"], 5)
        Enforces: slow_period - fast_period >= 5
    """

    constraint_type: str
    params: list[str]
    value: float

    def __post_init__(self) -> None:
        valid_types = {"sum_equals", "max_total", "ratio_bound", "difference_min", "exclusion"}
        if self.constraint_type not in valid_types:
            raise ValueError(
                f"Invalid constraint_type '{self.constraint_type}'. "
                f"Must be one of: {valid_types}"
            )


@dataclass
class SweepConfig:
    """
    Configuration for parallel parameter sweep.

    Attributes:
        strategy_type: Type of strategy ("multifactor", "ml_predicted", "momentum")
        param_ranges: Dict mapping param names to lists of values to test
        constraints: List of constraints on parameter combinations
        symbols: Symbols to include in backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        n_folds: Number of CV folds for Sharpe decay analysis
        n_jobs: Number of parallel jobs (-1 = all cores)
        scoring: Metric to optimize ("sharpe_ratio", "total_return", etc.)
        min_samples: Minimum samples per backtest
        aggregation: How to aggregate across folds ("median" or "mean")
    """

    strategy_type: str
    param_ranges: dict[str, list[Any]]
    constraints: list[SweepConstraint]
    symbols: list[str]
    start_date: date
    end_date: date
    n_folds: int = 5
    n_jobs: int = -1
    scoring: str = "sharpe_ratio"
    min_samples: int = 100
    aggregation: str = "median"

    def __post_init__(self) -> None:
        valid_strategies = {"multifactor", "ml_predicted", "momentum"}
        if self.strategy_type not in valid_strategies:
            raise ValueError(
                f"Invalid strategy_type '{self.strategy_type}'. "
                f"Must be one of: {valid_strategies}"
            )
        if self.aggregation not in ("median", "mean"):
            raise ValueError(
                f"aggregation must be 'median' or 'mean', got '{self.aggregation}'"
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_research/test_parameter_sweep.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/parameter_sweep.py tests/test_research/test_parameter_sweep.py
git commit -m "feat(research): add SweepConstraint and SweepConfig dataclasses"
```

---

### Task 2.2: Implement validate_constraints function

**Files:**
- Modify: `hrp/research/parameter_sweep.py`
- Modify: `tests/test_research/test_parameter_sweep.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_research/test_parameter_sweep.py

def test_validate_difference_min_constraint_passes():
    """Test difference_min constraint passes when satisfied."""
    from hrp.research.parameter_sweep import SweepConstraint, validate_constraints

    constraints = [SweepConstraint("difference_min", ["slow_period", "fast_period"], 5)]
    params = {"slow_period": 30, "fast_period": 20}

    assert validate_constraints(params, constraints) is True


def test_validate_difference_min_constraint_fails():
    """Test difference_min constraint fails when not satisfied."""
    from hrp.research.parameter_sweep import SweepConstraint, validate_constraints

    constraints = [SweepConstraint("difference_min", ["slow_period", "fast_period"], 5)]
    params = {"slow_period": 22, "fast_period": 20}  # diff = 2 < 5

    assert validate_constraints(params, constraints) is False


def test_validate_sum_equals_constraint():
    """Test sum_equals constraint validation."""
    from hrp.research.parameter_sweep import SweepConstraint, validate_constraints

    constraints = [SweepConstraint("sum_equals", ["w1", "w2", "w3"], 1.0)]

    assert validate_constraints({"w1": 0.5, "w2": 0.3, "w3": 0.2}, constraints) is True
    assert validate_constraints({"w1": 0.5, "w2": 0.3, "w3": 0.3}, constraints) is False


def test_validate_max_total_constraint():
    """Test max_total constraint validation."""
    from hrp.research.parameter_sweep import SweepConstraint, validate_constraints

    constraints = [SweepConstraint("max_total", ["a", "b"], 10)]

    assert validate_constraints({"a": 3, "b": 5}, constraints) is True
    assert validate_constraints({"a": 6, "b": 5}, constraints) is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_research/test_parameter_sweep.py::test_validate_difference_min_constraint_passes -v`
Expected: FAIL with "cannot import name 'validate_constraints'"

**Step 3: Write minimal implementation**

```python
# Add to hrp/research/parameter_sweep.py

def validate_constraints(
    params: dict[str, Any],
    constraints: list[SweepConstraint],
) -> bool:
    """
    Check if parameter combination satisfies all constraints.

    Args:
        params: Dict of parameter name -> value
        constraints: List of constraints to check

    Returns:
        True if all constraints are satisfied, False otherwise
    """
    for constraint in constraints:
        values = [params.get(p, 0) for p in constraint.params]

        if constraint.constraint_type == "difference_min":
            # First param - second param >= value
            if len(values) < 2:
                continue
            if values[0] - values[1] < constraint.value:
                return False

        elif constraint.constraint_type == "sum_equals":
            # Sum must equal value (with small tolerance)
            if abs(sum(values) - constraint.value) > 1e-6:
                return False

        elif constraint.constraint_type == "max_total":
            # Sum must be <= value
            if sum(values) > constraint.value:
                return False

        elif constraint.constraint_type == "ratio_bound":
            # Ratio between first two params <= value
            if len(values) < 2 or values[1] == 0:
                continue
            if values[0] / values[1] > constraint.value:
                return False

        elif constraint.constraint_type == "exclusion":
            # Not all params can be non-zero
            if all(v != 0 for v in values):
                return False

    return True
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_research/test_parameter_sweep.py -v -k "validate"
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add hrp/research/parameter_sweep.py tests/test_research/test_parameter_sweep.py
git commit -m "feat(research): implement validate_constraints function"
```

---

### Task 2.3: Create SweepResult dataclass with Sharpe decay fields

**Files:**
- Modify: `hrp/research/parameter_sweep.py`
- Modify: `tests/test_research/test_parameter_sweep.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_research/test_parameter_sweep.py

def test_sweep_result_creation():
    """Test SweepResult dataclass creation with Sharpe decay fields."""
    from hrp.research.parameter_sweep import SweepResult
    import numpy as np

    results_df = pd.DataFrame({
        "fast_period": [10, 20],
        "slow_period": [20, 30],
        "train_sharpe": [1.5, 1.2],
        "test_sharpe": [1.2, 1.0],
    })

    train_sharpe_matrix = pd.DataFrame({0: [1.5, 1.2], 1: [1.4, 1.1]})
    test_sharpe_matrix = pd.DataFrame({0: [1.2, 1.0], 1: [1.1, 0.9]})
    sharpe_diff_matrix = test_sharpe_matrix - train_sharpe_matrix
    sharpe_diff_median = sharpe_diff_matrix.median(axis=1)

    result = SweepResult(
        results_df=results_df,
        best_params={"fast_period": 10, "slow_period": 20},
        best_metrics={"sharpe_ratio": 1.2, "total_return": 0.15},
        train_sharpe_matrix=train_sharpe_matrix,
        test_sharpe_matrix=test_sharpe_matrix,
        sharpe_diff_matrix=sharpe_diff_matrix,
        sharpe_diff_median=sharpe_diff_median,
        constraint_violations=2,
        execution_time_seconds=5.5,
        generalization_score=0.75,
    )

    assert result.best_params["fast_period"] == 10
    assert result.generalization_score == 0.75
    assert len(result.sharpe_diff_median) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_research/test_parameter_sweep.py::test_sweep_result_creation -v`
Expected: FAIL with "cannot import name 'SweepResult'"

**Step 3: Write minimal implementation**

```python
# Add to hrp/research/parameter_sweep.py after SweepConfig

@dataclass
class SweepResult:
    """
    Result of parameter sweep with train/test analysis.

    Attributes:
        results_df: All param combos with per-fold metrics
        best_params: Best parameter combination found
        best_metrics: Metrics for best params

        Sharpe decay analysis (key from PyQuantNews):
        train_sharpe_matrix: Params as index, folds as columns
        test_sharpe_matrix: Test Sharpe per param/fold
        sharpe_diff_matrix: test - train (positive = good)
        sharpe_diff_median: Aggregated across folds

        Metadata:
        constraint_violations: Number of param combos that violated constraints
        execution_time_seconds: Total sweep time
        generalization_score: % of params where test >= train
    """

    results_df: pd.DataFrame
    best_params: dict[str, Any]
    best_metrics: dict[str, float]

    # Sharpe decay analysis
    train_sharpe_matrix: pd.DataFrame
    test_sharpe_matrix: pd.DataFrame
    sharpe_diff_matrix: pd.DataFrame
    sharpe_diff_median: pd.Series

    # Metadata
    constraint_violations: int
    execution_time_seconds: float
    generalization_score: float
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_research/test_parameter_sweep.py::test_sweep_result_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/parameter_sweep.py tests/test_research/test_parameter_sweep.py
git commit -m "feat(research): add SweepResult dataclass with Sharpe decay fields"
```

---

### Task 2.4: Implement compute_sharpe_diff_analysis function

**Files:**
- Modify: `hrp/research/parameter_sweep.py`
- Modify: `tests/test_research/test_parameter_sweep.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_research/test_parameter_sweep.py
import numpy as np


def test_compute_sharpe_diff_analysis():
    """Test Sharpe diff analysis computation."""
    from hrp.research.parameter_sweep import compute_sharpe_diff_analysis

    results_df = pd.DataFrame({
        "fast_period": [10, 10, 20, 20],
        "slow_period": [20, 20, 30, 30],
        "fold": [0, 1, 0, 1],
        "train_sharpe": [1.5, 1.4, 1.2, 1.1],
        "test_sharpe": [1.2, 1.3, 0.9, 0.8],  # diff: -0.3, -0.1, -0.3, -0.3
    })

    diff_matrix, diff_agg, gen_score = compute_sharpe_diff_analysis(
        results_df,
        param_columns=["fast_period", "slow_period"],
        aggregation="median",
    )

    # Should have 2 rows (unique param combos)
    assert len(diff_agg) == 2
    # First combo has positive median diff (1.3-1.4=-0.1 better than 1.2-1.5=-0.3)
    assert gen_score >= 0  # At least some generalize


def test_generalization_score_calculation():
    """Test generalization score = % of combos where test >= train."""
    from hrp.research.parameter_sweep import compute_sharpe_diff_analysis

    # All positive diffs = 100% generalization
    results_df = pd.DataFrame({
        "param": [1, 1, 2, 2],
        "fold": [0, 1, 0, 1],
        "train_sharpe": [1.0, 1.0, 1.0, 1.0],
        "test_sharpe": [1.1, 1.2, 1.0, 1.1],
    })

    _, _, gen_score = compute_sharpe_diff_analysis(
        results_df,
        param_columns=["param"],
        aggregation="median",
    )

    assert gen_score == 1.0  # 100% generalization
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_research/test_parameter_sweep.py::test_compute_sharpe_diff_analysis -v`
Expected: FAIL with "cannot import name 'compute_sharpe_diff_analysis'"

**Step 3: Write minimal implementation**

```python
# Add to hrp/research/parameter_sweep.py
import numpy as np


def compute_sharpe_diff_analysis(
    results_df: pd.DataFrame,
    param_columns: list[str],
    aggregation: str = "median",
) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Compute Sharpe ratio diff analysis across parameter combinations.

    This is the key diagnostic from PyQuantNews thread:
    sharpe_diff = test_sharpe - train_sharpe
    Blue (positive) = good generalization
    Red (negative) = overfitting

    Args:
        results_df: DataFrame with param columns, fold, train_sharpe, test_sharpe
        param_columns: Columns that define parameter combinations
        aggregation: "median" or "mean" across folds

    Returns:
        - sharpe_diff_matrix: Full matrix of test-train diffs per fold
        - sharpe_diff_agg: Aggregated (median/mean) diff per param combo
        - generalization_score: % of combos where agg_diff >= 0
    """
    # Compute diff
    results_df = results_df.copy()
    results_df["sharpe_diff"] = results_df["test_sharpe"] - results_df["train_sharpe"]

    # Pivot to matrix: params as index, folds as columns
    pivot_cols = param_columns + ["fold"]
    if not all(col in results_df.columns for col in pivot_cols):
        raise ValueError(f"Missing required columns. Need: {pivot_cols}")

    # Group by param combo and aggregate
    grouped = results_df.groupby(param_columns)["sharpe_diff"]

    if aggregation == "median":
        sharpe_diff_agg = grouped.median()
    else:
        sharpe_diff_agg = grouped.mean()

    # Create diff matrix (params x folds)
    sharpe_diff_matrix = results_df.pivot_table(
        values="sharpe_diff",
        index=param_columns,
        columns="fold",
        aggfunc="first",
    )

    # Generalization score: % where aggregated diff >= 0
    n_generalizing = (sharpe_diff_agg >= 0).sum()
    generalization_score = n_generalizing / len(sharpe_diff_agg) if len(sharpe_diff_agg) > 0 else 0.0

    return sharpe_diff_matrix, sharpe_diff_agg, float(generalization_score)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_research/test_parameter_sweep.py -v -k "sharpe_diff or generalization"
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/parameter_sweep.py tests/test_research/test_parameter_sweep.py
git commit -m "feat(research): implement compute_sharpe_diff_analysis function"
```

---

### Task 2.5: Implement parallel_parameter_sweep function

**Files:**
- Modify: `hrp/research/parameter_sweep.py`
- Modify: `tests/test_research/test_parameter_sweep.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_research/test_parameter_sweep.py

def test_parallel_parameter_sweep(test_db):
    """Test parallel_parameter_sweep runs and returns valid result."""
    from hrp.research.parameter_sweep import (
        SweepConfig,
        SweepConstraint,
        parallel_parameter_sweep,
    )
    from hrp.data.db import get_db
    from datetime import timedelta

    db = get_db(test_db)

    # Setup test data
    symbols = ["AAPL", "MSFT"]
    for sym in symbols:
        db.execute(
            "INSERT INTO symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
            (sym,),
        )

    start = date(2020, 1, 1)
    for i in range(500):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for sym in symbols:
            price = 100 + i * 0.1
            db.execute(
                """INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1000000, 'test')""",
                (sym, d, price * 0.99, price * 1.01, price * 0.98, price, price),
            )

    config = SweepConfig(
        strategy_type="momentum",
        param_ranges={"fast_period": [10, 20], "slow_period": [30, 40]},
        constraints=[SweepConstraint("difference_min", ["slow_period", "fast_period"], 5)],
        symbols=symbols,
        start_date=date(2020, 1, 1),
        end_date=date(2021, 6, 30),
        n_folds=2,
        n_jobs=1,  # Single-threaded for test
    )

    result = parallel_parameter_sweep(config)

    assert result.best_params is not None
    assert "fast_period" in result.best_params
    assert result.generalization_score >= 0
    assert result.generalization_score <= 1
    assert len(result.results_df) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_research/test_parameter_sweep.py::test_parallel_parameter_sweep -v`
Expected: FAIL with "cannot import name 'parallel_parameter_sweep'"

**Step 3: Write implementation** (This is a longer implementation)

```python
# Add to hrp/research/parameter_sweep.py
import itertools
import time
from loguru import logger
from joblib import Parallel, delayed

from hrp.research.backtest import get_price_data, run_backtest
from hrp.research.config import BacktestConfig
from hrp.research.strategies import generate_momentum_signals


def _generate_param_combinations(param_ranges: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations of parameters from ranges."""
    if not param_ranges:
        return [{}]

    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def _run_single_combo(
    params: dict[str, Any],
    prices: pd.DataFrame,
    symbols: list[str],
    start_date: date,
    end_date: date,
    fold_dates: list[tuple[date, date, date, date]],
    strategy_type: str,
) -> dict[str, Any]:
    """Run a single parameter combination across all folds."""
    fold_results = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(fold_dates):
        try:
            # Generate signals based on strategy type
            if strategy_type == "momentum":
                fast = params.get("fast_period", 20)
                slow = params.get("slow_period", 50)
                signals = generate_momentum_signals(prices, fast_period=fast, slow_period=slow)
            else:
                # Default: use params directly
                signals = generate_momentum_signals(prices, **params)

            # Train period backtest
            train_config = BacktestConfig(
                symbols=symbols,
                start_date=train_start,
                end_date=train_end,
            )
            train_signals = signals.loc[train_start:train_end]
            train_prices = prices.loc[train_start:train_end]
            if len(train_signals) > 0 and len(train_prices) > 0:
                train_result = run_backtest(train_signals, train_config, train_prices)
                train_sharpe = train_result.sharpe
            else:
                train_sharpe = 0.0

            # Test period backtest
            test_config = BacktestConfig(
                symbols=symbols,
                start_date=test_start,
                end_date=test_end,
            )
            test_signals = signals.loc[test_start:test_end]
            test_prices = prices.loc[test_start:test_end]
            if len(test_signals) > 0 and len(test_prices) > 0:
                test_result = run_backtest(test_signals, test_config, test_prices)
                test_sharpe = test_result.sharpe
            else:
                test_sharpe = 0.0

            fold_results.append({
                "fold": fold_idx,
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
            })

        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed for params {params}: {e}")
            fold_results.append({
                "fold": fold_idx,
                "train_sharpe": 0.0,
                "test_sharpe": 0.0,
            })

    return {"params": params, "fold_results": fold_results}


def parallel_parameter_sweep(
    config: SweepConfig,
    hypothesis_id: str | None = None,
) -> SweepResult:
    """
    Run parallel parameter sweep with constraint validation.

    Args:
        config: Sweep configuration
        hypothesis_id: Optional hypothesis ID for audit trail

    Returns:
        SweepResult with best params, Sharpe decay analysis, and metrics
    """
    start_time = time.time()
    logger.info(f"Starting parameter sweep for {config.strategy_type}")

    # Load price data
    prices = get_price_data(config.symbols, config.start_date, config.end_date)

    if prices.empty:
        raise ValueError("No price data available for specified date range")

    # Generate fold dates
    available_dates = sorted(prices.index.unique())
    n_dates = len(available_dates)
    min_train = max(100, n_dates // (config.n_folds + 1))
    test_size = (n_dates - min_train) // config.n_folds

    fold_dates = []
    for fold_idx in range(config.n_folds):
        test_start_idx = min_train + fold_idx * test_size
        test_end_idx = min(test_start_idx + test_size - 1, n_dates - 1)
        train_start = available_dates[0]
        train_end = available_dates[test_start_idx - 1]
        test_start = available_dates[test_start_idx]
        test_end = available_dates[test_end_idx]
        fold_dates.append((train_start, train_end, test_start, test_end))

    # Generate and filter parameter combinations
    all_combos = _generate_param_combinations(config.param_ranges)
    valid_combos = [c for c in all_combos if validate_constraints(c, config.constraints)]
    constraint_violations = len(all_combos) - len(valid_combos)

    logger.info(
        f"Testing {len(valid_combos)} valid combos "
        f"({constraint_violations} violated constraints)"
    )

    # Run sweeps (parallel or sequential)
    if config.n_jobs == 1:
        results = [
            _run_single_combo(
                params, prices, config.symbols, config.start_date, config.end_date,
                fold_dates, config.strategy_type
            )
            for params in valid_combos
        ]
    else:
        results = Parallel(n_jobs=config.n_jobs)(
            delayed(_run_single_combo)(
                params, prices, config.symbols, config.start_date, config.end_date,
                fold_dates, config.strategy_type
            )
            for params in valid_combos
        )

    # Compile results into DataFrame
    rows = []
    for r in results:
        params = r["params"]
        for fr in r["fold_results"]:
            row = {**params, **fr}
            rows.append(row)

    results_df = pd.DataFrame(rows)
    param_columns = list(config.param_ranges.keys())

    # Compute Sharpe decay analysis
    sharpe_diff_matrix, sharpe_diff_median, generalization_score = compute_sharpe_diff_analysis(
        results_df,
        param_columns=param_columns,
        aggregation=config.aggregation,
    )

    # Build train/test Sharpe matrices
    train_sharpe_matrix = results_df.pivot_table(
        values="train_sharpe",
        index=param_columns,
        columns="fold",
        aggfunc="first",
    )
    test_sharpe_matrix = results_df.pivot_table(
        values="test_sharpe",
        index=param_columns,
        columns="fold",
        aggfunc="first",
    )

    # Find best params (highest aggregated test Sharpe)
    agg_func = np.median if config.aggregation == "median" else np.mean
    test_sharpe_agg = results_df.groupby(param_columns)["test_sharpe"].apply(agg_func)

    best_idx = test_sharpe_agg.idxmax()
    if isinstance(best_idx, tuple):
        best_params = dict(zip(param_columns, best_idx))
    else:
        best_params = {param_columns[0]: best_idx}

    best_test_sharpe = test_sharpe_agg.loc[best_idx]
    best_metrics = {"sharpe_ratio": float(best_test_sharpe)}

    execution_time = time.time() - start_time
    logger.info(
        f"Sweep complete in {execution_time:.1f}s. "
        f"Best params: {best_params}, generalization: {generalization_score:.1%}"
    )

    return SweepResult(
        results_df=results_df,
        best_params=best_params,
        best_metrics=best_metrics,
        train_sharpe_matrix=train_sharpe_matrix,
        test_sharpe_matrix=test_sharpe_matrix,
        sharpe_diff_matrix=sharpe_diff_matrix,
        sharpe_diff_median=sharpe_diff_median,
        constraint_violations=constraint_violations,
        execution_time_seconds=execution_time,
        generalization_score=generalization_score,
    )
```

Note: You'll also need to add `generate_momentum_signals` to `hrp/research/strategies.py` if it doesn't exist. Check the existing strategies module for the correct pattern.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_research/test_parameter_sweep.py::test_parallel_parameter_sweep -v`
Expected: PASS (may need adjustments based on existing strategies module)

**Step 5: Commit**

```bash
git add hrp/research/parameter_sweep.py tests/test_research/test_parameter_sweep.py
git commit -m "feat(research): implement parallel_parameter_sweep with Sharpe decay"
```

---

## Phase 3: ATR-Based Trailing Stops

### Task 3.1: Add StopLossConfig to config.py

**Files:**
- Modify: `hrp/research/config.py`
- Modify: `tests/test_research/test_config.py` (or create if needed)

**Step 1: Write the failing test**

```python
# tests/test_research/test_config.py (create or add to)
"""Tests for research configuration."""

import pytest
from datetime import date


def test_stop_loss_config_creation():
    """Test StopLossConfig dataclass creation."""
    from hrp.research.config import StopLossConfig

    config = StopLossConfig(
        enabled=True,
        type="atr_trailing",
        atr_multiplier=2.5,
        atr_period=14,
    )

    assert config.enabled is True
    assert config.type == "atr_trailing"
    assert config.atr_multiplier == 2.5


def test_stop_loss_config_defaults():
    """Test StopLossConfig has sensible defaults."""
    from hrp.research.config import StopLossConfig

    config = StopLossConfig()

    assert config.enabled is False
    assert config.type == "atr_trailing"
    assert config.atr_multiplier == 2.0
    assert config.atr_period == 14
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_research/test_config.py::test_stop_loss_config_creation -v`
Expected: FAIL with "cannot import name 'StopLossConfig'"

**Step 3: Write minimal implementation**

```python
# Add to hrp/research/config.py after CostModel

@dataclass
class StopLossConfig:
    """
    Configuration for stop-loss mechanisms.

    Attributes:
        enabled: Whether stop-loss is active
        type: Stop type ("fixed_pct", "atr_trailing", "volatility_scaled")
        atr_multiplier: Multiplier for ATR-based stops
        atr_period: Period for ATR calculation
        fixed_pct: Fixed percentage for fixed stops
        lookback_for_high: Bars to look back for trailing high
    """

    enabled: bool = False
    type: str = "atr_trailing"
    atr_multiplier: float = 2.0
    atr_period: int = 14
    fixed_pct: float = 0.05
    lookback_for_high: int = 1

    def __post_init__(self) -> None:
        valid_types = {"fixed_pct", "atr_trailing", "volatility_scaled"}
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid stop type '{self.type}'. Must be one of: {valid_types}"
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_research/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/config.py tests/test_research/test_config.py
git commit -m "feat(research): add StopLossConfig dataclass"
```

---

### Task 3.2: Create stops module with compute_atr_stops

**Files:**
- Create: `hrp/research/stops.py`
- Create: `tests/test_research/test_stops.py`

**Step 1: Write the failing test**

```python
# tests/test_research/test_stops.py
"""Tests for trailing stop implementation."""

import pytest
import pandas as pd
import numpy as np
from datetime import date


def test_compute_atr_stops_returns_stop_levels():
    """Test ATR stops are computed below entry prices."""
    from hrp.research.stops import compute_atr_stops

    # Create sample OHLC data
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    prices = pd.DataFrame({
        ("AAPL", "high"): np.linspace(100, 110, 30) + np.random.randn(30) * 0.5,
        ("AAPL", "low"): np.linspace(100, 110, 30) - np.random.randn(30) * 0.5 - 1,
        ("AAPL", "close"): np.linspace(100, 110, 30),
    }, index=dates)

    # Entry signals on day 15
    entries = pd.DataFrame({
        "AAPL": [0] * 14 + [1] + [0] * 15,
    }, index=dates)

    stops = compute_atr_stops(
        prices=prices,
        entries=entries,
        atr_multiplier=2.0,
        atr_period=14,
    )

    assert isinstance(stops, pd.DataFrame)
    assert "AAPL" in stops.columns
    # Stop level should be below entry price after entry
    assert stops.loc[dates[15], "AAPL"] < prices.loc[dates[15], ("AAPL", "close")]


def test_stop_level_below_price():
    """Test stop is always below current price."""
    from hrp.research.stops import compute_atr_stops

    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    close_prices = np.linspace(100, 120, 50)
    prices = pd.DataFrame({
        ("AAPL", "high"): close_prices + 2,
        ("AAPL", "low"): close_prices - 2,
        ("AAPL", "close"): close_prices,
    }, index=dates)

    entries = pd.DataFrame({"AAPL": [0] * 20 + [1] + [0] * 29}, index=dates)

    stops = compute_atr_stops(prices, entries, atr_multiplier=2.0, atr_period=14)

    # After entry (day 20), stop should always be below close
    for i in range(20, 50):
        if not np.isnan(stops.iloc[i]["AAPL"]):
            assert stops.iloc[i]["AAPL"] < prices.iloc[i][("AAPL", "close")]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_research/test_stops.py::test_compute_atr_stops_returns_stop_levels -v`
Expected: FAIL with "No module named 'hrp.research.stops'"

**Step 3: Write minimal implementation**

```python
# hrp/research/stops.py
"""
Trailing stop implementation.

Provides ATR-based and fixed-percentage trailing stops for backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class StopResult:
    """Result of stop-loss check for a position."""

    triggered: bool
    trigger_date: date | None
    trigger_price: float | None
    stop_level: float
    pnl_at_stop: float | None


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_atr_stops(
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Compute ATR-based trailing stop levels.

    Args:
        prices: MultiIndex columns (symbol, field) with high/low/close
        entries: DataFrame of entry signals (1 = entry, 0 = no entry)
        atr_multiplier: Multiplier for ATR to set stop distance
        atr_period: Period for ATR calculation

    Returns:
        DataFrame of stop levels with same shape as entries
    """
    symbols = entries.columns.tolist()
    stop_levels = pd.DataFrame(index=entries.index, columns=symbols, dtype=float)

    for symbol in symbols:
        # Extract OHLC for this symbol
        try:
            high = prices[(symbol, "high")]
            low = prices[(symbol, "low")]
            close = prices[(symbol, "close")]
        except KeyError:
            logger.warning(f"Missing OHLC data for {symbol}")
            continue

        # Compute ATR
        atr = _compute_atr(high, low, close, atr_period)

        # Track position and trailing stop
        in_position = False
        trailing_high = 0.0
        stop_level = np.nan

        for i, idx in enumerate(entries.index):
            entry_signal = entries.loc[idx, symbol]
            current_close = close.loc[idx]
            current_atr = atr.loc[idx]

            if entry_signal == 1 and not in_position:
                # New entry
                in_position = True
                trailing_high = current_close
                stop_level = current_close - atr_multiplier * current_atr

            elif in_position:
                # Update trailing high and stop
                if current_close > trailing_high:
                    trailing_high = current_close
                    stop_level = trailing_high - atr_multiplier * current_atr

            stop_levels.loc[idx, symbol] = stop_level if in_position else np.nan

    return stop_levels
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_research/test_stops.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/stops.py tests/test_research/test_stops.py
git commit -m "feat(research): add compute_atr_stops function"
```

---

### Task 3.3: Implement apply_trailing_stops function

**Files:**
- Modify: `hrp/research/stops.py`
- Modify: `tests/test_research/test_stops.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_research/test_stops.py

def test_apply_trailing_stops_generates_exits():
    """Test apply_trailing_stops generates exit signals when stop hit."""
    from hrp.research.stops import apply_trailing_stops
    from hrp.research.config import StopLossConfig

    dates = pd.date_range("2020-01-01", periods=30, freq="B")

    # Price rises then falls
    close_prices = list(range(100, 115)) + list(range(114, 99, -1))
    prices = pd.DataFrame({
        ("AAPL", "high"): [p + 2 for p in close_prices],
        ("AAPL", "low"): [p - 2 for p in close_prices],
        ("AAPL", "close"): close_prices,
    }, index=dates)

    # Entry on day 5, no explicit exit
    signals = pd.DataFrame({
        "AAPL": [0] * 5 + [1] + [0] * 24,
    }, index=dates)

    stop_config = StopLossConfig(
        enabled=True,
        type="atr_trailing",
        atr_multiplier=1.5,
        atr_period=5,
    )

    adjusted_signals, stop_events = apply_trailing_stops(signals, prices, stop_config)

    # Should have some exit signals after price falls
    # The adjusted signals should show the position being closed
    assert adjusted_signals is not None
    assert len(stop_events) > 0


def test_stop_tracks_trailing_high():
    """Test stop level increases as price increases."""
    from hrp.research.stops import compute_atr_stops

    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    # Steadily increasing price
    close_prices = np.linspace(100, 130, 30)

    prices = pd.DataFrame({
        ("AAPL", "high"): close_prices + 1,
        ("AAPL", "low"): close_prices - 1,
        ("AAPL", "close"): close_prices,
    }, index=dates)

    entries = pd.DataFrame({"AAPL": [1] + [0] * 29}, index=dates)

    stops = compute_atr_stops(prices, entries, atr_multiplier=2.0, atr_period=5)

    # Stop should increase over time as price increases
    valid_stops = stops["AAPL"].dropna()
    assert valid_stops.iloc[-1] > valid_stops.iloc[0]
```

**Step 2: Run test**

Run: `pytest tests/test_research/test_stops.py::test_apply_trailing_stops_generates_exits -v`
Expected: FAIL with "cannot import name 'apply_trailing_stops'"

**Step 3: Write implementation**

```python
# Add to hrp/research/stops.py

from hrp.research.config import StopLossConfig


def apply_trailing_stops(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    stop_config: StopLossConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply trailing stops to signals, generating exit signals when stops are hit.

    Args:
        signals: Entry signals (1 = entry, 0 = no entry)
        prices: MultiIndex columns (symbol, field) with OHLC
        stop_config: Stop configuration

    Returns:
        - adjusted_signals: Signals with stop-based exits (-1 = exit)
        - stop_events: DataFrame of stop trigger events
    """
    if not stop_config.enabled:
        return signals, pd.DataFrame()

    symbols = signals.columns.tolist()
    adjusted_signals = signals.copy()
    stop_events = []

    # Compute stop levels
    stop_levels = compute_atr_stops(
        prices=prices,
        entries=signals,
        atr_multiplier=stop_config.atr_multiplier,
        atr_period=stop_config.atr_period,
    )

    for symbol in symbols:
        try:
            low = prices[(symbol, "low")]
            close = prices[(symbol, "close")]
        except KeyError:
            continue

        in_position = False
        entry_price = 0.0

        for idx in signals.index:
            signal = signals.loc[idx, symbol]
            stop_level = stop_levels.loc[idx, symbol]
            current_low = low.loc[idx]
            current_close = close.loc[idx]

            if signal == 1 and not in_position:
                in_position = True
                entry_price = current_close

            elif in_position and not np.isnan(stop_level):
                # Check if stop was hit (low touched stop)
                if current_low <= stop_level:
                    # Stop triggered
                    adjusted_signals.loc[idx, symbol] = -1  # Exit signal
                    in_position = False

                    pnl = (stop_level - entry_price) / entry_price

                    stop_events.append({
                        "symbol": symbol,
                        "date": idx,
                        "entry_price": entry_price,
                        "stop_level": stop_level,
                        "trigger_price": min(current_low, stop_level),
                        "pnl_pct": pnl,
                    })

    return adjusted_signals, pd.DataFrame(stop_events)
```

**Step 4: Run tests**

Run: `pytest tests/test_research/test_stops.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/stops.py tests/test_research/test_stops.py
git commit -m "feat(research): implement apply_trailing_stops function"
```

---

## Phase 4: Walk-Forward Visualization

### Task 4.1: Create walkforward_viz component

**Files:**
- Create: `hrp/dashboard/components/walkforward_viz.py`
- Create: `tests/test_dashboard/test_walkforward_viz.py`

**Step 1: Write the failing test**

```python
# tests/test_dashboard/test_walkforward_viz.py
"""Tests for walk-forward visualization components."""

import pytest
from datetime import date
from unittest.mock import patch, MagicMock


def test_render_walkforward_splits_with_valid_data():
    """Test render_walkforward_splits renders without error."""
    from hrp.dashboard.components.walkforward_viz import render_walkforward_splits
    from hrp.ml.validation import FoldResult, WalkForwardConfig

    fold_results = [
        FoldResult(
            fold_index=0,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 1),
            test_end=date(2021, 6, 30),
            metrics={"ic": 0.05, "mse": 0.001},
            model=None,
            n_train_samples=250,
            n_test_samples=125,
        ),
        FoldResult(
            fold_index=1,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            metrics={"ic": 0.04, "mse": 0.0012},
            model=None,
            n_train_samples=375,
            n_test_samples=125,
        ),
    ]

    config = WalkForwardConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d"],
        start_date=date(2020, 1, 1),
        end_date=date(2021, 12, 31),
        n_folds=2,
    )

    # Mock streamlit
    with patch("hrp.dashboard.components.walkforward_viz.st") as mock_st:
        mock_st.plotly_chart = MagicMock()

        # Should not raise
        render_walkforward_splits(fold_results, config)

        # Should have called plotly_chart
        mock_st.plotly_chart.assert_called()


def test_render_handles_empty_folds():
    """Test render handles empty fold list gracefully."""
    from hrp.dashboard.components.walkforward_viz import render_walkforward_splits
    from hrp.ml.validation import WalkForwardConfig

    config = WalkForwardConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d"],
        start_date=date(2020, 1, 1),
        end_date=date(2021, 12, 31),
        n_folds=2,
    )

    with patch("hrp.dashboard.components.walkforward_viz.st") as mock_st:
        mock_st.warning = MagicMock()

        render_walkforward_splits([], config)

        # Should show warning for empty folds
        mock_st.warning.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard/test_walkforward_viz.py -v`
Expected: FAIL with "No module named 'hrp.dashboard.components.walkforward_viz'"

**Step 3: Write minimal implementation**

```python
# hrp/dashboard/components/walkforward_viz.py
"""
Walk-forward split visualization components.

Provides interactive visualizations for walk-forward validation results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from hrp.ml.validation import FoldResult, WalkForwardConfig


def render_walkforward_splits(
    fold_results: list["FoldResult"],
    config: "WalkForwardConfig",
) -> None:
    """
    Render interactive walk-forward split visualization.

    Shows:
    - Timeline of train/test periods for each fold
    - Per-fold metrics (IC, MSE, R2) as annotations
    - Stability score indicator

    Args:
        fold_results: List of results from each fold
        config: Walk-forward configuration used
    """
    if not fold_results:
        st.warning("No fold results to display")
        return

    fig = go.Figure()

    colors = {"train": "#3366cc", "test": "#dc3912"}

    for fold in fold_results:
        # Training period
        fig.add_trace(go.Bar(
            name=f"Fold {fold.fold_index} Train",
            x=[f"Fold {fold.fold_index}"],
            y=[(fold.train_end - fold.train_start).days],
            base=0,
            marker_color=colors["train"],
            text=f"Train: {fold.train_start} to {fold.train_end}",
            hoverinfo="text",
            showlegend=fold.fold_index == 0,
            legendgroup="train",
        ))

        # Test period
        fig.add_trace(go.Bar(
            name=f"Fold {fold.fold_index} Test",
            x=[f"Fold {fold.fold_index}"],
            y=[(fold.test_end - fold.test_start).days],
            base=(fold.train_end - fold.train_start).days,
            marker_color=colors["test"],
            text=f"Test: {fold.test_start} to {fold.test_end}<br>IC: {fold.metrics.get('ic', 0):.4f}",
            hoverinfo="text",
            showlegend=fold.fold_index == 0,
            legendgroup="test",
        ))

    fig.update_layout(
        title=f"Walk-Forward Splits ({config.window_type.title()} Window)",
        xaxis_title="Fold",
        yaxis_title="Days",
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_fold_metrics_heatmap(
    fold_results: list["FoldResult"],
) -> None:
    """Render heatmap of metrics across folds."""
    if not fold_results:
        st.warning("No fold results to display")
        return

    import pandas as pd

    metrics_data = []
    for fold in fold_results:
        row = {"fold": fold.fold_index, **fold.metrics}
        metrics_data.append(row)

    df = pd.DataFrame(metrics_data).set_index("fold")

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=[f"Fold {i}" for i in df.index],
        colorscale="RdYlGn",
        text=df.values.round(4),
        texttemplate="%{text}",
        textfont={"size": 12},
    ))

    fig.update_layout(
        title="Metrics Across Folds",
        xaxis_title="Metric",
        yaxis_title="Fold",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_fold_comparison_chart(
    fold_results: list["FoldResult"],
) -> None:
    """Render bar chart comparing fold performance."""
    if not fold_results:
        st.warning("No fold results to display")
        return

    import pandas as pd

    ic_values = [f.metrics.get("ic", 0) for f in fold_results]
    fold_labels = [f"Fold {f.fold_index}" for f in fold_results]

    fig = go.Figure(data=[
        go.Bar(
            x=fold_labels,
            y=ic_values,
            marker_color=["#3366cc" if ic >= 0 else "#dc3912" for ic in ic_values],
            text=[f"{ic:.4f}" for ic in ic_values],
            textposition="outside",
        )
    ])

    fig.update_layout(
        title="Information Coefficient by Fold",
        xaxis_title="Fold",
        yaxis_title="IC",
        yaxis=dict(zeroline=True, zerolinecolor="gray"),
    )

    st.plotly_chart(fig, use_container_width=True)
```

**Step 4: Run tests**

Run: `pytest tests/test_dashboard/test_walkforward_viz.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/dashboard/components/walkforward_viz.py tests/test_dashboard/test_walkforward_viz.py
git commit -m "feat(dashboard): add walk-forward visualization components"
```

---

### Task 4.2: Create sharpe_decay_viz component

**Files:**
- Create: `hrp/dashboard/components/sharpe_decay_viz.py`
- Create: `tests/test_dashboard/test_sharpe_decay_viz.py`

**Step 1: Write the failing test**

```python
# tests/test_dashboard/test_sharpe_decay_viz.py
"""Tests for Sharpe decay heatmap visualization."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def test_render_sharpe_decay_heatmap():
    """Test Sharpe decay heatmap renders without error."""
    from hrp.dashboard.components.sharpe_decay_viz import render_sharpe_decay_heatmap
    from hrp.research.parameter_sweep import SweepResult

    # Create mock sweep result
    results_df = pd.DataFrame({
        "fast_period": [10, 10, 20, 20],
        "slow_period": [30, 30, 40, 40],
        "fold": [0, 1, 0, 1],
        "train_sharpe": [1.5, 1.4, 1.2, 1.1],
        "test_sharpe": [1.2, 1.3, 0.9, 0.8],
    })

    # Create matrices
    sharpe_diff_median = pd.Series([-0.2, -0.3], index=[(10, 30), (20, 40)])
    sharpe_diff_median.index = pd.MultiIndex.from_tuples(
        sharpe_diff_median.index, names=["fast_period", "slow_period"]
    )

    sweep_result = SweepResult(
        results_df=results_df,
        best_params={"fast_period": 10, "slow_period": 30},
        best_metrics={"sharpe_ratio": 1.25},
        train_sharpe_matrix=pd.DataFrame(),
        test_sharpe_matrix=pd.DataFrame(),
        sharpe_diff_matrix=pd.DataFrame(),
        sharpe_diff_median=sharpe_diff_median,
        constraint_violations=0,
        execution_time_seconds=5.0,
        generalization_score=0.5,
    )

    with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st:
        mock_st.plotly_chart = MagicMock()

        render_sharpe_decay_heatmap(
            sweep_result,
            param_x="fast_period",
            param_y="slow_period",
        )

        mock_st.plotly_chart.assert_called()


def test_generalization_summary_metrics():
    """Test generalization summary shows correct metrics."""
    from hrp.dashboard.components.sharpe_decay_viz import render_generalization_summary
    from hrp.research.parameter_sweep import SweepResult

    results_df = pd.DataFrame({"fast_period": [10], "slow_period": [20]})

    sweep_result = SweepResult(
        results_df=results_df,
        best_params={"fast_period": 10, "slow_period": 30},
        best_metrics={"sharpe_ratio": 1.2},
        train_sharpe_matrix=pd.DataFrame(),
        test_sharpe_matrix=pd.DataFrame(),
        sharpe_diff_matrix=pd.DataFrame(),
        sharpe_diff_median=pd.Series(),
        constraint_violations=0,
        execution_time_seconds=5.0,
        generalization_score=0.75,
    )

    with patch("hrp.dashboard.components.sharpe_decay_viz.st") as mock_st:
        mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        render_generalization_summary(sweep_result)

        # Should call columns
        mock_st.columns.assert_called()
```

**Step 2: Run test**

Run: `pytest tests/test_dashboard/test_sharpe_decay_viz.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# hrp/dashboard/components/sharpe_decay_viz.py
"""
Sharpe decay heatmap visualization (VectorBT PRO style).

Key diagnostic from PyQuantNews thread:
- Blue regions = test > train = good generalization
- Red regions = test < train = overfitting
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from hrp.research.parameter_sweep import SweepResult


def render_sharpe_decay_heatmap(
    sweep_result: "SweepResult",
    param_x: str,
    param_y: str,
    colorscale: str = "RdBu",
) -> None:
    """
    Render Sharpe ratio decay heatmap (VectorBT PRO style).

    Shows test_sharpe - train_sharpe across parameter combinations.
    Blue = positive (good), Red = negative (overfitting).

    Args:
        sweep_result: Result from parallel_parameter_sweep
        param_x: Parameter for X-axis (e.g., "fast_period")
        param_y: Parameter for Y-axis (e.g., "slow_period")
        colorscale: Plotly colorscale (RdBu recommended)
    """
    if sweep_result.sharpe_diff_median.empty:
        st.warning("No Sharpe decay data available")
        return

    # Unstack to 2D matrix for heatmap
    try:
        heatmap_data = sweep_result.sharpe_diff_median.unstack(param_x)
    except (KeyError, ValueError) as e:
        st.warning(f"Cannot create heatmap: {e}")
        return

    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        color_continuous_scale=colorscale,
        color_continuous_midpoint=0,
        labels={"x": param_x, "y": param_y, "color": "Sharpe Diff"},
        title="Sharpe Ratio Decay: Test - Train (Blue = Good Generalization)",
    )

    fig.update_layout(
        xaxis_title=param_x,
        yaxis_title=param_y,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_generalization_summary(
    sweep_result: "SweepResult",
) -> None:
    """
    Render summary metrics for parameter generalization.

    Shows:
    - % of parameter combos that generalize (test >= train)
    - Best generalizing parameters
    - Worst overfitting parameters
    """
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Generalization Score",
        f"{sweep_result.generalization_score:.1%}",
        help="% of param combos where test Sharpe >= train Sharpe"
    )

    col2.metric(
        "Best Params",
        str(sweep_result.best_params),
    )

    col3.metric(
        "Param Combos Tested",
        len(sweep_result.results_df.drop_duplicates(
            subset=[c for c in sweep_result.results_df.columns if c not in ["fold", "train_sharpe", "test_sharpe"]]
        )),
    )


def render_parameter_sensitivity_chart(
    sweep_result: "SweepResult",
    param_name: str,
) -> None:
    """
    Render sensitivity analysis for a single parameter.

    Shows how Sharpe diff varies as one parameter changes,
    averaging over other parameters.
    """
    if sweep_result.results_df.empty:
        st.warning("No results data available")
        return

    # Group by the single parameter and aggregate
    df = sweep_result.results_df.copy()
    df["sharpe_diff"] = df["test_sharpe"] - df["train_sharpe"]

    sensitivity = df.groupby(param_name)["sharpe_diff"].mean()

    fig = go.Figure(data=[
        go.Bar(
            x=sensitivity.index.tolist(),
            y=sensitivity.values,
            marker_color=["#3366cc" if v >= 0 else "#dc3912" for v in sensitivity.values],
            text=[f"{v:.3f}" for v in sensitivity.values],
            textposition="outside",
        )
    ])

    fig.update_layout(
        title=f"Sharpe Diff Sensitivity to {param_name}",
        xaxis_title=param_name,
        yaxis_title="Mean Sharpe Diff (test - train)",
        yaxis=dict(zeroline=True, zerolinecolor="gray"),
    )

    st.plotly_chart(fig, use_container_width=True)
```

**Step 4: Run tests**

Run: `pytest tests/test_dashboard/test_sharpe_decay_viz.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/dashboard/components/sharpe_decay_viz.py tests/test_dashboard/test_sharpe_decay_viz.py
git commit -m "feat(dashboard): add Sharpe decay heatmap visualization"
```

---

## Phase 5: HMM Regime Detection

### Task 5.1: Create HMMConfig and RegimeResult dataclasses

**Files:**
- Create: `hrp/ml/regime.py`
- Create: `tests/test_ml/test_regime.py`

**Step 1: Write the failing test**

```python
# tests/test_ml/test_regime.py
"""Tests for HMM regime detection."""

import pytest
from datetime import date


def test_hmm_config_creation():
    """Test HMMConfig dataclass creation."""
    from hrp.ml.regime import HMMConfig

    config = HMMConfig(
        n_regimes=3,
        features=["returns_20d", "volatility_20d"],
    )

    assert config.n_regimes == 3
    assert config.covariance_type == "full"  # default
    assert config.n_iter == 100  # default


def test_regime_result_creation():
    """Test RegimeResult dataclass creation."""
    from hrp.ml.regime import RegimeResult
    import pandas as pd
    import numpy as np

    regimes = pd.Series([0, 1, 0, 2, 1], name="regime")
    transition_matrix = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

    result = RegimeResult(
        regimes=regimes,
        transition_matrix=transition_matrix,
        regime_means={0: {"returns": 0.01}, 1: {"returns": -0.02}},
        regime_covariances={0: np.eye(2), 1: np.eye(2)},
        log_likelihood=-100.5,
        regime_durations={0: 5.0, 1: 3.0, 2: 4.0},
    )

    assert len(result.regimes) == 5
    assert result.transition_matrix.shape == (3, 3)
    assert result.log_likelihood == -100.5
```

**Step 2: Run test**

Run: `pytest tests/test_ml/test_regime.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# hrp/ml/regime.py
"""
HMM regime detection module.

Provides Hidden Markov Model-based market regime detection
for identifying bull, bear, sideways, and crisis states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass
class HMMConfig:
    """
    Configuration for HMM regime detection.

    Attributes:
        n_regimes: Number of hidden states to detect
        features: List of features to use for regime detection
        covariance_type: Type of covariance matrix ('full', 'diag', 'spherical')
        n_iter: Maximum iterations for EM algorithm
        random_state: Random seed for reproducibility
    """

    n_regimes: int = 3
    features: list[str] = field(default_factory=lambda: ["returns_20d", "volatility_20d"])
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {self.n_regimes}")
        if self.covariance_type not in ("full", "diag", "spherical", "tied"):
            raise ValueError(
                f"Invalid covariance_type '{self.covariance_type}'. "
                f"Must be 'full', 'diag', 'spherical', or 'tied'"
            )


@dataclass
class RegimeResult:
    """
    Result of regime detection.

    Attributes:
        regimes: Series of regime labels indexed by date
        transition_matrix: Probability of transitioning between regimes
        regime_means: Mean feature values for each regime
        regime_covariances: Covariance matrices for each regime
        log_likelihood: Model log-likelihood (for comparison)
        regime_durations: Average duration in each regime (in periods)
    """

    regimes: pd.Series
    transition_matrix: np.ndarray
    regime_means: dict[int, dict[str, float]]
    regime_covariances: dict[int, np.ndarray]
    log_likelihood: float
    regime_durations: dict[int, float]
```

**Step 4: Run tests**

Run: `pytest tests/test_ml/test_regime.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/regime.py tests/test_ml/test_regime.py
git commit -m "feat(ml): add HMMConfig and RegimeResult dataclasses"
```

---

### Task 5.2: Implement RegimeDetector class

**Files:**
- Modify: `hrp/ml/regime.py`
- Modify: `tests/test_ml/test_regime.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_ml/test_regime.py
import numpy as np
import pandas as pd


def test_regime_detector_fit_returns_self():
    """Test RegimeDetector.fit() returns self for chaining."""
    from hrp.ml.regime import HMMConfig, RegimeDetector

    config = HMMConfig(n_regimes=2, features=["returns"])

    # Create synthetic price data
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame({
        "close": 100 * np.cumprod(1 + np.random.randn(500) * 0.01),
    }, index=dates)

    detector = RegimeDetector(config)
    result = detector.fit(prices)

    assert result is detector  # Returns self


def test_regime_detector_predict_returns_series():
    """Test RegimeDetector.predict() returns regime Series."""
    from hrp.ml.regime import HMMConfig, RegimeDetector

    config = HMMConfig(n_regimes=2, features=["returns"])

    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame({
        "close": 100 * np.cumprod(1 + np.random.randn(500) * 0.01),
    }, index=dates)

    detector = RegimeDetector(config)
    detector.fit(prices)

    regimes = detector.predict(prices)

    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(prices)
    assert set(regimes.unique()).issubset({0, 1})


def test_transition_matrix_probabilities():
    """Test transition matrix rows sum to 1."""
    from hrp.ml.regime import HMMConfig, RegimeDetector

    config = HMMConfig(n_regimes=3, features=["returns"])

    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame({
        "close": 100 * np.cumprod(1 + np.random.randn(500) * 0.01),
    }, index=dates)

    detector = RegimeDetector(config)
    detector.fit(prices)

    trans_matrix = detector.get_transition_matrix()

    # Each row should sum to 1
    row_sums = trans_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
```

**Step 2: Run tests**

Run: `pytest tests/test_ml/test_regime.py::test_regime_detector_fit_returns_self -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# Add to hrp/ml/regime.py
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. Regime detection will not be available.")


class RegimeDetector:
    """
    Hidden Markov Model regime detector.

    Identifies market regimes (bull, bear, sideways) using
    an HMM fitted on price/return features.
    """

    def __init__(self, config: HMMConfig) -> None:
        """
        Initialize regime detector.

        Args:
            config: HMM configuration
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for regime detection. "
                "Install with: pip install hmmlearn"
            )

        self.config = config
        self.model: GaussianHMM | None = None
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}

    def _prepare_features(self, prices: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from prices."""
        features = {}

        if "close" in prices.columns:
            close = prices["close"]
        elif ("", "close") in prices.columns:
            close = prices[("", "close")]
        else:
            close = prices.iloc[:, 0]  # Use first column

        # Compute basic features
        if "returns" in self.config.features or "returns_20d" in self.config.features:
            features["returns"] = close.pct_change().fillna(0)

        if "volatility" in self.config.features or "volatility_20d" in self.config.features:
            features["volatility"] = close.pct_change().rolling(20).std().fillna(0)

        feature_df = pd.DataFrame(features)

        # Standardize
        for col in feature_df.columns:
            mean = feature_df[col].mean()
            std = feature_df[col].std()
            self._feature_means[col] = mean
            self._feature_stds[col] = std if std > 0 else 1.0
            feature_df[col] = (feature_df[col] - mean) / self._feature_stds[col]

        return feature_df.values

    def fit(self, prices: pd.DataFrame) -> "RegimeDetector":
        """
        Fit HMM to price data.

        Args:
            prices: DataFrame with 'close' column

        Returns:
            self (for chaining)
        """
        X = self._prepare_features(prices)

        self.model = GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
        )

        self.model.fit(X)
        logger.info(f"HMM fitted with {self.config.n_regimes} regimes")

        return self

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        """
        Predict regimes for price data.

        Args:
            prices: DataFrame with 'close' column

        Returns:
            Series of regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_features(prices)
        regimes = self.model.predict(X)

        return pd.Series(regimes, index=prices.index, name="regime")

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix.

        Returns:
            DataFrame with transition probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        labels = [f"Regime {i}" for i in range(self.config.n_regimes)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels,
        )

    def get_regime_result(self, prices: pd.DataFrame) -> RegimeResult:
        """
        Get full regime detection result.

        Args:
            prices: DataFrame with 'close' column

        Returns:
            RegimeResult with all regime information
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        regimes = self.predict(prices)

        # Compute regime means
        regime_means = {}
        for i in range(self.config.n_regimes):
            regime_means[i] = {
                f"feature_{j}": float(self.model.means_[i, j])
                for j in range(self.model.means_.shape[1])
            }

        # Compute regime covariances
        regime_covs = {i: self.model.covars_[i] for i in range(self.config.n_regimes)}

        # Compute average regime durations
        regime_durations = {}
        for i in range(self.config.n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                # Count consecutive runs
                runs = (mask != mask.shift()).cumsum()
                run_lengths = mask.groupby(runs).sum()
                regime_durations[i] = float(run_lengths[run_lengths > 0].mean())
            else:
                regime_durations[i] = 0.0

        return RegimeResult(
            regimes=regimes,
            transition_matrix=self.model.transmat_,
            regime_means=regime_means,
            regime_covariances=regime_covs,
            log_likelihood=float(self.model.score(self._prepare_features(prices))),
            regime_durations=regime_durations,
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_ml/test_regime.py -v`
Expected: PASS (if hmmlearn is installed)

**Step 5: Commit**

```bash
git add hrp/ml/regime.py tests/test_ml/test_regime.py
git commit -m "feat(ml): implement RegimeDetector class with HMM"
```

---

### Task 5.3: Add hmmlearn dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add dependency**

Add to `requirements.txt`:
```
hmmlearn>=0.3.0
```

**Step 2: Install**

```bash
pip install hmmlearn>=0.3.0
```

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add hmmlearn for regime detection"
```

---

## Final Steps

### Task F.1: Export new modules from package __init__.py files

**Files:**
- Modify: `hrp/ml/__init__.py`
- Modify: `hrp/research/__init__.py`
- Modify: `hrp/dashboard/components/__init__.py`

Add appropriate exports for all new modules.

### Task F.2: Run full test suite

```bash
pytest tests/ -v
```

Ensure all tests pass (target: 100%)

### Task F.3: Update documentation

Update these files:
- `CLAUDE.md`: Add examples for new features
- `docs/plans/Project-Status.md`: Update feature counts
- `docs/plans/Changelog.md`: Add v1.5.0 entry

### Task F.4: Final commit

```bash
git add -A
git commit -m "docs: complete VectorBT PRO features implementation"
```

---

## Implementation Order Summary

```
Phase 1 (8 tasks): Cross-Validated Optimization
    └── Phase 2 (5 tasks): Parallel Parameter Sweeps (depends on Phase 1 patterns)

Phase 3 (3 tasks): ATR Trailing Stops (independent)

Phase 4 (2 tasks): Walk-Forward Visualization (depends on Phase 1 results)

Phase 5 (3 tasks): HMM Regime Detection (independent)

Final (4 tasks): Integration, testing, docs
```

**Recommended execution:** 1 → 2 → 3 → 4 → 5 → Final

---

**Plan complete and saved to `docs/plans/2025-01-25-vectorbt-pro-implementation.md`. Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
