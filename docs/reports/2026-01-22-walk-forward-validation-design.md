# Walk-Forward Validation Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add walk-forward validation to the ML framework for robust model selection.

**Use Case:** Compare models/hyperparameters across temporal folds to identify the most robust configuration and detect overfitting.

---

## API Surface

```python
from hrp.ml import WalkForwardConfig, WalkForwardResult, walk_forward_validate

# Configure walk-forward
wf_config = WalkForwardConfig(
    model_type="ridge",
    target="returns_20d",
    features=["momentum_20d", "volatility_20d"],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    window_type="expanding",  # or "rolling"
    min_train_periods=252,    # ~1 year minimum training data
    hyperparameters={},
    feature_selection=True,
    max_features=20,
)

# Run validation
result = walk_forward_validate(
    config=wf_config,
    symbols=["AAPL", "MSFT", "GOOGL"],
    log_to_mlflow=False,
)

# Access results
print(result.aggregate_metrics)  # {'mean_mse': 0.002, 'std_mse': 0.001, ...}
print(result.fold_metrics)       # List of per-fold metric dicts
print(result.stability_score)    # std(metric) / mean(metric) - lower is better
```

---

## Data Structures

### WalkForwardConfig

```python
@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    model_type: str
    target: str
    features: list[str]
    start_date: date
    end_date: date
    n_folds: int = 5
    window_type: str = "expanding"  # "expanding" or "rolling"
    min_train_periods: int = 252    # Minimum training samples (~1 year)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_selection: bool = True
    max_features: int = 20

    def __post_init__(self):
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: '{self.model_type}'")
        if self.window_type not in ("expanding", "rolling"):
            raise ValueError("window_type must be 'expanding' or 'rolling'")
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
```

### FoldResult

```python
@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    metrics: dict[str, float]  # mse, mae, r2, ic
    model: Any
    n_train_samples: int
    n_test_samples: int
```

### WalkForwardResult

```python
@dataclass
class WalkForwardResult:
    """Aggregated result from walk-forward validation."""

    config: WalkForwardConfig
    fold_results: list[FoldResult]
    aggregate_metrics: dict[str, float]  # mean/std of each metric
    stability_score: float  # Lower is better (coefficient of variation)
    symbols: list[str]
```

---

## Fold Generation Logic

### Expanding Window (default)

```
Fold 1: [====TRAIN====][==TEST==]
Fold 2: [=======TRAIN=======][==TEST==]
Fold 3: [==========TRAIN==========][==TEST==]
```

Training data grows with each fold. More data available, but older patterns weighted equally.

### Rolling Window

```
Fold 1: [====TRAIN====][==TEST==]
Fold 2:      [====TRAIN====][==TEST==]
Fold 3:           [====TRAIN====][==TEST==]
```

Fixed training window slides forward. Adapts to regime changes, less data per fold.

### Constraints

- Test periods are non-overlapping (no data leakage)
- Each fold has at least `min_train_periods` training samples
- Gap between train_end and test_start: 1 day (no overlap)

### Algorithm

```python
def generate_folds(
    config: WalkForwardConfig,
    available_dates: list[date],
) -> list[tuple[date, date, date, date]]:
    """
    Generate (train_start, train_end, test_start, test_end) for each fold.

    1. Divide available dates into n_folds equal test periods
    2. For expanding: train_start is always start_date
    3. For rolling: train_start slides to maintain fixed window
    4. Ensure min_train_periods constraint is met
    """
```

---

## Metrics

### Per-Fold Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `mse` | Mean Squared Error | Standard regression loss |
| `mae` | Mean Absolute Error | Robust to outliers |
| `r2` | R² Score | Explained variance |
| `ic` | Information Coefficient | Rank correlation (Spearman) between predictions and actuals |

```python
def compute_fold_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "ic": spearmanr(y_true, y_pred).correlation,
    }
```

### Aggregate Metrics

For each metric, compute mean and standard deviation across folds:

```python
aggregate_metrics = {
    "mean_mse": np.mean([f.metrics["mse"] for f in fold_results]),
    "std_mse": np.std([f.metrics["mse"] for f in fold_results]),
    "mean_mae": np.mean([f.metrics["mae"] for f in fold_results]),
    "std_mae": np.std([f.metrics["mae"] for f in fold_results]),
    "mean_r2": np.mean([f.metrics["r2"] for f in fold_results]),
    "std_r2": np.std([f.metrics["r2"] for f in fold_results]),
    "mean_ic": np.mean([f.metrics["ic"] for f in fold_results]),
    "std_ic": np.std([f.metrics["ic"] for f in fold_results]),
}
```

### Stability Score

Coefficient of variation on MSE—lower means more consistent:

```python
stability_score = std_mse / mean_mse if mean_mse > 0 else float('inf')
```

Per HRP spec: stability_score > 1.0 flags the model as unstable (std exceeds mean).

---

## Main Function Flow

```python
def walk_forward_validate(
    config: WalkForwardConfig,
    symbols: list[str],
    log_to_mlflow: bool = False,
) -> WalkForwardResult:
    """
    Run walk-forward validation.

    Steps:
    1. Fetch all data for date range (single query)
    2. Generate fold date ranges
    3. For each fold:
       a. Split data into train/test by date
       b. Feature selection (on train only, avoid leakage)
       c. Train model
       d. Predict on test
       e. Compute metrics
       f. Store FoldResult
    4. Aggregate metrics across folds
    5. Compute stability score
    6. Optionally log to MLflow (parent run + child runs per fold)
    7. Return WalkForwardResult
    """
```

### Error Handling

- Skip fold if insufficient training data (log warning, continue)
- Raise if no folds complete successfully
- Handle NaN in metrics gracefully (e.g., if R² undefined)

### Code Reuse

From `hrp/ml/training.py`:
- `_fetch_features()` - Data loading
- `select_features()` - Feature selection

From `hrp/ml/models.py`:
- `get_model()` - Model instantiation
- `SUPPORTED_MODELS` - Validation

---

## File Structure

### New File: `hrp/ml/validation.py`

```
hrp/ml/
├── __init__.py      # Add new exports
├── models.py        # Unchanged
├── signals.py       # Unchanged
├── training.py      # Unchanged (reuse _fetch_features, select_features)
└── validation.py    # NEW
```

### Updates to `hrp/ml/__init__.py`

```python
from hrp.ml.validation import (
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
    walk_forward_validate,
)

__all__ = [
    # ... existing exports ...
    # Validation
    "WalkForwardConfig",
    "WalkForwardResult",
    "FoldResult",
    "walk_forward_validate",
]
```

### Test File: `tests/test_ml/test_validation.py`

Test coverage:
- Config validation (invalid window_type, n_folds < 2, invalid model_type)
- Fold generation (correct date ranges, no overlap, both window types)
- Metrics computation (known inputs → expected outputs)
- Stability score calculation
- Integration test with mocked feature data

---

## Dependencies

No new dependencies. Uses:
- `sklearn.metrics` (existing)
- `scipy.stats.spearmanr` (existing in environment)
- `pandas`, `numpy` (existing)

---

## Future Enhancements (Not in MVP)

- Trading metrics per fold (Sharpe, hit rate) via mini-backtests
- Hyperparameter optimization across folds (Optuna integration)
- Parallel fold execution for speed
- Purged/embargo cross-validation for overlapping prediction horizons
- Walk-forward for production signal generation

---

## Summary

| Aspect | Decision |
|--------|----------|
| **Use case** | Model selection—compare models across folds for robustness |
| **Fold definition** | Fixed `n_folds`, system divides date range automatically |
| **Window types** | Expanding (default) and rolling supported |
| **Metrics** | Prediction-only: MSE, MAE, R², IC per fold |
| **Aggregation** | Mean/std of each metric + stability score |
| **Output** | `WalkForwardResult` dataclass, optional MLflow logging |
| **File** | `hrp/ml/validation.py` |
| **Reuses** | `_fetch_features`, `select_features`, `get_model` from existing modules |
