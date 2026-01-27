# Implementation Plan: VectorBT PRO-Inspired Features for HRP

**Date**: 2025-01-25
**Source**: PyQuantNews thread on cross-validation for backtesting
**Status**: ✅ Implemented

## Executive Summary

This plan outlines the implementation of five features inspired by VectorBT PRO patterns to enhance HRP's optimization, backtesting, and regime detection capabilities. The implementation follows HRP's test-driven development approach, integrates with existing overfitting guards, and maintains the three-layer architecture.

---

## Features Overview

| Phase | Feature | Priority | Effort |
|-------|---------|----------|--------|
| 1 | Cross-Validated Optimization | **Critical** | Medium |
| 2 | Parallel Parameter Sweeps | **High** | Medium |
| 3 | ATR-Based Trailing Stops | Medium | Low |
| 4 | Walk-Forward Visualization | Medium | Low |
| 5 | HMM Regime Detection | Medium | Medium |

---

## Phase 1: Cross-Validated Optimization Framework

**Objective**: Add parameterized cross-validation optimization that integrates with existing walk-forward validation and overfitting guards.

### New Module: `hrp/ml/optimization.py`

**Public API**:
```python
@dataclass
class OptimizationConfig:
    """Configuration for cross-validated optimization."""
    model_type: str
    target: str
    features: list[str]
    param_grid: dict[str, list[Any]]  # e.g., {"alpha": [0.1, 1.0, 10.0]}
    start_date: date
    end_date: date
    n_folds: int = 5
    window_type: str = "expanding"  # expanding or rolling
    scoring_metric: str = "ic"  # ic, mse, sharpe
    constraints: dict[str, Any] | None = None
    max_trials: int = 50  # Integration with HyperparameterTrialCounter
    hypothesis_id: str | None = None

@dataclass
class OptimizationResult:
    """Result of cross-validated optimization."""
    best_params: dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    fold_results: list[FoldResult]
    all_trials: list[dict]
    hypothesis_id: str | None

def cross_validated_optimize(
    config: OptimizationConfig,
    symbols: list[str],
    log_to_mlflow: bool = True,
) -> OptimizationResult:
    """
    Run cross-validated parameter optimization.

    Combines VectorBT PRO's grid search with HRP's walk-forward validation
    and overfitting guards.
    """
```

**Key Implementation Details**:
1. Leverage existing `generate_folds()` from `hrp/ml/validation.py`
2. Integrate with `HyperparameterTrialCounter` to limit search space
3. Support early stopping when overfitting detected via `SharpeDecayMonitor`
4. Log all trials to lineage table for audit trail
5. Support both grid search and randomized search

**Tests**: `tests/test_ml/test_optimization.py`
- `test_config_creation_with_defaults`
- `test_config_validates_param_grid`
- `test_returns_best_params`
- `test_integrates_with_trial_counter`
- `test_respects_max_trials_limit`
- `test_early_stops_on_overfitting`
- `test_logs_to_mlflow`

---

## Phase 2: Parallel Parameter Sweeps with Constraints

**Objective**: Add efficient parallel parameter sweeps with constraint validation for multi-factor strategies, including **Sharpe decay analysis** to identify parameter combinations that generalize well.

### Key Insight from PyQuantNews Thread

The thread demonstrates computing `sharpe_ratio_diff = test_sharpe - train_sharpe` and visualizing as a heatmap:
- **Blue regions** = positive diff (test > train) = good generalization
- **Red regions** = negative diff (test < train) = overfitting
- Use **median** across folds for robustness

```python
# From PyQuantNews thread
sharpe_ratio_diff = test_sharpe_ratio - train_sharpe_ratio
sharpe_ratio_diff_median = sharpe_ratio_diff.groupby(
    ["fast_period", "slow_period"]
).median()
sharpe_ratio_diff_median.vbt.heatmap(
    trace_kwargs=dict(colorscale="RdBu")
).show_png()
```

### New Module: `hrp/research/parameter_sweep.py`

**Public API**:
```python
@dataclass
class SweepConstraint:
    """Constraint on parameter combinations."""
    constraint_type: str  # "sum_equals", "max_total", "ratio_bound", "difference_min"
    params: list[str]
    value: float
    # Example: SweepConstraint("difference_min", ["slow_period", "fast_period"], 5)
    # Enforces: slow_period - fast_period >= 5

@dataclass
class SweepConfig:
    """Configuration for parallel parameter sweep."""
    strategy_type: str  # "multifactor", "ml_predicted", "momentum"
    param_ranges: dict[str, list[Any]]
    constraints: list[SweepConstraint]
    symbols: list[str]
    start_date: date
    end_date: date
    n_folds: int = 5  # Number of CV folds
    n_jobs: int = -1
    scoring: str = "sharpe_ratio"
    min_samples: int = 100
    aggregation: str = "median"  # median or mean across folds

@dataclass
class SweepResult:
    """Result of parameter sweep with train/test analysis."""
    results_df: pd.DataFrame  # All param combos with per-fold metrics
    best_params: dict[str, Any]
    best_metrics: dict[str, float]

    # Sharpe decay analysis (key addition from PyQuantNews)
    train_sharpe_matrix: pd.DataFrame  # Params as index, folds as columns
    test_sharpe_matrix: pd.DataFrame
    sharpe_diff_matrix: pd.DataFrame  # test - train
    sharpe_diff_median: pd.Series  # Aggregated across folds

    # Metadata
    constraint_violations: int
    execution_time_seconds: float
    generalization_score: float  # % of params where test >= train

def parallel_parameter_sweep(
    config: SweepConfig,
    hypothesis_id: str | None = None,
) -> SweepResult:
    """Run parallel parameter sweep with constraint validation."""

def validate_constraints(
    params: dict[str, Any],
    constraints: list[SweepConstraint],
) -> bool:
    """Check if parameter combination satisfies all constraints."""

def compute_sharpe_diff_analysis(
    results_df: pd.DataFrame,
    param_columns: list[str],
    aggregation: str = "median",
) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Compute Sharpe ratio diff analysis across parameter combinations.

    Returns:
        - sharpe_diff_matrix: Full matrix of test-train diffs per fold
        - sharpe_diff_agg: Aggregated (median/mean) diff per param combo
        - generalization_score: % of combos where agg_diff >= 0
    """
```

**Constraint Types**:
- `sum_equals`: Feature weights must sum to a value
- `max_total`: Maximum number of active parameters
- `ratio_bound`: Ratio between parameters must be within bounds
- `exclusion`: Mutually exclusive parameters
- `difference_min`: Minimum difference between two params (e.g., slow > fast + 5)

**Tests**: `tests/test_research/test_parameter_sweep.py`
- `test_sum_equals_constraint`
- `test_max_total_constraint`
- `test_difference_min_constraint`
- `test_parallel_execution`
- `test_constraint_validation`
- `test_integrates_with_overfitting_guard`
- `test_sharpe_diff_analysis`
- `test_generalization_score_calculation`

---

## Phase 3: ATR-Based Trailing Stops

**Objective**: Add ATR-based trailing stop functionality to backtests.

### Modify: `hrp/research/config.py`

```python
@dataclass
class StopLossConfig:
    """Configuration for stop-loss mechanisms."""
    enabled: bool = False
    type: str = "atr_trailing"  # "fixed_pct", "atr_trailing", "volatility_scaled"
    atr_multiplier: float = 2.0
    atr_period: int = 14
    fixed_pct: float = 0.05
    lookback_for_high: int = 1

@dataclass
class BacktestConfig:
    # ... existing fields ...
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
```

### New Module: `hrp/research/stops.py`

```python
@dataclass
class StopResult:
    """Result of stop-loss check for a position."""
    triggered: bool
    trigger_date: date | None
    trigger_price: float | None
    stop_level: float
    pnl_at_stop: float | None

def compute_atr_stops(
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Compute ATR-based trailing stop levels."""

def apply_trailing_stops(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    stop_config: StopLossConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply trailing stops to signals."""

def calculate_stop_statistics(
    stop_events: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, float]:
    """Calculate stop-loss statistics for reporting."""
```

**Tests**: `tests/test_research/test_stops.py`
- `test_stop_level_below_price`
- `test_stop_tracks_trailing_high`
- `test_generates_exit_signals`
- `test_backtest_with_atr_trailing_stop`

---

## Phase 4: Walk-Forward Split & Parameter Heatmap Visualization

**Objective**: Add interactive visualizations for walk-forward splits AND **Sharpe decay heatmaps** to identify robust parameter regions.

### Key Visualization from PyQuantNews Thread

The heatmap shows Sharpe ratio differences (test - train) across parameter combinations:
- X-axis: `fast_period` (10-50)
- Y-axis: `slow_period` (15-50)
- Color: `RdBu` scale (Blue = good generalization, Red = overfitting)
- Triangular shape due to constraint `slow > fast`

### New Component: `hrp/dashboard/components/walkforward_viz.py`

```python
def render_walkforward_splits(
    fold_results: list[FoldResult],
    config: WalkForwardConfig,
) -> None:
    """
    Render interactive walk-forward split visualization.

    Shows:
    - Timeline of train/test periods for each fold
    - Per-fold metrics (IC, MSE, R2) as annotations
    - Stability score indicator
    """

def render_fold_metrics_heatmap(
    fold_results: list[FoldResult],
) -> None:
    """Render heatmap of metrics across folds."""

def render_fold_comparison_chart(
    fold_results: list[FoldResult],
) -> None:
    """Render bar chart comparing fold performance."""
```

### New Component: `hrp/dashboard/components/sharpe_decay_viz.py`

```python
def render_sharpe_decay_heatmap(
    sweep_result: SweepResult,
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
    import plotly.express as px

    # Pivot sharpe_diff_median to 2D matrix
    heatmap_data = sweep_result.sharpe_diff_median.unstack(param_x)

    fig = px.imshow(
        heatmap_data,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale=colorscale,
        color_continuous_midpoint=0,  # Center at zero
        labels={"x": param_x, "y": param_y, "color": "Sharpe Diff"},
        title="Sharpe Ratio Decay: Test - Train (Blue = Good Generalization)",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_generalization_summary(
    sweep_result: SweepResult,
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
        len(sweep_result.results_df),
    )

def render_parameter_sensitivity_chart(
    sweep_result: SweepResult,
    param_name: str,
) -> None:
    """
    Render sensitivity analysis for a single parameter.

    Shows how Sharpe diff varies as one parameter changes,
    holding others at their median values.
    """
```

### Modify: `hrp/dashboard/pages/experiments.py`

Add new tabs:

**Tab 1: Walk-Forward Splits**
- Configuration form for model/features/date range
- Interactive Plotly timeline visualization
- Stability score metrics display
- Per-fold performance comparison

**Tab 2: Parameter Optimization** (NEW)
- Parameter sweep configuration
- **Sharpe decay heatmap** (key visualization)
- Generalization score summary
- Best/worst parameter regions highlighted

**Tests**: `tests/test_dashboard/test_walkforward_viz.py`
- `test_render_splits_with_valid_data`
- `test_render_handles_empty_folds`

**Tests**: `tests/test_dashboard/test_sharpe_decay_viz.py`
- `test_render_sharpe_decay_heatmap`
- `test_heatmap_centers_at_zero`
- `test_generalization_summary_metrics`
- `test_handles_triangular_constraint_data`

---

## Phase 5: HMM Regime Detection Module

**Objective**: Add Hidden Markov Model regime detection for market state identification.

### New Module: `hrp/ml/regime.py`

```python
from enum import Enum

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"

@dataclass
class HMMConfig:
    """Configuration for HMM regime detection."""
    n_regimes: int = 3
    features: list[str] = field(default_factory=lambda: ["returns_20d", "volatility_20d"])
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42

@dataclass
class RegimeResult:
    """Result of regime detection."""
    regimes: pd.Series
    transition_matrix: np.ndarray
    regime_means: dict[int, dict[str, float]]
    regime_covariances: dict[int, np.ndarray]
    log_likelihood: float
    regime_durations: dict[int, float]

class RegimeDetector:
    """Hidden Markov Model regime detector."""

    def __init__(self, config: HMMConfig): ...
    def fit(self, prices: pd.DataFrame) -> "RegimeDetector": ...
    def predict(self, prices: pd.DataFrame) -> pd.Series: ...
    def get_transition_matrix(self) -> pd.DataFrame: ...
```

### Integration with Robustness Checks

Update `hrp/risk/robustness.py`:

```python
def check_regime_stability_hmm(
    returns: pd.Series,
    prices: pd.DataFrame,
    strategy_metrics_by_date: pd.DataFrame,
) -> RobustnessResult:
    """Check strategy performance across HMM-detected regimes."""
```

**New Dependency**: `hmmlearn>=0.3.0`

**Tests**: `tests/test_ml/test_regime.py`
- `test_fit_returns_self`
- `test_predict_returns_series`
- `test_regime_mapping_correct`
- `test_transition_matrix_probabilities`
- `test_check_regime_stability_hmm`

---

## Key Takeaway from PyQuantNews Thread

> "Although you might have developed a promising strategy on paper, cross-validating it is essential to confirm its consistent performance over time and to ensure it's not merely a result of random fluctuations."

The **Sharpe decay heatmap** is the critical diagnostic tool:
- Visualizes which parameter regions generalize vs overfit
- Blue = test > train = robust parameters
- Red = test < train = overfitting
- Use median across folds for robustness
- Triangular heatmaps naturally show parameter constraints

---

## Files Summary

### New Files

| File | Phase | Description |
|------|-------|-------------|
| `hrp/ml/optimization.py` | 1 | Cross-validated optimization |
| `hrp/research/parameter_sweep.py` | 2 | Parallel parameter sweeps + Sharpe decay analysis |
| `hrp/research/stops.py` | 3 | Trailing stop implementation |
| `hrp/dashboard/components/walkforward_viz.py` | 4 | Split visualization |
| `hrp/dashboard/components/sharpe_decay_viz.py` | 4 | **Sharpe decay heatmap** (key viz) |
| `hrp/ml/regime.py` | 5 | HMM regime detection |
| `tests/test_ml/test_optimization.py` | 1 | Optimization tests |
| `tests/test_research/test_parameter_sweep.py` | 2 | Sweep tests + Sharpe diff tests |
| `tests/test_research/test_stops.py` | 3 | Stop tests |
| `tests/test_dashboard/test_walkforward_viz.py` | 4 | Viz tests |
| `tests/test_dashboard/test_sharpe_decay_viz.py` | 4 | Heatmap viz tests |
| `tests/test_ml/test_regime.py` | 5 | Regime tests |

### Modified Files

| File | Phase | Changes |
|------|-------|---------|
| `hrp/research/config.py` | 3 | Add `StopLossConfig` |
| `hrp/research/backtest.py` | 3 | Integrate trailing stops |
| `hrp/dashboard/pages/experiments.py` | 4 | Add walk-forward tab |
| `hrp/risk/robustness.py` | 5 | Add regime stability check |
| `requirements.txt` | 5 | Add `hmmlearn` |

---

## Integration Requirements

### Overfitting Guards

All optimization features must integrate with existing guards:

```python
from hrp.risk.overfitting import (
    HyperparameterTrialCounter,
    SharpeDecayMonitor,
    TestSetGuard,
)

# In optimization.py / parameter_sweep.py:
counter = HyperparameterTrialCounter(hypothesis_id, max_trials=50)
decay_monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
guard = TestSetGuard(hypothesis_id, max_evaluations=3)
```

### MLflow Logging

All new modules must log to MLflow:

```python
import mlflow

with mlflow.start_run(run_name="cv_optimization"):
    mlflow.log_param("model_type", config.model_type)
    mlflow.log_param("n_trials", len(results))
    mlflow.log_metric("best_score", best_score)
    mlflow.log_dict(best_params, "best_params.json")
```

### Lineage Logging

All significant events must be logged:

```python
from hrp.research.lineage import log_event

log_event(
    event_type="optimization_completed",
    details={"best_params": best_params, "n_trials": n_trials},
    hypothesis_id=hypothesis_id,
    actor="system",
)
```

---

## Implementation Order

```
Phase 1: Cross-Validated Optimization
    └── Phase 2: Parallel Parameter Sweeps (depends on 1)
            └── Phase 4: Walk-Forward Visualization (depends on 1)

Phase 3: ATR Trailing Stops (independent)

Phase 5: HMM Regime Detection (independent, can run parallel)
```

**Recommended sequence**: 1 → 2 → 3 → 4 → 5

---

## Success Criteria

1. All tests pass (`pytest tests/ -v` = 100%)
2. Each feature integrates with existing overfitting guards
3. MLflow logging for all optimization runs
4. Dashboard visualizations render correctly
5. Documentation updated in CLAUDE.md, Project-Status.md, Changelog.md

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting from optimization | Integrate with `HyperparameterTrialCounter`, `SharpeDecayMonitor` |
| Computational cost | Configurable parallelization, early stopping |
| HMM instability | Multiple random seeds, stability checks |
| Backward compatibility | All features opt-in, existing API unchanged |
