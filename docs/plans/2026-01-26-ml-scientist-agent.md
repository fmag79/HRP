# Agent Definition: ML Scientist

**Date:** January 26, 2026
**Status:** Implemented
**Type:** Custom (deterministic - extends `ResearchAgent`)

---

## Identity

| Attribute | Value |
|-----------|-------|
| **Name** | ML Scientist |
| **Actor ID** | `agent:ml-scientist` |
| **Type** | Custom (deterministic) |
| **Role** | Model development, walk-forward validation, hyperparameter optimization |
| **Implementation** | `hrp/agents/research_agents.py` |
| **Trigger** | Lineage event (after Alpha Researcher) + MCP on-demand |
| **Upstream** | Alpha Researcher (promotes hypotheses to 'testing' status) |
| **Downstream** | ML Quality Sentinel (audits experiments for overfitting) |

---

## Purpose

Validates hypotheses through rigorous ML experimentation. The ML Scientist:

1. **Picks up hypotheses** in 'testing' status
2. **Runs walk-forward validation** with proper train/test splits
3. **Evaluates model stability** across time folds
4. **Updates hypothesis status** to 'validated' or 'rejected'
5. **Triggers downstream** ML Quality Sentinel via lineage events

---

## Core Capabilities

### 1. Hypothesis Validation Pipeline

```python
from hrp.agents import MLScientist

# Run validation on testing hypotheses
scientist = MLScientist(
    n_folds=5,
    window_type='expanding',  # or 'rolling'
    stability_threshold=1.0,  # CV threshold for stability
)
result = scientist.run()

print(f"Tested: {result['hypotheses_tested']}")
print(f"Validated: {result['hypotheses_validated']}")
print(f"Rejected: {result['hypotheses_rejected']}")
```

### 2. Walk-Forward Validation

For each hypothesis:
- Extract feature specification from hypothesis
- Run walk-forward validation with configurable folds
- Calculate stability score (CV of IC across folds)
- Log all metrics to MLflow

### 3. Model Training

Supports multiple model types:
- `ridge` - Ridge regression (default)
- `lasso` - Lasso regression
- `random_forest` - Random Forest
- `lightgbm` - LightGBM
- `xgboost` - XGBoost

### 4. Hypothesis Status Updates

| Condition | New Status | Action |
|-----------|------------|--------|
| `stability_score <= threshold` & positive IC | `validated` | Proceed to deployment review |
| `stability_score > threshold` | `rejected` | Unstable across time |
| IC sign flips across folds | `rejected` | Inconsistent direction |
| Mean IC < 0 | `rejected` | No predictive power |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hypothesis_ids` | `None` (all testing) | Specific hypotheses to validate |
| `n_folds` | `5` | Number of walk-forward folds |
| `window_type` | `'expanding'` | `'expanding'` or `'rolling'` |
| `stability_threshold` | `1.0` | Max CV for stability (lower is better) |
| `model_type` | `'ridge'` | ML model to train |
| `feature_selection` | `True` | Enable automatic feature selection |
| `max_features` | `20` | Max features after selection |
| `n_jobs` | `-1` | Parallel fold processing |

---

## Outputs

### 1. Model Experiment Result

```python
@dataclass
class ModelExperimentResult:
    hypothesis_id: str
    model_type: str
    n_folds: int
    fold_results: list[dict]  # Per-fold metrics
    mean_ic: float
    stability_score: float  # CV of IC across folds
    is_stable: bool
    passed_validation: bool
    mlflow_run_id: str
```

### 2. ML Scientist Report

```python
@dataclass
class MLScientistReport:
    report_date: date
    hypotheses_tested: int
    hypotheses_validated: int
    hypotheses_rejected: int
    experiments: list[ModelExperimentResult]
    duration_seconds: float
```

### 3. MLflow Logging

For each hypothesis:
- Parent run: hypothesis-level metrics
- Child runs: per-fold results
- Metrics: IC, MSE, R², Sharpe (if applicable)
- Parameters: model type, feature list, window config
- Artifacts: feature importance, fold predictions

### 4. Lineage Events

- `ML_SCIENTIST_VALIDATION`: Per-hypothesis validation event
- `AGENT_RUN_COMPLETE`: Triggers ML Quality Sentinel
- `HYPOTHESIS_STATUS_CHANGE`: Status updates logged

### 5. Research Note

- Location: `docs/research/YYYY-MM-DD-ml-scientist.md`
- Contents: Validation results, stability analysis, recommendations

---

## Trigger Model

### Primary: Lineage Event Trigger

```python
# ML Scientist listens for Alpha Researcher completion
scheduler.register_lineage_trigger(
    event_type="AGENT_RUN_COMPLETE",
    actor_filter="agent:alpha-researcher",
    callback=trigger_ml_scientist,
)
```

### Secondary: MCP On-Demand

```python
# MCP tool: run_ml_scientist
result = run_ml_scientist(
    hypothesis_id="HYP-2026-042",
    n_folds=5,
    model_type="ridge",
)
```

---

## Validation Criteria

### Stability Score Calculation

```python
# CV = standard deviation / mean
stability_score = np.std(fold_ics) / np.abs(np.mean(fold_ics))

# Lower is better:
# < 0.5: Very stable
# 0.5-1.0: Stable
# 1.0-2.0: Moderately unstable
# > 2.0: Unstable (reject)
```

### Pass/Fail Logic

```python
def should_validate(result: ModelExperimentResult) -> bool:
    # Must have positive mean IC
    if result.mean_ic <= 0:
        return False

    # Must be stable across folds
    if result.stability_score > stability_threshold:
        return False

    # No excessive sign flips
    positive_folds = sum(1 for f in result.fold_results if f['ic'] > 0)
    if positive_folds < len(result.fold_results) * 0.6:
        return False

    return True
```

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Hypothesis Registry** | Reads 'testing' hypotheses, updates status |
| **ML Framework** | Uses `walk_forward_validate()` from `hrp/ml/` |
| **MLflow** | Logs experiments, metrics, artifacts |
| **Lineage** | Logs events, triggers downstream agents |
| **Feature Store** | Retrieves features for model training |
| **ML Quality Sentinel** | Downstream - audits experiments |

---

## Example Research Note

```markdown
# ML Scientist Report - 2026-01-26

## Summary
- Hypotheses tested: 4
- Validated: 2
- Rejected: 2
- Duration: 847.2s

## Results

### HYP-2026-042: momentum_20d predicts monthly returns
**Status:** VALIDATED

| Fold | IC | MSE | R² |
|------|----|----|-----|
| 1 | 0.048 | 0.0023 | 0.012 |
| 2 | 0.052 | 0.0021 | 0.015 |
| 3 | 0.041 | 0.0025 | 0.009 |
| 4 | 0.055 | 0.0020 | 0.018 |
| 5 | 0.044 | 0.0024 | 0.011 |

- **Mean IC:** 0.048
- **Stability Score:** 0.41 (very stable)
- **Model:** Ridge regression

### HYP-2026-043: rsi_14d mean reversion
**Status:** REJECTED

- **Mean IC:** 0.012
- **Stability Score:** 2.3 (unstable)
- **Reason:** High variability across folds, IC sign flips in fold 3

## Recommendations
- HYP-2026-042 ready for risk review
- HYP-2026-043 needs thesis refinement or different feature combination

---
*Generated by ML Scientist (agent:ml-scientist)*
```

---

## Performance Notes

- **Parallel fold processing**: Uses all CPU cores for fold validation
- **Typical duration**: 3-5 minutes per hypothesis (5-fold validation)
- **Memory usage**: ~2-4GB for full S&P 500 feature matrix

---

## Document History

- **2026-01-26:** Initial agent definition created
