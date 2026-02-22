# Design: TASK-017 Phase 5 - Advanced Backtesting UI

**Author:** Athena
**Date:** 2026-02-14
**Status:** Ready for Implementation
**Priority:** High
**Pipeline:** Athena → Forge → Gauntlet → Sentinel

---

## Overview

Create a dashboard page that exposes the Optuna optimization infrastructure and allows interactive parameter exploration. This UI enables users to:
- Configure and run cross-validated hyperparameter optimization
- Visualize optimization progress and results
- Analyze fold stability and overfitting risk
- Compare optimization runs across studies
- Export results for further analysis

**Key Innovation:** Live parameter preview showing impact before running optimization, reducing wasted computation on poor configurations.

---

## Current State

### Existing Infrastructure

**Optimization Module** (`hrp/ml/optimization.py`):
- ✅ `OptimizationConfig` dataclass with full parameter space support
- ✅ `OptimizationResult` dataclass with trial history and fold results
- ✅ `cross_validated_optimize()` function with Optuna integration
- ✅ 4 samplers: grid, random, TPE, CMAES
- ✅ Median pruning support
- ✅ Overfitting guards via `HyperparameterTrialCounter` and `SharpeDecayMonitor`
- ✅ MLflow logging integration
- ✅ SQLite study persistence for resume capability

**Dashboard Infrastructure**:
- ✅ Streamlit app with routing in `hrp/dashboard/app.py`
- ✅ Phase 4 Risk Limits page provides UI patterns to follow
- ✅ Platform API for database access
- ✅ Strategy config components in `hrp/dashboard/components/strategy_config.py`

### Missing Components

- ❌ Optimization API layer (business logic)
- ❌ Optimization UI page (`hrp/dashboard/pages/13_Optimization.py`)
- ❌ Optimization controls components
- ❌ Parameter importance visualization
- ❌ Study history management

---

## Proposed Changes

### Architecture Decision: Focus on Optuna Only

The original design considered both parameter sweep grid search and Optuna. However:

1. **Parameter sweep not implemented:** `parameter_sweep.py`'s `_evaluate_single_combination()` raises `NotImplementedError`
2. **Optuna is production-ready:** Fully integrated with 4 samplers, pruning, and MLflow
3. **Simpler UX:** Single optimization approach reduces cognitive load
4. **Better performance:** Bayesian optimization (TPE) converges faster than grid search

**Decision:** Focus entirely on Optuna optimization. Do NOT implement parameter sweep grid search.

---

## Components Affected

### 1. NEW: `hrp/api/optimization_api.py`

**Purpose:** Business logic layer for optimization operations. Follows the `RiskConfigAPI` pattern from Phase 4.

**Key Classes:**

```python
@dataclass
class OptimizationPreview:
    """Preview of optimization configuration without running."""
    estimated_time_seconds: float
    estimated_cost_estimate: str  # "Low (~1m)", "Medium (~5m)", "High (~15m+)"
    parameter_space_summary: dict[str, str]  # Human-readable param ranges
    recommended_sampler: str  # Based on parameter space
    warnings: list[str]  # e.g., "High trial count may take >10m"

class OptimizationAPI:
    """API for optimization configuration and execution."""
    def __init__(self, api: PlatformAPI):
        self.api = api

    def get_default_param_space(self, model_type: str) -> dict[str, BaseDistribution]:
        """Get default Optuna parameter space for model type."""

    def get_available_strategies(self) -> list[str]:
        """Get list of strategies with ML models."""

    def estimate_execution_time(self, config: OptimizationConfig) -> float:
        """Estimate execution time based on configuration."""

    def preview_configuration(self, config: OptimizationConfig) -> OptimizationPreview:
        """Preview optimization impact without running."""

    def run_optimization(
        self,
        config: OptimizationConfig,
        symbols: list[str],
        progress_callback: Callable[[int, int], None] | None = None
    ) -> OptimizationResult:
        """Run optimization with progress updates."""

    def list_studies(self, hypothesis_id: str | None = None) -> list[dict]:
        """List Optuna studies from storage."""

    def get_study_details(self, study_name: str) -> dict:
        """Get detailed study information including parameter importance."""

    def delete_study(self, study_name: str) -> bool:
        """Delete a study from storage."""
```

**Design Rationale:**
- Separation of concerns: UI layer doesn't touch Optuna directly
- Reusable API: Future MCP servers can use this for optimization
- Testable: Business logic isolated from Streamlit

---

### 2. NEW: `hrp/dashboard/pages/13_Optimization.py`

**Purpose:** Main optimization dashboard page. Follows Phase 4's risk_limits.py patterns.

**Page Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│ Strategy Optimization                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ [Sidebar - Configuration]                                  │
│ Strategy:    [multifactor_ml ▼]                             │
│   └─ Model:       [ridge ▼]                                │
│ Sampler:     [TPE ▼] (grid/random/tpe/cmaes)              │
│ Trials:      [50] (slider 10-200)                          │
│ CV Folds:    [5]  (slider 3-10)                            │
│ Scoring:     [IC ▼] (ic/r2/mse/mae/sharpe)                │
│ Window:      [expanding ▼] (expanding/rolling)             │
│ Pruning:     [✓] enabled                                   │
│ Date Range:  [2020-01-01] to [2025-12-31]                  │
│                                                             │
│ Feature Selection:                                          │
│ [✓] momentum_20d  [✓] volatility_60d  [✓] rsi_14d          │
│ [ ] quality_roe    [ ] value_pe_ratio                       │
│                                                             │
│ [Configuration Preview]                                     │
│ Estimated time: ~3 minutes                                  │
│ Cost estimate: Medium                                       │
│ Recommended sampler: TPE                                    │
│ ⚠️ High trial count (150) may take >10m                     │
│                                                             │
│ [Run Optimization]  ← button                                │
│                                                             │
│ [Tabs]                                                     │
│ [Results] [Fold Analysis] [Study History]                   │
│                                                             │
│ Tab 1: Results                                              │
│ ┌──────────────────────────────────────────────────┐       │
│ │ Best Parameters                                   │       │
│ │ alpha: 0.75                                        │       │
│ │                                                       │       │
│ │ Best Score: 0.0234 (IC)                            │       │
│ │                                                       │       │
│ │ Optimization Progress Chart                         │       │
│ │ [Line chart: trial # vs score with best line]      │       │
│ │                                                       │       │
│ │ Parameter Importance (from Optuna)                  │       │
│ │ [Bar chart: param name vs importance]              │       │
│ └──────────────────────────────────────────────────┘       │
│                                                             │
│ Tab 2: Fold Analysis                                         │
│ ┌──────────────────────────────────────────────────┐       │
│ │ Fold-wise Metrics                                 │       │
│ │ [Grouped bar: train vs test IC per fold]          │       │
│ │                                                       │       │
│ │ Stability Score: 0.032 (CV of test IC)            │       │
│ │ Overfitting Risk: Low (train/test gap: 0.005)     │       │
│ └──────────────────────────────────────────────────┘       │
│                                                             │
│ Tab 3: Study History                                        │
│ ┌──────────────────────────────────────────────────┐       │
│ │ Previous Runs                                     │       │
│ │ Study ID | Date | Model | Trials | Best Score    │       │
│ │ abc-123  | ... | ridge | 50     | 0.0234        │       │
│ │ def-456  | ... | lasso | 100    | 0.0256        │       │
│ │                                                       │       │
│ │ [Compare] [Delete] [Export CSV]                    │       │
│ └──────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
1. **Sidebar Configuration:** All optimization controls in sidebar (follows Phase 4 pattern)
2. **Live Preview:** Shows estimated time and warnings before running
3. **Progress Bar:** Streamlit progress callback during optimization
4. **Three Tabs:** Results, Fold Analysis, Study History
5. **Export:** Download results as CSV

---

### 3. NEW: `hrp/dashboard/components/optimization_controls.py`

**Purpose:** Reusable UI components for optimization configuration.

**Components:**

```python
def render_strategy_selector(api: PlatformAPI) -> str:
    """Render strategy dropdown and return selected strategy."""

def render_model_selector() -> str:
    """Render model type dropdown (ridge/lasso/elasticnet/random_forest/lightgbm)."""

def render_sampler_selector() -> str:
    """Render Optuna sampler selector (grid/random/tpe/cmaes)."""

def render_trials_slider(default: int = 50, max_val: int = 200) -> int:
    """Render trials slider with cost estimate."""

def render_folds_slider(default: int = 5, max_val: int = 10) -> int:
    """Render CV folds slider."""

def render_scoring_selector() -> str:
    """Render scoring metric selector (ic/r2/mse/mae/sharpe)."""

def render_date_range() -> tuple[date, date]:
    """Render start/end date pickers."""

def render_feature_selector(
    api: PlatformAPI,
    default_features: list[str] | None = None
) -> list[str]:
    """Render multi-select for features with grouping by category."""

def render_optimization_preview(preview: OptimizationPreview) -> None:
    """Render preview card with estimated time and warnings."""

def render_results_tab(result: OptimizationResult) -> None:
    """Render Results tab with best params, progress chart, importance."""

def render_fold_analysis_tab(result: OptimizationResult) -> None:
    """Render Fold Analysis tab with stability metrics."""

def render_study_history_tab(studies: list[dict]) -> None:
    """Render Study History tab with comparison options."""
```

**Design Rationale:**
- Reusable across pages (e.g., could be embedded in strategy config)
- Isolated for testing (each component can be unit tested)
- Follows existing `strategy_config.py` patterns

---

### 4. MODIFIED: `hrp/dashboard/app.py`

**Change:** Add routing for optimization page.

```python
# Add to page routing
pages = {
    "Home": pages.home,
    "Data Health": pages.data_health,
    # ... existing pages ...
    "Risk Limits": pages.risk_limits.render_risk_limits,
    "Optimization": pages.optimization.render_optimization_page,  # NEW
}
```

---

## Data Flow

### 1. Initial Page Load

```
User requests /Optimization
  ↓
app.py routes to 13_Optimization.py
  ↓
render_optimization_page(api)
  ↓
Initialize session state:
  - optimization_config = None
  - optimization_result = None
  - last_study_id = None
  ↓
Render sidebar with default values
  ↓
Generate OptimizationPreview with defaults
  ↓
Show preview card (estimated time: ~3m)
```

### 2. User Updates Configuration

```
User changes sampler from TPE to Grid
  ↓
Streamlit triggers on_change callback
  ↓
Call preview_configuration() with new config
  ↓
Update preview card (estimated time: ~8m)
  ↓
Show warning: "Grid sampler with 200 trials may take >15m"
```

### 3. User Runs Optimization

```
User clicks "Run Optimization"
  ↓
Validate configuration (via OptimizationAPI)
  ↓
Show progress bar: st.progress(0)
  ↓
Call OptimizationAPI.run_optimization(progress_callback=update_progress)
  ↓
For each trial:
  progress_callback(trial_idx, n_trials)
  st.progress(trial_idx / n_trials)
  ↓
Optimization complete → OptimizationResult
  ↓
Store in session_state["optimization_result"]
  ↓
Switch to Results tab automatically
  ↓
Render best params, progress chart, parameter importance
```

### 4. User Studies History

```
User clicks Study History tab
  ↓
Call OptimizationAPI.list_studies()
  ↓
Load Optuna studies from SQLite storage
  ↓
Render table with Study ID, Date, Model, Trials, Best Score
  ↓
User clicks "Compare" for 2 studies
  ↓
Call OptimizationAPI.get_study_details() for each
  ↓
Show side-by-side parameter comparison
```

---

## Implementation Tasks

### Phase 1: API Layer (Priority: P0)

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 1 | Create `OptimizationAPI` class with `__init__` | `optimization_api.py` | Instantiates with PlatformAPI |
| 2 | Implement `get_default_param_space()` | `optimization_api.py` | Returns correct distributions for each model |
| 3 | Implement `get_available_strategies()` | `optimization_api.py` | Returns strategies with ML models |
| 4 | Implement `estimate_execution_time()` | `optimization_api.py` | Heuristic based on trials × folds |
| 5 | Implement `preview_configuration()` | `optimization_api.py` | Returns OptimizationPreview with warnings |
| 6 | Implement `run_optimization()` with progress callback | `optimization_api.py` | Calls `cross_validated_optimize()`, updates progress |
| 7 | Implement `list_studies()` | `optimization_api.py` | Reads SQLite storage, returns study metadata |
| 8 | Implement `get_study_details()` with importance | `optimization_api.py` | Loads study, computes param importance |
| 9 | Add unit tests for OptimizationAPI | `tests/api/test_optimization_api.py` | All methods tested |

**Estimated Complexity:** Medium

---

### Phase 2: UI Components (Priority: P0)

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 1 | Create `optimization_controls.py` module | `optimization_controls.py` | Imports correctly |
| 2 | Implement `render_strategy_selector()` | `optimization_controls.py` | Returns selected strategy |
| 3 | Implement `render_model_selector()` | `optimization_controls.py` | Returns selected model |
| 4 | Implement `render_sampler_selector()` | `optimization_controls.py` | Returns selected sampler |
| 5 | Implement `render_trials_slider()` | `optimization_controls.py` | Returns trial count |
| 6 | Implement `render_folds_slider()` | `optimization_controls.py` | Returns fold count |
| 7 | Implement `render_scoring_selector()` | `optimization_controls.py` | Returns selected metric |
| 8 | Implement `render_date_range()` | `optimization_controls.py` | Returns (start, end) dates |
| 9 | Implement `render_feature_selector()` | `optimization_controls.py` | Returns selected features |
| 10 | Implement `render_optimization_preview()` | `optimization_controls.py` | Shows preview card |
| 11 | Implement `render_results_tab()` | `optimization_controls.py` | Shows results visualizations |
| 12 | Implement `render_fold_analysis_tab()` | `optimization_controls.py` | Shows fold stability |
| 13 | Implement `render_study_history_tab()` | `optimization_controls.py` | Shows study list |

**Estimated Complexity:** Medium

---

### Phase 3: Main Page (Priority: P0)

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 1 | Create `13_Optimization.py` page | `pages/13_Optimization.py` | Loads in Streamlit |
| 2 | Implement `render_optimization_page()` entry point | `13_Optimization.py` | Renders sidebar and tabs |
| 3 | Integrate sidebar controls using components | `13_Optimization.py` | All selectors render correctly |
| 4 | Implement live preview on configuration change | `13_Optimization.py` | Preview updates on input change |
| 5 | Implement "Run Optimization" button with progress | `13_Optimization.py` | Progress bar updates during run |
| 6 | Implement auto-switch to Results tab after run | `13_Optimization.py` | Tab switches automatically |
| 7 | Implement Results tab visualization | `13_Optimization.py` | Shows best params, charts |
| 8 | Implement Fold Analysis tab | `13_Optimization.py` | Shows fold-wise metrics |
| 9 | Implement Study History tab | `13_Optimization.py` | Lists and compares studies |
| 10 | Add export functionality (CSV download) | `13_Optimization.py` | Downloads CV results |
| 11 | Add routing to `app.py` | `app.py` | "Optimization" link works |

**Estimated Complexity:** Medium-High

---

### Phase 4: Polish & Testing (Priority: P1)

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 1 | Manual UI testing - all controls and visualizations | - | No console errors |
| 2 | Integration test - small optimization (5 trials, 3 folds) | - | Results display correctly |
| 3 | Test edge cases - no results, optimization failure | - | Graceful error messages |
| 4 | Test progress callback - verify bar updates | - | Progress bar moves |
| 5 | Test preview accuracy - compare estimated vs actual | - | Within 50% tolerance |
| 6 | Test study persistence - restart and reload | - | Studies list correctly |
| 7 | Add tooltips and help text for all inputs | `13_Optimization.py` | Users understand options |
| 8 | Add loading states for expensive operations | `13_Optimization.py` | Spinners show correctly |
| 9 | Responsive design - test on mobile viewport | - | Layout usable |

**Estimated Complexity:** Low-Medium

---

## Testing Strategy

### Unit Tests

**OptimizationAPI:**
- `test_get_default_param_space()` - Verify distributions for each model
- `test_estimate_execution_time()` - Verify heuristics (50 trials × 5 folds ≈ 3m)
- `test_preview_configuration()` - Verify warnings generated
- `test_list_studies()` - Verify SQLite reading
- `test_get_study_details()` - Verify importance calculation

**Optimization Controls:**
- `test_render_strategy_selector()` - Mock Streamlit, verify returns
- `test_render_model_selector()` - Verify all models available
- `test_render_feature_selector()` - Verify feature grouping

### Integration Tests

- Run small optimization (5 trials, 3 folds) → verify results display
- Test progress callback updates → verify progress bar moves
- Test study persistence → restart app → verify studies load
- Test export → verify CSV downloads correctly

### Manual UI Testing

- All controls render and respond
- Preview updates on configuration change
- Progress bar shows during optimization
- Results tab shows best params and charts
- Fold analysis shows stability metrics
- Study history lists previous runs

### Edge Cases

- No studies exist → show "No previous runs"
- Optimization fails → show error message
- Invalid date range → validation error
- Too many trials → warning in preview
- Empty feature selection → error before run

---

## Edge Cases & Error Handling

### 1. No Data Available

**Scenario:** User selects date range with no data.

**Handling:**
- Validate before running: `if all_data.empty: raise ValueError(...)`
- Show error: "No data found for selected date range. Try a different range."
- Disable "Run" button if validation fails

### 2. Optimization Failure

**Scenario:** Optuna throws exception during optimization.

**Handling:**
- Wrap in try-except in `run_optimization()`
- Log full error to logger
- Show user-friendly message: "Optimization failed: {error}"
- Preserve any partial results if available

### 3. Study Storage Not Writable

**Scenario:** SQLite file permission denied.

**Handling:**
- Catch `PermissionError` in `list_studies()`
- Fall back to in-memory study (no persistence)
- Show warning: "Study storage not available. Results will not persist."

### 4. Progress Callback Timeout

**Scenario:** Progress callback fails (e.g., Streamlit session disconnected).

**Handling:**
- Make callback optional: `progress_callback: Callable | None = None`
- If callback raises, log warning but continue optimization
- Don't fail optimization just because progress update fails

### 5. Too Many Trials

**Scenario:** User sets 200 trials with grid sampler.

**Handling:**
- Show warning in preview: "High trial count may take >15m"
- Allow but confirm with `st.confirmation_dialog()`
- Provide "Run in background" option (future enhancement)

---

## Phase 4 Learnings Incorporated

From Phase 4 Risk Limits implementation, applying these patterns:

1. **API Layer Pattern:**
   - Create dedicated `OptimizationAPI` (like `RiskConfigAPI`)
   - Business logic isolated from UI
   - Reusable for future MCP servers

2. **Dataclasses for Configuration:**
   - Use `OptimizationConfig` (already exists)
   - Create `OptimizationPreview` dataclass
   - Type safety and runtime validation

3. **Live Preview Without Committing:**
   - `preview_configuration()` shows estimated time and warnings
   - User sees impact before expensive computation
   - Reduces wasted runs on poor configurations

4. **Session State Management:**
   - Store editable config in `st.session_state`
   - Cache results to avoid re-running
   - Reset to defaults available

5. **Progressive Disclosure:**
   - Sidebar for configuration (collapsible)
   - Tabs for different views (Results, Folds, History)
   - Expandable sections for details

6. **Visual Feedback:**
   - Color-coded cards (green for good, yellow for warnings)
   - Progress bar during long operations
   - Dynamic styling based on results

7. **Error Handling:**
   - Try-except blocks in all API methods
   - Meaningful error messages to users
   - Graceful fallback (e.g., default limits)

8. **Validation Before Action:**
   - Validate config before `run_optimization()`
   - Check data availability
   - Warn about expensive operations

9. **Reset Functionality:**
   - "Reset to Defaults" button
   - Quick way to start over
   - Useful for exploration

---

## Design Decisions

### Decision 1: Focus on Optuna Only

**Rationale:**
- Parameter sweep grid search not implemented
- Optuna is production-ready with better algorithms
- Simpler UX (single optimization approach)

**Trade-off:** No grid search option, but Optuna covers use cases.

### Decision 2: Sidebar Configuration

**Rationale:**
- Follows Phase 4 pattern (risk limits)
- Keeps main area free for visualizations
- Collapsible on mobile

**Trade-off:** Less visible than top-of-page config, but cleaner layout.

### Decision 3: Progress Callback

**Rationale:**
- User feedback during long optimization (can take 5-15m)
- Streamlit progress bar is standard pattern
- Allows cancellation (future)

**Trade-off:** Adds complexity to `run_optimization()`, but worth it for UX.

### Decision 4: Three Tabs

**Rationale:**
- Progressive disclosure: Results first, details later
- Reduces cognitive load
- Follows standard dashboard pattern

**Trade-off:** Extra clicks to see details, but cleaner initial view.

### Decision 5: Estimated Time Preview

**Rationale:**
- Set expectations before running
- Helps user choose appropriate trial count
- Warns about expensive configurations

**Trade-off:** Heuristic may be inaccurate, but better than no estimate.

---

## Open Questions

None at this time. Design is ready for implementation.

---

## Next Steps

1. **Athena:** Log this design to Convex documents
2. **Athena:** Send handoff message to Forge
3. **Forge:** Implement Phase 1 (API Layer)
4. **Forge:** Implement Phase 2 (UI Components)
5. **Forge:** Implement Phase 3 (Main Page)
6. **Forge:** Implement Phase 4 (Polish & Testing)
7. **Gauntlet:** Run integration tests
8. **Sentinel:** Code review
9. **Fernando:** Final approval and deployment

---

## Appendix: Default Parameter Spaces

```python
def get_default_param_space(model_type: str) -> dict[str, BaseDistribution]:
    """Get default Optuna parameter space for a model type."""
    if model_type == "ridge":
        return {
            "alpha": FloatDistribution(0.01, 100.0, log=True)
        }
    elif model_type == "lasso":
        return {
            "alpha": FloatDistribution(0.001, 10.0, log=True)
        }
    elif model_type == "elasticnet":
        return {
            "alpha": FloatDistribution(0.001, 10.0, log=True),
            "l1_ratio": FloatDistribution(0.0, 1.0),
        }
    elif model_type == "random_forest":
        return {
            "n_estimators": IntDistribution(50, 500, step=50),
            "max_depth": IntDistribution(3, 15),
            "min_samples_leaf": IntDistribution(5, 50),
        }
    elif model_type == "lightgbm":
        return {
            "n_estimators": IntDistribution(50, 500, step=50),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "max_depth": IntDistribution(3, 12),
            "num_leaves": IntDistribution(15, 127),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

---

## Appendix: Estimated Time Heuristic

```python
def estimate_execution_time(config: OptimizationConfig) -> float:
    """
    Estimate execution time in seconds.

    Heuristic:
    - Base time per fold: ~10s (data fetch + model training)
    - Multiply by n_trials × n_folds
    - Adjust for model complexity:
      - Linear models: 1x
      - Random forest: 2x
      - LightGBM: 1.5x
    - Adjust for sampler:
      - TPE/CMAES: 1x (pruning helps)
      - Random: 1.2x
      - Grid: 2x (no pruning, many wasted trials)
    """
    base_time_per_fold = 10  # seconds

    model_complexity = {
        "ridge": 1.0,
        "lasso": 1.0,
        "elasticnet": 1.0,
        "random_forest": 2.0,
        "lightgbm": 1.5,
    }.get(config.model_type, 1.5)

    sampler_overhead = {
        "tpe": 1.0,
        "cmaes": 1.0,
        "random": 1.2,
        "grid": 2.0,
    }.get(config.sampler, 1.0)

    estimated_time = (
        config.n_trials
        * config.n_folds
        * base_time_per_fold
        * model_complexity
        * sampler_overhead
    )

    return estimated_time
```
