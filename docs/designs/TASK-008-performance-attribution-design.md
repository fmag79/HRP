# Design: TASK-008 — Strategy Performance Attribution

## Overview

Decompose portfolio returns into explainable components — which factors, features, and decisions drove performance? Answers "why did the strategy make/lose money?"

Three attribution layers:
1. **Factor Attribution** — Brinson-Fachler sector/factor decomposition
2. **Feature Importance** — ML feature importance tracking (SHAP/permutation)
3. **Decision Attribution** — Trade-level P&L decomposition

## Current State (Updated 2026-02-10)

**Status:** Not yet implemented. Design complete. Research brief from Recon available at `docs/research/TASK-008-research-brief.md`.

Forge is currently working on TASK-006 (VaR/CVaR) and this is the next major task after that.

## Architecture — New Files

```
hrp/
  data/
    attribution/                     # NEW: Attribution module
      __init__.py
      factor_attribution.py          # Factor-level return decomposition
      feature_importance.py          # ML feature importance tracking
      decision_attribution.py        # Trade-level decision attribution
      attribution_config.py          # Configuration
    features/
      attribution_features.py        # NEW: Attribution as computed features
  dashboard/
    pages/
      performance_attribution.py     # NEW: Streamlit dashboard page
  tests/
    test_attribution/                # NEW: Test suite
      test_factor_attribution.py
      test_feature_importance.py
      test_decision_attribution.py
```

## Three Attribution Layers

### Layer 1: Factor Attribution (`factor_attribution.py`)

Decomposes portfolio returns by risk factors using **Brinson-Fachler** methodology.

**Effects:**
- **Allocation Effect:** Did we overweight/underweight the right sectors?
- **Selection Effect:** Did we pick good assets within each sector?
- **Interaction Effect:** Cross-term between allocation and selection
- **Total Effect:** Sum of all effects = active return vs benchmark

**Also supports:**
- Market/beta attribution
- Sector attribution
- Style factor attribution (value, momentum, size, quality)

**Key Classes:**
- `BrinsonAttribution` — Classic Brinson-Fachler decomposition
- `FactorAttribution` — Regression-based factor decomposition (Fama-French style)
- `AttributionResult` dataclass: `factor`, `effect_type`, `contribution_pct`, `contribution_dollar`

**Brinson-Fachler Formulas:**
```
Allocation_i  = (w_p_i - w_b_i) × (r_b_i - R_b)
Selection_i   = w_b_i × (r_p_i - r_b_i)
Interaction_i = (w_p_i - w_b_i) × (r_p_i - r_b_i)
Active Return = Σ(Allocation_i + Selection_i + Interaction_i)
```
Where: `w_p` = portfolio weight, `w_b` = benchmark weight, `r_p` = portfolio return, `r_b` = benchmark return, `R_b` = total benchmark return.

### Layer 2: Feature Importance (`feature_importance.py`)

Tracks which ML features drove portfolio decisions over time.

**Methods:**
- **Permutation importance** — Model-agnostic default (shuffle feature, measure degradation)
- **SHAP values** — For tree-based models (optional, if shap installed)
- **Rolling importance** — How feature importance changes over sliding windows
- **Feature interaction** — Which feature pairs matter most together

**Key Classes:**
- `FeatureImportanceTracker` — Wraps SHAP/permutation importance
- `RollingImportance` — Tracks importance over sliding windows
- `ImportanceResult` dataclass: `feature_name`, `importance_score`, `direction`, `period`

### Layer 3: Decision Attribution (`decision_attribution.py`)

Attributes P&L to individual trading decisions.

**Decomposition:**
- **Entry timing:** How much did entry timing contribute?
- **Exit timing:** Did we exit too early/late?
- **Position sizing:** Were position sizes optimal?
- **Rebalancing:** Did rebalancing add or destroy value?

**Key Classes:**
- `DecisionAttributor` — Analyzes trade-level P&L contributions
- `TradeDecision` dataclass: `trade_id`, `asset`, `entry_date`, `exit_date`, `pnl`, `timing_contribution`, `sizing_contribution`
- `RebalanceAnalyzer` — Measures value-add of rebalancing events

## Dashboard Layout (`performance_attribution.py`)

### 7 Sections:
1. **Summary Bar** — Total return, benchmark return, active return, attribution period
2. **Waterfall Chart** — **THE main visualization.** Shows: Allocation + Selection + Interaction + Residual = Active Return
3. **Factor Contribution Table** — Per-factor contribution to returns
4. **Feature Importance Heatmap** — Rolling feature importance over time (color = importance)
5. **Decision Attribution Timeline** — P&L waterfall per trade decision
6. **Sector/Style Attribution** — Pie/bar charts showing sector/style decomposition
7. **Period Comparison** — Attribution across time windows (1m, 3m, 6m, YTD, 1y)

### Sidebar Controls:
- Strategy selector
- Benchmark selector
- Attribution period (date range)
- Factor model (Brinson / regression)
- Feature importance window size

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Factor method | Brinson-Fachler | Industry standard, well-understood |
| Feature importance default | Permutation importance | Model-agnostic, works with any model |
| SHAP | Optional upgrade | Only if tree-based model and shap installed |
| Visualization centerpiece | Waterfall chart | Most intuitive way to show return decomposition |
| Computation frequency | Daily, cached | Attribution computed once per day, cached for dashboard |

## Dependencies

| Package | Status | Usage |
|---------|--------|-------|
| numpy, pandas, scipy | Already in HRP | Core calculations |
| plotly | Already in HRP | Waterfall charts (`go.Waterfall()`), heatmaps |
| scikit-learn | Already in HRP | Permutation importance |
| shap | **Optional** | SHAP values for tree models (install only if needed) |

## Implementation Tasks

### Phase 1: Factor Attribution Engine (Priority: High)
- [ ] Task 1.1: Create attribution module structure + config (low)
- [ ] Task 1.2: Implement Brinson-Fachler attribution (medium)
- [ ] Task 1.3: Implement regression-based factor decomposition (medium)
- [ ] Task 1.4: Unit tests for factor attribution (medium)

### Phase 2: Feature Importance Tracking (Priority: High)
- [ ] Task 2.1: Implement permutation importance tracker (medium)
- [ ] Task 2.2: Implement SHAP integration — conditional on model type (medium)
- [ ] Task 2.3: Implement rolling importance with sliding window (medium)
- [ ] Task 2.4: Unit tests for feature importance (medium)

### Phase 3: Decision Attribution (Priority: Medium)
- [ ] Task 3.1: Implement trade-level P&L decomposition (medium)
- [ ] Task 3.2: Implement timing and sizing attribution (medium)
- [ ] Task 3.3: Implement rebalancing value-add analysis (medium)
- [ ] Task 3.4: Unit tests for decision attribution (medium)

### Phase 4: Dashboard Visualization (Priority: Medium)
- [ ] Task 4.1: Create page skeleton with sidebar and summary bar (low)
- [ ] Task 4.2: Implement waterfall chart — THE key visualization (medium)
- [ ] Task 4.3: Factor contribution table (low)
- [ ] Task 4.4: Feature importance heatmap (medium)
- [ ] Task 4.5: Decision attribution timeline (medium)
- [ ] Task 4.6: Period comparison view (low)

### Phase 5: Integration
- [ ] Task 5.1: Feature registry integration — attribution as computed features (low)
- [ ] Task 5.2: End-to-end integration test (medium)
- [ ] Task 5.3: Add to dashboard navigation (low)

**Estimated: 2-3 Forge sessions**

## Testing Strategy

### Critical Invariants:
- **Brinson effects sum to active return:** `Σ(allocation + selection + interaction) = portfolio_return - benchmark_return`
- **Factor R-squared reasonable:** 0.7-0.95 for equity portfolios
- **Feature importance scores:** Non-negative and sum to 1.0
- **Trade P&L decomposition:** `timing + sizing + residual = actual_trade_pnl`

### Test Categories:
- Unit tests for each attribution layer independently
- Cross-validation: Brinson results vs regression results should broadly agree
- Edge cases: Single-asset portfolio, no trades, zero benchmark return
- Integration: Full pipeline from returns → attribution → features → dashboard

## Data Dependencies

| Data | Source | Status |
|------|--------|--------|
| Portfolio positions/weights | Existing backtest output | ✅ Available |
| Benchmark weights | **Needs data source** | ⚠️ Question |
| Trade history (entry/exit) | **Needs logging** | ⚠️ Question |
| Feature values | Feature store | ✅ Available |
| Asset returns | Data pipeline | ✅ Available |
| Asset metadata (sector, style) | **Needs classification** | ⚠️ Question |

## Watch Out For

1. **Brinson requires benchmark weights** — Need a benchmark data source (S&P 500 weights, equal-weight, etc.)
2. **SHAP can be slow** — Cache aggressively or use TreeSHAP for tree models
3. **Decision attribution needs trade history** — Verify trades are logged with entry/exit timestamps
4. **Waterfall chart in Plotly:** Use `go.Waterfall()` directly, not workarounds
5. **Attribution must sum:** Always validate the summation invariant
6. **Asset metadata:** Need sector/industry classification for each asset — may need to add to data pipeline

## Open Questions

1. What benchmark should we use? (S&P 500, equal-weight universe, custom?)
2. Is trade history currently logged with entry/exit timestamps?
3. Are ML features stored persistently for historical importance analysis?
4. Should attribution be computed on-the-fly or pre-computed daily?

## Handoff Notes for Forge

**Start with Phase 1 (Factor Attribution)** — it is foundational and self-contained.

**Key sequence:**
1. Create module structure (1.1) → essential scaffolding
2. Brinson-Fachler (1.2) → the core algorithm, well-documented in finance literature
3. Regression-based (1.3) → Fama-French style, can reuse sklearn OLS
4. Tests (1.4) → Critical: verify summation invariant

**Before Phase 3 (Decision Attribution):** Verify that trade history data is available. If trades aren't logged, this phase will need to be deferred or a trade logging mechanism added first.
