# Design: TASK-006 — Advanced Risk Metrics (VaR & CVaR)

## Overview

Add Value-at-Risk and Conditional VaR calculations to the HRP project. Three methodologies: Parametric (closed-form), Historical Simulation (empirical), and Monte Carlo (simulated). Results feed into the feature registry and a dedicated Streamlit dashboard page.

## Current State (Updated 2026-02-10)

### Implemented (Phase 1 + Feature Registry)
- `hrp/data/risk/var_calculator.py` — VaRCalculator with all 3 methods ✅
- `hrp/data/risk/risk_config.py` — VaRConfig, VaRMethod, Distribution enums ✅
- `hrp/data/risk/__init__.py` — Module exports ✅
- `hrp/data/features/risk_features.py` — Feature registry integration ✅
- `tests/test_risk/test_var_calculator.py` — Core tests ✅
- `tests/test_risk/test_risk_features.py` — Feature tests ✅
- Commit: `333becb` (feat(risk): Implement Phase 1 VaR/CVaR calculator)

### Remaining
- Monte Carlo engine is already integrated into `var_calculator.py` (design originally proposed separate file — Forge combined, which is fine for v1)
- Dashboard page: `hrp/dashboard/pages/risk_metrics.py` — NOT YET BUILT
- Polish, stress testing, backtesting — NOT YET DONE

## Architecture

### New Module: `hrp/data/risk/`
```
hrp/data/risk/
  __init__.py              # Module exports
  var_calculator.py        # VaRCalculator class with 3 methods
  risk_config.py           # VaRConfig, VaRMethod, Distribution enums
```

### Feature Registry Integration
```
hrp/data/features/
  risk_features.py         # 5 registered features: var_95_1d, cvar_95_1d, var_99_1d, mc_var_95_1d, var_95_10d
```

### Dashboard (TODO)
```
hrp/dashboard/pages/
  risk_metrics.py          # Streamlit page with 7 sections
```

## Three VaR Methods

### 1. Parametric (Normal or T-Distribution)
- `VaR = -(μ*T + z*σ*√T)` where z = distribution quantile
- CVaR via PDF at quantile / tail probability
- T-distribution CVaR includes scaling factor `(df + z²) / (df - 1)`

### 2. Historical Simulation
- 1-day: Direct empirical quantile of loss distribution
- Multi-day: Overlapping N-day cumulative returns (NOT sqrt(T) scaling)
- Minimum 30 observations + horizon length

### 3. Monte Carlo
- Fits t-distribution to historical returns via `scipy.stats.t.fit()`
- Simulates `n_simulations` (default 10,000) paths
- Each path: sum of `time_horizon` daily returns
- VaR/CVaR from simulated loss distribution

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Library | Pure numpy/scipy | No external risk libraries needed |
| Result type | VaRResult dataclass | Structured output with metadata |
| MC distribution | T-distribution default | Captures fat tails in financial returns |
| Multi-day historical | Overlapping windows | More accurate than sqrt(T) scaling |
| Scope | Portfolio-level only (v1) | Asset-level VaR is a v2 feature |
| MC integration | Combined in var_calculator.py | Forge decision — simpler than separate file |

## VaR Sign Convention

**Positive values = losses.** VaR of 0.05 means "with 95% confidence, losses won't exceed 5% of portfolio." This is the industry-standard convention. The calculator negates returns internally.

## Registered Features

| Feature Name | Config | Description |
|-------------|--------|-------------|
| `var_95_1d` | 95%, 1-day, parametric | Standard 1-day VaR |
| `cvar_95_1d` | 95%, 1-day, parametric | Expected tail loss |
| `var_99_1d` | 99%, 1-day, parametric | Conservative VaR |
| `mc_var_95_1d` | 95%, 1-day, Monte Carlo | Simulation-based VaR |
| `var_95_10d` | 95%, 10-day, parametric | Regulatory VaR (Basel III) |

## Dashboard Specification (Phase 4 — TODO)

### 7 Sections:
1. **KPI Cards** — VaR/CVaR at 95%/99% with dollar amounts
2. **Return Distribution** — Histogram with VaR/CVaR vertical lines
3. **Rolling VaR Time Series** — VaR over rolling windows
4. **Monte Carlo Fan Chart** — Simulated return paths with confidence bands
5. **Method Comparison Table** — Parametric vs Historical vs MC side-by-side
6. **Stress Test Scenarios** — Historical stress periods (2008, 2020, etc.)
7. **VaR Backtesting** — Count violations (actual losses > VaR) with Kupiec test

### Sidebar Controls:
- Confidence level slider (90-99%)
- Time horizon selector (1, 5, 10, 21 days)
- Method selection (all three + comparison)
- Date range selector
- Portfolio value input

## Implementation Tasks

### Phase 1: Core VaR Calculator ✅ COMPLETE
- [x] Task 1.1: Create risk module structure + config (low)
- [x] Task 1.2: Implement parametric VaR/CVaR (medium)
- [x] Task 1.3: Implement historical simulation VaR (medium)
- [x] Task 1.4: Unit tests for VaR calculator (medium)

### Phase 2: Monte Carlo Engine ✅ COMPLETE (combined into var_calculator.py)
- [x] Task 2.1: Monte Carlo simulation with path generation (medium)
- [x] Task 2.2: T-distribution fitting via scipy.stats.t.fit (medium)
- [x] Task 2.3: calculate_all_methods() comparison function (low)
- [x] Task 2.4: Unit tests for Monte Carlo (medium)
- [x] Task 2.5: Reproducibility with random seed (low) — implicit via numpy

### Phase 3: Feature Registry Integration ✅ COMPLETE
- [x] Task 3.1: Create risk_features.py with 5 registered features
- [x] Task 3.2: Unit tests for risk features
- [x] Task 3.3: Integration with existing feature pipeline

### Phase 4: Dashboard Visualization ❌ TODO
- [ ] Task 4.1: Create page skeleton with sidebar and KPI cards (low)
- [ ] Task 4.2: Return distribution histogram with VaR/CVaR lines (medium)
- [ ] Task 4.3: Rolling VaR time series chart (medium)
- [ ] Task 4.4: Monte Carlo fan chart (medium)
- [ ] Task 4.5: Method comparison table (low)
- [ ] Task 4.6: Stress test scenarios section (medium)
- [ ] Task 4.7: VaR backtesting with violation counting (medium)

### Phase 5: Polish & Integration ❌ TODO
- [ ] Task 5.1: Caching with @st.cache_data for MC simulations (low)
- [ ] Task 5.2: End-to-end integration test (medium)
- [ ] Task 5.3: Add to dashboard navigation sidebar (low)

## Testing Strategy

- **Unit tests:** Each VaR method individually, edge cases (few returns, NaN handling)
- **Invariant tests:** CVaR ≥ VaR always, VaR increases with confidence level
- **Cross-method tests:** All methods should agree within tolerance for normal data
- **Backtesting:** Actual violation rate ≈ expected rate (Kupiec test)

## Watch Out For

1. `scipy.stats.norm.ppf()` sign conventions — lower tail gives negative values
2. VaR sign convention: positive = loss (we negate returns internally)
3. Historical VaR multi-day: overlapping windows, NOT sqrt(T) scaling
4. `@st.cache_data` for MC simulations — key on inputs, not random state
5. Monte Carlo reproducibility: consider setting `np.random.seed()` for consistent results in tests

## Open Questions

- Should VaR backtesting (Kupiec test) be in the calculator or dashboard only?
- Asset-level VaR decomposition for v2?
- Should we expose VaR via the REST API as well?
