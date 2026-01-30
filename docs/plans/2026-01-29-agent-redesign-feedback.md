# Agent Redesign: Feedback Implementation

**Date:** January 29, 2026
**Status:** Design Complete
**Purpose:** Address feedback on agent pipeline design

---

## Executive Summary

This document implements five key improvements to the HRP research agent pipeline:

1. **Split Alpha Researcher** → Alpha Researcher (specs) + Code Materializer (code)
2. **Adaptive IC thresholds** by strategy class (cross-sectional, time-series, ML)
3. **Formal Stability Score** definition (`stability_score_v1`)
4. **Pre-Backtest Review** stage (Quant Developer early check, warnings only)
5. **HMM-based structural regimes** for scenario analysis

**Impact:** 10 agents → 11 agents + 1 review stage. Cleaner separation of concerns, more realistic thresholds, better execution feasibility checks.

---

## Section 1: Updated Agent Pipeline

### Current Pipeline (10 agents)

```
Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel
→ Quant Developer → Pipeline Orchestrator → Validation Analyst
→ Risk Manager → CIO Agent → Human CIO
```

### Updated Pipeline (11 agents + 1 review stage)

```
Signal Scientist
    ↓
Alpha Researcher (strategy specs only, no code)
    ↓
Code Materializer (spec → executable code, no logic changes) ← NEW AGENT
    ↓
ML Scientist
    ↓
ML Quality Sentinel
    ↓
Quant Developer Pre-Backtest Review (warnings only) ← NEW STAGE
    ↓
Quant Developer (full production backtests)
    ↓
Pipeline Orchestrator
    ↓
Validation Analyst
    ↓
Risk Manager
    ↓
CIO Agent
    ↓
Human CIO
```

---

## Section 2: New Agent - Code Materializer

### Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Code Materializer |
| **Actor ID** | `agent:code-materializer` |
| **Type** | Custom (deterministic - extends `ResearchAgent`) |
| **Role** | Translates strategy specifications into executable code |
| **Trigger** | Lineage event (after Alpha Researcher) |
| **Upstream** | Alpha Researcher (strategy specs) |
| **Downstream** | ML Scientist |

### Purpose

The Code Materializer performs **mechanical translation** of strategy specifications into executable code.

### Core Responsibilities

1. **Reads strategy specs** from Alpha Researcher
2. **Generates boilerplate code** for strategy execution
3. **Validates syntactic correctness** (code runs, no syntax errors)
4. **Enforces implementation constraints** (no lookahead, point-in-time data)

### Key Constraint

**No Logic Changes:** Code Materializer cannot interpret, modify, or "optimize" strategy logic. It translates specs as written.

### Translation Examples

| Input (from Alpha Researcher) | Output (Code Materializer) |
|------------------------------|---------------------------|
| "Long top decile of momentum_20d" | `def generate_signals(prices): return momentum_20d.rank(pct=True) >= 0.9` |
| "Max 10% sector exposure" | `SectorConstraint(max_exposure=0.10)` |
| "Weekly rebalance on Mondays" | `RebalanceSchedule(frequency='weekly', day_of_week=0)` |

### What It Does NOT Do

- ❌ Interpret economic rationale
- ❌ Optimize signal logic
- ❌ Choose features
- ❌ Judge performance
- ❌ Modify strategy parameters

---

## Section 3: Updated Decision Gates & Metrics

### 3.1 Adaptive IC Thresholds

Signal Scientist and ML Quality Sentinel will now apply **strategy-class-aware IC thresholds**:

| Strategy Type | IC Pass | IC Kill | Application |
|---------------|---------|---------|-------------|
| **Cross-sectional factor** | ≥ 0.015 | < 0.005 | Value, quality, low-vol factors |
| **Time-series momentum** | ≥ 0.02 | < 0.01 | Trend-following strategies |
| **ML composite** | ≥ 0.025 | < 0.01 | Multi-feature ML models |

**Implementation:** Alpha Researcher must tag each hypothesis with `strategy_class` during review.

### 3.2 Stability Score - Formal Definition

**Location:** `hrp/research/metrics.py`

**Version:** `stability_score_v1`

**Formula:**

```python
def calculate_stability_score_v1(
    fold_sharpes: list[float],
    fold_drawdowns: list[float],
    mean_fold_ic: float,
) -> float:
    """
    Stability Score v1 - Lower is better.

    Components:
    1. Sharpe coefficient of variation (CV)
    2. Drawdown dispersion penalty
    3. Sign flip penalty

    Returns:
        float: Stability score (≤ 1.0 is stable)
    """
    import numpy as np

    # Component 1: Sharpe CV
    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes)
    sharpe_cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float('inf')

    # Component 2: Drawdown dispersion
    mean_dd = np.mean(fold_drawdowns)
    std_dd = np.std(fold_drawdowns)
    dd_dispersion = (std_dd / mean_dd) if mean_dd > 0 else 0

    # Component 3: Sign flip penalty
    positive_ic = sum(1 for ic in [mean_fold_ic] if ic > 0)
    negative_ic = sum(1 for ic in [mean_fold_ic] if ic < 0)
    sign_flip_penalty = 0.5 if positive_ic > 0 and negative_ic > 0 else 0

    stability_score = sharpe_cv + dd_dispersion + sign_flip_penalty

    return stability_score
```

**Threshold:** `stability_score_v1 ≤ 1.0` → Stable

---

## Section 4: New Stage - Quant Developer Pre-Backtest Review

### Purpose

A lightweight **execution feasibility sanity check** that runs between ML Quality Sentinel and full Production Backtesting.

### When It Runs

```
ML Quality Sentinel (passes)
    ↓
Quant Developer Pre-Backtest Review (warnings only) ← NEW
    ↓
Quant Developer Production Backtesting (full run)
```

### What It Checks

| Check | Description | Action |
|-------|-------------|--------|
| **Data Availability** | Required features exist in database | Warning if missing |
| **Point-in-Time Validity** | Features can be computed as of required dates | Warning if violated |
| **Execution Frequency** | Rebalance cadence is achievable | Warning if unrealistic |
| **Universe Liquidity** | Target symbols have sufficient liquidity | Warning if illiquid |
| **Cost Model Applicability** | Strategy can handle IBKR-style costs | Warning if costs would dominate |

### Key Characteristic

**Warnings Only:** This stage cannot block or veto. It generates warnings logged to lineage and visible in CIO Agent reports.

### Example Warnings

```python
# Warning 1: Insufficient history
"WARNING: momentum_252d requires 252 days of history.
 AAPL IPO was 1980-12-12, backtest start is 1981-01-01.
 Minimum viable date: 1981-12-15."

# Warning 2: Illiquid universe
"WARNING: 15/100 symbols in universe have avg daily volume < $1M.
 Consider micro-cap liquidity filter."

# Warning 3: Cost sensitivity
"WARNING: High-turnover strategy (estimated 180% annual).
 At 15 bps total cost, estimated net drag = 27 bps/year."
```

### Output Format

```python
@dataclass
class PreBacktestReviewResult:
    hypothesis_id: str
    passed: bool  # Always True (warnings only)
    warnings: list[str]
    data_issues: list[str]
    execution_notes: list[str]
    reviewed_at: datetime
```

---

## Section 5: Structural Regimes for Scenario Analysis

### Approach

**Data-driven HMM regime detection exclusively** — no hardcoded crisis dates.

### HMM Configuration

**Two separate HMM models:**

| HMM Model | Purpose | States | Features |
|-----------|---------|--------|----------|
| **Volatility HMM** | Classify vol regimes | 2 (Low, High) | `volatility_20d` |
| **Trend HMM** | Classify market behavior | 2 (Bull, Bear) | `returns_20d` |

**Combined Regime Matrix:**

| Vol \ Trend | Bull | Bear |
|-------------|------|------|
| **Low** | Low Vol Bull | Low Vol Bear |
| **High** | High Vol Bull | High Vol Bear (Crisis) |

### Pipeline Orchestrator Integration

**Current:** Requires ≥ 3 scenarios, checks Sharpe CV ≤ 0.30

**Updated:** Structural regime scenarios become **mandatory scenario buckets**

| Requirement | Specification |
|-------------|---------------|
| **Minimum scenarios** | 4 (one per regime matrix cell) |
| **Sharpe CV threshold** | ≤ 0.30 across all regimes |
| **Regime coverage** | Must test in all 4 regime types |

### HMM Training Example

```python
from hrp.ml import HMMConfig, RegimeDetector

# Volatility HMM
vol_config = HMMConfig(
    n_regimes=2,
    features=['volatility_20d'],
    covariance_type='full',
)

# Trend HMM
trend_config = HMMConfig(
    n_regimes=2,
    features=['returns_20d'],
    covariance_type='full',
)

# Fit on market index (SPY)
spy_prices = get_prices(['SPY'], start='2010-01-01', end='2023-12-31')

vol_detector = RegimeDetector(vol_config)
vol_detector.fit(spy_prices)

trend_detector = RegimeDetector(trend_config)
trend_detector.fit(spy_prices)

# Classify historical periods into regimes
regime_labels = combine_regime_labels(
    vol_detector.predict(spy_prices),
    trend_detector.predict(spy_prices),
)
# Returns: "low_vol_bull", "high_vol_bear", etc.
```

### Example Scenario Set

| Scenario | Regime Type | Date Range | Market Context |
|----------|-------------|------------|----------------|
| 1 | Low Vol Bull | 2017-01 to 2017-12 | Steady uptrend, calm |
| 2 | Low Vol Bear | 2015-06 to 2016-01 | Slow grinding decline |
| 3 | High Vol Bull | 2020-04 to 2020-12 | Recovery from COVID crash |
| 4 | High Vol Bear | 2008-09 to 2008-12 | Financial crisis |

---

## Section 6: Implementation Priority

### Phase 1: High Impact (Do First)

| Change | Complexity | Impact | Files |
|--------|------------|--------|-------|
| **Code Materializer agent** | Medium | High | New: agent doc, implementation |
| **Alpha Researcher split** | Low | High | Update: agent doc |
| **Adaptive IC thresholds** | Medium | High | Update: Signal Scientist, ML Quality Sentinel |

### Phase 2: Foundation (Do Second)

| Change | Complexity | Impact | Files |
|--------|------------|--------|-------|
| **Stability Score v1** | Low | Medium | New: `hrp/research/metrics.py` |
| **Pre-Backtest Review** | Medium | Medium | Update: Quant Developer |
| **Strategy classification** | Low | Medium | Update: Alpha Researcher, schema |

### Phase 3: Enhancement (Do Third)

| Change | Complexity | Impact | Files |
|--------|------------|--------|-------|
| **HMM structural regimes** | High | Medium | Update: Pipeline Orchestrator |
| **Scenario generation** | Medium | Medium | Update: Pipeline Orchestrator |

### Dependencies

```
Phase 1 (Code Materializer)
    ↓
Phase 2 (Stability Score, Pre-Backtest)
    ↓
Phase 3 (HMM Regimes)
```

---

## Section 7: Complete Agent List (11 agents)

| # | Agent | Actor ID | Type | Trigger |
|---|-------|----------|------|---------|
| 1 | Signal Scientist | `agent:signal-scientist` | SDK | Weekly (Mon 7 PM) |
| 2 | Alpha Researcher | `agent:alpha-researcher` | SDK | Event: Signal Scientist |
| 3 | **Code Materializer** | `agent:code-materializer` | Custom | Event: Alpha Researcher |
| 4 | ML Scientist | `agent:ml-scientist` | SDK | Event: Code Materializer |
| 5 | ML Quality Sentinel | `agent:ml-quality-sentinel` | Custom | Event: ML Scientist |
| 6 | Quant Developer (Pre-Review) | `agent:quant-developer` | Custom | Event: ML Quality Sentinel |
| 7 | Quant Developer (Backtest) | `agent:quant-developer` | Custom | Continuation |
| 8 | Pipeline Orchestrator | `agent:pipeline-orchestrator` | SDK | Event: Quant Developer |
| 9 | Validation Analyst | `agent:validation-analyst` | SDK | Event: Pipeline Orchestrator |
| 10 | Risk Manager | `agent:risk-manager` | Custom | Event: Validation Analyst |
| 11 | CIO Agent | `agent:cio` | Custom | Weekly (Fri 5 PM) |
| - | Human CIO | `user` | Human | Event: CIO Agent |

---

## Section 8: Key Takeaways

### What Changed

| Aspect | Before | After |
|--------|--------|-------|
| **Total agents** | 10 | 11 |
| **Alpha Researcher** | Specs + code | Specs only |
| **Code generation** | Alpha Researcher | Code Materializer (new) |
| **IC thresholds** | Uniform (0.03) | Adaptive by strategy class |
| **Stability Score** | Implicit | Formal, versioned definition |
| **Quant Developer** | Production backtests only | Pre-review + backtests |
| **Scenario analysis** | 3+ scenarios | 4 structural regimes (HMM) |

### What Stayed the Same

- Event-driven architecture via lineage
- Kill gates at Pipeline Orchestrator
- Human CIO final approval
- Risk Manager veto power
- MLflow experiment tracking

---

## Document History

- **2026-01-29:** Initial design created based on feedback review
