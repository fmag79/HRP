# Agent Definition: Quant Developer

**Date:** January 29, 2026
**Status:** Implemented
**Type:** Custom (deterministic - extends `ResearchAgent`)

---

## Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Quant Developer |
| **Actor ID** | `agent:quant-developer` |
| **Type** | Custom (deterministic) |
| **Role** | Backtesting, execution cost modeling, performance simulation |
| **Implementation** | `hrp/agents/quant_developer.py` |
| **Trigger** | Lineage event (after ML Quality Sentinel) + MCP on-demand |
| **Upstream** | Alpha Researcher (strategy specs), ML Scientist (validated models) |
| **Downstream** | Pipeline Orchestrator (coordinates experiments) |

---

## Purpose

Implements production backtests with realistic execution costs. The Quant Developer:

1. **Implements strategy specs** - Translates economic specs into executable backtests
2. **Applies cost modeling** - IBKR-style commission, slippage, market impact
3. **Runs parameter variations** - Tests robustness across configurations
4. **Analyzes cost sensitivity** - Measures performance impact of execution costs
5. **Optimizes performance** - Parallel execution, caching, vectorization
6. **Triggers downstream** Pipeline Orchestrator via lineage events

---

## Core Capabilities

### 1. Production Backtesting Pipeline

```python
from hrp.agents import QuantDeveloper

# Backtest strategy from spec
developer = QuantDeveloper(
    hypothesis_ids=["HYP-2026-001"],
    run_parameter_variations=True,
    run_time_splits=True,
)
result = developer.run()

print(f"Backtests completed: {result['backtests_completed']}")
print(f"Baseline Sharpe: {result['baseline']['sharpe']:.2f}")
print(f"Post-cost Sharpe: {result['post_cost_sharpe']:.2f}")
print(f"Cost sensitivity: {result['cost_sensitivity']}")
# MLflow run logged with full configuration
```

### 2. IBKR-Style Cost Modeling

**Default Cost Structure:**
- **Commission:** 5 basis points (0.05%) per trade
- **Slippage:** 10 basis points (0.10%) per trade
- **Market Impact:** Size-dependent penalty (square root model)

```python
def calculate_total_cost(
    notional: float,
    commission_bps: float = 5.0,
    slippage_bps: float = 10.0,
    position_size_pct: float = 0.05,
) -> float:
    """Calculate total execution cost for a trade."""
    # Commission: fixed % of notional
    commission = notional * (commission_bps / 10000)

    # Slippage: fixed % of notional (execution lag)
    slippage = notional * (slippage_bps / 10000)

    # Market Impact: size-dependent penalty
    # Square root model: impact ∝ sqrt(size / ADV)
    size_ratio = position_size_pct
    market_impact = notional * 0.001 * (size_ratio ** 0.5)

    return commission + slippage + market_impact
```

### 3. Parameter Variations

Tests strategy robustness across configuration space:

```python
# Default parameter grid
lookback_variations = [10, 20, 40]  # days
top_pct_variations = [0.05, 0.10, 0.15]  # top 5%, 10%, 15%
cost_variations = [0.5, 1.0, 1.5, 2.0]  # cost multipliers

# Generates 3 × 3 = 9 baseline variations
# Plus 4 cost sensitivity variations per baseline
```

### 4. Time Period Analysis

**Regime-Based Performance:**
- Bull market periods (e.g., 2019-2021)
- Bear market periods (e.g., 2022)
- High volatility periods (e.g., COVID crash)
- Low volatility periods

### 5. Cost Sensitivity Analysis

**Stress Test:**
| Cost Multiplier | Scenario | Expected Sharpe Impact |
|-----------------|----------|------------------------|
| 0.5x | Best case execution | +10-20% Sharpe |
| 1.0x | Baseline (IBKR model) | Baseline Sharpe |
| 1.5x | Stress test | -10-20% Sharpe |
| 2.0x | Extreme stress | -20-40% Sharpe |

**Rule:** If Sharpe degrades >50% when costs double, strategy is too cost-sensitive.

---

## Configuration

```python
@dataclass
class QuantDeveloperConfig:
    hypothesis_ids: list[str] | None = None  # Specific IDs, or None for all audited

    # Cost model settings (IBKR-style)
    default_cost_model: str = "ibkr"
    commission_bps: float = 5.0  # 5 bps per trade
    slippage_bps: float = 10.0  # 10 bps per trade
    enable_market_impact: bool = True
    market_impact_model: str = "square_root"

    # Parameter variations
    run_parameter_variations: bool = True
    lookback_variations: list[int] = field(default_factory=lambda: [10, 20, 40])
    top_pct_variations: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    cost_variations: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])

    # Time period analysis
    run_time_splits: bool = True

    # Performance settings
    max_parallel_backtests: int = 4  # For Mac M4
    enable_numba: bool = True
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hypothesis_ids` | `None` (all audited) | Specific hypotheses to backtest |
| `default_cost_model` | `ibkr` | Cost model: ibkr, custom |
| `commission_bps` | `5.0` | Commission rate in basis points |
| `slippage_bps` | `10.0` | Slippage rate in basis points |
| `enable_market_impact` | `True` | Apply size-dependent market impact |
| `run_parameter_variations` | `True` | Test parameter robustness |
| `run_time_splits` | `True` | Analyze regime-specific performance |
| `max_parallel_backtests` | `4` | Parallel execution limit |

---

## Outputs

### 1. Backtest Result

```python
@dataclass
class BacktestResult:
    experiment_id: str
    sharpe: float
    returns: float
    max_drawdown: float
    win_rate: float
    trades: int
    turnover: float
    equity_curve: pd.Series
    trade_list: pd.DataFrame
```

### 2. Quant Developer Summary

```python
@dataclass
class QuantDeveloperSummary:
    hypothesis_id: str
    baseline: BacktestResult
    parameter_variations: list[BacktestResult]
    time_splits: dict[str, BacktestResult]
    cost_sensitivity: dict[float, float]  # {cost_multiplier: sharpe}
    infra_metadata: dict
```

### 3. Infrastructure Notes

- Location: `docs/research/YYYY-MM-DD-quant-developer.md`
- Contents: Engineering decisions, performance optimizations, known limitations

### 4. Lineage Events

- `QUANT_DEVELOPER_BACKTEST_COMPLETE`: Per-hypothesis completion event
- `AGENT_RUN_COMPLETE`: Triggers Pipeline Orchestrator

---

## Trigger Model

### Primary: Lineage Event Trigger

```python
# Quant Developer listens for ML Quality Sentinel completion
scheduler.register_lineage_trigger(
    event_type="ML_QUALITY_SENTINEL_AUDIT",
    actor_filter="agent:ml-quality-sentinel",
    callback=trigger_quant_developer,
)
```

### Secondary: MCP On-Demand

```python
# MCP tool: run_quant_developer
result = run_quant_developer(hypothesis_id="HYP-2026-001")
```

---

## Pre-Backtest Review (NEW)

### Purpose

Lightweight execution feasibility sanity check before expensive backtests.

### When It Runs

Between ML Quality Sentinel and full Production Backtesting.

### Checks

| Check | Description | Action |
|-------|-------------|--------|
| Data Availability | Required features exist | Warning if missing |
| Point-in-Time Validity | Features computable as of dates | Warning if violated |
| Execution Frequency | Rebalance cadence achievable | Warning if unrealistic |
| Universe Liquidity | Sufficient liquidity | Warning if illiquid |
| Cost Model | Can handle IBKR costs | Warning if dominant |

### Output

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

## Implementation Details

### Look-Ahead Bias Prevention

```python
# Always apply purge/embargo periods
config = WalkForwardConfig(
    purge_days=5,   # Execution lag
    embargo_days=10,  # Implementation delay
)
```

### Point-in-Time Data

```python
# Use get_features() with as_of_date parameter
features = api.get_features(
    symbols=['AAPL'],
    features=['momentum_20d'],
    as_of_date='2023-01-15',  # Prevents future leakage
)
```

### Reproducibility

```python
# Set random seed for determinism
np.random.seed(42)
results = backtest(strategy, data, seed=42)
```

---

## Performance Benchmarks

| Operation | Target | Notes |
|-----------|--------|-------|
| Single backtest (5y, 500 symbols) | < 30s | With realistic costs |
| Parameter sweep (25 combinations) | < 5 min | Parallel execution |
| Cost sensitivity (4 variations) | < 1 min | Sequential |
| Memory usage | < 4GB | Typical workload |

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Strategy Specs** | Reads from `docs/strategies/` or hypothesis metadata |
| **VectorBT** | Backtest engine integration |
| **Platform API** | Data access via `get_prices()`, `get_features()` |
| **MLflow** | Logs all backtest runs with full configuration |
| **Lineage** | Logs events, triggers downstream |
| **Pipeline Orchestrator** | Downstream - coordinates parallel experiments |

---

## Explicit Non-Responsibilities

The Quant Developer does NOT:

- ❌ Invent strategies (Alpha Researcher's job)
- ❌ Select features (ML Scientist's job)
- ❌ Interpret performance results (Validation Analyst's job)
- ❌ Approve deployment (CIO Agent's job)
- ❌ Generate economic rationale (Alpha Researcher's job)
- ❌ Train ML models (ML Scientist's job)

---

## Document History

- **2026-01-29:** Initial agent definition created
