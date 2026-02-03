# CIO Agent Specification

> **Agent Type:** Decision Support (T1 - Core Research Infrastructure)
> **Status:** Implemented (Phase 1 - 2026-01-27)
> **Version:** 1.0.0

---

## Overview

The **CIO (Chief Investment Officer) Agent** is a strategic decision-making agent that evaluates validated hypotheses and manages a simulated paper trading portfolio. It serves as the final decision gate before real capital allocation, ensuring only robust strategies advance to live trading.

### Core Responsibilities

1. **Hypothesis Evaluation** - Score hypotheses across 4 dimensions and recommend CONTINUE/CONDITIONAL/KILL/PIVOT decisions
2. **Paper Portfolio Management** - Equal-risk allocation with constraint checking
3. **Advisory Mode** - All decisions require user approval; agent never auto-executes trades

### Identity

| Attribute | Value |
|-----------|-------|
| **Name** | CIO Agent (Chief Investment Officer) |
| **Actor ID** | `agent:cio` |
| **Type** | SDK (Claude-powered decision support) |
| **Role** | Strategic hypothesis scoring, deployment decisions, paper portfolio management |
| **Trigger** | Weekly scheduled (Friday 5 PM ET) + MCP on-demand |
| **Upstream** | Risk Manager (provides risk-assessed hypotheses), Validation Analyst (stress-tested hypotheses) |
| **Downstream** | User/CIO (final human approval for paper trading deployment) |

---

## Decision Framework

### 4-Dimension Scoring

| Dimension | Metrics | Scoring Type | Weight |
|-----------|---------|--------------|--------|
| **Statistical** | Sharpe, stability, IC, fold CV | Linear interpolation | 25% |
| **Risk** | Max DD, volatility, regime stability, Sharpe decay | Linear + Binary | 25% |
| **Economic** | Thesis strength, regime alignment, interpretability, uniqueness | Ordinal + Linear | 25% |
| **Cost** | Slippage survival, turnover, capacity, execution complexity | Ordinal | 25% |

### Decision Thresholds

| Total Score | Decision | Criteria |
|-------------|----------|----------|
| ≥ 0.75 | **CONTINUE** | Strong across all dimensions |
| 0.50 - 0.74 | **CONDITIONAL** | Requires mitigation before proceeding |
| < 0.50 | **KILL** | Insufficient quality |
| *Any* | **PIVOT** | Critical failure detected (overrides score) |

### Critical Failures (Triggers PIVOT)

- Max drawdown > 35%
- Sharpe decay > 75%
- Target leakage correlation > 95%

---

## Scoring Details

### Statistical Dimension

| Metric | Bad | Target | Good | Scoring |
|--------|-----|--------|------|---------|
| Sharpe | 0.5 | 1.0 | 1.5 | Linear: (value - 0.5) / (1.5 - 0.5) |
| Stability | 2.0 | 1.0 | 0.5 | Piecewise linear: 2.0→0, 1.0→0.5, 0.5→1.0 |
| IC | 0.01 | 0.03 | 0.05 | Linear: (value - 0.01) / (0.05 - 0.01) |
| Fold CV | 3.0 | 2.0 | 1.0 | Linear: (3.0 - value) / (3.0 - 1.0) |

### Risk Dimension

| Metric | Bad | Target | Good | Scoring |
|--------|-----|--------|------|---------|
| Max DD | 30% | 20% | 10% | Piecewise linear |
| Volatility | 25% | 15% | 10% | Piecewise linear |
| Regime Stable | - | - | - | Binary: 1.0 if ≥2/3 regimes profitable |
| Sharpe Decay | - | - | - | Binary: 1.0 if ≤50% |

### Economic Dimension

| Metric | Levels | Scoring |
|--------|--------|---------|
| Thesis Strength | weak, moderate, strong | Ordinal: 0.0, 0.5, 1.0 |
| Regime Alignment | mismatch, neutral, aligned | Ordinal: 0.0, 0.5, 1.0 |
| Feature Interpretability | >5, 3-5, <3 black-box features | Linear: >5→0, 3-5→0.5, <3→1.0 |
| Uniqueness | duplicate, related, novel | Ordinal: 0.0, 0.5, 1.0 |

**Claude API Integration:** Thesis strength and regime alignment assessed via Claude API with fallback to moderate/neutral if unavailable.

### Cost Dimension

| Metric | Levels | Scoring |
|--------|--------|---------|
| Slippage Survival | collapse, degraded, stable | Ordinal: 0.0, 0.5, 1.0 |
| Turnover | 100%, 50%, 20% | Piecewise linear |
| Capacity | low (<$1M), medium ($1-10M), high (>$10M) | Ordinal: 0.0, 0.5, 1.0 |
| Execution Complexity | high, medium, low | Ordinal: 0.0, 0.5, 1.0 |

---

## Portfolio Management

### Equal-Risk Allocation

```
weight = target_risk_contribution / volatility
```

- **Target Risk Contribution:** 3% (default, configurable)
- **Max Weight Cap:** 5% per position
- **Proportional Scaling:** When all positions exceed cap, scale proportionally

### Portfolio Constraints

| Constraint | Limit | Violation Action |
|-------------|-------|-------------------|
| Gross Exposure | 100% | Flag violation |
| Sector Concentration | 30% | Flag violation |
| Annual Turnover | 50% | Flag violation |
| Max Drawdown | 15% | Flag violation |

### Paper Portfolio Parameters

- **Capital:** $1,000,000 (simulated)
- **Max Positions:** 20
- **Min Weight Threshold:** 1%

---

## Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `paper_portfolio` | Current allocations |
| `paper_portfolio_trades` | Simulated trade log |
| `cio_decisions` | Decision records with approval status |
| `model_cemetery` | Killed strategies archive |

---

## API Reference

### Class: `CIOAgent`

**Parent:** `SDKAgent`

**Attributes:**
- `agent_name` = "cio"
- `agent_version` = "1.0.0"
- `DEFAULT_THRESHOLDS` (dict): Default scoring thresholds

#### Methods

##### `score_hypothesis()`

```python
def score_hypothesis(
    hypothesis_id: str,
    experiment_data: dict,
    risk_data: dict,
    economic_data: dict,
    cost_data: dict,
) -> CIOScore
```

Score a hypothesis across all 4 dimensions.

**Parameters:**
- `experiment_data`: Statistical metrics from MLflow (sharpe, stability_score, mean_ic, fold_cv)
- `risk_data`: Risk metrics (max_drawdown, volatility, regime_stable, sharpe_decay)
- `economic_data`: Economic rationale (thesis, current_regime, black_box_count, uniqueness, agent_reports)
- `cost_data`: Cost realism (slippage_survival, turnover, capacity, execution_complexity)

**Returns:** `CIOScore` with all dimension scores and computed decision

---

### Dataclasses

#### `CIOScore`

```python
@dataclass
class CIOScore:
    hypothesis_id: str
    statistical: float  # 0-1
    risk: float  # 0-1
    economic: float  # 0-1
    cost: float  # 0-1
    critical_failure: bool = False

    @property
    def total(self) -> float: ...  # Average of 4 dimensions

    @property
    def decision(self) -> Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]: ...
```

#### `CIODecision`

```python
@dataclass
class CIODecision:
    hypothesis_id: str
    decision: Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]
    score: CIOScore
    rationale: str
    evidence: dict
    paper_allocation: Optional[float] = None  # For CONTINUE decisions
    pivot_direction: Optional[str] = None  # For PIVOT decisions
```

#### `CIOReport`

```python
@dataclass
class CIOReport:
    report_date: date
    decisions: list[CIODecision]
    portfolio_state: dict  # {nav, cash, positions_count, ...}
    market_regime: str
    next_actions: list[dict]  # {priority, action, deadline}
    report_path: str
```

---

## Usage Examples

### Basic Hypothesis Scoring

```python
from hrp.agents import CIOAgent
from unittest.mock import patch

with patch("hrp.agents.cio.PlatformAPI"):
    agent = CIOAgent(job_id="cio-001", actor="agent:cio")

score = agent.score_hypothesis(
        hypothesis_id="HYP-2026-001",
        experiment_data={"sharpe": 1.5, "stability_score": 0.6, "mean_ic": 0.045, "fold_cv": 1.2},
        risk_data={"max_drawdown": 0.12, "volatility": 0.11, "regime_stable": True, "sharpe_decay": 0.30},
        economic_data={"thesis": "Strong momentum", "current_regime": "Bull Market", "black_box_count": 2, "uniqueness": "novel"},
        cost_data={"slippage_survival": "stable", "turnover": 0.25, "capacity": "high", "execution_complexity": "low"},
    )

print(f"Decision: {score.decision}")  # CONTINUE
print(f"Score: {score.total:.2f}")   # 0.85
```

### Custom Thresholds

```python
custom_thresholds = {
    "min_sharpe": 1.5,  # Stricter Sharpe requirement
    "max_drawdown": 0.15,  # Tighter drawdown limit
}

agent = CIOAgent(
    job_id="cio-002",
    actor="agent:cio",
    thresholds=custom_thresholds,
)
```

### Portfolio Allocation

```python
strategies = [
    {"hypothesis_id": "HYP-001", "volatility": 0.10},
    {"hypothesis_id": "HYP-002", "volatility": 0.15},
]

weights = agent._calculate_position_weights(
    strategies=strategies,
    target_risk_contribution=0.03,
    max_weight_cap=0.05,
)
# Returns: {"HYP-001": 0.05, "HYP-002": 0.03} (capped proportionally)
```

---

## Dependencies

### Internal
- `hrp.agents.sdk_agent.SDKAgent` - Base class for Claude-powered agents
- `hrp.api.platform.PlatformAPI` - Database and platform access
- `hrp.data.db.DatabaseManager` - Direct database operations

### External
- `anthropic` - Claude API for economic assessment (optional, with fallback)

---

## Testing

### Test Files

| File | Purpose | Test Count |
|------|---------|------------|
| `tests/test_agents/test_cio_dataclasses.py` | Dataclass tests | 9 |
| `tests/test_agents/test_cio_agent.py` | Initialization tests | 3 |
| `tests/test_agents/test_cio_scoring.py` | Dimension scoring tests | 29 |
| `tests/test_agents/test_cio_portfolio.py` | Portfolio allocation tests | 8 |
| `tests/test_agents/test_cio_exports.py` | Package export tests | 3 |
| `tests/test_data/test_cio_schema.py` | Database schema tests | 6 |

### Test Coverage

- **Total Tests:** 58 CIO Agent tests
- **Pass Rate:** 100% (58/58 passing)
- **Code Coverage:** ~66% (hrp/agents/cio.py)

---

## Design Decisions

### Why 4 Equal Dimensions?

Each dimension captures a critical aspect of trading strategy viability:

1. **Statistical** - Does the backtest hold up under scrutiny?
2. **Risk** - Can we survive drawdowns and regime changes?
3. **Economic** - Is there a sound rationale for why this works?
4. **Cost** - Can this scale to real capital without destroying returns?

Equal weighting prevents any single dimension from dominating decisions.

### Why Advisory Mode?

- Prevents automated capital allocation without human oversight
- Allows CIO Agent to surface recommendations while user retains authority
- Aligns with project principle: "Agents cannot approve deployments"

### Why Equal-Risk Weighting?

- Traditional equal-weighting ignores volatility differences
- Equal-risk ensures each position contributes similar risk to portfolio
- Lower volatility strategies get larger allocations (more capital efficient)

### Why Piecewise Linear Scoring?

- Some metrics have uniform progression (Sharpe: 0.5→1.0→1.5 maps to 0→0.5→1.0)
- Others have non-uniform progression (stability: 2.0→1.0→0.5 maps to 0→0.5→1.0)
- Piecewise linear handles both cases correctly

---

## Future Enhancements (Phase 2)

### Deferred Features

- **Report Generation:** Weekly markdown reports with decision summaries
- **Scheduler Integration:** Automated weekly reviews via IngestionScheduler
- **Full Workflow:** `run_weekly_review()` orchestration method
- **MCP Server Integration:** CIO decisions accessible via MCP protocol
- **Adaptive Threshold Tuning:** Dynamic threshold adjustment based on market conditions

### Planned Improvements

- **Portfolio Rebalancing:** Automated rebalance signals when drift exceeds tolerance
- **Transaction Cost Modeling:** More sophisticated slippage and market impact estimation
- **Regime Detection Integration:** HMM-based regime detection for economic alignment
- **Multi-Period Optimization:** Evaluate decisions across different market regimes

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-27 | Initial implementation - Phase 1 complete |

---

## Related Documentation

- **Design:** `docs/plans/2026-01-26-cio-agent-design.md` - Original design document
- **TDD Plan:** `docs/plans/2026-01-27-cio-agent-tdd.md` - Test-driven implementation plan
- **Implementation:** `docs/plans/2026-01-27-cio-agent-implementation.md` - Implementation summary

---

**Last Updated:** 2026-01-27
**Maintained By:** CIO Agent Team
