# CIO Agent Implementation Plan

**Date:** 2026-01-27
**Status:** Ready for Implementation
**Phase:** 1 - Core Decision Framework + Paper Portfolio
**Author:** CIO Agent Design (based on specification 2026-01-26-cio-agent-design.md)

---

## Overview

The **CIO Agent** is a strategic decision-making agent that sits at the top of the HRP research pipeline. It synthesizes outputs from all research agents to make **CONTINUE/CONDITIONAL/KILL/PIVOT** decisions on hypotheses and manages a **simulated paper trading portfolio**.

**Scope for Phase 1:**
- 4-dimension scoring framework (statistical, risk, economic, cost)
- Decision logic with critical failure detection
- Paper portfolio management (equal-risk weighting, $1M capital)
- Report generation (markdown + email)
- Scheduler integration (weekly reviews)
- Database schema for paper portfolio and decisions

**Deferred to Phase 2:**
- MCP server integration for interactive commands
- Event-driven alert integration
- Adaptive threshold tuning

---

## Architecture

### Position in Agent Pipeline

```
SignalScientist → AlphaResearcher → MLScientist → MLQualitySentinel → ValidationAnalyst
                                                                  ↓
                                                            ┌─────────────┐
                                                            │   CIO Agent  │ ◄─── User (approve/reject)
                                                            └─────────────┘
                                                                  ↓
                                                            Research Decisions
                                                            Paper Portfolio
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `CIOAgent` | `hrp/agents/cio.py` | Main orchestrator |
| `CIOScore` | `hrp/agents/cio.py` | 4-dimension scoring dataclass |
| `CIODecision` | `hrp/agents/cio.py` | Single decision with rationale |
| `CIOReport` | `hrp/agents/cio.py` | Complete weekly report |
| Database Tables | Schema migrations | Paper portfolio + decisions |

### Design Principle

**Advisory Mode:** The CIO Agent never modifies database state directly. All decisions require user approval before execution.

---

## 4-Dimension Scoring Framework

### Scoring Algorithm

```python
total_score = (statistical + risk + economic + cost) / 4
```

Each dimension is scored 0-1, with equal weight (25%).

### Dimension 1: Statistical Quality (25%)

Linear scoring between thresholds:

| Metric | Bad (0 pts) | Target (0.5 pts) | Good (1 pt) |
|--------|-------------|------------------|-------------|
| Walk-forward Sharpe | 0.5 | 1.0 | 1.5 |
| Stability Score | 2.0 | 1.0 | 0.5 |
| Mean IC | 0.01 | 0.03 | 0.05 |
| Fold Stability (CV) | 3.0 | 2.0 | 1.0 |

**Formula:** `score = (value - bad) / (good - bad)`, clamped to [0, 1]

### Dimension 2: Risk Profile (25%)

| Metric | Bad (0 pts) | Target (0.5 pts) | Good (1 pt) | Type |
|--------|-------------|------------------|-------------|------|
| Max Drawdown | 30% | 20% | 10% | Linear |
| Volatility | 25% | 15% | 10% | Linear |
| Regime Stability | <2/3 regimes | - | ≥2/3 regimes | Binary |
| Sharpe Decay | >50% | - | ≤50% | Binary |

### Dimension 3: Economic Rationale (25%)

| Factor | Scoring |
|--------|---------|
| Thesis Strength | Weak→0, Moderate→0.5, Strong→1.0 |
| Regime Alignment | Mismatch→0, Neutral→0.5, Aligned→1.0 |
| Feature Interpretability | >5 black-box→0, 3-5→0.5, <3→1.0 |
| Uniqueness | Duplicate→0, Related→0.5, Novel→1.0 |

**Uses Claude API** to assess thesis strength and regime alignment from agent reports.

### Dimension 4: Cost Realism (25%)

| Factor | Bad (0 pts) | Target (0.5 pts) | Good (1 pt) |
|--------|-------------|------------------|-------------|
| 2x Slippage Survival | Sharpe collapses | Sharpe degrades | Sharpe stable |
| Turnover | 100% | 50% | 20% |
| Capacity Estimate | <$1M | $1-10M | >$10M |
| Execution Complexity | High | Medium | Low |

---

## Decision Logic

### Decision Mapping

| Total Score | Decision | Action |
|-------------|----------|--------|
| ≥ 0.75 | **CONTINUE** | Allocate to paper portfolio |
| 0.50 - 0.74 | **CONDITIONAL** | Re-evaluate next cycle, specific concerns |
| < 0.50 | **KILL** | Archive to model cemetery |
| Critical failure | **PIVOT** | Redirect research effort |

### Critical Failures (Override Score)

- Sharpe decay > 75%
- Target leakage detected (correlation > 0.95)
- Max drawdown > 35%
- Zero profitable regimes

### Default Thresholds

```python
DEFAULT_THRESHOLDS = {
    "min_sharpe": 1.0,
    "max_drawdown": 0.20,
    "sharpe_decay_limit": 0.50,
    "min_ic": 0.03,
    "max_turnover": 0.50,
    "critical_sharpe_decay": 0.75,
    "critical_target_leakage": 0.95,
    "critical_max_drawdown": 0.35,
    "min_profitable_regimes": 2,
}
```

---

## Paper Portfolio Management

### Portfolio Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Capital | $1,000,000 | Hypothetical, standard unit |
| Max Positions | 20 | Diversification |
| Max Weight/Position | 5% | Risk containment |
| Rebalance Frequency | Weekly (Sunday) | Balance responsiveness/costs |
| Min Weight Threshold | 1% | Avoid dust positions |

### Allocation Algorithm

Equal-risk weighting:

```python
position_weight = min(
    target_risk_contribution / position_volatility,
    max_weight_cap  # 5%
)
```

### Portfolio Constraints

| Constraint | Limit |
|------------|-------|
| Gross Exposure | 100% (long-only) |
| Sector Concentration | ≤ 30% single sector |
| Turnover Limit | ≤ 50% annual |
| Max Drawdown Limit | 15% portfolio-level |

### Rebalancing Triggers

1. Scheduled: Weekly review (Sunday 8 PM ET)
2. Drift: Position weight deviates > 20% from target
3. New Strategy: CONTINUE decision adds position
4. Kill Event: Strategy fails, capital redeployed
5. Regime Shift: Market regime change

---

## Database Schema

### Paper Portfolio Tables

```sql
-- Current allocations
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR NOT NULL UNIQUE,
    weight DECIMAL(5, 4),  -- Position weight (0-1)
    entry_price DECIMAL(10, 4),
    entry_date DATE,
    current_price DECIMAL(10, 4),
    unrealized_pnl DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily history
CREATE TABLE IF NOT EXISTS paper_portfolio_history (
    id INTEGER PRIMARY KEY,
    as_of_date DATE,
    nav DECIMAL(12, 2),
    cash DECIMAL(12, 2),
    total_positions INTEGER,
    sharpe_ratio DECIMAL(5, 2),
    max_drawdown DECIMAL(5, 3),
    returns_daily DECIMAL(8, 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trade log
CREATE TABLE IF NOT EXISTS paper_portfolio_trades (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR,
    action VARCHAR,  -- 'ADD', 'REMOVE', 'REBALANCE'
    weight_before DECIMAL(5, 4),
    weight_after DECIMAL(5, 4),
    price DECIMAL(10, 4),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### CIO Decision Tables

```sql
-- Decision records
CREATE TABLE IF NOT EXISTS cio_decisions (
    id INTEGER PRIMARY KEY,
    decision_id VARCHAR UNIQUE,
    report_date DATE,
    hypothesis_id VARCHAR,
    decision VARCHAR,  -- CONTINUE, CONDITIONAL, KILL, PIVOT
    score_total DECIMAL(4, 2),
    score_statistical DECIMAL(4, 2),
    score_risk DECIMAL(4, 2),
    score_economic DECIMAL(4, 2),
    score_cost DECIMAL(4, 2),
    rationale TEXT,
    approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR,
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Killed strategies
CREATE TABLE IF NOT EXISTS model_cemetery (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR UNIQUE,
    killed_date DATE,
    reason TEXT,
    final_score DECIMAL(4, 2),
    experiment_count INTEGER,
    archived_by VARCHAR
);
```

---

## Report Format

### Decision Report Structure

**Location:** `docs/reports/YYYY-MM-DD/HH-MM-cio-decision.md`

```markdown
# CIO Decision Report
**Date:** 2026-01-26
**Reporting Period:** Week 04 (2026-01-20 to 2026-01-26)
**Agent:** CIOAgent (auto-generated)
**Status:** Awaiting Approval

## Executive Summary

| Metric | Value |
|--------|-------|
| Hypotheses Reviewed | 5 |
| CONTINUE | 2 |
| CONDITIONAL | 1 |
| KILL | 1 |
| PIVOT | 1 |
| Paper Portfolio NAV | $1,023,456 (+2.35%) |
| Portfolio Sharpe | 1.24 |

## Market Regime Context

Current: **Bull Market** (Volatility: Low, Trend: Up)
Regime Duration: 45 days
Recommended Bias: Momentum strategies favored

## Research Decisions

### CONTINUE: HYP-2026-001 (Momentum predicts returns)

**Final Score:** 0.82/1.0

| Dimension | Score | Details |
|-----------|-------|---------|
| Statistical | 0.85 | Sharpe: 1.42, Stability: 0.68, IC: 0.042 |
| Risk | 0.78 | MaxDD: 14.2%, Vol: 12.1%, 3/3 regimes profitable |
| Economic | 0.88 | Strong thesis, regime-aligned, interpretable features |
| Cost | 0.75 | Survives 2x slippage, turnover: 38% |

**Evidence:**
- MLflow Run: `abc123` (walk-forward, 5 folds)
- ValidationAnalyst: `docs/research/2026-01-25-validation-analyst.md`

**Paper Portfolio Action:** Allocate 4.2% of capital ($42,000)

## Paper Portfolio Status

### Current Composition
| Ticker | Weight | Entry Date | Unrealized P&L | Strategy |
|--------|--------|------------|----------------|----------|
| AAPL | 4.2% | 2026-01-15 | +3.2% | HYP-2026-001 |

### Performance Attribution (MTD)
| Strategy | Weight | Return | Attribution |
|----------|--------|--------|-------------|
| HYP-2026-001 | 42% | +3.8% | +1.60% |
| **Total** | 100% | **+2.35%** | |

## Next 10 Actions

| Priority | Action | Assigned To | Deadline |
|----------|--------|-------------|----------|
| 1 | Approve/reject CONTINUE decisions | User | 2026-01-27 |
| 2 | Execute approved paper portfolio trades | CIO | Upon approval |
```

### Email Notification

**Trigger:** After report generation

**Template:**
```
Subject: CIO Decision Report - 2 CONTINUE, 1 KILL - Approval Required

Hello,

The CIO Agent has completed its weekly review and is awaiting your approval.

SUMMARY
- 2 strategies recommended for CONTINUE
- 1 strategy recommended for KILL
- 1 strategy requires PIVOT
- Paper portfolio: +2.35% this week (Sharpe: 1.24)

DECISIONS REQUIRING APPROVAL:
1. CONTINUE: HYP-2026-001 (Momentum) - Score: 0.82
2. CONTINUE: HYP-2026-004 (Value) - Score: 0.76

Full report: docs/reports/2026-01-26/09-00-cio-decision.md

To approve: Reply "APPROVE" or use MCP chat
To review: Access via MCP or read the full report

Regards,
CIO Agent
HRP Platform
```

---

## Implementation Tasks

### File Structure

```
hrp/agents/
├── cio.py                    # CIOAgent, CIOScore, CIODecision, CIOReport
├── __init__.py               # Export CIOAgent
└── scheduler.py              # Add setup_weekly_cio_review()

tests/test_agents/
├── test_cio.py               # Unit tests for scoring, decisions
└── test_cio_integration.py   # Integration tests for full workflow

docs/reports/
└── YYYY-MM-DD/
    └── HH-MM-cio-decision.md # Generated reports
```

### Task Checklist

#### 1. Core Agent Implementation
- [ ] Create `hrp/agents/cio.py` with dataclasses
  - [ ] `CIOScore` (statistical, risk, economic, cost, total)
  - [ ] `CIODecision` (hypothesis_id, decision, score, rationale)
  - [ ] `CIOReport` (report_date, decisions, portfolio_state)
  - [ ] `CIOAgent` class extending `SDKAgent`
- [ ] Implement `__init__` with PlatformAPI and thresholds
- [ ] Implement `run_weekly_review()` orchestration method

#### 2. Scoring Implementation
- [ ] `_score_statistical_dimension()` - Extract metrics from MLflow/experiments
- [ ] `_score_risk_dimension()` - MaxDD, volatility, regime stability, Sharpe decay
- [ ] `_score_economic_dimension()` - Claude API assessment of thesis/regime
- [ ] `_score_cost_dimension()` - Slippage, turnover, capacity, complexity
- [ ] `_calculate_total_score()` - Average 4 dimensions, check critical failures
- [ ] `_get_decision_from_score()` - Map to CONTINUE/CONDITIONAL/KILL/PIVOT

#### 3. Paper Portfolio Management
- [ ] `_calculate_position_weights()` - Equal-risk allocation
- [ ] `_check_portfolio_constraints()` - Validate concentration, turnover, drawdown
- [ ] `_update_portfolio_allocations()` - Add/remove/rebalance positions
- [ ] `_track_portfolio_history()` - Daily NAV snapshot
- [ ] `_execute_paper_trade()` - Log simulated trades

#### 4. Report Generation
- [ ] `_generate_markdown_report()` - Create structured markdown report
- [ ] `_send_email_notification()` - Send approval request email
- [ ] `_get_market_regime_context()` - Current regime, duration, bias
- [ ] `_generate_next_actions()` - Prioritized action list

#### 5. Database Operations
- [ ] Create schema migration for 5 new tables
- [ ] `_load_validated_hypotheses()` - Query candidates for review
- [ ] `_fetch_experiment_data()` - Get MLflow metrics, lineage events
- [ ] `_save_decision_to_db()` - Persist decision with approval status
- [ ] `_load_paper_portfolio()` - Get current allocations
- [ ] `_save_paper_portfolio_state()` - Update allocations after approval
- [ ] `_archive_killed_hypothesis()` - Move to model cemetery

#### 6. Scheduler Integration
- [ ] Add `setup_weekly_cio_review()` to `IngestionScheduler`
- [ ] Add `--with-cio-review` CLI flag to `run_scheduler.py`
- [ ] Add `_run_cio_review()` job execution method
- [ ] Update launchd plist for Sunday 8 PM execution

#### 7. Testing
- [ ] Unit tests for each scoring dimension (mocked data)
- [ ] Unit tests for decision logic (boundary cases)
- [ ] Unit tests for portfolio allocation algorithm
- [ ] Integration test for weekly review workflow
- [ ] Integration test for approval flow
- [ ] Test report generation with realistic data

#### 8. Documentation
- [ ] Update CLAUDE.md with CIO Agent usage examples
- [ ] Add CIO Agent to Research Agents table in Project-Status.md
- [ ] Update Feature Registry with F-049 (CIO Agent) → ✅ done

---

## Weekly Review Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Scheduler Trigger (Sunday 8 PM)                              │
│    setup_weekly_cio_review() → _run_cio_review()                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Data Collection                                               │
│    - Query validated hypotheses (status='validated')             │
│    - Fetch experiments (MLflow metrics)                          │
│    - Fetch lineage events (validation_passed/failed)            │
│    - Fetch agent reports (AlphaResearcher, MLScientist, etc.)   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Scoring (4 Dimensions)                                        │
│    - Statistical: Sharpe, stability, IC, fold CV                │
│    - Risk: MaxDD, volatility, regime stability, Sharpe decay    │
│    - Economic: Claude API for thesis strength, regime alignment │
│    - Cost: Slippage, turnover, capacity, complexity             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Decision Generation                                           │
│    - Check critical failures (auto-PIVOT)                        │
│    - Map total score to CONTINUE/CONDITIONAL/KILL               │
│    - For CONTINUE: Calculate paper portfolio allocation          │
│    - For KILL: Prepare model cemetery entry                     │
│    - For PIVOT: Generate redirect direction                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Paper Portfolio Update                                        │
│    - Calculate new allocations (equal-risk weighting)           │
│    - Check portfolio constraints (concentration, turnover)      │
│    - Generate trade list (ADD/REMOVE/REBALANCE)                 │
│    - Update state (pending user approval)                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Report Generation                                             │
│    - Write markdown report to docs/reports/                     │
│    - Send email notification with summary                       │
│    - Persist decisions to cio_decisions table                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Await User Approval                                          │
│    - User reviews report (via MCP or markdown file)             │
│    - User approves/rejects decisions                            │
│    - Approved: Execute paper portfolio trades (simulated)       │
│    - Rejected: Keep status unchanged, log to history            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Dependencies

| Module | Purpose |
|--------|---------|
| `hrp.api.platform.PlatformAPI` | Database access, hypothesis/experiment queries |
| `hrp.agents.base.SDKAgent` | Base class with Claude SDK integration |
| `hrp.notifications.email` | Email notifications for approval requests |
| `anthropic.Anthropic` | Claude API for economic rationale assessment |
| `apscheduler.schedulers.background.BackgroundScheduler` | Weekly scheduling |
| `hrp.data.db` | Direct database access for paper portfolio tables |
| `hrp.research.lineage` | Lineage event queries for agent outputs |

---

## Example Usage

### Running Weekly Review

```python
from hrp.agents.cio import CIOAgent

cio = CIOAgent()
report = cio.run_weekly_review()

print(f"Decisions: {len(report.decisions)}")
print(f"Portfolio NAV: ${report.portfolio_state['nav']:,.0f}")
print(f"Report: {report.report_path}")
```

### Approving Decisions

```python
# Via PlatformAPI
from hrp.api.platform import PlatformAPI

api = PlatformAPI()
api.approve_cio_decisions(
    decision_id="2026-01-26",
    hypotheses=["HYP-2026-001", "HYP-2026-004"]
)
```

### Viewing Paper Portfolio

```python
portfolio = api.get_paper_portfolio()
print(f"Positions: {len(portfolio['positions'])}")
print(f"Total NAV: ${portfolio['nav']:,.0f}")
print(f"Sharpe: {portfolio['sharpe']:.2f}")
```

---

## Success Criteria

- [ ] All validated hypotheses are scored weekly
- [ ] Decision scores are reproducible (same inputs → same scores)
- [ ] Paper portfolio allocations respect all constraints
- [ ] Reports are generated with all required sections
- [ ] Email notifications are sent on schedule
- [ ] User approval flow works end-to-end
- [ ] Unit test coverage > 80%
- [ ] Integration tests pass with realistic data

---

## Next Steps

1. **Create implementation plan** (this document) ✅
2. **Create git worktree** for isolated development
3. **Implement core classes** (CIOAgent, CIOScore, CIODecision, CIOReport)
4. **Implement scoring functions** (4 dimensions)
5. **Implement paper portfolio** (allocation, tracking)
6. **Add database schema** (migrations)
7. **Implement report generation** (markdown + email)
8. **Add scheduler integration** (weekly reviews)
9. **Write tests** (unit + integration)
10. **Update documentation** (CLAUDE.md)

---

**End of Implementation Plan**
