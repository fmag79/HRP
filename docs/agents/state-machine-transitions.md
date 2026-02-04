# Agent State Machine Transitions

> Complete documentation of hypothesis states, pipeline stages, and agent responsibilities in the HRP decision pipeline.

## Overview

The HRP agent pipeline processes hypotheses through multiple stages, with each agent having specific responsibilities for state transitions and event emissions. This document provides an end-to-end view of all state machines.

---

## 1. Hypothesis Status States

### Valid Status Values

| Status | Description |
|--------|-------------|
| `draft` | Initial state after hypothesis creation |
| `testing` | Moved to ML validation |
| `validated` | Passed ML/quality checks |
| `rejected` | Failed validation |
| `deployed` | Live in production |
| `deleted` | Soft-deleted |

### State Transition Diagram

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
┌─────────┐    ┌─────────┐    ┌───────────┐    ┌──────────┐  │
│  draft  │───▶│ testing │───▶│ validated │───▶│ deployed │  │
└─────────┘    └─────────┘    └───────────┘    └──────────┘  │
     │              │              │                 │        │
     │              │              │                 │        │
     │              ▼              ▼                 ▼        │
     │         ┌─────────┐   ┌─────────┐       ┌─────────┐   │
     │         │rejected │   │rejected │       │validated│   │
     │         └─────────┘   └─────────┘       └─────────┘   │
     │                                                       │
     └───────────────────▶ deleted ◀─────────────────────────┘
```

### Transition Rules

| From | To | Agent | Condition | Event |
|------|-----|-------|-----------|-------|
| `draft` | `testing` | Alpha Researcher | Economic rationale approved | `ALPHA_RESEARCHER_REVIEW` |
| `testing` | `validated` | ML Scientist | IC ≥ 0.03 AND Stability ≤ 1.0 | `ML_SCIENTIST_VALIDATION` |
| `testing` | `rejected` | ML Scientist | IC < 0.03 OR Stability > 1.5 | `EXPERIMENT_COMPLETED` |
| `validated` | `deployed` | Human CIO | Final approval | `DEPLOYMENT_APPROVED` |
| `validated` | `rejected` | Risk Manager | Risk veto issued | `RISK_VETO` |
| `deployed` | `validated` | System/Manual | Undeploy | - |
| `*` | `deleted` | User/System | Manual deletion | `HYPOTHESIS_DELETED` |

---

## 2. Pipeline Stages

Pipeline stages track hypothesis position independently of status.

### Stage Progression

```
1. created           ─▶ Hypothesis just created
2. signal_discovery  ─▶ Signal Scientist analyzing
3. alpha_review      ─▶ Alpha Researcher reviewing
4. ml_training       ─▶ ML Scientist validating
5. quality_audit     ─▶ ML Quality Sentinel auditing
6. quant_backtest    ─▶ Quant Developer backtesting
7. kill_gate         ─▶ Kill Gate Enforcer checking
8. stress_test       ─▶ Validation Analyst testing
9. risk_review       ─▶ Risk Manager assessing
10. cio_review       ─▶ CIO Agent scoring
11. human_approval   ─▶ Awaiting human decision
12. deployed         ─▶ Live in production
13. archived         ─▶ Rejected/deleted (terminal)
```

---

## 3. Agent Responsibilities

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ENTRY TIER                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Signal Scientist                                                        │
│  └─ Creates draft hypotheses from IC analysis                           │
│  └─ Emits: SIGNAL_SCAN_COMPLETE                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ECONOMIC REVIEW                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Alpha Researcher                                                        │
│  └─ Promotes draft → testing                                            │
│  └─ Generates strategy specifications                                    │
│  └─ Emits: ALPHA_RESEARCHER_REVIEW, ALPHA_RESEARCHER_COMPLETE           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML VALIDATION                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ML Scientist                                                            │
│  └─ Runs walk-forward validation (5+ folds)                             │
│  └─ Promotes testing → validated (or rejects)                           │
│  └─ Emits: ML_SCIENTIST_VALIDATION, EXPERIMENT_COMPLETED                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       QUALITY ASSURANCE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ML Quality Sentinel                                                     │
│  └─ Audits for overfitting, leakage, instability                        │
│  └─ Flags but does NOT change status                                    │
│  └─ Emits: ML_QUALITY_SENTINEL_AUDIT, HYPOTHESIS_FLAGGED               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION BACKTEST                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Quant Developer                                                         │
│  └─ Generates production backtests + parameter variations               │
│  └─ Enriches metadata (does NOT change status)                          │
│  └─ Emits: QUANT_DEVELOPER_BACKTEST_COMPLETE                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      KILL GATE (Optional)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Kill Gate Enforcer                                                      │
│  └─ Applies early termination gates                                     │
│  └─ Blocks unpromising hypotheses                                       │
│  └─ Emits: KILL_GATE_ENFORCER_COMPLETE, KILL_GATE_TRIGGERED            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       STRESS TESTING                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Validation Analyst                                                      │
│  └─ Parameter sensitivity, time stability, regime stability             │
│  └─ Can demote validated → testing on failure                           │
│  └─ Emits: VALIDATION_ANALYST_REVIEW, VALIDATION_ANALYST_COMPLETE      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    INDEPENDENT RISK VETO                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Risk Manager                                                            │
│  └─ Portfolio risk assessment                                           │
│  └─ Can VETO but cannot APPROVE                                         │
│  └─ Emits: RISK_MANAGER_ASSESSMENT, RISK_REVIEW_COMPLETE, RISK_VETO    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CIO ADVISORY SCORING                                │
├─────────────────────────────────────────────────────────────────────────┤
│  CIO Agent                                                               │
│  └─ 4-dimensional scoring (Statistical, Risk, Economic, Cost)           │
│  └─ Recommends: CONTINUE, CONDITIONAL, KILL, PIVOT                      │
│  └─ Emits: CIO_AGENT_DECISION                                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       HUMAN APPROVAL                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Human CIO (Manual)                                                      │
│  └─ Final deployment decision                                           │
│  └─ Only entity that can approve deployment                             │
│  └─ Emits: DEPLOYMENT_APPROVED / DEPLOYMENT_REJECTED                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Lineage Event Types

### Event Categories

**Hypothesis Lifecycle:**
- `HYPOTHESIS_CREATED` - Hypothesis first created
- `HYPOTHESIS_UPDATED` - Metadata/outcome changed
- `HYPOTHESIS_DELETED` - Soft-deleted
- `HYPOTHESIS_FLAGGED` - Critical issues detected

**Experiment Events:**
- `EXPERIMENT_RUN` - Walk-forward validation started
- `EXPERIMENT_LINKED` - MLflow run linked to hypothesis
- `EXPERIMENT_COMPLETED` - ML training completed

**Validation Events:**
- `VALIDATION_PASSED` - Out-of-sample validation passed
- `VALIDATION_FAILED` - Out-of-sample validation failed

**Deployment Events:**
- `DEPLOYMENT_APPROVED` - Human CIO approval
- `DEPLOYMENT_REJECTED` - Human CIO rejection

### Events by Agent

| Agent | Events Emitted |
|-------|----------------|
| Signal Scientist | `SIGNAL_SCAN_COMPLETE`, `AGENT_RUN_COMPLETE` |
| Alpha Researcher | `ALPHA_RESEARCHER_REVIEW`, `ALPHA_RESEARCHER_COMPLETE` |
| ML Scientist | `ML_SCIENTIST_VALIDATION`, `EXPERIMENT_COMPLETED` |
| ML Quality Sentinel | `ML_QUALITY_SENTINEL_AUDIT`, `HYPOTHESIS_FLAGGED` |
| Quant Developer | `QUANT_DEVELOPER_BACKTEST_COMPLETE` |
| Kill Gate Enforcer | `KILL_GATE_ENFORCER_COMPLETE`, `KILL_GATE_TRIGGERED` |
| Validation Analyst | `VALIDATION_ANALYST_REVIEW`, `VALIDATION_ANALYST_COMPLETE` |
| Risk Manager | `RISK_MANAGER_ASSESSMENT`, `RISK_REVIEW_COMPLETE`, `RISK_VETO` |
| CIO Agent | `CIO_AGENT_DECISION` |

---

## 5. Validation Thresholds

### ML Scientist (Promotion to Validated)

| Threshold | Value | Action |
|-----------|-------|--------|
| IC minimum | 0.03 | Below → rejected |
| IC suspicious | 0.15 | Above → flag for leakage |
| Stability max | 1.0 | Above → rejected |

### ML Quality Sentinel (Kill Gates)

| Check | Critical Threshold | Action |
|-------|-------------------|--------|
| Sharpe decay | > 50% | Flag overfitting |
| Feature-target correlation | > 0.95 | Flag leakage |
| Feature count | > 50 | Flag dimensionality curse |
| Fold CV instability | > 2.0 | Flag inconsistency |

### Kill Gate Enforcer

| Gate | Threshold | Action |
|------|-----------|--------|
| Baseline Sharpe | < 0.5 | Kill |
| Train Sharpe | > 3.0 | Kill (suspicious) |
| Max drawdown | > 30% | Kill |
| Feature count | > 50 | Kill |
| Instability score | > 1.5 | Kill |

### Risk Manager (Veto Authority)

| Check | Limit | Veto If Exceeded |
|-------|-------|------------------|
| Max drawdown | 25% | Yes |
| Sector exposure | 30% | Yes |
| Single position | 10% | Yes |
| Min diversification | 10 positions | Yes (if fewer) |
| Volatility | 25% | Warning |
| Turnover | 50% | Warning |

### CIO Agent (Scoring)

| Decision | Score Threshold | Condition |
|----------|-----------------|-----------|
| CONTINUE | ≥ 0.75 | No critical failures |
| CONDITIONAL | 0.50 - 0.74 | Needs more work |
| KILL | < 0.50 | Below minimum |
| PIVOT | Any | Critical failure detected |

---

## 6. Trigger Chain (Event-Driven Pipeline)

```
Signal Scientist completes
    └─ Emits: SIGNAL_SCAN_COMPLETE
        └─ Triggers: Alpha Researcher

Alpha Researcher completes
    └─ Emits: ALPHA_RESEARCHER_COMPLETE
        └─ Triggers: ML Scientist

ML Scientist completes
    └─ Emits: EXPERIMENT_COMPLETED
        └─ Triggers: ML Quality Sentinel

ML Quality Sentinel completes
    └─ Emits: ML_QUALITY_SENTINEL_AUDIT
        └─ Triggers: Quant Developer

Quant Developer completes
    └─ Emits: QUANT_DEVELOPER_BACKTEST_COMPLETE
        └─ Triggers: Kill Gate Enforcer (optional)

Kill Gate Enforcer completes
    └─ Emits: KILL_GATE_ENFORCER_COMPLETE
        └─ Triggers: Validation Analyst

Validation Analyst completes
    └─ Emits: VALIDATION_ANALYST_COMPLETE
        └─ Triggers: Risk Manager

Risk Manager completes
    └─ Emits: RISK_MANAGER_ASSESSMENT
        └─ Triggers: CIO Agent

CIO Agent completes
    └─ Emits: CIO_AGENT_DECISION
        └─ Awaits: Human approval
```

---

## 7. Permissions Matrix

| Action | Agents | Risk Manager | CIO Agent | Human CIO |
|--------|--------|--------------|-----------|-----------|
| Create hypotheses | Yes | No | No | Yes |
| Run experiments | Yes | No | No | Yes |
| Analyze results | Yes | Yes | Yes | Yes |
| Recommend decisions | Yes | Yes | Yes | Yes |
| Issue veto | No | **Yes** | No | Yes |
| Approve deployment | No | No | No | **Yes** |
| Modify deployed | No | No | No | Yes |

**Key Principle:** Risk Manager has independent veto authority but cannot approve. Only human CIO can approve deployments.

---

## 8. Idempotency & Deduplication

### Lineage-Based Idempotency

All agents check lineage before processing to prevent double-processing:

```python
# Example: Quant Developer checks for existing backtest
if hypothesis.metadata.get("quant_developer_backtest"):
    logger.info(f"Skipping {hypothesis_id} - already backtested")
    continue
```

### Event Deduplication Window

- 60-second window for duplicate event detection
- Same (event_type, hypothesis_id, actor) within window = duplicate

### Metadata Markers

Each agent writes a marker to hypothesis metadata when complete:

| Agent | Metadata Key |
|-------|--------------|
| Alpha Researcher | `alpha_researcher_review` |
| ML Scientist | `ml_scientist_results` |
| ML Quality Sentinel | `ml_quality_sentinel_audit` |
| Quant Developer | `quant_developer_backtest` |
| Validation Analyst | `validation_analyst_review` |
| Risk Manager | `risk_manager_review` |
| CIO Agent | `cio_agent_decision` |

---

## Document History

- **2026-02-03:** Initial comprehensive state machine documentation created
