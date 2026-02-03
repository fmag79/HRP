# Agent Definition: Pipeline Orchestrator

**Date:** January 29, 2026
**Status:** Implemented
**Type:** Custom (deterministic - extends `ResearchAgent`)

---

## Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Pipeline Orchestrator |
| **Actor ID** | `agent:pipeline-orchestrator` |
| **Type** | Custom (deterministic) |
| **Role** | Experiment coordination, baseline management, early kill gates |
| **Implementation** | `hrp/agents/pipeline_orchestrator.py` |
| **Trigger** | Lineage event (after Quant Developer) + Scheduled (daily) |
| **Upstream** | Quant Developer (backtest results) |
| **Downstream** | Validation Analyst (stress testing) |

---

## Purpose

Coordinates parallel experiment execution with intelligent resource management and early stopping. The Pipeline Orchestrator:

1. **Runs baselines first** - Establish performance floor before expensive experiments
2. **Queues parallel experiments** - Execute multiple configs efficiently
3. **Applies early kill gates** - Save compute by stopping unpromising runs early
4. **Logs all artifacts** - Full MLflow lineage for reproducibility
5. **Tracks resource usage** - Monitor compute, time saved from kill gates
6. **Triggers downstream** Validation Analyst via lineage events

---

## Core Capabilities

### 1. Baseline Execution (Sequential)

Run simple baselines first to establish performance floor:

```python
from hrp.agents import PipelineOrchestrator

orchestrator = PipelineOrchestrator(
    run_baselines_first=True,
    baseline_types=[
        "equal_weight_long_short",
        "buy_and_hold_spy",
        "market_cap_weighted",
    ],
)
result = orchestrator.run()

print(f"Baselines run: {result['baselines_run']}")
print(f"Experiments passed baseline: {result['experiments_passed']}")
print(f"Hypotheses killed early: {result['hypotheses_killed']}")
```

**Baseline Configuration:**
- Standard period: 5 years (2019-2023)
- Standard universe: S&P 500 (ex-financials, ex-REITs)
- Standard cost model: IBKR (5 bps commission, 10 bps slippage)
- Standard rebalance: weekly
- Standard position limit: 20 positions max, 5% max position size

### 2. Early Kill Gates

Apply conservative thresholds to terminate unpromising research early:

**Kill Gate Criteria:**

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Baseline Sharpe** | < 0.5 | Below market, not worth pursuing |
| **Train Sharpe** | > 3.0 | Suspiciously good, likely overfit |
| **Max Drawdown** | > 30% | Excessive risk, institutional-grade limit |
| **Feature Count** | > 50 | Too many features, curse of dimensionality |
| **Sharpe Decay** | > 50% | Train >> test, severe overfitting |
| **Win Rate** | < 40% | Poor hit rate, unlikely to improve |

**Kill Gate Actions:**
- Log kill event to MLflow with rationale
- Update hypothesis status to "rejected"
- Write kill gate report to `docs/reports/YYYY-MM-DD-kill-gate-{hyp_id}.md`
- Free up resources for next experiment
- Notify user via email if configured

### 3. Parallel Experiment Execution

Build experiment queue from parameter grids and execute in parallel:

```python
# Build experiment queue from validated hypotheses
queue = orchestrator._build_experiment_queue(hypothesis_id)

# Run experiments in parallel with resource management
results = orchestrator._run_parallel_experiments(
    queue,
    max_parallel_jobs=4,
    batch_size=10,
)

# Each experiment checked against kill gates during execution
# Early stopping if Sharpe decay > 50%, feature count > 50, etc.
```

**Resource Management:**
- CPU cores: Use all available (n_jobs=-1)
- Memory: Limit concurrent experiments based on data size
- Disk: Cache intermediate results to avoid re-computation
- Network: Batch API calls to reduce overhead

### 4. Kill Gate Reports

Every killed hypothesis generates a report:

```markdown
# Kill Gate Report: {Hypothesis Title}

**Hypothesis ID**: HYP-2026-XXX
**Kill Date**: YYYY-MM-DD HH:MM:SS
**Killed By**: agent:pipeline_orchestrator
**Reason**: Early termination to save compute resources

## Hypothesis Summary
{Brief description of hypothesis and economic thesis}

## Baseline Results
- **Sharpe Ratio**: {value} (threshold: 0.5)
- **Max Drawdown**: {value} (threshold: 30%)
- **Train Sharpe**: {value} (threshold: 3.0)
- **Sharpe Decay**: {value}% (threshold: 50%)
- **Feature Count**: {value} (threshold: 50)
- **Win Rate**: {value}% (threshold: 40%)

## Kill Gate Decision
**FAILED**: {metric_name} = {value} {operator} {threshold}

**Rationale**: {Detailed explanation}

## Resource Savings
- **Compute Time Saved**: {estimated hours}
- **Cost Savings**: ${estimated cost}
- **Experiments Avoided**: {count}

## Artifacts
- MLflow Run: {run_id}
- Baseline Plot: {artifact_path}
```

---

## Configuration

```python
@dataclass
class PipelineOrchestratorConfig:
    hypothesis_ids: list[str] | None = None  # None = all ready for orchestration

    # Baseline settings
    run_baselines_first: bool = True
    baseline_period_years: int = 5
    baseline_universe: str = "sp500"

    # Parallel execution settings
    enable_parallel: bool = True
    max_parallel_jobs: int = -1  # -1 for auto-detect
    batch_size: int = 10  # Experiments per batch
    retry_failed: bool = True
    max_retries: int = 2

    # Kill gate settings
    enable_kill_gates: bool = True
    kill_after_baseline: bool = True
    kill_gate_thresholds: dict = field(default_factory=lambda: {
        "min_sharpe": 0.5,
        "max_train_sharpe": 3.0,
        "max_drawdown": 0.30,
        "max_features": 50,
        "max_sharpe_decay": 0.50,
        "min_win_rate": 0.40,
    })

    # Resource settings
    max_memory_gb: float | None = None  # Auto-detect if None
    cache_dir: str | None = None  # Default: ~/hrp-data/cache/

    # MLflow settings
    mlflow_tracking_uri: str = "sqlite:///~/hrp-data/mlflow/mlflow.db"
    mlflow_experiment: str = "research_pipeline"

    # Reporting settings
    write_kill_gate_reports: bool = True
    write_resource_summary: bool = True
    send_kill_notifications: bool = True
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hypothesis_ids` | `None` (all ready) | Specific hypotheses to orchestrate |
| `run_baselines_first` | `True` | Run baselines before parameter sweeps |
| `enable_parallel` | `True` | Enable parallel experiment execution |
| `max_parallel_jobs` | `-1` | Max concurrent jobs (-1 = all cores) |
| `enable_kill_gates` | `True` | Apply early termination criteria |
| `kill_after_baseline` | `True` | Check kill gates after baseline |
| `write_kill_gate_reports` | `True` | Generate kill gate reports |

---

## Outputs

### 1. Baseline Results

- Location: MLflow runs tagged `baseline`
- Metrics: Sharpe, drawdown, returns, turnover
- Artifacts: Equity curve, trade list, signal distribution
- Used for kill gate decisions

### 2. Parallel Experiments

- Location: MLflow runs with parent-child relationships
- Parameter grid search results
- Performance heatmaps (Sharpe vs parameters)
- Used for sensitivity analysis

### 3. Kill Gate Reports

- Location: `docs/reports/YYYY-MM-DD-kill-gate-{hyp_id}.md`
- Rationale for termination
- Metrics that failed thresholds
- Recommendations for improvement (if any)

### 4. Resource Summary

- Location: `docs/reports/YYYY-MM-DD-pipeline-summary.md`
- Total experiments run
- Experiments killed early
- Compute time saved
- Cost savings estimate

### 5. MLflow Artifacts

- All experiments logged with full configuration
- Lineage from hypothesis → experiments → decisions
- Reproducible artifact storage

---

## Structural Regime Scenarios

**Updated:** Pipeline Orchestrator uses HMM-based structural regimes.

### Regime Matrix

| Vol \ Trend | Bull | Bear |
|-------------|------|------|
| **Low** | Low Vol Bull | Low Vol Bear |
| **High** | High Vol Bull | High Vol Bear (Crisis) |

### Requirements

| Requirement | Specification |
|-------------|---------------|
| Minimum scenarios | 4 (one per regime type) |
| Sharpe CV threshold | ≤ 0.30 across all regimes |
| Regime coverage | Must test in all 4 regime types |

---

## Trigger Model

### Primary: Lineage Event Trigger

```python
# Pipeline Orchestrator listens for Quant Developer completion
scheduler.register_lineage_trigger(
    event_type="QUANT_DEVELOPER_BACKTEST_COMPLETE",
    actor_filter="agent:quant-developer",
    callback=trigger_pipeline_orchestrator,
)
```

### Secondary: Scheduled Run

```python
# Daily orchestration at 6 AM ET
scheduler.setup_daily_pipeline_orchestrator(
    orchestration_time='06:00',
)
```

### Tertiary: MCP On-Demand

```python
# MCP tool: run_pipeline_orchestrator
result = run_pipeline_orchestrator(hypothesis_id="HYP-2026-001")
```

---

## Resource Optimization Strategies

### 1. Intelligent Caching

```python
# Cache frequently accessed data
with cache_enabled():
    prices = api.get_prices(symbols, start, end)
    features = api.get_features(symbols, feature_list, dates)
```

### 2. Batch Processing

```python
# Process experiments in batches to reduce overhead
for batch in chunks(experiment_queue, batch_size=10):
    parallel_execute(batch, n_jobs=-1)
```

### 3. Early Termination

```python
# Check kill gates after baseline
if not passes_kill_gates(baseline_result):
    log_kill_event(hypothesis_id)
    continue  # Skip parameter sweep
```

### 4. Resource Monitoring

```python
# Monitor memory usage and adjust batch size
if memory_usage() > threshold:
    reduce_batch_size()
    gc.collect()
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Pipeline Orchestrator                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Receive Hypothesis Queue                               │
│     ┌───────────────────────────────────┐                  │
│     │ HYP-001, HYP-002, HYP-003        │                  │
│     └───────────────────────────────────┘                  │
│                    ↓                                        │
│  2. Run Baselines (Sequential)                            │
│     ┌───────────────────────────────────┐                  │
│     │ HYP-001: Sharpe=0.3 ✗ KILLED      │                  │
│     │ HYP-002: Sharpe=0.8 ✓ PASSED      │                  │
│     │ HYP-003: Sharpe=2.5 ✗ KILLED      │                  │
│     └───────────────────────────────────┘                  │
│                    ↓                                        │
│  3. Build Experiment Queue (for passed)                    │
│     ┌───────────────────────────────────┐                  │
│     │ HYP-002: 50 parameter variations  │                  │
│     └───────────────────────────────────┘                  │
│                    ↓                                        │
│  4. Run Parallel Experiments                               │
│     ┌───────────────────────────────────┐                  │
│     │ Workers: 8 (n_jobs=-1)           │                  │
│     │ Batches: 5 (10 experiments each) │                  │
│     └───────────────────────────────────┘                  │
│                    ↓                                        │
│  5. Apply Kill Gates (per experiment)                      │
│     ┌───────────────────────────────────┐                  │
│     │ 35 passed, 15 killed early        │                  │
│     └───────────────────────────────────┘                  │
│                    ↓                                        │
│  6. Generate Reports                                       │
│     ┌───────────────────────────────────┐                  │
│     │ Kill gate reports                 │                  │
│     │ Resource summary                  │                  │
│     │ MLflow artifacts                  │                  │
│     └───────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Quant Developer** | Receives backtest results for orchestration |
| **Platform API** | Hypothesis registry access |
| **MLflow** | Logs all experiments and artifacts |
| **Lineage** | Logs events, triggers downstream |
| **Validation Analyst** | Downstream - stress tests passed experiments |

---

## Explicit Non-Responsibilities

The Pipeline Orchestrator does NOT:

- ❌ Create strategies (Alpha Researcher's job)
- ❌ Implement backtests (Quant Developer's job)
- ❌ Train ML models (ML Scientist's job)
- ❌ Judge performance qualitatively (Applies thresholds, doesn't interpret)
- ❌ Modify strategy logic (Executes, doesn't design)
- ❌ Approve deployment (CIO Agent's job)

---

## Success Metrics

- Kill gates save >50% compute time on average
- Zero false negatives (no good strategies killed)
- Parallel execution achieves >80% CPU utilization
- Resource reports generated for all runs
- MLflow lineage complete for all experiments

---

## Document History

- **2026-01-29:** Initial agent definition created
