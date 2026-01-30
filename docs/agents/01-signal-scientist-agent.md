# Agent Definition: Signal Scientist

**Date:** January 26, 2026
**Status:** Implemented
**Type:** Custom (deterministic - extends `ResearchAgent`)

---

## Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Signal Scientist |
| **Actor ID** | `agent:signal-scientist` |
| **Type** | Custom (deterministic) |
| **Role** | Feature engineering, signal discovery, predictive pattern identification |
| **Implementation** | `hrp/agents/research_agents.py` |
| **Trigger** | Scheduled (weekly) + MCP on-demand |
| **Downstream** | Alpha Researcher (refines discovered signals into hypotheses) |

---

## Purpose

Automated discovery of predictive signals in the feature universe. The Signal Scientist:

1. **Scans features** for predictive power via Information Coefficient (IC)
2. **Creates draft hypotheses** for signals exceeding threshold
3. **Logs discoveries** to MLflow and lineage system
4. **Triggers downstream** Alpha Researcher via lineage events

---

## Core Capabilities

### 1. Signal Discovery Pipeline

```python
from hrp.agents import SignalScientist

# Run signal scan on all features
agent = SignalScientist(
    symbols=None,  # None = all universe symbols
    features=None,  # None = all 44 features
    forward_horizons=[5, 10, 20],  # prediction horizons (days)
    ic_threshold=0.03,  # minimum IC to create hypothesis
    create_hypotheses=True,
)
result = agent.run()

print(f"Signals found: {result['signals_found']}")
print(f"Hypotheses created: {result['hypotheses_created']}")
print(f"MLflow run: {result['mlflow_run_id']}")
```

### 2. IC Calculation

For each feature-horizon combination:
- Calculate rank correlation between feature and forward returns
- Apply statistical significance test
- Filter by IC threshold

### 3. Hypothesis Creation

For each significant signal:
- Creates draft hypothesis with auto-generated thesis
- Links feature and horizon metadata
- Sets status to "draft" for Alpha Researcher review

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbols` | `None` (all) | Symbols to scan |
| `features` | `None` (all 44) | Features to test |
| `forward_horizons` | `[5, 10, 20]` | Prediction horizons in days |
| `ic_threshold` | `0.03` | Minimum IC to flag as signal |
| `min_samples` | `252` | Minimum samples for IC calculation |
| `create_hypotheses` | `True` | Auto-create draft hypotheses |
| `n_jobs` | `-1` | Parallel jobs (all cores) |

---

## Outputs

### 1. Signal Scan Results

```python
@dataclass
class SignalScanResult:
    feature: str
    horizon: int
    ic: float
    ic_pvalue: float
    n_samples: int
    is_significant: bool
    hypothesis_id: str | None  # If hypothesis created
```

### 2. Signal Scan Report

```python
@dataclass
class SignalScanReport:
    scan_date: date
    symbols_scanned: int
    features_scanned: int
    horizons_tested: list[int]
    signals_found: int
    hypotheses_created: int
    duration_seconds: float
    mlflow_run_id: str
```

### 3. MLflow Logging

- Parent run for scan session
- Child runs for each feature-horizon
- Metrics: IC, p-value, sample count
- Tags: feature name, horizon, significance

### 4. Lineage Events

- `SIGNAL_SCAN_COMPLETE`: Logged when scan finishes
- Triggers Alpha Researcher via event watcher

### 5. Research Note

- Location: `docs/research/YYYY-MM-DD-signal-scientist.md`
- Contents: Summary of signals found, IC rankings, hypotheses created

---

## Scheduling

### Weekly Scan (Default)

```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()
scheduler.setup_weekly_signal_scan(
    scan_time='19:00',  # Monday 7 PM ET (after feature computation)
    day_of_week='mon',
    ic_threshold=0.03,
    create_hypotheses=True,
)
scheduler.start()
```

### On-Demand via MCP

```python
# MCP tool: run_signal_scan
result = run_signal_scan(
    features=['momentum_20d', 'rsi_14d'],
    horizons=[5, 10],
    ic_threshold=0.02,
)
```

---

## Performance Optimization

The Signal Scientist includes performance optimizations:

1. **Data Pre-loading**: Loads all price/feature data once at start
2. **Parallel IC Calculation**: Uses joblib for parallel feature-horizon computation
3. **Batch Hypothesis Creation**: Creates hypotheses in batches to reduce DB round-trips

Typical scan time: ~2-3 minutes for full 44-feature, 3-horizon scan on S&P 500.

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Feature Store** | Reads computed features via `api.get_features()` |
| **Hypothesis Registry** | Creates draft hypotheses via `api.create_hypothesis()` |
| **MLflow** | Logs scan results and metrics |
| **Lineage** | Logs `AGENT_RUN_COMPLETE` event |
| **Scheduler** | Weekly scheduled execution |
| **Alpha Researcher** | Downstream - triggered by lineage event |

---

## Example Research Note

```markdown
# Signal Scientist Report - 2026-01-26

## Summary
- Symbols scanned: 503
- Features tested: 44
- Horizons: [5, 10, 20] days
- Signals found: 7
- Hypotheses created: 7
- Duration: 142.3s

## Top Signals by IC

| Feature | Horizon | IC | p-value | Hypothesis |
|---------|---------|----|---------|-----------|
| momentum_20d | 20 | 0.051 | 0.001 | HYP-2026-042 |
| rsi_14d | 10 | 0.043 | 0.003 | HYP-2026-043 |
| volatility_60d | 20 | -0.038 | 0.008 | HYP-2026-044 |
| returns_252d | 20 | 0.035 | 0.015 | HYP-2026-045 |

## Next Steps
- Alpha Researcher will review draft hypotheses
- Hypotheses promoted to 'testing' proceed to ML Scientist

---
*Generated by Signal Scientist (agent:signal-scientist)*
```

---

## Document History

- **2026-01-26:** Initial agent definition created
