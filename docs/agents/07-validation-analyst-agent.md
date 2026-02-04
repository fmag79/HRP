# Validation Analyst Agent

**Status:** ✅ IMPLEMENTED (2026-01-26) | **Report Upgraded:** 2026-02-03

## Agent Definition

### Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Validation Analyst |
| **Actor ID** | `agent:validation-analyst` |
| **Type** | Hybrid (deterministic tests + optional Claude reasoning) |
| **Role** | Pre-deployment stress testing, parameter sensitivity, regime analysis, execution cost validation |
| **Trigger** | Lineage event (after ML Quality Sentinel / Quant Developer) + MCP on-demand |
| **Upstream** | Quant Developer (produces backtest metrics, period/regime splits) |
| **Downstream** | Risk Manager (portfolio-level risk review before CIO decision) |
| **Module** | `hrp/agents/validation_analyst.py` |

### Purpose

Final validation gate before hypotheses can proceed to Risk Manager review. The Validation Analyst stress tests strategies to ensure they are robust and not overfit:

1. **Parameter Sensitivity** — Tests if strategy degrades gracefully when parameters are varied (±20%)
2. **Time Stability** — Verifies strategy is profitable across multiple time periods (≥67% of periods)
3. **Regime Robustness** — Ensures strategy performs in ≥2 of 3 market regimes (bull/bear/sideways)
4. **Execution Cost Impact** — Validates net returns remain positive after realistic IBKR costs

---

## Validation Thresholds

| Check | Threshold | Severity on Fail |
|-------|-----------|------------------|
| Parameter Sensitivity | Varied Sharpe ≥ 50% of baseline | CRITICAL |
| Time Stability | ≥67% periods profitable, Sharpe CV ≤ 1.0 | CRITICAL |
| Regime Robustness | ≥2 of 3 regimes profitable | CRITICAL |
| Execution Costs | Net return > 0; Cost < 50% of gross → warning | CRITICAL/WARNING |

---

## Report Output (Medallion Standard)

The Validation Analyst produces institutional-grade research reports written to:
```
~/hrp-data/output/research/YYYY-MM-DD/YYYY-MM-DDTHHMMSS-07-validation-analyst.md
```

### Report Sections

1. **Executive Summary**
   - Pass/fail verdict with aggregate statistics
   - KPI dashboard (validated, passed, failed, critical, warnings)
   - Alert banners for failures

2. **Stress Test Overview**
   - Check type breakdown table (passed/failed/pass rate per check type)
   - System health gauges

3. **Statistical Distribution** (when data available)
   - Parameter stability analysis (mean ratio, std dev, min ratio)
   - Time stability analysis (profitable period ratio, Sharpe CV)
   - Regime robustness analysis (profitable regime count)
   - Execution cost impact (cost/gross ratio, net returns)

4. **Validation Thresholds**
   - Current threshold configuration

5. **Per-Hypothesis Analysis**
   - Hypothesis context (title, thesis excerpt)
   - Metadata summary table
   - Detailed check results with metrics:
     - Parameter sensitivity: baseline vs. variation Sharpe ratios
     - Time stability: period-by-period breakdown table
     - Regime stability: bull/bear/sideways performance table
     - Execution costs: trade counts, cost breakdown, net return

6. **Recommendations**
   - Actionable items based on failure patterns
   - Check-specific guidance for common issues

---

## Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Quant Developer │ ──▶ │ Validation      │ ──▶ │ Risk Manager    │
│                 │     │ Analyst         │     │                 │
│ period_metrics  │     │                 │     │                 │
│ regime_metrics  │     │ Stress tests    │     │ Portfolio risk  │
│ param_variations│     │ Pass/fail       │     │ Veto authority  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Input (from Quant Developer metadata)

- `period_metrics`: List of yearly performance dicts
- `regime_metrics`: Dict of bull/bear/sideways performance
- `param_experiments`: Dict of baseline + parameter variations
- `num_trades`, `avg_trade_value`, `gross_return`: For cost estimation

### Output

- **Passed hypotheses**: Status remains `validated`, proceed to Risk Manager
- **Failed hypotheses**: Status demoted to `testing`, require refinement
- **Lineage events**: `VALIDATION_ANALYST_REVIEW`, `VALIDATION_ANALYST_COMPLETE`
- **Research note**: Detailed markdown report

---

## Usage

### Programmatic

```python
from hrp.agents import ValidationAnalyst

analyst = ValidationAnalyst(
    hypothesis_ids=["HYP-2026-001"],  # Optional: specific hypotheses
    param_sensitivity_threshold=0.5,   # Min ratio of varied/baseline Sharpe
    min_profitable_periods=0.67,       # 2/3 periods must be profitable
    min_profitable_regimes=2,          # At least 2 of 3 regimes profitable
    commission_bps=5.0,                # IBKR-style commission
    slippage_bps=10.0,                 # Estimated slippage
    send_alerts=True,                  # Email on failures
)
result = analyst.run()

print(f"Validated: {result['hypotheses_validated']}")
print(f"Passed: {result['hypotheses_passed']}")
print(f"Failed: {result['hypotheses_failed']}")
```

### MCP Tool

```python
# Via hrp-research MCP server
mcp__hrp_research__run_validation_analyst(
    hypothesis_ids=["HYP-2026-001"],  # Optional
)
```

### Scheduled

Automatically triggered after ML Quality Sentinel / Quant Developer completes via lineage event subscription.

---

## Robustness Module Integration

The Validation Analyst uses `hrp/risk/robustness.py` for deterministic checks:

```python
from hrp.risk.robustness import (
    check_parameter_sensitivity,  # Tests Sharpe degradation under param changes
    check_time_stability,         # Tests period-by-period profitability
    check_regime_stability,       # Tests regime-by-regime performance
)
```

Each function returns a `RobustnessResult` with:
- `passed`: Boolean
- `checks`: Dict of detailed metrics
- `failures`: List of failure reason strings

---

## Example Report Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 HRP | Hedgefund Research Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ✅ Validation Analyst Report — 2026-02-03

> 🔬 Pre-deployment stress testing | 2 hypotheses | 1 validated

## Executive Summary

🟡 **MODERATE VALIDATION RATE** — Mixed robustness across hypothesis pool

## 📊 Key Metrics

┌──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 🔬 Validated      │ ✅ Passed         │ ❌ Failed         │ 🔴 Critical       │ ⚠️ Warnings      │
│        2         │        1         │        1         │        1         │        0         │
│ hypotheses       │ 50% pass rate    │ demoted          │ blocking issues  │ flagged concerns │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘

### Validation Check Results

| Check Type | Passed | Failed | Pass Rate | Status |
|------------|--------|--------|-----------|--------|
| parameter_sensitivity | 2 | 0 | 100% | ✅ |
| time_stability | 1 | 1 | 50% | ⚠️ |

## 📋 Hypothesis Analysis

### ❌ HYP-2026-020 — **FAILED**

**Earnings Quality Momentum**

| Attribute | Value |
|-----------|-------|
| **Validation Result** | FAILED |
| **Critical Issues** | 1 |
| **Checks Performed** | 2 |

**❌ Time Stability** — 🔴 CRITICAL

*Time stability failed: Only 8/12 periods profitable (66.7% < 67.0%)*

| Period | Sharpe | Return | Profitable |
|--------|--------|--------|------------|
| 2015 | 0.00 | 0.0% | ❌ |
| 2016 | 0.13 | 1.3% | ✅ |
...
```

---

## State Transitions

```
┌─────────┐         ┌───────────┐         ┌───────────┐
│ testing │ ──────▶ │ validated │ ──────▶ │ Risk Mgr  │
└─────────┘  pass   └───────────┘  pass   └───────────┘
     ▲                    │
     │                    │ fail
     └────────────────────┘
         (demote to testing)
```

- **Pass**: Hypothesis proceeds to Risk Manager review
- **Fail**: Hypothesis demoted to `testing` status for refinement

---

## Configuration

### Environment Variables

None specific to Validation Analyst.

### Default Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `param_sensitivity_threshold` | 0.50 | Min ratio of varied/baseline Sharpe |
| `min_profitable_periods` | 0.67 | Min fraction of periods that must be profitable |
| `min_profitable_regimes` | 2 | Min regimes (of 3) that must be profitable |
| `commission_bps` | 5.0 | Commission per trade (IBKR estimate) |
| `slippage_bps` | 10.0 | Slippage per trade estimate |

---

## Testing

```bash
pytest tests/test_agents/test_validation_analyst.py -v
```

19 tests covering:
- Dataclass creation and properties
- Initialization with defaults and custom thresholds
- Parameter sensitivity checks
- Time stability checks
- Regime stability checks
- Execution cost estimation
- Execute method with mocked hypotheses
