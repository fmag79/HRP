# Research Agents: Day-to-Day Operations Projection

**Date:** January 25, 2026
**Status:** Draft - Brainstorm Documentation
**Related:** [Research Agents Design](2026-01-25-research-agents-design.md)

---

## Overview

This document projects how the 8-agent research team would operate day-to-day, leveraging HRP platform capabilities. It covers the ML experimentation pipeline, timeline from cold start to presented strategies, and the CIO's role.

---

## Daily Operations: The Research Machine

### 6:00 PM ET - Market Close Cascade

**Automated Pipeline (existing infrastructure):**
```
IngestionScheduler triggers:
├── 6:00 PM  PriceIngestionJob    → Fresh OHLCV for S&P 500
├── 6:05 PM  UniverseUpdateJob    → Check for index changes
├── 6:10 PM  FeatureComputationJob → Recalculate all 44 features
└── 6:15 PM  DataQualityJob       → Validate data, send alerts if issues
```

**Then the agents wake up:**

### 6:30 PM - ML Quality Sentinel
- Scans today's feature computations for anomalies
- Checks for data drift vs historical distributions
- Flags anything suspicious → writes to lineage with `actor='agent:ml-quality-sentinel'`

### 7:00 PM - Signal Scientist
- Runs nightly signal scans using fresh features
- Tests correlation of features vs forward returns
- If IC > threshold on any signal:
  ```python
  api.create_hypothesis(
      title="RSI divergence predicts 5-day returns",
      thesis="...",
      prediction="IC > 0.03 sustained",
      falsification="IC < 0.01 or unstable across regimes",
      actor='agent:signal-scientist'
  )
  ```
- Logs findings to MLflow experiment `signal-scans/YYYY-MM-DD`

---

## Weekly Cycle

| Day | Agent Activity |
|-----|----------------|
| **Monday** | Alpha Researcher reviews draft hypotheses, refines falsification criteria, promotes to `testing` |
| **Tuesday-Wednesday** | ML Scientist runs walk-forward validation on `testing` hypotheses |
| **Thursday** | Validation Analyst stress tests + Risk Manager reviews |
| **Friday** | Report Generator compiles weekly summary for CIO review |
| **Saturday** | Fundamentals ingestion (weekly job) |
| **Sunday** | System idle |

### Monday - Alpha Researcher
- Reviews all draft hypotheses from Signal Scientist
- Adds regime awareness: "Does this signal work in high-vol vs low-vol?"
- Refines falsification criteria to be testable
- Promotes promising ones to `status='testing'`

### Tuesday/Wednesday - ML Scientist
- Picks up hypotheses in `testing` status
- Runs walk-forward validation:
  ```python
  result = walk_forward_validate(
      config=WalkForwardConfig(
          model_type='ridge',
          features=['momentum_20d', 'rsi_14d'],
          n_folds=5,
          n_jobs=-1,  # Parallel processing
      ),
      symbols=universe,
      log_to_mlflow=True,
  )
  ```
- Updates hypothesis with results, stability scores

### Thursday - Validation Analyst + Risk Manager

**Validation Analyst:**
- Runs parameter sensitivity on validated models
- Stress tests across market regimes (2008, 2020, 2022)
- Checks execution realism (can we actually trade this?)

**Risk Manager (independent):**
- Reviews all strategies approaching `validated` status
- Checks portfolio-level impact if strategy deployed
- Can flag concerns or veto, but **cannot approve**

### Friday - Report Generator + CIO

**Report Generator produces:**
```
Weekly Research Report - Week of Jan 20, 2026

HYPOTHESES SUMMARY
- 3 new hypotheses created (Signal Scientist)
- 2 promoted to testing (Alpha Researcher)
- 1 passed walk-forward (ML Scientist)
- 1 passed validation (Validation Analyst)
- 0 vetoed by Risk Manager

AWAITING CIO DECISION
┌─────────────────────────────────────────────────────┐
│ HYP-2026-042: Momentum + Low Vol Factor            │
│ Status: Validated ✓                                 │
│ Sharpe: 1.2 | Max DD: 12% | Stability: 0.7         │
│ Risk Manager: No concerns                          │
│ Action Required: APPROVE / REJECT for paper trade  │
└─────────────────────────────────────────────────────┘

EXPERIMENTS THIS WEEK
- 12 walk-forward runs
- 847 hyperparameter trials tracked
- 2 leakage warnings (addressed)

NEXT WEEK PRIORITIES
- Signal Scientist: Test alternative data signals
- Alpha Researcher: Investigate sector momentum
```

---

## ML Experimentation: Who Does What

| Stage | Agent | What They Do | HRP Tools Used |
|-------|-------|--------------|----------------|
| **1. Signal Discovery** | Signal Scientist | Tests feature predictiveness (IC analysis) | `get_features`, correlation scans |
| **2. Hypothesis Creation** | Alpha Researcher | Formalizes signal into testable hypothesis | `create_hypothesis` |
| **3. Model Training** | ML Scientist | Trains models, tunes hyperparameters | `walk_forward_validate`, MLflow |
| **4. Training Audit** | ML Quality Sentinel | Checks leakage, overfitting, Sharpe decay | `HyperparameterTrialCounter`, `TargetLeakageValidator` |
| **5. Strategy Backtest** | Quant Developer | Runs full backtest with realistic costs | `run_backtest`, `generate_ml_predicted_signals` |
| **6. Stress Testing** | Validation Analyst | Parameter sensitivity, regime stress | `check_parameter_sensitivity` |
| **7. Risk Review** | Risk Manager | Portfolio fit, drawdown limits | `validate_strategy` |

---

## ML Experimentation Loop (Detail)

```
Signal Scientist                    ML Scientist
     │                                   │
     │ "momentum_20d has IC=0.04"        │
     │                                   │
     ▼                                   │
Alpha Researcher                         │
     │                                   │
     │ Creates HYP-2026-001              │
     │ status='testing'                  │
     │                                   │
     └──────────────────────────────────►│
                                         │
                    ┌────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │         ML SCIENTIST LOOP             │
        │                                       │
        │  for model_type in [ridge, lasso,    │
        │                      rf, lightgbm]:   │
        │                                       │
        │    for features in feature_combos:   │
        │                                       │
        │      walk_forward_validate(          │
        │        model_type=model_type,        │
        │        features=features,            │
        │        n_folds=5,                    │
        │      )                               │
        │                                       │
        │      log_to_mlflow()                 │
        │      counter.log_trial()  ← max 50   │
        │                                       │
        └───────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │       ML QUALITY SENTINEL             │
        │                                       │
        │  - Check Sharpe decay (train vs test)│
        │  - Verify no target leakage          │
        │  - Validate feature count < 30       │
        │  - Flag if stability_score > 1.0     │
        │                                       │
        └───────────────────────────────────────┘
                    │
                    ▼
              Best model selected
              (highest IC, stable, no leakage)
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │        QUANT DEVELOPER                │
        │                                       │
        │  signals = generate_ml_predicted_     │
        │    signals(                           │
        │      prices,                          │
        │      model_type="ridge",              │
        │      features=[...],                  │
        │      signal_method="rank",            │
        │      top_pct=0.1,                     │
        │    )                                  │
        │                                       │
        │  result = run_backtest(signals, ...)  │
        │                                       │
        └───────────────────────────────────────┘
                    │
                    ▼
            Full backtest metrics:
            Sharpe, drawdown, turnover, costs
```

---

## Fresh Start Timeline: Cold Start to Presented Strategies

### Day 0 (Setup)
```
6:00 PM - Data already loaded (prices, features for S&P 500)
        - Hypothesis registry empty
        - MLflow experiments empty
```

### Day 1 - Signal Discovery
```
Signal Scientist runs feature scans:

for feature in ALL_44_FEATURES:
    for forward_return in [5d, 10d, 20d]:
        ic = calculate_information_coefficient(feature, forward_return)
        if ic > 0.02:
            log_promising_signal(feature, ic)

Output: 8-12 promising signals identified
        (e.g., momentum_20d, rsi_14d, volatility_60d inversely)
```

### Day 2 - Hypothesis Formation
```
Alpha Researcher reviews signals, creates hypotheses:

HYP-2026-001: "Momentum + Low Volatility"
HYP-2026-002: "RSI Mean Reversion"
HYP-2026-003: "Multi-factor Composite"

Status: 'testing' → triggers ML Scientist
```

### Days 3-5 - ML Experimentation (The Heavy Lifting)

**This is where the bulk of computation happens:**

```python
# ML Scientist runs for each hypothesis
for hypothesis in hypotheses_in_testing:

    # Test multiple model types
    for model_type in ['ridge', 'lasso', 'random_forest', 'lightgbm']:

        # Test feature combinations
        for feature_set in generate_feature_combinations(hypothesis.signals):

            # Walk-forward validation (5 folds × 8 years)
            result = walk_forward_validate(
                config=WalkForwardConfig(
                    model_type=model_type,
                    features=feature_set,
                    start_date=date(2015, 1, 1),
                    end_date=date(2023, 12, 31),
                    n_folds=5,
                    window_type='expanding',
                    n_jobs=-1,  # Parallel
                ),
                symbols=sp500_universe,  # ~450 stocks
                log_to_mlflow=True,
            )

            # Track trial count (max 50 per hypothesis)
            counter.log_trial(model_type, params, result.mean_ic)
```

**Per hypothesis, ML Scientist tests:**
- 4 model types × 5-10 feature combos = 20-40 experiments
- Each experiment: 5-fold walk-forward = ~2-5 min with parallel processing
- Total per hypothesis: ~1-3 hours of compute

**ML Quality Sentinel runs after each batch:**
```python
# Check every completed experiment
for experiment in todays_experiments:

    # Sharpe decay check
    decay = SharpeDecayMonitor(max_decay_ratio=0.5)
    decay.check(train_sharpe, test_sharpe)

    # Leakage check
    leakage = TargetLeakageValidator(correlation_threshold=0.95)
    leakage.check(features_df, target)

    # Feature count check
    validator = FeatureCountValidator(max_threshold=50)
    validator.check(feature_count, sample_count)
```

### Days 6-7 - Backtesting & Validation

**Quant Developer** runs full backtests on best models:

```python
# For each hypothesis with passing ML validation
signals = generate_ml_predicted_signals(
    prices,
    model_type="ridge",  # Winner from ML experiments
    features=["momentum_20d", "volatility_60d"],
    signal_method="rank",
    top_pct=0.1,
    train_lookback=252,
    retrain_frequency=21,
)

result = run_backtest(signals, config, prices)
# Returns: Sharpe, max_drawdown, win_rate, turnover, etc.
```

**Validation Analyst** stress tests:

```python
# Parameter sensitivity
experiments = {
    "baseline": backtest_with_lookback(20),
    "short": backtest_with_lookback(10),
    "long": backtest_with_lookback(40),
}
robustness = check_parameter_sensitivity(experiments)

# Regime stress testing
for regime in ['2008_crisis', '2020_covid', '2022_rates']:
    run_backtest(signals, regime_period)
```

### Day 8 - Risk Review & Report

**Risk Manager** validates:
```python
result = validate_strategy({
    "sharpe": 1.1,
    "num_trades": 500,
    "max_drawdown": 0.15,
    "win_rate": 0.54,
})
# Checks: Sharpe > 0.5, trades > 100, DD < 25%, etc.
```

**Report Generator** compiles findings for CIO review.

---

## Timeline Summary

| Day | Activity | Output |
|-----|----------|--------|
| **1** | Signal Scientist scans features | 8-12 promising signals |
| **2** | Alpha Researcher creates hypotheses | 3-5 formal hypotheses |
| **3-5** | ML Scientist runs experiments | 100-200 ML experiments logged |
| **3-5** | ML Quality Sentinel audits | Leakage/overfitting flags |
| **6** | Quant Developer backtests winners | Full backtest metrics |
| **7** | Validation Analyst stress tests | Robustness reports |
| **8** | Risk Manager reviews | Approved/flagged strategies |
| **8** | Report Generator compiles | **Strategies presented to CIO** |

**Result: ~8 calendar days from cold start to first ML strategies ready for review.**

---

## Accelerating the Timeline

| Optimization | Impact |
|--------------|--------|
| Run Signal Scientist + Alpha Researcher same day | -1 day |
| Parallel ML experiments across hypotheses | -1 day |
| Pre-define feature combinations (skip search) | -1 day |
| Use only `ridge` + `lightgbm` (skip slower models) | Faster experiments |

**Aggressive timeline: 5 days** to first presented strategies.

---

## Ongoing Cadence (After Initial Run)

Once the system is warm:

```
Week N:
├── Mon: Signal Scientist finds 2 new signals
├── Tue: Alpha Researcher creates 1 hypothesis
├── Wed-Thu: ML Scientist tests (existing pipeline warm)
├── Fri: CIO reviews 1-2 new strategies + ongoing refinements

Throughput: 1-3 new validated strategies per week
```

---

## CIO Role: Where You Fit

### What Agents CANNOT Do (by design)
- Deploy any strategy to paper or live trading
- Override Risk Manager vetoes
- Modify deployed strategies
- Approve capital allocation

### What the CIO Does

**1. Weekly Review (~30 min Friday)**
- Read Report Generator's summary
- Review validated hypotheses awaiting decision
- Approve/reject for paper trading

**2. Strategic Direction**
- "Focus on low-turnover strategies this month"
- "Investigate why momentum stopped working in Q4"
- Agents pick up these directives

**3. Ad-hoc Queries via MCP**
```
CIO: "Why did HYP-2026-042 fail in 2022?"

→ Quant Developer pulls the backtest
→ Alpha Researcher adds regime context
→ Response: "Strategy relies on momentum continuation;
   2022 had 4 regime shifts that broke the signal"
```

**4. Final Authority**
```
Pipeline: Discovery → Modeling → Validation → Risk Review
                                                    ↓
                                           CIO APPROVAL
                                                    ↓
                                            Paper Trading
```

### Typical Week

| Day | CIO Time | What's Happening |
|-----|----------|------------------|
| Mon | 0 min | Agents processing weekend findings |
| Tue | 0 min | ML training running autonomously |
| Wed | 5 min | Glance at alerts if any leakage/risk flags |
| Thu | 0 min | Validation + Risk review in progress |
| Fri | 30 min | Read report, approve/reject, set direction |
| Sat | 0 min | Fundamentals ingestion (weekly) |
| Sun | 0 min | System idle |

**Total weekly commitment: ~35 minutes** for a full quant research operation.

---

## Information Flow Diagram

```
                    ┌─────────────────┐
                    │   CIO (User)    │
                    │  Weekly review  │
                    │  Final approval │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │ Report Generator│◄───────────│  Risk Manager   │
    │ (synthesizes)   │            │  (independent)  │
    └────────┬────────┘            └────────┬────────┘
             │                              │
    ┌────────┴────────────────────┬─────────┴────┐
    ▼                             ▼              ▼
┌──────────┐  ┌──────────────┐  ┌────────────────┐
│   ML     │  │  Validation  │  │    Quant       │
│Scientist │  │   Analyst    │  │   Developer    │
└────┬─────┘  └──────────────┘  └────────────────┘
     │                 ▲
     ▼                 │
┌──────────┐           │
│ML Quality│           │
│ Sentinel │           │
└──────────┘           │
                       │
    ┌──────────────────┴────────────────┐
    ▼                                   ▼
┌────────────────┐            ┌─────────────────┐
│Alpha Researcher│◄───────────│ Signal Scientist│
│(strategy)      │            │ (features)      │
└────────────────┘            └─────────────────┘
        ▲                             ▲
        └─────────────────────────────┘
              Shared: Hypothesis Registry
                      MLflow Experiments
                      Lineage System
                      Feature Store
```

---

## Next Steps

1. [ ] Decide which 2-3 agents to build first
2. [ ] Define agent infrastructure (base class, scheduling, MCP integration)
3. [ ] Implement initial agents
4. [ ] Test coordination through shared workspace

---

## Document History

- **2026-01-25:** Initial brainstorm documentation from conversation

My recommendation for first 2-3 agents:                                                                                                                                           
  ┌──────────┬─────────────────────┬───────────────────────────────────────────────────────────────────┐                                                                            
  │ Priority │        Agent        │                             Rationale                             │                                                                            
  ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤                                                                            
  │ 1st      │ Signal Scientist    │ Starts the pipeline - discovers signals that feed everything else │                                                                            
  ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤                                                                            
  │ 2nd      │ ML Scientist        │ Core value - runs the experiments that produce strategies         │                                                                            
  ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤                                                                            
  │ 3rd      │ ML Quality Sentinel │ Safety net - prevents garbage from propagating                    │                                                                            
  └──────────┴─────────────────────┴───────────────────────────────────────────────────────────────────┘                                                                            
  Why this order:                                                                                                                                                                   
  1. Signal Scientist can work standalone (scans features, logs findings)                                                                                                           
  2. ML Scientist consumes Signal Scientist output and produces the actual strategies                                                                                               
  3. ML Quality Sentinel catches problems early before wasting compute downstream                                                                                                   
                                                                                                                                                                                    
  What we defer:                                                                                                                                                                    
  - Alpha Researcher: You can manually refine hypotheses initially                                                                                                                  
  - Quant Developer: Backtesting can be triggered manually                                                                                                                          
  - Validation/Risk/Report: Later stages, less urgent                                                                                                                               
                                                                                                                                                                                    
  Which agents would you like to start with?                                                                                                                                        
                                                                                                                                                                                    
✻ Sautéed for 1m 41s                                                                                                                                                                
                                                                                                                                                                                    
❯ let's start with signal scientist. give me a full description of the agent for review before we build it        