# HRP - Hedgefund Research Platform

## Project Overview

Personal quantitative research platform for systematic trading strategy development.
Long-only US equities, daily timeframe, institutional rigor.

## Architecture

Three-layer architecture:
1. **Data Layer** - DuckDB storage, ingestion pipelines, feature store
2. **Research Layer** - VectorBT backtesting, MLflow experiments, hypothesis registry
3. **Control Layer** - Streamlit dashboard, MCP servers, scheduled agents

External access goes through `hrp/api/platform.py`. Internal Data Layer modules (`hrp/data/`) may access the database directly via `hrp/data/db.py`.

## Key Principles

1. **Research-First** - Every strategy starts as a formal hypothesis
2. **Reproducibility** - All experiments versioned and re-runnable
3. **Statistical Rigor** - Walk-forward validation, significance testing enforced
4. **Audit Trail** - Full lineage from hypothesis to deployment

## Agent Permissions

| Action | Agent | User |
|--------|-------|------|
| Create/run hypotheses | ✅ | ✅ |
| Run backtests | ✅ | ✅ |
| Analyze results | ✅ | ✅ |
| **Deploy strategies** | ❌ | ✅ |

Agents cannot approve deployments or modify deployed strategies.

## Code Conventions

- Python 3.11+
- Type hints required
- Black formatting (100 char line length)
- External database access through `hrp/api/platform.py`
- Data Layer modules may use `hrp/data/db.py` directly
- Log all significant actions to lineage table

## Common Tasks

### Run a backtest
```python
from hrp.api.platform import PlatformAPI
api = PlatformAPI()
experiment_id = api.run_backtest(config, hypothesis_id='HYP-2026-001')
```

### Create a hypothesis
```python
# Note: hypothesis module uses function-based API (not class-based)
hypothesis_id = api.create_hypothesis(
    title="Momentum predicts returns",
    thesis="Stocks with high 12-month returns continue outperforming",
    prediction="Top decile momentum > SPY by 3% annually",
    falsification="Sharpe < SPY or p-value > 0.05",
    actor='user'  # or 'agent:discovery'
)
```

### Query data
```python
prices = api.get_prices(['AAPL', 'MSFT'], start_date, end_date)
features = api.get_features(['AAPL'], ['momentum_20d', 'volatility_60d'], date)
```

### Available Features (44 total)

| Category | Features |
|----------|----------|
| **Returns** | `returns_1d`, `returns_5d`, `returns_20d`, `returns_60d`, `returns_252d` |
| **Momentum** | `momentum_20d`, `momentum_60d`, `momentum_252d` |
| **Volatility** | `volatility_20d`, `volatility_60d` |
| **Volume** | `volume_20d`, `volume_ratio`, `obv` |
| **Oscillators** | `rsi_14d`, `cci_20d`, `roc_10d`, `stoch_k_14d`, `stoch_d_14d`, `williams_r_14d`, `mfi_14d` |
| **Trend** | `atr_14d`, `adx_14d`, `macd_line`, `macd_signal`, `macd_histogram`, `trend` |
| **Moving Averages** | `sma_20d`, `sma_50d`, `sma_200d`, `ema_12d`, `ema_26d` |
| **EMA Signals** | `ema_crossover` |
| **Price Ratios** | `price_to_sma_20d`, `price_to_sma_50d`, `price_to_sma_200d` |
| **Bollinger Bands** | `bb_upper_20d`, `bb_lower_20d`, `bb_width_20d` |
| **VWAP** | `vwap_20d` |
| **Fundamental** | `market_cap`, `pe_ratio`, `pb_ratio`, `dividend_yield`, `ev_ebitda` |

### Ingest Fundamental Data
```python
from hrp.data.ingestion.fundamentals import ingest_snapshot_fundamentals
from datetime import date

# Ingest current P/E, P/B, market cap, dividend yield, EV/EBITDA
result = ingest_snapshot_fundamentals(
    symbols=['AAPL', 'MSFT'],
    as_of_date=date.today(),
)
print(f"Inserted {result['records_inserted']} fundamental records")
```

### Run data quality checks
```python
# Run quality checks via PlatformAPI
result = api.run_quality_checks(as_of_date=date.today(), send_alerts=True)
# Returns: health_score, critical_issues, warning_issues, passed

# Get quality trend for analysis
trend = api.get_quality_trend(days=30)
# Returns: dates, health_scores, critical_issues, warning_issues

# Get data health summary for dashboard
summary = api.get_data_health_summary()
# Returns: symbol_count, date_range, total_records, data_freshness, ingestion_summary
```

### Backfill EMA/VWAP features
```python
from hrp.data.backfill import backfill_features_ema_vwap

# Backfill historical EMA/VWAP features
result = backfill_features_ema_vwap(
    symbols=['AAPL', 'MSFT'],
    start=date(2020, 1, 1),
    end=date(2026, 1, 24),
    batch_size=10,
    progress_file=Path("~/hrp-data/backfill_ema_vwap_progress.json"),
)
# Returns: symbols_success, symbols_failed, failed_symbols, rows_inserted

# Or use CLI
python -m hrp.data.backfill --ema-vwap --universe --start 2020-01-01 --end 2026-01-24
```

### Generate time-series fundamentals
```python
from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries

# Create daily fundamental time-series with point-in-time correctness
result = backfill_fundamentals_timeseries(
    symbols=['AAPL', 'MSFT'],
    start=date(2023, 10, 1),
    end=date(2023, 12, 31),
    metrics=['revenue', 'eps', 'book_value'],
    source='yfinance',
)
# Returns: symbols_success, symbols_failed, failed_symbols, rows_inserted
# Features created: ts_revenue, ts_eps, ts_book_value

# Schedule weekly time-series fundamentals (Sunday 6 AM ET)
from hrp.agents.scheduler import IngestionScheduler
scheduler = IngestionScheduler()
scheduler.setup_weekly_fundamentals_timeseries(
    fundamentals_time='06:00',
    day_of_week='sun',
)
scheduler.start()
```

### Validate data before operations
```python
from hrp.data.quality.validation import DataValidator, validate_before_operation

# Validate price data before ingestion
prices_df = pd.DataFrame({
    "open": [180.0],
    "high": [182.0],
    "low": [178.0],
    "close": [180.5],
    "volume": [10000000],
})

validation = DataValidator.validate_price_data(prices_df)
if not validation.is_valid:
    print(f"Validation failed: {validation.errors}")

# Or use context manager for automatic validation
with validate_before_operation(
    DataValidator.validate_price_data,
    on_failure="raise",
    prices_df=prices_df,
):
    # Operation only executes if validation passes
    ingest_prices(...)

# Validate feature computation inputs
result = DataValidator.validate_feature_computation_inputs(
    prices_df=price_data,
    feature_name="momentum_20d",
    min_history_days=20,
)

# Validate universe health
result = DataValidator.validate_universe_data(
    symbols=['AAPL', 'MSFT'],
    as_of_date=date.today(),
    db_path=db_path,
    require_prices=True,
    max_staleness_days=3,
)
```

### Data retention policy
```python
from hrp.data.retention import RetentionEngine, DataCleanupJob

# Get retention tier for a date
engine = RetentionEngine()
tier = engine.get_tier_for_date('prices', date(2023, 1, 1))
# Returns: RetentionTier.COLD

# Get cleanup candidates
candidates = engine.get_cleanup_candidates(
    data_type='prices',
    as_of_date=date.today(),
)

# Estimate cleanup impact
impact = engine.estimate_cleanup_impact('prices')
print(f"Records to clean: {impact['total_records']}")

# Run cleanup job (dry-run mode first!)
job = DataCleanupJob(dry_run=True)
results = job.run()
print(f"Would delete: {sum(r.records_deleted for r in results.values())} records")

# Schedule weekly cleanup
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()
scheduler.setup_weekly_cleanup(
    cleanup_time='02:00',  # 2 AM ET Sunday
    dry_run=True,  # Always start with dry_run=True
)

# Retention tiers:
# - HOT (90d): Frequently accessed active data
# - WARM (1y): Recent historical data
# - COLD (3y): Long-term storage for backtesting
# - ARCHIVE (5y+): Compressed archival
```

### Track data lineage
```python
from hrp.data.lineage import FeatureLineage, DataProvenance, get_data_flow, get_feature_dependencies, get_impact_analysis

# Track feature computation
lineage = FeatureLineage()
lineage_id = lineage.track_computation(
    feature_name="momentum_20d",
    symbols=['AAPL', 'MSFT'],
    computation_date=date.today(),
    computation_source="batch",
    input_features=["close"],
    input_symbols=["AAPL", "MSFT"],
    computation_params={"window": 20},
    rows_computed=500,
    duration_ms=150.0,
)

# Get computation history
history = lineage.get_computation_history(feature_name="momentum_20d")
for record in history:
    print(f"{record.symbol}: {record.date} - {record.computation_source}")

# Get feature statistics
stats = lineage.get_feature_statistics("momentum_20d")
print(f"Unique symbols: {stats['unique_symbols']}")
print(f"Computation days: {stats['computation_days']}")

# Trace computation dependencies
chain = lineage.get_computation_chain(
    feature_name="momentum_20d",
    symbol="AAPL",
    computation_date=date.today(),
)
for step in chain:
    print(f"{step['feature']} <- {step['input_features']}")

# Track data provenance
provenance = DataProvenance()
provenance_id = provenance.track_source(
    data_type="prices",
    record_identifier="prices_AAPL_2024-01-15",
    source_system="yfinance",
    source_timestamp=datetime.now(),
    data_content={"close": 180.5, "volume": 10000000},
)

# Add transformation to history
provenance.add_transformation(
    provenance_id=provenance_id,
    transformation_type="outlier_removal",
    transformation_params={"method": "sigma_clip", "threshold": 3.0},
)

# Add quality check results
provenance.add_quality_check(
    provenance_id=provenance_id,
    check_name="validation",
    check_result={"passed": True, "checks_performed": 5},
)

# Verify data integrity with SHA-256 hash
is_valid = provenance.verify_integrity(
    provenance_id,
    data_content={"close": 180.5, "volume": 10000000},
)
print(f"Data integrity: {is_valid}")

# Query utilities
flow = get_data_flow("prices_AAPL_2024-01-15")
for step in flow:
    print(f"{step['source_system']} -> {step['transformations']}")

deps = get_feature_dependencies("momentum_20d")
print(f"Inputs: {deps['inputs']}")
print(f"Derived features: {deps['derived']}")

impact = get_impact_analysis({
    "symbol": "AAPL",
    "date": date(2024, 1, 15),
    "issue_type": "missing_data",
    "data_type": "prices",
})
print(f"Affected features: {impact['affected_features']}")
print(f"Remediation: {impact['remediation']}")
```

### Schedule daily data ingestion
```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()
scheduler.setup_daily_ingestion(
    symbols=['AAPL', 'MSFT'],  # None for all universe symbols
    price_job_time='18:00',    # 6 PM ET (after market close)
    universe_job_time='18:05', # 6:05 PM ET (after prices loaded)
    feature_job_time='18:10',  # 6:10 PM ET (after universe updated)
)
scheduler.start()
```

### Run a job manually
```python
from hrp.agents.jobs import PriceIngestionJob, FeatureComputationJob, UniverseUpdateJob, FundamentalsIngestionJob

job = PriceIngestionJob(symbols=['AAPL'], start=date.today() - timedelta(days=7))
result = job.run()  # Returns status, records_fetched, records_inserted

# Or update the universe
universe_job = UniverseUpdateJob()
result = universe_job.run()  # Fetches S&P 500 from Wikipedia, applies exclusions

# Or run fundamentals ingestion (revenue, EPS, book value with point-in-time correctness)
fundamentals_job = FundamentalsIngestionJob(symbols=['AAPL', 'MSFT'], source='yfinance')
result = fundamentals_job.run()  # Uses SimFin (primary) or YFinance (fallback)
```

### Schedule weekly fundamentals ingestion
```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()
scheduler.setup_weekly_fundamentals(
    fundamentals_time='10:00',  # Saturday 10 AM ET
    day_of_week='sat',
    source='simfin',  # or 'yfinance'
)
scheduler.start()
```

### Run Signal Scientist for automated signal discovery
```python
from hrp.agents import SignalScientist

# Run on-demand signal scan
agent = SignalScientist(
    symbols=None,  # None = all universe symbols
    features=None,  # None = all 44 features
    forward_horizons=[5, 10, 20],  # days
    ic_threshold=0.03,  # minimum IC to create hypothesis
    create_hypotheses=True,
)
result = agent.run()

print(f"Signals found: {result['signals_found']}")
print(f"Hypotheses created: {result['hypotheses_created']}")
print(f"MLflow run: {result['mlflow_run_id']}")
```

### Schedule weekly signal scan
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

### Enable event-driven agent coordination
```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()

# Set up all scheduled jobs
scheduler.setup_daily_ingestion()
scheduler.setup_weekly_signal_scan()

# Enable automatic agent chaining via lineage events:
# Signal Scientist → Alpha Researcher → Pipeline Orchestrator → ML Scientist → ML Quality Sentinel
scheduler.setup_research_agent_triggers(poll_interval_seconds=60)

# Start scheduler with event watcher
scheduler.start_with_triggers()
```

### Run Alpha Researcher for hypothesis review
```python
from hrp.agents import AlphaResearcher

# Reviews draft hypotheses using Claude API
# Promotes promising ones to 'testing' status
researcher = AlphaResearcher()
result = researcher.run()

print(f"Analyzed: {result['hypotheses_analyzed']}")
print(f"Promoted: {result['promoted_to_testing']}")
# Research note written to docs/research/YYYY-MM-DD-alpha-researcher.md
```

### Strategy Generation
```python
from hrp.agents import AlphaResearcher, AlphaResearcherConfig

# Enable strategy generation with 3 new concepts per run
config = AlphaResearcherConfig(
    enable_strategy_generation=True,
    generation_target_count=3,
    generation_sources=["claude_ideation", "literature_patterns", "pattern_mining"],
    write_strategy_docs=True,
    strategy_docs_dir="docs/strategies",
)

researcher = AlphaResearcher(config=config)
result = researcher.run()

print(f"Strategies generated: {result['strategies_generated']}")
print(f"Strategy docs written: {result['strategy_docs_written']}")
# Output: Strategies generated: 3, Strategy docs written: ["docs/strategies/post_earnings_drift.md", ...]
```

### Run ML Scientist for hypothesis validation
```python
from hrp.agents import MLScientist

# Validates hypotheses in 'testing' status using walk-forward validation
scientist = MLScientist(
    n_folds=5,
    window_type='expanding',
    stability_threshold=1.0,  # Lower is better
)
result = scientist.run()

print(f"Tested: {result['hypotheses_tested']}")
print(f"Validated: {result['hypotheses_validated']}")
print(f"Rejected: {result['hypotheses_rejected']}")
# Research note written to docs/research/YYYY-MM-DD-ml-scientist.md
```

### Run ML Quality Sentinel for experiment auditing
```python
from hrp.agents import MLQualitySentinel

# Audits recent experiments for overfitting signals
sentinel = MLQualitySentinel(
    audit_window_days=7,
    send_alerts=True,
)
result = sentinel.run()

print(f"Audited: {result['experiments_audited']}")
print(f"Critical issues: {result['critical_issues']}")
print(f"Warnings: {result['warnings']}")
# Checks: Sharpe decay, target leakage, feature count, fold stability
# Research note written to docs/research/YYYY-MM-DD-ml-quality-sentinel.md
```

### Run Validation Analyst for pre-deployment stress testing
```python
from hrp.agents import ValidationAnalyst

# Stress tests validated hypotheses before deployment
analyst = ValidationAnalyst(
    hypothesis_ids=["HYP-2026-001"],  # None = all audited hypotheses
    param_sensitivity_threshold=0.5,   # Min ratio of varied/baseline Sharpe
    min_profitable_periods=0.67,       # 2/3 of time periods profitable
    min_profitable_regimes=2,          # At least 2 of 3 regimes profitable
    send_alerts=True,
)
result = analyst.run()

print(f"Validated: {result['hypotheses_validated']}")
print(f"Passed: {result['hypotheses_passed']}")
print(f"Failed: {result['hypotheses_failed']}")
# Checks: Parameter sensitivity, time stability, regime stability, execution costs
# Research note written to docs/research/YYYY-MM-DD-validation-analyst.md
```

### Run Report Generator for automated research summaries
```python
from hrp.agents import ReportGenerator

# Generate daily research report
daily = ReportGenerator(report_type="daily")
result = daily.run()

print(f"Report type: {result['report_type']}")
print(f"Report saved: {result['report_path']}")
print(f"Token usage: {result['token_usage']}")
# Report saved to docs/reports/YYYY-MM-DD/HH-MM-daily.md

# Generate weekly research report
weekly = ReportGenerator(report_type="weekly")
result = weekly.run()
# Report saved to docs/reports/YYYY-MM-DD/HH-MM-weekly.md
```

### Schedule daily/weekly research reports
```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()

# Daily report at 7 AM ET
scheduler.setup_daily_report(report_time='07:00')

# Weekly report on Sunday at 8 PM ET
scheduler.setup_weekly_report(report_time='20:00')

scheduler.start()
```

### Use CIO Agent for hypothesis decision-making
```python
from hrp.agents import CIOAgent
from unittest.mock import patch

with patch("hrp.agents.cio.PlatformAPI"):
    agent = CIOAgent(job_id="cio-weekly-001", actor="agent:cio")

# Score a hypothesis across all 4 dimensions
experiment_data = {
    "sharpe": 1.5,
    "stability_score": 0.6,
    "mean_ic": 0.045,
    "fold_cv": 1.2,
}
risk_data = {
    "max_drawdown": 0.12,
    "volatility": 0.11,
    "regime_stable": True,
    "sharpe_decay": 0.30,
}
economic_data = {
    "thesis": "Strong momentum effect persists",
    "current_regime": "Bull Market",
    "black_box_count": 2,
    "uniqueness": "novel",
    "agent_reports": {},
}
cost_data = {
    "slippage_survival": "stable",
    "turnover": 0.25,
    "capacity": "high",
    "execution_complexity": "low",
}

score = agent.score_hypothesis(
    hypothesis_id="HYP-2026-001",
    experiment_data=experiment_data,
    risk_data=risk_data,
    economic_data=economic_data,
    cost_data=cost_data,
)

print(f"Decision: {score.decision}")  # CONTINUE, CONDITIONAL, KILL, or PIVOT
print(f"Total Score: {score.total:.2f}")  # 0.75+ for CONTINUE
print(f"Statistical: {score.statistical:.2f}")
print(f"Risk: {score.risk:.2f}")
print(f"Economic: {score.economic:.2f}")
print(f"Cost: {score.cost:.2f}")
```

### Run weekly CIO Agent review (scheduled)
```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()

# Schedule weekly CIO review (Friday 5 PM ET)
scheduler.setup_weekly_cio_review(
    review_time='17:00',  # 5 PM ET
    day_of_week='fri',
)

scheduler.start()

# The CIO Agent will:
# 1. Fetch all validated hypotheses from database
# 2. Score each across 4 dimensions (Statistical, Risk, Economic, Cost)
# 3. Generate decision (CONTINUE/CONDITIONAL/KILL/PIVOT)
# 4. Save to cio_decisions table with rationale
# 5. Generate markdown report in docs/reports/YYYY-MM-DD/
```

### Run Risk Manager for portfolio risk assessment
```python
from hrp.agents import RiskManager

# Assess all validated hypotheses with conservative risk limits
agent = RiskManager(
    max_drawdown=0.15,  # 15% max drawdown
    max_correlation=0.60,  # 0.60 max correlation with existing positions
    max_sector_exposure=0.25,  # 25% max sector exposure
    send_alerts=True,  # Send email for vetoes
)
result = agent.run()

print(f"Assessed: {result['hypotheses_assessed']}")
print(f"Passed: {result['hypotheses_passed']}")
print(f"Vetoed: {result['hypotheses_vetoed']}")
# Research note written to docs/research/YYYY-MM-DD-risk-manager.md

# Risk Manager performs:
# 1. Drawdown risk check - Max drawdown limits, drawdown duration
# 2. Concentration risk - Position diversification, sector exposure limits
# 3. Correlation check - Ensures new strategies add diversification value
# 4. Risk limits validation - Volatility, turnover, leverage checks
# 5. Independent veto - Can veto strategies but CANNOT approve deployment
# 6. Portfolio impact calculation - Assesses impact of adding to paper portfolio

# Risk Manager uses conservative institutional defaults:
# - MAX_MAX_DRAWDOWN = 0.20 (20%)
# - MAX_DRAWDOWN_DURATION_DAYS = 126 (6 months)
# - MAX_POSITION_CORRELATION = 0.70
# - MAX_SECTOR_EXPOSURE = 0.30 (30%)
# - MAX_SINGLE_POSITION = 0.10 (10%)
# - MIN_DIVERSIFICATION = 10 positions
# - TARGET_POSITIONS = 20 positions
```

### Pipeline Orchestrator
```python
from hrp.agents import PipelineOrchestrator, PipelineOrchestratorConfig

# Configure with kill gates
config = PipelineOrchestratorConfig(
    enable_early_kill=True,
    min_baseline_sharpe=0.5,
    max_train_sharpe=3.0,
    max_drawdown_threshold=0.30,
    max_feature_count=50,
)

orchestrator = PipelineOrchestrator(
    hypothesis_ids=["HYP-2026-001"],
    config=config,
)

result = orchestrator.run()

print(f"Processed: {result['report']['hypotheses_processed']}")
print(f"Killed by gates: {result['report']['hypotheses_killed']}")
print(f"Time saved: {result['report']['time_saved_seconds']:.0f}s")
```


### Run a multi-factor strategy backtest
```python
from hrp.research.strategies import generate_multifactor_signals
from hrp.research.backtest import get_price_data, run_backtest
from hrp.research.config import BacktestConfig

# Load prices
prices = get_price_data(['AAPL', 'MSFT', 'GOOGL'], start_date, end_date)

# Generate signals: combine momentum (positive) and volatility (negative)
signals = generate_multifactor_signals(
    prices,
    feature_weights={
        "momentum_20d": 1.0,    # Favor high momentum
        "volatility_60d": -0.5,  # Penalize high volatility
    },
    top_n=10,
)

# Run backtest
config = BacktestConfig(symbols=['AAPL', 'MSFT', 'GOOGL'], start_date=start_date, end_date=end_date)
result = run_backtest(signals, config, prices)
```

### Run an ML-predicted strategy backtest
```python
from hrp.research.strategies import generate_ml_predicted_signals

# Generate signals using ML model predictions
signals = generate_ml_predicted_signals(
    prices,
    model_type="ridge",  # ridge, lasso, random_forest, lightgbm, xgboost
    features=["momentum_20d", "volatility_60d"],
    signal_method="rank",  # rank, threshold, zscore
    top_pct=0.1,           # Top 10% for rank method
    train_lookback=252,    # 1 year training window
    retrain_frequency=21,  # Monthly retraining
)

result = run_backtest(signals, config, prices)
```

### Run walk-forward validation
```python
from hrp.ml import WalkForwardConfig, walk_forward_validate
from datetime import date

config = WalkForwardConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_20d', 'rsi_14d'],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    window_type='expanding',  # or 'rolling'
    feature_selection=True,
    max_features=20,
    n_jobs=-1,  # Use all cores for parallel fold processing (3-4x speedup)
)

result = walk_forward_validate(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    log_to_mlflow=True,
)

# Check results
print(f"Stability Score: {result.stability_score:.4f}")  # Lower is better
print(f"Mean IC: {result.mean_ic:.4f}")  # Information coefficient
print(f"Model is stable: {result.is_stable}")  # stability_score <= 1.0

# Per-fold results
for fold in result.fold_results:
    print(f"Fold {fold.fold_index}: IC={fold.metrics['ic']:.4f}, "
          f"MSE={fold.metrics['mse']:.6f}")
```

### Use overfitting guards
```python
from hrp.risk import TestSetGuard, validate_strategy, check_parameter_sensitivity

# Test set discipline (limits to 3 evaluations per hypothesis)
guard = TestSetGuard(hypothesis_id='HYP-2026-001')

with guard.evaluate(metadata={"experiment": "final_validation"}):
    metrics = model.evaluate(test_data)

print(f"Evaluations remaining: {guard.remaining_evaluations}")

# Validate strategy meets minimum criteria
result = validate_strategy({
    "sharpe": 0.80,
    "num_trades": 200,
    "max_drawdown": 0.18,
    "win_rate": 0.52,
})

if result.passed:
    print(f"✅ Validation passed! Confidence: {result.confidence_score:.2f}")
else:
    print(f"❌ Failed: {result.failed_criteria}")

# Check parameter robustness
experiments = {
    "baseline": {"sharpe": 0.80, "params": {"lookback": 20}},
    "var_1": {"sharpe": 0.75, "params": {"lookback": 16}},
    "var_2": {"sharpe": 0.82, "params": {"lookback": 24}},
}

robustness = check_parameter_sensitivity(experiments, baseline_key="baseline")
print(f"Parameter stability: {'✅ PASS' if robustness.passed else '❌ FAIL'}")

# Sharpe decay monitoring (detect train/test overfitting)
from hrp.risk.overfitting import SharpeDecayMonitor

monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
result = monitor.check(train_sharpe=1.5, test_sharpe=1.0)
if not result.passed:
    print(f"Sharpe decay warning: {result.message}")

# Hyperparameter trial tracking (limit HP search)
from hrp.risk.overfitting import HyperparameterTrialCounter

counter = HyperparameterTrialCounter(hypothesis_id='HYP-2026-001', max_trials=50)
counter.log_trial(
    model_type='ridge',
    hyperparameters={'alpha': 1.0},
    metric_name='val_r2',
    metric_value=0.85,
)
print(f"HP trials remaining: {counter.remaining_trials}")
best = counter.get_best_trial()

# Feature count validation (prevent overfitting from too many features)
from hrp.risk.overfitting import FeatureCountValidator

validator = FeatureCountValidator(warn_threshold=30, max_threshold=50)
result = validator.check(feature_count=25, sample_count=1000)
if not result.passed:
    raise OverfittingError(result.message)

# Target leakage detection (catch data leakage before training)
from hrp.risk.overfitting import TargetLeakageValidator

leakage_validator = TargetLeakageValidator(correlation_threshold=0.95)
result = leakage_validator.check(features_df, target_series)
if not result.passed:
    print(f"Leakage detected: {result.suspicious_features}")
```

### Run cross-validated optimization
```python
from hrp.ml import OptimizationConfig, cross_validated_optimize
from datetime import date

config = OptimizationConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_20d', 'rsi_14d'],
    param_grid={'alpha': [0.1, 1.0, 10.0, 100.0]},
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    scoring_metric='ic',  # ic, mse, or sharpe
    max_trials=50,  # Integrates with HyperparameterTrialCounter
    hypothesis_id='HYP-2026-001',
)

result = cross_validated_optimize(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    log_to_mlflow=True,
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### Run parameter sweep with Sharpe decay analysis
```python
from hrp.research.parameter_sweep import SweepConfig, parallel_parameter_sweep
from datetime import date

config = SweepConfig(
    param_grid={
        "fast_period": [10, 15, 20, 25, 30],
        "slow_period": [20, 30, 40, 50, 60],
    },
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date=date(2018, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=3,
    strategy_type="momentum_crossover",
    constraints=[
        # Ensure slow_period > fast_period by at least 5
        {"type": "difference_min", "params": ["slow_period", "fast_period"], "min_diff": 5}
    ],
    n_jobs=-1,  # Use all CPU cores
)

result = parallel_parameter_sweep(config)

# Sharpe decay analysis: Blue = good, Red = overfitting
print(f"Generalization Score: {result.generalization_score:.1%}")
print(f"Best params: {result.best_params}")

# Access Sharpe decay matrix for visualization
# result.sharpe_diff_matrix: test_sharpe - train_sharpe per fold
# result.sharpe_diff_median: aggregated across folds
```

### ML Model Registry & Deployment
```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# Register a trained model in the Model Registry
model_version = api.register_model(
    model=trained_model,
    model_name="momentum_strategy",
    model_type="ridge",
    features=["momentum_20d", "volatility_60d"],
    target="returns_20d",
    metrics={"sharpe": 0.85, "ic": 0.07},
    hyperparameters={"alpha": 1.0},
    training_date=date.today(),
    hypothesis_id="HYP-2026-001",
)
# Returns: "1" (model version)

# Get current production model
prod_model = api.get_production_model("momentum_strategy")
print(f"Production version: {prod_model['model_version']}")

# Deploy model to staging (with validation checks)
result = api.deploy_model(
    model_name="momentum_strategy",
    model_version="1",
    validation_data=validation_df,
    environment="staging",
    actor="user",
)
# Returns: {"status": "success", "validation_passed": True, ...}

# Promote to production (with shadow mode)
result = api.deploy_model(
    model_name="momentum_strategy",
    model_version="1",
    validation_data=validation_df,
    environment="production",
)
# Returns: {"status": "success", ...}

# Rollback if needed
api.rollback_model(
    model_name="momentum_strategy",
    to_version="1",
    actor="user",
    reason="Performance degradation",
)
```

### ML Production Drift Monitoring
```python
from hrp.api.platform import PlatformAPI

api = PlatformAPI()

# Check for model drift (KL divergence, PSI, IC decay)
drift_results = api.check_model_drift(
    model_name="momentum_strategy",
    current_data=current_features_df,
    reference_data=historical_features_df,
    predictions_col="prediction",
    target_col="returns_20d",
    reference_ic=0.07,  # IC from training
)

# Check summary
print(f"Drift detected: {drift_results['summary']['drift_detected']}")
print(f"Checks performed: {drift_results['summary']['total_checks']}")

# Individual drift results
for check_name, result in drift_results.items():
    if check_name != "summary":
        print(f"{check_name}: drift={result['is_drift_detected']}, value={result['metric_value']:.4f}")

# Alert thresholds:
# - KL divergence > 0.2 (prediction drift)
# - PSI > 0.2 (feature drift)
# - IC decay > 0.2 (concept drift)
```

### ML Model Inference
```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# Generate predictions using deployed model
predictions = api.predict_model(
    model_name="momentum_strategy",
    symbols=["AAPL", "MSFT", "GOOGL"],
    as_of_date=date.today(),
    # model_version=None  # None for production model
)
# Returns DataFrame: symbol, date, prediction, model_name, model_version

# Retrieve historical predictions
history = api.get_model_predictions(
    model_name="momentum_strategy",
    start_date=date(2026, 1, 1),
    end_date=date(2026, 1, 31),
)
# Returns DataFrame with prediction metrics over time
```

### Purge/Embargo Periods in Walk-Forward Validation
```python
from hrp.ml import WalkForwardConfig, walk_forward_validate
from datetime import date

# Add purge/embargo to prevent temporal leakage
config = WalkForwardConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_60d'],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    purge_days=5,      # Gap between train and test (execution lag)
    embargo_days=10,   # Initial test period excluded (implementation delay)
)

result = walk_forward_validate(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
)

# Purge: prevents look-ahead bias from execution lag
# Embargo: accounts for implementation delay
# Both default to 0 (backward compatible with existing behavior)
```

### Configure ATR trailing stops
```python
from hrp.research.config import BacktestConfig, StopLossConfig
from hrp.research.stops import apply_trailing_stops

# Add stop-loss to backtest config
stop_config = StopLossConfig(
    enabled=True,
    type="atr_trailing",  # "fixed_pct", "atr_trailing", "volatility_scaled"
    atr_multiplier=2.0,
    atr_period=14,
)

config = BacktestConfig(
    symbols=['AAPL', 'MSFT'],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    stop_loss=stop_config,
)

# Or apply trailing stops to signals manually
from hrp.research.stops import apply_trailing_stops

modified_signals, stop_events = apply_trailing_stops(
    signals=signals,
    prices=prices,
    stop_config=stop_config,
)
```

### HMM regime detection
```python
from hrp.ml import HMMConfig, RegimeDetector, MarketRegime

# Configure and fit regime detector
config = HMMConfig(
    n_regimes=3,
    features=['returns_20d', 'volatility_20d'],
    covariance_type='full',
)

detector = RegimeDetector(config)
detector.fit(prices)

# Predict regimes
regimes = detector.predict(prices)  # Series of regime labels

# Get regime statistics
result = detector.get_regime_statistics(prices)
print(f"Current regime: {result.current_regime}")
print(f"Regime durations: {result.regime_durations}")

# Check strategy stability across regimes
from hrp.risk.robustness import check_regime_stability_hmm

stability = check_regime_stability_hmm(
    returns=strategy_returns,
    prices=prices,
    strategy_metrics_by_date=metrics_df,
    n_regimes=3,
    min_regimes_profitable=2,
)
print(f"Regime stability: {'PASS' if stability.passed else 'FAIL'}")
```

## File Locations

- Database: `~/hrp-data/hrp.duckdb`
- MLflow: `~/hrp-data/mlflow/`
- Logs: `~/hrp-data/logs/`

## Testing

```bash
pytest tests/ -v
# Pass rate: 99.85% (2,661 passed, 4 failed, 1 skipped)
```

## Performance Metrics (Empyrical-powered)

Backtests return comprehensive metrics via `calculate_metrics()`:

| Category | Metrics |
|----------|---------|
| **Returns** | `total_return`, `cagr` |
| **Risk** | `volatility`, `downside_volatility`, `max_drawdown` |
| **Risk-Adjusted** | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` |
| **Trade Stats** | `win_rate`, `avg_win`, `avg_loss`, `profit_factor` |
| **Benchmark** | `alpha`, `beta`, `tracking_error`, `information_ratio` |
| **Tail Risk** | `value_at_risk`, `conditional_value_at_risk` |
| **Advanced** | `omega_ratio`, `tail_ratio`, `stability` |

```python
from hrp.research.metrics import calculate_metrics, format_metrics

metrics = calculate_metrics(returns, benchmark_returns, risk_free_rate=0.05)
print(format_metrics(metrics))
```

## Services

| Service | Command | Port |
|---------|---------|------|
| Dashboard | `streamlit run hrp/dashboard/app.py` | 8501 |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///~/hrp-data/mlflow/mlflow.db` | 5000 |
| Scheduler | `python -m hrp.agents.run_scheduler` | - |
| Scheduler (full) | `python -m hrp.agents.run_scheduler --with-research-triggers --with-signal-scan --with-quality-sentinel --with-daily-report --with-weekly-report` | - |

### Scheduler CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--with-research-triggers` | off | Enable event-driven agent pipeline |
| `--trigger-poll-interval` | 60 | Lineage event poll interval (seconds) |
| `--with-signal-scan` | off | Enable weekly signal scan (Monday 7 PM ET) |
| `--signal-scan-time` | 19:00 | Time for signal scan (HH:MM) |
| `--signal-scan-day` | mon | Day for signal scan |
| `--ic-threshold` | 0.03 | Minimum IC to create hypothesis |
| `--with-quality-sentinel` | off | Enable daily ML Quality Sentinel (6 AM ET) |
| `--sentinel-time` | 06:00 | Time for quality sentinel |
| `--with-daily-report` | off | Enable daily research report (7 AM ET) |
| `--daily-report-time` | 07:00 | Time for daily report (HH:MM) |
| `--with-weekly-report` | off | Enable weekly research report (Sunday 8 PM ET) |
| `--weekly-report-time` | 20:00 | Time for weekly report (HH:MM) |

### Scheduler Management (launchd)

```bash
# Check status
launchctl list | grep hrp

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Stop scheduler
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Start scheduler
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Restart after config changes
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist && \
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HRP_DB_PATH` | Database path (default: `~/hrp-data/hrp.duckdb`) | No |
| `RESEND_API_KEY` | Resend API key for email notifications | For alerts |
| `NOTIFICATION_EMAIL` | Email address for notifications | For alerts |
| `NOTIFICATION_FROM_EMAIL` | From address (default: `noreply@hrp.local`) | No |
| `SIMFIN_API_KEY` | SimFin API key for fundamentals (falls back to YFinance) | For fundamentals |

## Current Scope

- Universe: S&P 500 (excluding financials, REITs, penny stocks)
- Direction: Long-only
- Timeframe: Daily
- Broker: Interactive Brokers

## Project Structure

```
hrp/
├── api/            # Platform API (single entry point)
├── data/           # Data layer (DuckDB, ingestion, features)
├── research/       # Research engine (backtest, hypothesis, lineage, strategies)
├── ml/             # ML framework (training, validation, signals)
├── risk/           # Risk management (limits, validation)
├── dashboard/      # Streamlit dashboard
│   └── components/ # Reusable UI components (strategy config)
├── mcp/            # Claude MCP servers
├── agents/         # Scheduled agents
├── notifications/  # Email alerts
├── execution/      # Live trading, broker integration (Tier 4)
├── monitoring/     # System health, ops alerting (Tier 3)
└── utils/          # Shared utilities
```

## Where Does New Code Go?

| Adding... | Put it in... | Example |
|-----------|--------------|---------|
| New data provider | `hrp/data/sources/` | `simfin_source.py` |
| New ingestion pipeline | `hrp/data/ingestion/` | `fundamentals.py` |
| New computed feature | `hrp/data/features/definitions.py` + `computation.py` | `pe_ratio` |
| New strategy/signal type | `hrp/research/strategies/` | pairs trading signals |
| New ML model type | `hrp/ml/` | transformer model |
| New risk check | `hrp/risk/` | correlation limits |
| New dashboard page | `hrp/dashboard/pages/` | execution monitor |
| New scheduled job | `hrp/agents/jobs.py` | weekly rebalance |
| Expose via API | `hrp/api/platform.py` | new method |
| Live trading (future) | `hrp/execution/` | order manager |
| System monitoring (future) | `hrp/monitoring/` | uptime checks |

**The Rule:** Data layer modules access `hrp/data/db.py` directly. Everything else goes through `hrp/api/platform.py`.

## Development Status

| Tier | Focus | Status |
|------|-------|--------|
| **Foundation** | Data + Research Core | ✅ 100% |
| **Intelligence** | ML + Agents | 🟡 90% |
| **Production** | Security + Ops | ⏳ 0% |
| **Trading** | Live Execution | 🔮 0% |

See `docs/plans/Project-Status.md` for detailed tier-based status

# Project Specific Guidelines

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.










