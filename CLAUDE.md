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
result = api.run_quality_checks(as_of_date=date.today(), send_alerts=True)
# Returns: health_score, critical_issues, warning_issues, passed
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
# Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel
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
# Pass rate: 100% (2,101 passed, 18 skipped)
```

## Services

| Service | Command | Port |
|---------|---------|------|
| Dashboard | `streamlit run hrp/dashboard/app.py` | 8501 |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///~/hrp-data/mlflow/mlflow.db` | 5000 |
| Scheduler | `python -m hrp.agents.run_scheduler` (or use launchd - see cookbook) | - |

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

# Project Instructions

## Always run tests after edits
after making any code changes, run 'pytest tests/ -v'

## Things to Remember
Before writing any code:
1. state how you will verify this change works (test, bash command, browser check, etc.)
2. Write the test or verification step first
3. Then implement the code
4. Run verification and iterate until it passes




