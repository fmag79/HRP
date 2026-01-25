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
experiment_id = api.run_backtest(config, hypothesis_id='HYP-2025-001')
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
    feature_job_time='18:10',  # 6:10 PM ET (after prices loaded)
)
scheduler.start()
```

### Run a job manually
```python
from hrp.agents.jobs import PriceIngestionJob, FeatureComputationJob

job = PriceIngestionJob(symbols=['AAPL'], start=date.today() - timedelta(days=7))
result = job.run()  # Returns status, records_fetched, records_inserted
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
guard = TestSetGuard(hypothesis_id='HYP-2025-001')

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
```

## File Locations

- Database: `~/hrp-data/hrp.duckdb`
- MLflow: `~/hrp-data/mlflow/`
- Logs: `~/hrp-data/logs/`

## Testing

```bash
pytest tests/ -v
# Pass rate: ~86% (902 passed, 141 failed, 105 errors)
# Known issue: FK constraint violations in test fixtures during cleanup
```

## Services

| Service | Command | Port |
|---------|---------|------|
| Dashboard | `streamlit run hrp/dashboard/app.py` | 8501 |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///~/hrp-data/mlflow/mlflow.db` | 5000 |
| Scheduler | `python run_scheduler.py` (or use launchd - see cookbook) | - |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HRP_DB_PATH` | Database path (default: `~/hrp-data/hrp.duckdb`) | No |
| `RESEND_API_KEY` | Resend API key for email notifications | For alerts |
| `NOTIFICATION_EMAIL` | Email address for notifications | For alerts |
| `NOTIFICATION_FROM_EMAIL` | From address (default: `noreply@hrp.local`) | No |

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
└── utils/          # Shared utilities
```

## Development Status

Currently implementing Phase 0: Foundation
- See `docs/plans/2025-01-19-hrp-spec.md` for full specification
- See `docs/plans/Roadmap.md` for implementation roadmap

# Project Instructions

## Always run tests after edits
after making any code changes, run 'npm test'

## Things to Remember
Before writing any code:
1. state how you will verify this change works (test, bash command, browser check, etc.)
2. Write the test or verification step first
3. Then implement the code
4. Run verification and iterate until it passes