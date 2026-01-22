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
api.create_hypothesis(
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

## File Locations

- Database: `~/hrp-data/hrp.duckdb`
- MLflow: `~/hrp-data/mlflow/`
- Logs: `~/hrp-data/logs/`

## Testing

```bash
pytest tests/ -v
```

## Services

| Service | Command | Port |
|---------|---------|------|
| Dashboard | `streamlit run hrp/dashboard/app.py` | 8501 |
| MLflow UI | `mlflow ui --backend-store-uri ~/hrp-data/mlflow/mlflow.db` | 5000 |
| Scheduler | `python -m hrp.agents.cli start` | - |

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
├── research/       # Research engine (backtest, hypothesis, lineage)
├── ml/             # ML framework (training, validation)
├── risk/           # Risk management (limits, validation)
├── dashboard/      # Streamlit dashboard
├── mcp/            # Claude MCP servers
├── agents/         # Scheduled agents
├── notifications/  # Email alerts
└── utils/          # Shared utilities
```

## Development Status

Currently implementing Phase 0: Foundation
- See `docs/plans/2025-01-19-hrp-spec.md` for full specification
- See `docs/plans/Roadmap.md` for implementation roadmap
