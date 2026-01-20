# HRP - Hedgefund Research Platform

## Project Overview

Personal quantitative research platform for systematic trading strategy development.
Long-only US equities, daily timeframe, institutional rigor.

## Architecture

Three-layer architecture:
1. **Data Layer** - DuckDB storage, ingestion pipelines, feature store
2. **Research Layer** - VectorBT backtesting, MLflow experiments, hypothesis registry
3. **Control Layer** - Streamlit dashboard, MCP servers, scheduled agents

All external access goes through `hrp/api/platform.py`. Never access DuckDB directly.

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
- All database access through `hrp/api/platform.py`
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
