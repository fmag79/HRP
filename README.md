# HRP - Hedgefund Research Platform

Personal, professional-grade quantitative research platform for systematic trading strategy development.

## Features

- **Research-First Workflow** - Formal hypothesis → experiment → validation pipeline
- **Institutional Rigor** - Walk-forward validation, statistical significance testing, audit trails
- **ML-Ready** - Ridge, Lasso, ElasticNet, RandomForest, LightGBM with MLflow tracking
- **Agent-Native** - Claude integration via MCP for AI-assisted research
- **Local-First** - Runs entirely on your Mac, data stays private

## Quick Start

### Prerequisites

- Python 3.11+
- macOS (tested on M4 Mac Mini)

### Installation

```bash
# Clone the repository
git clone https://github.com/fmag79/hrp.git
cd hrp

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Create data directory
mkdir -p ~/hrp-data/{mlflow,logs,backups,cache}

# Initialize database
python -m hrp.data.schema --init

# Run tests
pytest tests/ -v
```

### Running the Dashboard

```bash
streamlit run hrp/dashboard/app.py
# Open http://localhost:8501
```

### Running MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///$HOME/hrp-data/mlflow/mlflow.db --port 5010
# Open http://localhost:5010
```

### Running the Scheduler

```bash
# Foreground (testing)
python run_scheduler.py

# Background service (production - macOS)
# See docs/cookbook.md section 7.2 for full launchd setup
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROL LAYER                                │
│   Streamlit Dashboard │ MCP Servers │ Scheduled Agents          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Platform API       │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                    RESEARCH LAYER                               │
│   VectorBT (Backtest) │ MLflow (Experiments) │ Hypothesis Reg   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                    DATA LAYER                                   │
│   DuckDB (Storage) │ Ingestion Pipelines │ Feature Store        │
└─────────────────────────────────────────────────────────────────┘
```

## Current Scope

| Dimension | In Scope | Out of Scope |
|-----------|----------|--------------|
| Asset class | US equities | ETFs, crypto, futures |
| Direction | Long-only | Short selling |
| Timeframe | Daily | Intraday |
| Universe | S&P 500 (ex-financials, REITs) | International |
| Deployment | Research only | Paper/live trading (future) |

## Usage Examples

### Running a Backtest

```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# Create hypothesis
hypothesis_id = api.create_hypothesis(
    title="Momentum predicts returns",
    thesis="Stocks with high 12-month momentum continue outperforming",
    prediction="Top decile momentum > SPY by 3% annually",
    falsification="Sharpe < SPY or p-value > 0.05",
    actor='user'
)

# Run backtest
experiment_id = api.run_backtest(
    config={
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
    },
    hypothesis_id=hypothesis_id
)
```

### Walk-Forward Validation

```python
from hrp.ml import WalkForwardConfig, walk_forward_validate

# Configure walk-forward validation
config = WalkForwardConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_20d', 'rsi_14d'],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    window_type='expanding',  # Train on increasing data
    feature_selection=True,
    max_features=20,
)

# Run validation
result = walk_forward_validate(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    log_to_mlflow=True,
)

# Analyze results
print(f"Stability Score: {result.stability_score:.4f}")
print(f"Mean IC: {result.mean_ic:.4f}")
print(f"Model is stable: {result.is_stable}")
```

## Documentation

- [Project Status](docs/plans/Project-Status.md)
- [Cookbook](docs/operations/cookbook.md) - Practical guide with examples
- [Decision Pipeline](docs/agents/decision-pipeline.md) - Agent architecture and signal-to-deployment flow
- [Deployment Guide](docs/operations/deployment.md) - Scheduler & service setup

## Development Status

| Tier | Status | Description |
|------|--------|-------------|
| Tier 1: Foundation | ✅ Complete | Data + Research Core (100%) |
| Tier 2: Intelligence | ✅ Complete | ML + Agents (100%) |
| Tier 3: Production | ⏳ Planned | Security + Ops (0%) |
| Tier 4: Trading | 🔮 Future | Live Execution (0%) |

### Research Agents (11 Implemented)
- Signal Scientist - Automated IC analysis and hypothesis creation
- Alpha Researcher - Claude-powered hypothesis review
- Code Materializer - Strategy code generation
- ML Scientist - Walk-forward validation and model training
- ML Quality Sentinel - Experiment auditing and overfitting detection
- Validation Analyst - Pre-deployment stress testing
- Risk Manager - Independent portfolio risk oversight with veto authority
- Quant Developer - Hyperparameter sweep and optimization
- Pipeline Orchestrator - End-to-end pipeline coordination with kill gates
- CIO Agent - Strategic 4-dimension hypothesis scoring
- Report Generator - Automated daily/weekly research summaries

## License

MIT
