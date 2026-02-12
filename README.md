# HRP - Hedgefund Research Platform

Personal, professional-grade quantitative research platform for systematic trading strategy development.

## Features

- **Research-First Workflow** - Formal hypothesis → experiment → validation pipeline
- **Institutional Rigor** - Walk-forward validation, statistical significance testing, audit trails
- **ML-Ready** - Ridge, Lasso, ElasticNet, RandomForest, LightGBM with MLflow tracking
- **10-Agent Pipeline** - Automated signal discovery through CIO scoring
- **Multi-Broker Trading** - IBKR and Robinhood with VaR-aware position sizing
- **Real-Time Data** - Polygon WebSocket intraday ingestion with 7 computed features
- **Performance Attribution** - Brinson-Fachler, Fama-French, SHAP feature importance
- **NLP Sentiment** - SEC EDGAR filing analysis via Claude API
- **Agent-Native** - Claude integration via MCP for AI-assisted research
- **Local-First** - Runs entirely on your Mac, data stays private

## Quick Start

### Prerequisites

- Python 3.11+
- macOS (tested on Apple Silicon)
- Homebrew (for system dependencies)

### Installation

```bash
# Clone the repository
git clone https://github.com/fmag-labs/HRP.git
cd HRP

# Run the interactive setup script
./scripts/setup.sh
```

The setup script walks you through 11 phases:

| Phase | What it does |
|-------|-------------|
| Pre-flight | Checks OS, Python >=3.11, detects uv/Homebrew |
| System Deps | Installs libomp (LightGBM/XGBoost) |
| Python Env | Creates venv, installs dependencies |
| Directories | Creates `~/hrp-data/` structure |
| .env Config | Interactive API key / environment setup |
| Database | Initializes DuckDB schema |
| Fix Configs | Updates `.mcp.json` and launchd plist paths |
| Auth | Creates dashboard login user |
| Data Bootstrap | Loads universe + 2 years of prices/features for top 20 stocks |
| Launchd | Optional: installs scheduled jobs |
| Verification | Runs all checks, prints PASS/FAIL summary |

Safe to re-run. Use `./scripts/setup.sh --check` for verification only.

### Manual Installation

<details>
<summary>If you prefer manual setup over the interactive script</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys — see .env.example for all ~50 configurable variables
mkdir -p ~/hrp-data/{mlflow,logs,auth,backups,cache,optuna,output,config}
python -m hrp.data.schema --init
pytest tests/ -v
```

</details>

### Bootstrap Data

The setup script bootstraps 2 years of daily price data and 45 computed features for the **top 20 most traded S&P 500 stocks** (~2-5 minutes, no API key required — uses Yahoo Finance):

```
AAPL  MSFT  NVDA  AMZN  META  TSLA  GOOGL  GOOG  AMD  AVGO
NFLX  COST  ADBE  CRM   PEP   CSCO  INTC   QCOM  TMUS INTU
```

To load the full S&P 500 universe (~400 stocks) after setup:

```bash
python -m hrp.agents.run_job --job prices     # ~20-30 min
python -m hrp.agents.run_job --job features
```

### Enabling Automated Reports

The bootstrap loads data but does **not** start scheduled agents. To get the full research pipeline running with automated reports:

1. **Add API keys to `.env`:**
   - `ANTHROPIC_API_KEY` — powers Claude-based agents (Signal Scientist, Alpha Researcher, CIO, Report Generator)
   - `RESEND_API_KEY` + `NOTIFICATION_EMAIL` — delivers reports via email

2. **Start the scheduler with agents:**
   ```bash
   ./scripts/startup.sh start --full        # scheduler + all research agents
   # or install as background services:
   ./scripts/manage_launchd.sh install       # launchd jobs (macOS)
   ```

3. **Pipeline flow:**
   ```
   Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel
   → Quant Developer → Kill Gate Enforcer → Validation Analyst → Risk Manager
   → CIO Agent → Report Generator → email
   ```

Without `ANTHROPIC_API_KEY`, Claude-powered agents (Alpha Researcher, CIO, Report Generator) will not run.

### Running Services

```bash
# Start all services (dashboard, MLflow, scheduler)
./scripts/startup.sh start

# Or individually
./scripts/startup.sh start --dashboard-only   # http://localhost:8501
./scripts/startup.sh start --mlflow-only       # http://localhost:5010

# Check status / stop
./scripts/startup.sh status
./scripts/startup.sh stop
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
| Timeframe | Daily + Intraday (minute bars) | Sub-second |
| Universe | S&P 500 (ex-financials, REITs) | International |
| Broker | Interactive Brokers, Robinhood | Others |

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

config = WalkForwardConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_20d', 'rsi_14d'],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    window_type='expanding',
    feature_selection=True,
    max_features=20,
)

result = walk_forward_validate(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    log_to_mlflow=True,
)

print(f"Stability Score: {result.stability_score:.4f}")
print(f"Mean IC: {result.mean_ic:.4f}")
print(f"Model is stable: {result.is_stable}")
```

## Documentation

- [Project Status](docs/plans/Project-Status.md) - Development roadmap and tier status
- [Cookbook](docs/operations/cookbook.md) - Practical guide with examples
- [Decision Pipeline](docs/agents/decision-pipeline.md) - Agent architecture and signal-to-deployment flow
- [State Machine](docs/agents/state-machine-transitions.md) - Hypothesis lifecycle documentation
- [Scheduler Guide](docs/setup/Scheduler-Configuration-Guide.md) - launchd job configuration
- [VaR Risk Metrics](docs/operations/var-risk-metrics.md) - VaR/CVaR calculator and dashboard
- [Trading Setup](docs/operations/tier4-trading-setup.md) - IBKR and Robinhood broker configuration
- [Deployment Guide](docs/operations/deployment.md) - Production deployment procedures

## Development Status

| Tier | Status | Description |
|------|--------|-------------|
| Tier 1: Foundation | Complete | Data + Research Core |
| Tier 2: Intelligence | Complete | ML + Agents + NLP Sentiment |
| Tier 3: Production | Complete | Security + Ops + Setup Script |
| Tier 4: Trading | Complete | Live Execution (IBKR + Robinhood) |
| Tier 5: Advanced Analytics | Complete | VaR/CVaR, Attribution, Real-time Data |

### Research Agents (10 Implemented)

| Agent | Purpose |
|-------|---------|
| Signal Scientist | Automated IC analysis and hypothesis creation |
| Alpha Researcher | Claude-powered hypothesis review |
| ML Scientist | Walk-forward validation and model training |
| ML Quality Sentinel | Experiment auditing and overfitting detection |
| Quant Developer | Production backtesting with costs |
| Kill Gate Enforcer | End-to-end pipeline with kill gates |
| Validation Analyst | Pre-deployment stress testing |
| Risk Manager | Independent portfolio risk oversight with veto authority |
| CIO Agent | Strategic 4-dimension hypothesis scoring |
| Report Generator | Automated daily/weekly research summaries |

**Pipeline:** Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel → Quant Developer → Kill Gate Enforcer → Validation Analyst → Risk Manager → CIO Agent → **Human CIO**

### Dashboard Pages (13)

| Page | Purpose |
|------|---------|
| Home | System status, recent activity |
| Data Health | Quality scores, anomalies, trends |
| Ingestion Status | Job history, source status |
| Hypotheses | Create, view, update hypotheses |
| Experiments | MLflow integration, comparison |
| Pipeline Progress | Kanban view of hypothesis pipeline |
| Agents Monitor | Real-time agent status and timeline |
| Job Health | Job execution health, error tracking |
| Ops | CPU/memory/disk, alert thresholds |
| Trading | Portfolio, positions, trades, drift |
| Risk Metrics | VaR/CVaR analysis, breach tracking |
| Performance Attribution | Brinson-Fachler, factor contributions, SHAP |
| Backtest Performance | Equity curves, drawdowns, exports |

## Testing

```bash
pytest tests/ -v    # 3,193 tests
```

## License

MIT
