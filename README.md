# HRP - Hedgefund Research Platform

Personal, professional-grade quantitative research platform for systematic trading strategy development.

## Features

- **Research-First Workflow** - Formal hypothesis → experiment → validation pipeline
- **Institutional Rigor** - Walk-forward validation, statistical significance testing, audit trails
- **ML-Ready** - LightGBM, XGBoost, sklearn integration with MLflow tracking
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
mlflow ui --backend-store-uri sqlite:///~/hrp-data/mlflow/mlflow.db
# Open http://localhost:5000
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
| Deployment | Paper trading | Live trading (future) |

## Documentation

- [Full Specification](docs/plans/2025-01-19-hrp-spec.md)
- [Implementation Roadmap](docs/plans/Roadmap.md)

## Development Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0: Foundation | ✅ Complete | Repository, schema, data ingestion |
| Phase 1: Core Research | ✅ Complete | Backtesting, MLflow, metrics |
| Phase 2: Hypothesis | ✅ Complete | Hypothesis registry, lineage |
| Phase 3: Dashboard | ✅ Complete | Streamlit UI |

### Current Data
- **10 symbols loaded**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, V, UNH, JNJ
- **17,720 price records** (2019-2025)
- **Database**: ~/hrp-data/hrp.duckdb

## License

MIT
