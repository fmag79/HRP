# HRP - Hedgefund Research Platform

## Project Overview

Personal quantitative research platform for systematic trading strategy development.
Long-only US equities, daily timeframe, institutional rigor.

- Universe: S&P 500 (excluding financials, REITs, penny stocks)
- Broker: Interactive Brokers
- Database: DuckDB at `~/hrp-data/hrp.duckdb`
- MLflow: `~/hrp-data/mlflow/`
- Logs: `~/hrp-data/logs/`

## Architecture

Three-layer architecture:
1. **Data Layer** - DuckDB storage, ingestion pipelines, feature store
2. **Research Layer** - VectorBT backtesting, MLflow experiments, hypothesis registry
3. **Control Layer** - Streamlit dashboard, MCP servers, scheduled agents

**The Rule:** External access goes through `hrp/api/platform.py`. Data layer modules (`hrp/data/`) may access `hrp/data/db.py` directly. Everything else uses the API.

## Key Principles

1. **Research-First** - Every strategy starts as a formal hypothesis
2. **Reproducibility** - All experiments versioned and re-runnable
3. **Statistical Rigor** - Walk-forward validation, significance testing enforced
4. **Audit Trail** - Full lineage from hypothesis to deployment

## Agent Permissions

| Action | Agent | User |
|--------|-------|------|
| Create/run hypotheses | Yes | Yes |
| Run backtests | Yes | Yes |
| Analyze results | Yes | Yes |
| **Deploy strategies** | **No** | Yes |

Agents cannot approve deployments or modify deployed strategies.

## Code Conventions

- Python 3.11+, type hints required
- Black formatting (100 char line length)
- Log all significant actions to lineage table

## API Quick Reference

All external access through `hrp/api/platform.py`:

```python
from hrp.api.platform import PlatformAPI
api = PlatformAPI()

# Core operations
api.run_backtest(config, hypothesis_id='HYP-2026-001')
api.create_hypothesis(title, thesis, prediction, falsification, actor='user')
api.get_prices(['AAPL'], start_date, end_date)
api.get_features(['AAPL'], ['momentum_20d'], date)
api.run_quality_checks(as_of_date, send_alerts=True)
api.link_experiment(hypothesis_id, experiment_id)  # Link MLflow run to hypothesis
api.update_hypothesis(hypothesis_id, status, outcome, actor)  # Guards: validated requires experiments

# ML operations
api.register_model(model, model_name, model_type, features, target, metrics, ...)
api.deploy_model(model_name, model_version, validation_data, environment, actor)
api.predict_model(model_name, symbols, as_of_date)
api.check_model_drift(model_name, current_data, reference_data, ...)

# Generic DB access (for ad-hoc queries outside data layer)
api.query_readonly(sql, params)       # Returns DataFrame (SELECT/WITH only)
api.fetchone_readonly(sql, params)    # Returns single row tuple
api.fetchall_readonly(sql, params)    # Returns list of tuples
api.execute_write(sql, params)        # INSERT/UPDATE/DELETE
```

## Available Features (54 total)

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
| **Fundamental** | `market_cap`, `pe_ratio`, `pb_ratio`, `dividend_yield`, `ev_ebitda`, `shares_outstanding` |
| **Fundamental TS** | `ts_revenue`, `ts_eps`, `ts_book_value`, `ts_market_cap`, `ts_pe_ratio`, `ts_pb_ratio`, `ts_dividend_yield`, `ts_ev_ebitda`, `ts_shares_outstanding` |

## Agents

All agents follow `agent.run()` pattern. Agent pipeline chain:
**Signal Scientist -> Alpha Researcher -> ML Scientist -> ML Quality Sentinel -> Quant Developer -> Kill Gate Enforcer**

| Agent | Purpose | Key module |
|-------|---------|------------|
| SignalScientist | Automated signal discovery (IC-based) | `hrp.agents` |
| AlphaResearcher | Reviews draft hypotheses, promotes to testing | `hrp.agents` |
| MLScientist | Walk-forward validation of testing hypotheses | `hrp.agents` |
| MLQualitySentinel | Audits experiments for overfitting | `hrp.agents` |
| ValidationAnalyst | Pre-deployment stress testing | `hrp.agents` |
| KillGateEnforcer | Enforces kill gates on hypotheses | `hrp.agents` |
| CIOAgent | Scores hypotheses across 4 dimensions (Statistical/Risk/Economic/Cost) | `hrp.agents` |
| RiskManager | Portfolio risk assessment, independent veto power | `hrp.agents` |
| ReportGenerator | Daily/weekly research summaries | `hrp.agents` |

## Key Modules

| Module | Purpose |
|--------|---------|
| `hrp.research.strategies` | `generate_multifactor_signals()`, `generate_ml_predicted_signals()` |
| `hrp.research.backtest` | `get_price_data()`, `run_backtest()` |
| `hrp.research.metrics` | `calculate_metrics()`, `calculate_stability_score_v1()` |
| `hrp.research.parameter_sweep` | `parallel_parameter_sweep()` with Sharpe decay analysis |
| `hrp.ml` | `WalkForwardConfig`, `walk_forward_validate()`, `cross_validated_optimize()` |
| `hrp.ml` | `HMMConfig`, `RegimeDetector` for regime detection |
| `hrp.risk` | `TestSetGuard`, `validate_strategy()`, `check_parameter_sensitivity()` |
| `hrp.risk.overfitting` | `SharpeDecayMonitor`, `HyperparameterTrialCounter`, `FeatureCountValidator`, `TargetLeakageValidator` |
| `hrp.research.config` | `BacktestConfig`, `StopLossConfig` (types: fixed_pct, atr_trailing, volatility_scaled) |
| `hrp.data.quality.validation` | `DataValidator` for price/feature/universe validation |
| `hrp.data.retention` | `RetentionEngine` (tiers: HOT 90d, WARM 1y, COLD 3y, ARCHIVE 5y+) |
| `hrp.data.lineage` | `FeatureLineage`, `DataProvenance` for audit trails |
| `hrp.data.ingestion` | Price, feature, universe, fundamentals ingestion jobs |

## Walk-Forward Validation

Supports purge/embargo periods to prevent temporal leakage:
- `purge_days`: gap between train and test (execution lag)
- `embargo_days`: initial test period excluded (implementation delay)
- `n_jobs=-1`: parallel fold processing (3-4x speedup)
- Stability Score v1: combines Sharpe CV, drawdown dispersion, sign flip penalty (lower is better, <= 1.0 is stable)

## Services

| Service | Command | Port |
|---------|---------|------|
| Dashboard | `streamlit run hrp/dashboard/app.py` | 8501 |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///~/hrp-data/mlflow/mlflow.db` | 5000 |
| Single job | `python -m hrp.agents.run_job --job prices` | - |
| Scheduler (legacy) | `python -m hrp.agents.run_scheduler` | - |

Job scheduling: Individual launchd plists in `launchd/`, managed via `scripts/manage_launchd.sh install|uninstall|status|reload`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HRP_DB_PATH` | Database path (default: `~/hrp-data/hrp.duckdb`) | No |
| `HRP_DATA_DIR` | Data directory (default: `~/hrp-data/`) | No |
| `RESEND_API_KEY` | Resend API key for email notifications | For alerts |
| `NOTIFICATION_EMAIL` | Email address for notifications | For alerts |
| `NOTIFICATION_FROM_EMAIL` | From address (default: `noreply@hrp.local`) | No |
| `SIMFIN_API_KEY` | SimFin API key for fundamentals (falls back to YFinance) | For fundamentals |

## Project Structure

```
hrp/
├── api/            # Platform API (single entry point)
├── data/           # Data layer (DuckDB, ingestion, features)
├── research/       # Research engine (backtest, hypothesis, lineage, strategies)
├── ml/             # ML framework (training, validation, signals)
├── risk/           # Risk management (limits, validation)
├── dashboard/      # Streamlit dashboard
├── mcp/            # Claude MCP servers
├── agents/         # Scheduled agents
├── notifications/  # Email alerts
├── execution/      # Live trading, broker integration (Tier 4)
├── monitoring/     # System health, ops alerting (Tier 3)
└── utils/          # Shared utilities
```

## Where Does New Code Go?

| Adding... | Put it in... |
|-----------|--------------|
| New data provider | `hrp/data/sources/` |
| New ingestion pipeline | `hrp/data/ingestion/` |
| New computed feature | `hrp/data/features/definitions.py` + `computation.py` |
| New strategy/signal type | `hrp/research/strategies/` |
| New ML model type | `hrp/ml/` |
| New risk check | `hrp/risk/` |
| New dashboard page | `hrp/dashboard/pages/` |
| New scheduled job | `hrp/agents/jobs.py` |
| Expose via API | `hrp/api/platform.py` |

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/architecture/data-pipeline-diagram.md` | Data pipeline: sources → jobs → DuckDB → agents → outputs |
| `docs/agents/decision-pipeline.md` | Agent decision workflow: 11 stages, kill gates, scoring, human approval |
| `docs/agents/01-*.md` through `docs/agents/10-*.md` | Individual agent specifications (numbered by pipeline order) |
| `docs/plans/Project-Status.md` | Development roadmap and tier status |
| `docs/setup/Scheduler-Configuration-Guide.md` | launchd job configuration |

## Testing

```bash
pytest tests/ -v
```

## Development Status

| Tier | Focus | Status |
|------|-------|--------|
| **Foundation** | Data + Research Core | 100% |
| **Intelligence** | ML + Agents | 90% |
| **Production** | Security + Ops | 0% |
| **Trading** | Live Execution | 0% |

See `docs/plans/Project-Status.md` for details.

# Development Guidelines

1. **Think Before Coding** - State assumptions explicitly. If uncertain, ask. If multiple interpretations exist, present them. Push back when warranted.

2. **Simplicity First** - Minimum code that solves the problem. No features beyond what was asked. No abstractions for single-use code. If 200 lines could be 50, rewrite it.

3. **Surgical Changes** - Touch only what you must. Don't improve adjacent code. Match existing style. Remove only orphans YOUR changes created.

4. **Goal-Driven Execution** - Transform tasks into verifiable goals. State a brief plan with verification steps. Loop until verified.
