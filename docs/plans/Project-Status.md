# HRP Project Status

## Quick Status

| Tier | Focus | Status |
|------|-------|--------|
| **Foundation** | Data + Research Core | Complete |
| **Intelligence** | ML + Agents | Complete |
| **Intelligence Extensions** | NLP + Bayesian Optimization | Bayesian: Complete, NLP: Not started |
| **Production** | Security + Ops | Complete |
| **Trading** | Live Execution | Complete |

---

## Tier 1: Foundation (Complete)

### Database & Schema

- DuckDB storage: 13 tables, 3 sequences, 17 indexes
- Thread-safe connection pooling (max 5 connections, 30s acquire timeout)
- FK constraints, NOT NULL, CHECK constraints, event type validation
- Idempotent migrations: agent_token_usage identity, CIO table FK removal, sector columns

### Core Research Loop

- Platform API: Single entry point, 45+ public methods (`hrp/api/platform.py`), "The Rule" enforced (no direct `get_db` outside data layer)
- Backtest engine: VectorBT with split/dividend adjustment, trailing stops, benchmark comparison
- Experiment tracking: MLflow integration
- Hypothesis registry: Full lifecycle (draft → testing → validated/rejected → deployed), validation guard enforced at API layer, MAX-based ID generation (gap-safe)
- Lineage system: Complete audit trail with actor tracking
- Trading calendar: NYSE calendar via `exchange_calendars`
- Corporate actions: Split + dividend adjustment
- Point-in-time fundamentals: `get_fundamentals_as_of()` prevents look-ahead bias

### Data Pipeline

- Universe management: S&P 500 from Wikipedia, exclusion rules (financials, REITs, penny stocks)
- Multi-source ingestion: Polygon.io (primary) + Yahoo Finance (fallback)
- Feature store: 45 technical/fundamental indicators with versioning
- Scheduled jobs: Individual launchd jobs — Prices (18:00) → Universe (18:05) → Features (18:10)
- Weekly fundamentals: Saturday 10 AM ET (SimFin + YFinance fallback)
- Data quality: 5 check types (anomaly, completeness, gaps, stale, volume)
- Backup system: Automated weekly (Saturday 2 AM ET), SHA-256 verification, 30-day retention
- Email notifications: Via Resend for failures and summaries
- Historical backfill: Progress tracking, resumability

### Dashboard

| Page | Features |
|------|----------|
| Home | System status, recent activity |
| Data Health | Ingestion status, quality metrics, anomalies |
| Ingestion Status | Source status, last fetch times |
| Hypotheses | Browse, create, update, lifecycle management |
| Experiments | MLflow integration, comparison, artifacts |
| Pipeline Progress | Kanban view of hypothesis pipeline, agent launcher |
| Agents Monitor | Real-time status + historical timeline for all 10 agents |
| Job Health | Job execution health, error tracking |

---

## Tier 2: Intelligence (Complete)

### ML Framework

- Models: Ridge, Lasso, ElasticNet, RandomForest, LightGBM
- Walk-forward validation: Expanding/rolling windows, purge/embargo periods, parallel fold processing, MLflow run ID propagation
- Feature selection: Mutual information, correlation filtering, caching
- Signal generation: Rank-based, threshold, z-score methods
- Model registry: Versioning, stage management (staging/production), lineage tracking
- Inference: Batch prediction, drift monitoring

### Statistical Validation

- T-tests for excess returns significance
- Bootstrap confidence intervals for Sharpe ratio
- Multiple hypothesis correction (Bonferroni + Benjamini-Hochberg FDR)
- Minimum criteria enforcement

### Overfitting Guards

- Test set guard: 3-evaluation limit per hypothesis
- Sharpe decay monitor: Train/test performance gap detection
- Feature count validator: Limits (warn >30, fail >50) with samples-per-feature ratio
- Hyperparameter trial counter: Database-tracked (default 50 max)
- Target leakage validator: Correlation checks and suspicious feature names

### Robustness Testing

- Parameter sensitivity: Vary params ±20%, measure degradation
- Time period stability: Test on 3+ subperiods
- Regime analysis: Bull/bear/sideways performance (HMM-based)

### Trading Strategies

- Multi-factor signal generation with configurable weights
- ML-predicted signal generation (model selection, signal methods)
- Stop losses: Fixed %, ATR trailing, volatility-scaled

### Research Agents (10 Implemented)

| Agent | Type | Purpose |
|-------|------|---------|
| Signal Scientist | Custom | IC analysis, hypothesis creation |
| Alpha Researcher | SDK (Claude) | Hypothesis review, economic rationale |
| ML Scientist | Custom | Walk-forward validation, model training |
| ML Quality Sentinel | Custom | Experiment auditing, overfitting detection |
| Quant Developer | Custom | Production backtesting with costs |
| Kill Gate Enforcer | Custom | End-to-end pipeline with kill gates |
| Validation Analyst | Custom | Parameter sensitivity, regime stress tests |
| Risk Manager | Custom | Independent risk oversight, veto authority |
| CIO Agent | SDK (Claude) | 4-dimension hypothesis scoring (Statistical/Risk/Economic/Cost) |
| Report Generator | SDK (Claude) | Daily/weekly research summaries |

**Pipeline:** Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel → Quant Developer → Kill Gate Enforcer → Validation Analyst → Risk Manager → CIO Agent

**Coordination:** Event-driven via individual launchd jobs (agent-pipeline polls every 15 min) + time-based launchd scheduling

### MCP Server (32 Tools)

Hypothesis management (6), data access (6), backtesting (4), ML training (4), quality/health (4), agents (8)

---

## Tier 2.5: Intelligence Extensions (Bayesian Complete, NLP Not Started)

### Bayesian Hyperparameter Optimization ✅ Complete

- **Status:** Merged to main (2026-02-04)
- **Implementation:** Optuna TPE sampler replaces grid/random search
- **Key Changes:**
  - `OptimizationConfig` now uses Optuna distributions (`FloatDistribution`, `IntDistribution`, `CategoricalDistribution`)
  - Samplers: `tpe` (default), `cmaes`, `grid`, `random`
  - MedianPruner for early trial termination
  - Study persistence to `~/hrp-data/optuna/` (SQLite)
- **Files:** `hrp/ml/optimization.py`, `hrp/utils/config.py`
- **Tests:** 21 tests passing in `tests/test_ml/test_optimization.py`

### Fundamental NLP

Text-based features from earnings calls, SEC filings, and news.

| Phase | Scope | Details |
|-------|-------|---------|
| **1** | SEC EDGAR ingestion | 10-Q/10-K text → new data source + ingestion job |
| **2** | Earnings sentiment features | FinBERT or Claude API → 6-8 new features |
| **3** | News sentiment aggregation | Rolling sentiment signals |

**New files:**

- `hrp/data/sources/sec_edgar_source.py` — EDGAR full-text retrieval
- `hrp/data/ingestion/nlp_text.py` — Text ingestion job
- `hrp/data/features/nlp_features.py` — Sentiment/NLP feature computation

**Schema:** New `raw_text_data` table; NLP features stored in existing `features` table.

**Integration:** Zero changes to backtest engine, ML pipeline, or risk system — uses existing feature store pattern.

---

## Database Architecture Roadmap

| Phase | Architecture | Enables |
|-------|-------------|---------|
| **Current** | DuckDB single-file, individual launchd jobs (short-lived locks) | MCP + dashboard without lock contention |
| **Tier 3** | DuckDB WAL mode, read-only connections everywhere possible | Concurrent reads during writes |
| **Tier 4** | Split storage: PostgreSQL (mutable metadata: hypotheses, lineage, experiments) + DuckDB (analytics: prices, features, backtests) | Concurrent writes for intraday agents |

---

## Tier 3: Production (Complete)

### Phase 1: Essential Security (Complete)

- **Environment System**: `HRP_ENVIRONMENT` enum (development/staging/production) controlling security behaviors
- **Dashboard Authentication**: bcrypt password hashing, session cookies, login/logout flow
- **Security Validators**: XSS prevention, path traversal detection, filename sanitization (`hrp/utils/security.py`)
- **Secrets Validation**: `validate_secrets()` in `hrp/utils/secrets.py` for environment variable checks
- **Auth CLI**: User management via `python -m hrp.dashboard.auth_cli` (list-users, add-user, remove-user, reset-password)
- **Environment Variables**: `HRP_AUTH_ENABLED`, `HRP_AUTH_COOKIE_KEY`, `HRP_AUTH_USERS_FILE`

### Phase 2: Ops (Complete)

- **Ops Server**: FastAPI server with `/health`, `/ready`, `/metrics` endpoints (`hrp/ops/`)
- **Prometheus Metrics**: `MetricsCollector` for system (CPU, memory, disk) and data pipeline metrics
- **Configurable Thresholds**: `OpsThresholds` dataclass with YAML + env var support (`HRP_THRESHOLD_*`)
- **Ops Dashboard**: Streamlit page for system monitoring (`hrp/dashboard/pages/9_Ops.py`)
- **Startup Validation**: `fail_fast_startup()` for production secret checks (`hrp/utils/startup.py`)
- **Secret Filtering**: `filter_secrets()` to mask API keys in logs (`hrp/utils/log_filter.py`)
- **Connection Pooling**: `ConnectionPool` with retry/backoff for DuckDB (`hrp/data/connection_pool.py`)
- **Job Locking**: File-based locking with stale detection (`hrp/utils/locks.py`)
- **read_only Enforcement**: Dashboard and monitoring queries use read-only connections
- **Integration Tests**: Golden path tests for hypothesis lifecycle (`tests/integration/`)
- **CLI Entrypoint**: Unified `hrp` command via `hrp/cli.py`
- **Version Alignment**: `hrp.__version__` from pyproject.toml metadata
- **launchd Service**: `launchd/com.hrp.ops-server.plist` for ops server as background service
- **Service Management**: `scripts/manage_launchd.sh` for install/uninstall/status/reload of all HRP jobs
- **Interactive Setup**: `scripts/setup.sh` — 11-phase onboarding (pre-flight, system deps, venv, directories, .env, DB init, config fixes, auth, data bootstrap, launchd, verification)

### Phase 2: Documentation (Complete)

- **Ops Server Guide**: `docs/operations/ops-server.md` — health endpoints, Prometheus setup, launchd service
- **Alert Thresholds Guide**: `docs/operations/alert-thresholds.md` — OpsThresholds dataclass, env vars, YAML config
- **Deployment Guide**: Updated with Tier 3 env vars, startup validation, ops server reference
- **Monitoring Guide**: Updated with ops server integration, Prometheus queries
- **Cookbook**: Section 12 refactored to cross-reference focused docs (no duplication)
- **Backup/Restore**: Updated with ops server health verification before/after restore

## Tier 4: Trading (Complete)

### Live Execution

- **IBKR Broker Integration**: Connection manager with paper trading support (`hrp/execution/broker.py`)
- **Order Management**: Market and limit orders, status tracking (`hrp/execution/orders.py`)
- **Position Tracking**: Broker sync, P&L calculations, persistence (`hrp/execution/positions.py`)
- **Signal Conversion**: ML predictions → orders with risk limits (`hrp/execution/signal_converter.py`)

### Scheduled Jobs

| Job | Schedule | Purpose |
|-----|----------|---------|
| `predictions` | Daily 6:15 PM | Generate predictions for deployed models |
| `drift-monitor` | Daily 7:00 PM | Check for model drift, optional rollback |
| `live-trader` | Daily 6:30 PM | Execute trades (DISABLED by default) |

### Trading Agents

- **DailyPredictionJob**: Generates predictions for all deployed strategies
- **LiveTradingAgent**: Executes trades based on signals (dry-run by default)
- **DriftMonitorJob**: Monitors models for drift, optional auto-rollback

### Database

| Table | Purpose |
|-------|---------|
| `executed_trades` | Trade history with broker order IDs |
| `live_positions` | Current positions synced from broker |

### Dashboard

- **Trading Page**: Portfolio overview, positions, trades, model drift status

### API Methods

- `get_live_positions()` - Query current positions
- `get_executed_trades()` - Trade history with filters
- `record_trade()` - Persist trade execution
- `get_portfolio_value()` - Portfolio metrics

### Safety Features

- Dry-run mode by default
- Position limits (max 20 positions, 10% each)
- Minimum order value ($100)
- Drift monitoring before execution
- Paper trading default configuration

### Documentation

- `docs/operations/ibkr-setup-guide.md` - IBKR configuration
- `docs/operations/tier4-trading-setup.md` - Complete setup guide
