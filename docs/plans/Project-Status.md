# HRP Project Status

## Quick Status

| Tier | Focus | Status |
|------|-------|--------|
| **Foundation** | Data + Research Core | Complete |
| **Intelligence** | ML + Agents | Complete |
| **Production** | Security + Ops | Not started |
| **Trading** | Live Execution | Not started |

---

## Tier 1: Foundation (Complete)

### Database & Schema

- DuckDB storage: 13 tables, 3 sequences, 17 indexes
- Thread-safe connection pooling (max 5 connections)
- FK constraints, NOT NULL, CHECK constraints, event type validation

### Core Research Loop

- Platform API: Single entry point, 35+ public methods (`hrp/api/platform.py`)
- Backtest engine: VectorBT with split/dividend adjustment, trailing stops, benchmark comparison
- Experiment tracking: MLflow integration
- Hypothesis registry: Full lifecycle (draft → testing → validated/rejected → deployed)
- Lineage system: Complete audit trail with actor tracking
- Trading calendar: NYSE calendar via `exchange_calendars`
- Corporate actions: Split + dividend adjustment
- Point-in-time fundamentals: `get_fundamentals_as_of()` prevents look-ahead bias

### Data Pipeline

- Universe management: S&P 500 from Wikipedia, exclusion rules (financials, REITs, penny stocks)
- Multi-source ingestion: Polygon.io (primary) + Yahoo Finance (fallback)
- Feature store: 44 technical indicators with versioning
- Scheduled jobs: APScheduler — Prices (18:00) → Universe (18:05) → Features (18:10)
- Weekly fundamentals: Saturday 10 AM ET (SimFin + YFinance fallback)
- Data quality: 5 check types (anomaly, completeness, gaps, stale, volume)
- Backup system: Automated daily, SHA-256 verification, 30-day retention
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
| Agents Monitor | Real-time status + historical timeline for all 11 agents |
| Job Health | Job execution health, error tracking |

---

## Tier 2: Intelligence (Complete)

### ML Framework

- Models: Ridge, Lasso, ElasticNet, RandomForest, LightGBM
- Walk-forward validation: Expanding/rolling windows, purge/embargo periods, parallel fold processing
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

### Research Agents (11 Implemented)

| Agent | Type | Purpose |
|-------|------|---------|
| Signal Scientist | Custom | IC analysis, hypothesis creation |
| Alpha Researcher | SDK (Claude) | Hypothesis review, economic rationale |
| Code Materializer | Custom | Strategy code generation |
| ML Scientist | Custom | Walk-forward validation, model training |
| ML Quality Sentinel | Custom | Experiment auditing, overfitting detection |
| Validation Analyst | Custom | Parameter sensitivity, regime stress tests |
| Risk Manager | Custom | Independent risk oversight, veto authority |
| Quant Developer | Custom | Hyperparameter sweep, optimization |
| Pipeline Orchestrator | Custom | End-to-end pipeline with kill gates |
| CIO Agent | SDK (Claude) | 4-dimension hypothesis scoring (Statistical/Risk/Economic/Cost) |
| Report Generator | SDK (Claude) | Daily/weekly research summaries |

**Pipeline:** Signal Scientist → Alpha Researcher → Code Materializer → ML Scientist → ML Quality Sentinel → Validation Analyst → Risk Manager → CIO Agent

**Coordination:** Event-driven via LineageEventWatcher + APScheduler time-based triggers

### MCP Server (22 Tools)

Hypothesis management (5), data access (5), backtesting (4), ML training (3), quality/health (3), agents (2)

---

## Tier 3: Production (Not Started)

- Dashboard authentication
- Security hardening (XSS, path traversal, secret management)
- Health endpoints and metrics collection
- Alert thresholds

## Tier 4: Trading (Not Started)

- IBKR paper trading connection
- Order execution (signal → order conversion)
- Position tracking and P&L
- Live vs backtest comparison
