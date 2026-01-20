# HRP Implementation Roadmap

## Overview

This roadmap addresses the identified gaps and flaws in the specification while maintaining a pragmatic, phased approach. Each version delivers working functionality while progressively addressing production-critical issues.

**Philosophy:** Ship working software early, iterate based on real usage, fix critical issues before they become problems.

---

## Version 1: MVP Research Platform (Foundation + Core Loop)

**Goal:** Working research platform with critical fixes. Safe for single-user, development use.

**Timeline:** 2-3 months  
**Exit Criteria:** Can run backtests end-to-end, log to MLflow, view results in dashboard. All critical concurrency and data integrity issues fixed.

### Critical Fixes (Must Include)

#### 1. Database Integrity & Concurrency
- [ ] **Connection Pooling** — Implement DuckDB connection pool in `hrp/data/db.py`
  - Max 5 connections, reuse connections, proper cleanup
  - Thread-safe connection management
- [ ] **Foreign Key Constraints** — Add FK constraints to schema
  - `prices.symbol` → `universe.symbol`
  - `features.symbol` → `prices.symbol`
  - `lineage.hypothesis_id` → `hypotheses.hypothesis_id`
  - `lineage.experiment_id` → MLflow run validation
- [ ] **Database Indexes** — Create indexes on frequently queried columns
  - `prices(symbol, date)` — composite index
  - `features(symbol, date, feature_name)`
  - `lineage(timestamp, hypothesis_id)`
  - `universe(symbol, date)`
- [ ] **Schema Constraints** — Add NOT NULL and CHECK constraints
  - `features.version` → NOT NULL
  - `prices.close` → CHECK (close > 0)
  - `prices.date` → CHECK (date <= CURRENT_DATE)
  - `fundamentals.value` → CHECK (value IS NOT NULL OR metric LIKE '%ratio%')

#### 2. Financial Accuracy Fixes
- [ ] **Holiday Calendar** — Integrate `exchange_calendars` package
  - NYSE calendar for trading days
  - Filter signals to trading days only
  - Adjust backtest date ranges
- [ ] **Corporate Action Handling** — Implement split/dividend adjustments
  - Apply splits to historical prices in backtests
  - Adjust for dividends (ex-dividend date)
  - Store adjustment factors in `corporate_actions` table
- [ ] **Point-in-Time Fundamentals** — Ensure `report_date` is used correctly
  - Only use fundamentals available on or before trade date
  - Query helper function: `get_fundamentals_as_of(symbol, date)`

#### 3. Input Validation & Error Handling
- [ ] **Platform API Validation** — Add input validation to all API methods
  - Validate symbols (whitelist, format)
  - Validate dates (not future, valid range)
  - Validate numeric parameters (ranges, types)
- [ ] **Error Recovery** — Basic retry logic for ingestion
  - Exponential backoff for API failures
  - Partial failure handling (log failures, continue with successes)
  - Dead letter queue for failed records

### Core Deliverables (From Spec Phases 0-3)

- [ ] Phase 0: Foundation (with fixes above)
  - Repository setup, dependencies
  - DuckDB schema with constraints and indexes
  - Basic data ingestion (Yahoo Finance)
- [ ] Phase 1: Core Research Loop
  - Platform API with validation
  - VectorBT backtest wrapper
  - MLflow integration
  - Simple momentum strategy
- [ ] Phase 2: Hypothesis & Lineage
  - Hypothesis registry
  - Lineage tracking
  - Basic validation checks
- [ ] Phase 3: Dashboard MVP
  - Streamlit dashboard
  - Home, Data Health, Hypotheses, Experiments pages

### Testing Requirements

- [ ] Unit tests for Platform API (70%+ coverage)
- [ ] Integration test: full backtest flow
- [ ] Test fixtures: synthetic data generator
- [ ] Database migration tests

### Known Limitations (Acceptable for v1)

- Single-user only (concurrency handled but not optimized)
- No authentication (localhost only)
- Basic error recovery (retry 3x, then fail)
- Yahoo Finance only (free data source)
- No data archival (accept disk growth)

---

## Version 2: Production-Ready Data Pipeline

**Goal:** Robust, reliable data ingestion. Ready for automated daily updates.

**Timeline:** 1-2 months after v1  
**Exit Criteria:** Daily ingestion runs reliably, data quality checks passing, failures handled gracefully.

### Critical Fixes

#### 1. Production-Grade Ingestion
- [ ] **Ingestion Orchestration** — Dependency management between jobs
  - Prices must complete before features compute
  - Universe must update before prices ingested
  - Use dependency graph (e.g., `airflow` lightweight or custom)
- [ ] **Data Quality Framework** — Comprehensive checks
  - Automated data quality reports
  - Alerting on anomalies (email notifications)
  - Data completeness tracking
- [ ] **Backup & Recovery** — Production backup strategy
  - Automated daily backups (DuckDB + MLflow)
  - Backup verification (checksums, restore tests)
  - Documented restore procedure
- [ ] **Error Monitoring** — Observability and alerting
  - Structured logging (JSON logs)
  - Error aggregation and reporting
  - Email alerts for critical failures

#### 2. Data Source Upgrades
- [ ] **OpenBB Integration** — Unified data access layer
  - Replace fragmented data sources with OpenBB SDK
  - Unified API for price, fundamental, and alternative data
  - Built-in support for multiple providers (Yahoo, Polygon, FRED, etc.)
  - Cleaner abstraction for data source switching
- [ ] **Polygon.io Integration** — Replace/backup Yahoo Finance
  - Rate limiting and retry logic
  - Corporate action data from Polygon
  - Fallback to Yahoo Finance on failures
- [ ] **Historical Data Backfill** — Initial load strategy
  - S&P 500 universe, 15+ years of history
  - Incremental backfill with rate limits
  - Progress tracking and resumability

#### 3. Feature Store Enhancements
- [ ] **Incremental Feature Computation** — Only compute new dates
  - Detect what's already computed
  - Skip redundant calculations
- [ ] **Feature Versioning** — Track feature computation versions
  - Schema versioning in code
  - Automatic migration on version change

### Deliverables

- [ ] Phase 4: Full Data Pipeline (enhanced)
  - S&P 500 universe management
  - Polygon.io integration
  - Feature store with versioning
  - Scheduled ingestion (cron or lightweight scheduler)
  - Data quality dashboard

### Testing Requirements

- [ ] End-to-end ingestion tests
- [ ] Data quality test suite
- [ ] Backup/restore procedure tested
- [ ] Failure scenario tests (API down, network issues)

---

## Version 3: Enhanced Validation & ML Framework

**Goal:** Full statistical rigor, ML capabilities, comprehensive risk management.

**Timeline:** 2-3 months after v2  
**Exit Criteria:** ML training pipeline working, full validation framework enforced, risk limits integrated.

### Critical Fixes

#### 1. Advanced Validation Framework
- [ ] **PyFolio + Empyrical Integration** — Industry-standard performance analysis
  - Replace custom metrics with Empyrical (battle-tested calculations)
  - PyFolio tearsheets for comprehensive performance reports
  - Drawdown analysis, rolling returns, exposure analysis
  - Professional-quality visualizations for hypothesis validation
- [ ] **Statistical Significance Testing** — Comprehensive test suite
  - T-tests for excess returns
  - Bootstrap confidence intervals
  - Multiple hypothesis correction (Bonferroni, Benjamini-Hochberg)
- [ ] **Robustness Testing** — Automated sensitivity analysis
  - Parameter sensitivity checks
  - Time period stability analysis
  - Regime analysis (bull/bear/sideways)
- [ ] **Test Set Discipline** — Enforce guardrails
  - Hard limit on test set access (3x per hypothesis)
  - Test set lock mechanism
  - Prevent model selection on test set

#### 2. Enhanced Risk Management
- [ ] **Position Sizing Algorithms** — Multiple methods
  - Equal-weight baseline
  - Volatility-adjusted sizing
  - Signal-scaled sizing
  - Kelly criterion (optional)
- [ ] **Transaction Cost Model** — More realistic
  - Volume-dependent market impact
  - Illiquid stock spread adjustments
  - Large order impact modeling
- [ ] **Sector Classification** — GICS sectors
  - Sector data source integration
  - Sector exposure tracking
  - Sector concentration limits

#### 3. ML Framework
- [ ] **ML Training Pipeline** — Full implementation
  - Model registry (LightGBM, XGBoost, linear models)
  - Training pipeline with walk-forward validation
  - Feature selection (mutual information, LASSO)
  - Signal generation from predictions
- [ ] **Overfitting Guards** — Comprehensive checks
  - Train/test Sharpe decay monitoring
  - Feature count limits
  - Hyperparameter trial limits
  - Walk-forward consistency checks

### Deliverables

- [ ] Phase 5: ML Framework (enhanced)
  - ML model registry
  - Training pipeline with validation
  - Walk-forward validation
  - Overfitting guards
- [ ] Phase 8: Risk & Validation (enhanced)
  - Full statistical validation
  - Risk limits enforcement
  - Robustness testing
  - Validation reports

### Testing Requirements

- [ ] ML pipeline integration tests
- [ ] Validation framework tests
- [ ] Risk limit enforcement tests
- [ ] Statistical test correctness verification

---

## Version 4: Agent Integration & Automation

**Goal:** Claude integration via MCP, scheduled agents for autonomous research.

**Timeline:** 1-2 months after v3  
**Exit Criteria:** Claude can run research via MCP, scheduled agents working reliably, all actions properly logged.

### Critical Fixes

#### 1. Agent Safety & Permissions
- [ ] **Rate Limiting** — Prevent agent resource exhaustion
  - Backtest rate limits per agent
  - Query rate limits
  - Resource quotas (max concurrent backtests)
- [ ] **Input Validation** — Sanitize agent inputs
  - Symbol whitelist validation
  - Date range limits
  - Parameter bounds checking
- [ ] **Action Logging** — Complete audit trail
  - All agent actions logged to lineage
  - Agent reasoning captured (when available)
  - Resource usage tracking

#### 2. Agent Reliability
- [ ] **Agent Error Handling** — Graceful failures
  - Retry logic for transient failures
  - Dead letter queue for failed hypotheses
  - Agent health monitoring
- [ ] **Agent Monitoring** — Observability
  - Agent activity dashboard
  - Recent actions log
  - Performance metrics (hypotheses created, experiments run)

### Deliverables

- [ ] Phase 6: Agent Integration
  - MCP server implementation
  - Claude Code configuration
  - Agent permission enforcement
  - Rate limiting and quotas
- [ ] Phase 7: Scheduled Agents
  - Scheduler setup (APScheduler)
  - Data Monitor agent
  - Discovery agent
  - Validation agent
  - Report agent
  - Email notifications

### Testing Requirements

- [ ] MCP server integration tests
- [ ] Agent permission tests
- [ ] Rate limiting tests
- [ ] Scheduled agent end-to-end tests

---

## Version 5: Production Hardening & Security

**Goal:** Secure, monitored, production-ready platform. Ready for remote access.

**Timeline:** 1-2 months after v4  
**Exit Criteria:** Authentication working, monitoring in place, security hardened, ready for remote access.

### Critical Fixes

#### 1. Security & Access Control
- [ ] **Dashboard Authentication** — Basic auth or session-based
  - Simple password protection (local deployment)
  - Session management
  - Secure password storage (hashed, salted)
- [ ] **API Key Management** — Secure secret handling
  - Environment variable validation
  - Key rotation strategy documentation
  - Secrets management best practices
- [ ] **Input Sanitization** — Prevent injection attacks
  - SQL injection prevention (parameterized queries)
  - XSS prevention in dashboard
  - Path traversal prevention

#### 2. Monitoring & Observability
- [ ] **Health Checks** — System health monitoring
  - Database health (connection test, disk space)
  - MLflow health check
  - Ingestion job health
  - Dashboard endpoint
- [ ] **Metrics Collection** — Basic metrics
  - Backtest execution times
  - Ingestion job durations
  - Error rates
  - API call counts
- [ ] **Alerting** — Critical failure alerts
  - Email alerts for data ingestion failures
  - Dashboard alerts for high errors
  - Disk space warnings

#### 3. Operational Excellence
- [ ] **Documentation** — Operational runbooks
  - Deployment guide
  - Troubleshooting guide
  - Backup/restore procedures
  - Disaster recovery plan
- [ ] **Performance Optimization** — Address bottlenecks
  - Query optimization (profiling slow queries)
  - Caching layer for frequently accessed data
  - Memory optimization for large backtests

### Deliverables

- [ ] Authentication system
- [ ] Monitoring dashboard
- [ ] Health check endpoints
- [ ] Operational documentation
- [ ] Performance optimizations

### Testing Requirements

- [ ] Security audit (input validation, injection tests)
- [ ] Authentication tests
- [ ] Monitoring integration tests
- [ ] Performance benchmarks

---

## Later: Advanced Features & Optimizations

**Goal:** Nice-to-haves, optimizations, advanced capabilities. Only if needed.

### Potential Features

#### Data & Features
- [ ] **Data Versioning** — Track price data corrections
  - Version history for price updates
  - Reproducibility for experiments with old data
- [ ] **Data Archival** — Manage disk space
  - Archive old data to compressed files
  - Query interface for archived data
- [ ] **Advanced Features** — Cross-sectional features
  - Momentum ranks, volatility percentiles
  - Factor loadings (if factor data available)
- [ ] **Survivorship Bias Mitigation** — Historical index membership
  - Track historical S&P 500 constituents
  - Point-in-time universe in backtests

#### Quant Tools Integration
- [ ] **AlphaLens** — Factor/signal analysis
  - Evaluate signals before backtesting
  - Factor IC, turnover analysis
  - Signal decay analysis
- [ ] **RiskFolio-Lib** — Portfolio optimization
  - Mean-variance optimization
  - Risk parity allocation
  - Maximum diversification

#### Research & ML
- [ ] **Ensemble Models** — Combine multiple models
  - Stacking, blending
  - Ensemble backtests
- [ ] **Alternative Strategies** — Beyond momentum
  - Mean reversion strategies
  - Factor models
  - Sector rotation
- [ ] **Walk-Forward Optimization** — Dynamic parameter tuning
  - Rolling window optimization
  - Expanding window optimization

#### Infrastructure
- [ ] **Caching Layer** — Redis or in-memory cache
  - Cache universe queries
  - Cache recent features
  - Cache experiment results
- [ ] **Distributed Backtests** — Parallel execution
  - Split backtests across multiple cores
  - Distributed VectorBT (if needed)
- [ ] **Database Scaling** — If DuckDB becomes bottleneck
  - Consider PostgreSQL for write-heavy workloads
  - Keep DuckDB for analytical queries

#### Trading & Deployment
- [ ] **Phase 9: Paper Trading** — Live deployment
  - IBKR integration
  - Order execution
  - Position tracking
  - Live vs backtest comparison
- [ ] **Live Trading** — Production deployment (future)
  - Real money execution
  - Risk monitoring
  - Performance attribution

---

## Version Summary

| Version | Focus | Critical Fixes | Timeline | Status |
|---------|-------|----------------|----------|--------|
| **v1** | MVP Research Platform | Database integrity, concurrency, financial accuracy | 2-3 months | 🔴 Not Started |
| **v2** | Production Data Pipeline | Ingestion orchestration, backups, monitoring | 1-2 months | 🔴 Not Started |
| **v3** | Validation & ML Framework | Statistical rigor, ML pipeline, risk management | 2-3 months | 🔴 Not Started |
| **v4** | Agent Integration | MCP servers, scheduled agents, safety | 1-2 months | 🔴 Not Started |
| **v5** | Production Hardening | Security, monitoring, operational excellence | 1-2 months | 🔴 Not Started |
| **Later** | Advanced Features | Optimizations, advanced strategies, live trading | TBD | 🔴 Not Started |

---

## Implementation Principles

1. **Ship Early, Iterate Often** — Get v1 working before optimizing
2. **Fix Critical Issues First** — Address data integrity, concurrency before nice-to-haves
3. **Test as You Build** — Don't defer testing to the end
4. **Document Decisions** — Keep ADRs (Architecture Decision Records) for major choices
5. **Measure Before Optimizing** — Profile performance, fix actual bottlenecks
6. **Security by Default** — Don't add security as an afterthought
7. **Operational Readiness** — Every feature needs monitoring and error handling

---

## Risk Mitigation

### High-Risk Areas

1. **DuckDB Concurrency** — Single-file database may hit limits
   - **Mitigation:** Implement connection pooling in v1, monitor closely
   - **Plan B:** Migrate to PostgreSQL if needed (later)

2. **Data Quality** — Bad data invalidates all research
   - **Mitigation:** Comprehensive validation in v2, automated checks
   - **Plan B:** Manual review process, data quality dashboard

3. **Overfitting** — ML models may overfit without guardrails
   - **Mitigation:** Strict validation framework in v3, test set discipline
   - **Plan B:** Manual review of all validated hypotheses

4. **Agent Safety** — Autonomous agents could create problems
   - **Mitigation:** Rate limiting, permission model, human review in v4
   - **Plan B:** Disable agents, manual research only

---

## Success Metrics

### v1 Success Criteria
- ✅ Can run backtest end-to-end without errors
- ✅ All critical data integrity issues fixed
- ✅ Dashboard displays results correctly
- ✅ 70%+ test coverage

### v2 Success Criteria
- ✅ Daily ingestion runs for 30 days without manual intervention
- ✅ Data quality checks passing >95% of the time
- ✅ Backup/restore procedure tested and documented

### v3 Success Criteria
- ✅ ML pipeline produces validated models
- ✅ Validation framework prevents invalidated hypotheses
- ✅ Risk limits enforced in all backtests

### v4 Success Criteria
- ✅ Claude can complete full research loop via MCP
- ✅ Scheduled agents run reliably for 30 days
- ✅ All agent actions properly logged

### v5 Success Criteria
- ✅ System accessible remotely with authentication
- ✅ Health checks passing, monitoring operational
- ✅ Zero security vulnerabilities in basic audit

---

## Notes

- **Prioritization:** This roadmap addresses critical flaws first, then builds features. Adjust priorities based on actual usage patterns.
- **Flexibility:** Each version should be usable independently. Don't block v1 features waiting for v2.
- **Documentation:** Update this roadmap as you discover new requirements or constraints.
