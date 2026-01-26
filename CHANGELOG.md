## [Unreleased]

### Added
- **Signal Scientist Research Agent**: First automated research agent for feature/signal discovery:
  - **ResearchAgent Base Class** (`hrp/agents/research_agents.py`): Abstract base extending `IngestionJob` with actor tracking, lineage logging, and Platform API access
  - **SignalScientist Agent**: Automated IC analysis across all 44 features against forward returns
  - **Rolling IC Calculation**: 60-day windows with Spearman correlation for robust signal detection
  - **Multi-Horizon Analysis**: Tests predictiveness at 5, 10, and 20 day forward return horizons
  - **Two-Factor Combinations**: Scans 5 theoretically-motivated feature pairs (momentum+vol, trend+mean-reversion, etc.)
  - **Automatic Hypothesis Creation**: Creates draft hypotheses for signals with IC > 0.03 and IC IR > 0.3
  - **IC Thresholds**: Weak (0.02), Moderate (0.03), Strong (0.05)
  - **MLflow Integration**: Logs all scan results with parameters, metrics, and signal rankings
  - **Email Notifications**: Sends summary email with top signals and hypotheses created
  - **Scheduler Integration**: `setup_weekly_signal_scan()` method for Monday 7 PM ET scheduling
  - **Dataclasses**: `SignalScanResult` and `SignalScanReport` for structured output
- **Research Agents Design**: Comprehensive design brainstorm for multi-agent quant research team
  - Researched real hedge fund structures (DE Shaw, Two Sigma, Citadel, Renaissance)
  - Proposed 3 architecture options: 8, 10, or 12 specialized AI agents
  - Key design: autonomous agents with shared workspace (hypotheses, MLflow, lineage)
  - Recommended 8-agent structure: Alpha Researcher, Signal Scientist, ML Scientist, ML Quality Sentinel, Quant Developer, Risk Manager, Validation Analyst, Report Generator
  - Design document: `docs/plans/2025-01-25-research-agents-design.md`

### Testing
- 38 new tests for Signal Scientist implementation:
  - `tests/test_agents/test_signal_scientist.py`: 21 tests (IC calculation, hypothesis creation, MLflow, email)
  - `tests/test_agents/test_research_agents.py`: 11 tests (base class, dataclasses)
  - `tests/test_agents/test_scheduler.py`: 6 new tests (weekly signal scan setup)
- Expanded test coverage for core modules:
  - `tests/test_data/test_simfin_source.py`: 38 tests (SimFin API, rate limiting)
  - `tests/test_data/test_computation.py`: 64 tests (44 feature functions, FeatureComputer class)
  - `tests/test_data/test_prices_ingestion.py`: 19 tests (ingest_prices, upsert, stats)
  - `tests/test_data/test_fundamentals_ingestion.py`: 34 tests (point-in-time, adapters)
  - `tests/test_ml/test_validation.py`: 51 tests (walk-forward validation, fold processing)
- Coverage improvements: simfin_source.py 0%→88%, computation.py 0%→77%, validation.py 71%→93%
- 1,680 tests passing (100% pass rate)

## [1.4.1] - 2026-01-25

### Fixed
- **FundamentalsIngestionJob**: Fixed `AttributeError` when running with `symbols=None` - was calling non-existent `get_current_members()` method on `UniverseManager`, now correctly uses `get_universe_at_date(date.today())`
- **SnapshotFundamentalsJob**: Same fix applied
- **Schema precision**: Expanded `features.value` column from `DECIMAL(18,6)` to `DECIMAL(24,6)` to support trillion-dollar market caps (Apple's $3.6T was overflowing)

## [1.4.0] - 2026-01-25

### Added
- **Weekly Fundamentals Ingestion**: Complete system for fetching quarterly fundamental data (revenue, EPS, book value, net income, total assets, total liabilities) with point-in-time correctness for backtesting:
  - **SimFin Source** (`hrp/data/sources/simfin_source.py`): Primary data source using `publish_date` for true point-in-time correctness, with rate limiting (60 req/hour for free tier)
  - **YFinance Fallback** (`hrp/data/ingestion/fundamentals.py`): Fallback adapter with 45-day conservative buffer for point-in-time estimates
  - **Point-in-Time Validation**: `_validate_point_in_time()` filters records where `period_end > report_date` to prevent look-ahead bias
  - **FundamentalsIngestionJob** (`hrp/agents/jobs.py`): Scheduled job with retry logic, logging, and email notifications on failure
  - **Scheduler Integration**: `setup_weekly_fundamentals()` method for Saturday 10 AM ET scheduling
  - **CLI Support**: `run_scheduler.py` with `--fundamentals-time`, `--fundamentals-day`, `--fundamentals-source`, `--no-fundamentals` flags
- **SimFin API Key Config**: Added `SIMFIN_API_KEY` environment variable support in `hrp/utils/config.py`
- **Fundamentals CLI**: `python -m hrp.data.ingestion.fundamentals --stats` for diagnostics

### Testing
- 31 new tests for fundamentals ingestion:
  - `tests/test_data/test_fundamentals_ingestion.py`: 18 tests (point-in-time validation, upsert, adapter behavior)
  - `tests/test_agents/test_fundamentals_job.py`: 13 tests (job init, execute, run, notifications)
- 1,456 tests passing (100% pass rate)

## [1.3.2] - 2026-01-25

### Testing
- **Test Coverage Improvement**: Added 119 new tests targeting low-coverage modules:
  - `hrp/utils/timing.py`: 23 tests (51% → 100%)
  - `hrp/notifications/templates.py`: 17 tests (0% → 100%)
  - `hrp/research/benchmark.py`: 20 tests (57% → 100%)
  - `hrp/research/mlflow_utils.py`: 17 tests (14% → 74%)
  - `hrp/ml/training.py`: 19 tests added (52% → 90%)
  - `hrp/data/ingestion/features.py`: 23 tests (18% → 65%)
- **New Test Files**:
  - `tests/test_utils/test_timing.py`
  - `tests/test_notifications/test_templates.py`
  - `tests/test_research/test_benchmark.py`
  - `tests/test_research/test_mlflow_utils.py`
  - `tests/test_data/test_ingestion_features.py`
- 1,401 tests passing (100% pass rate)

## [1.3.1] - 2026-01-25

### Added
- **Strategy Presets**: 4 named multi-factor strategy presets in `hrp/research/strategies.py`:
  - **Mean Reversion**: rsi_14d (-1.0), price_to_sma_20d (-1.0), bb_width_20d (+1.0)
  - **Trend Following**: trend (+1.0), adx_14d (+1.0), macd_histogram (+1.0)
  - **Quality Momentum**: momentum_60d (+1.0), volatility_60d (-1.0), atr_14d (-0.5)
  - **Volume Breakout**: volume_ratio (+1.0), obv (+1.0), momentum_20d (+0.5)
- **Preset Helper Function**: `get_preset_strategy()` for programmatic access to preset configurations
- **Dashboard Preset UI**: Preset dropdown in multi-factor strategy configuration with auto-populated weights
- **Roadmap Items**: Added F-040 to F-043 (GARP, Sector Rotation, Risk Parity, Pairs Trading) as parked features

### Testing
- 7 new preset strategy tests (all passing)
- 1,399 tests passing (100% pass rate)

## [1.3.0] - 2026-01-25

### Added
- **ML Pipeline Parallelization**: Walk-forward validation now supports parallel fold processing via `n_jobs` parameter in `WalkForwardConfig`:
  - `n_jobs=1` (default): Sequential processing (backward compatible)
  - `n_jobs=-1`: Use all CPU cores
  - `n_jobs=N`: Use N parallel workers
  - Uses `joblib.Parallel` with process-based execution
- **Feature Selection Caching**: New `FeatureSelectionCache` class in `hrp/ml/validation.py`:
  - Caches feature selection results across folds in sequential mode
  - Reduces redundant mutual information computation
  - ~15-20% speedup for walk-forward validation
- **Vectorized Feature Computation**: 6 new vectorized feature functions in `hrp/data/features/computation.py`:
  - `compute_returns_1d`, `compute_returns_5d`, `compute_returns_20d`
  - `compute_momentum_60d`, `compute_volatility_20d`, `compute_volume_20d`
  - All use pandas vectorized operations across all symbols simultaneously
- **Batch Feature Ingestion**: New `compute_features_batch()` in `hrp/data/ingestion/features.py`:
  - Processes all symbols in single vectorized pass (vs symbol-by-symbol loop)
  - ~10x speedup for daily feature computation jobs
  - Uses DuckDB DataFrame registration for efficient bulk inserts
- **Timing Utilities**: New `hrp/utils/timing.py` module:
  - `TimingMetrics` dataclass for tracking execution times
  - `timed_section()` context manager for code instrumentation
  - `Timer` class for manual timing control
  - Walk-forward validation now logs `data_fetch` and `fold_processing` timing

### Changed
- **FeatureComputationJob**: Now uses `compute_features_batch()` for vectorized processing
- **FEATURE_FUNCTIONS registry**: Extended from 2 to 8 feature computation functions
- **_upsert_features**: Refactored to use DuckDB DataFrame registration instead of row-by-row inserts

### Performance
- Walk-forward validation: **3-4x speedup** with parallel folds (`n_jobs=-1`)
- Feature ingestion: **~10x speedup** with batch processing
- Feature selection: **~15-20% speedup** with caching in sequential mode

### Testing
- **522 tests passing** (100% pass rate)
- No regressions introduced

## [1.2.0] - 2026-01-24

### Added
- **Extended Overfitting Guards**: Comprehensive overfitting prevention system in `hrp/risk/overfitting.py`:
  - **SharpeDecayMonitor**: Detects train/test performance gaps (already in 1.1.2)
  - **FeatureCountValidator**: Limits feature count (warn >30, fail >50) with samples-per-feature ratio check
  - **HyperparameterTrialCounter**: Tracks HP search trials in database with configurable limit (default 50)
  - **TargetLeakageValidator**: Detects high correlations (>0.95) and suspicious feature names (future, next, etc.)
- **Training Pipeline Integration**: Automatic validation hooks in `train_model()`:
  - FeatureCountValidator check before training
  - TargetLeakageValidator check before training
  - Raises `OverfittingError` on violations
- **New Tests**: 19 additional tests (16 validator tests, 3 integration tests)
- **Database Schema**: `hyperparameter_trials` table for HP search tracking (added in 1.1.2)

### Changed
- Updated CLAUDE.md with usage examples for all new overfitting validators
- Updated cookbook with extended overfitting guards recipes
- Updated Project-Status.md with complete overfitting guard feature list

### Testing
- **1,259 tests passing** (100% pass rate)
- No regressions introduced

## [1.1.2] - 2026-01-24

### Added
- **FeatureCountValidator**: New validator in `hrp/risk/overfitting.py` to prevent overfitting by limiting feature count in ML models:
  - Configurable warning threshold (default 30 features)
  - Configurable maximum threshold (default 50 features)
  - Features-per-sample ratio check (warns if < 10 samples per feature)
  - Returns detailed validation results with pass/warning status
- **FeatureCountValidator Tests**: 5 comprehensive tests covering all validation scenarios

## [1.1.1] - 2026-01-24

### Added
- **Automatic Universe Scheduling Implementation**: Complete implementation of `UniverseUpdateJob` class in `hrp/agents/jobs.py` with:
  - Daily S&P 500 constituent updates at 6:05 PM ET
  - Automatic exclusion rule application (financials, REITs, penny stocks)
  - Full retry logic with exponential backoff for transient failures
  - Comprehensive logging to `ingestion_log` table
  - Email notifications on failures
  - CLI support: `python -m hrp.agents.cli run-now --job universe`
  - 6 new comprehensive tests with 100% pass rate
- **Three-Stage Ingestion Pipeline**: Enhanced daily schedule with universe updates between price ingestion and feature computation (18:00 ET → 18:05 ET → 18:10 ET)
- **Universe Management Documentation**: New section 2.4 in cookbook with complete universe management recipes (point-in-time queries, sector breakdown, historical tracking)
- **Scheduler Runner Script**: Updated `run_scheduler.py` with `--universe-time` flag for configurable universe job timing
- **Implementation Report**: Created `UNIVERSE_SCHEDULING_IMPLEMENTATION.md` documenting full implementation details and test results
- **Documentation Update Report**: Created `docs/reports/2026-01-24-universe-scheduling-documentation-update.md` tracking all documentation changes

### Changed
- **Scheduler Integration**: Updated `setup_daily_ingestion()` in `hrp/agents/scheduler.py` to include `universe_job_time` parameter
- **Documentation Comprehensive Update**: 
  - `docs/plans/Project-Status.md`: Enhanced Version 2 section with universe scheduling details (~158 lines changed)
  - `docs/plans/2025-01-19-hrp-spec.md`: Updated daily schedule and implementation status (~67 lines changed)
  - `docs/operations/cookbook.md`: Added universe recipes and updated all job examples (~247 lines changed)
  - `CLAUDE.md`: Updated common tasks with universe scheduling examples
- **Test Suite Updates**: Updated all agent and smoke tests to expect 3 scheduled jobs instead of 2
- **Cookbook Reorganization**: Renumbered sections 2.4-2.7 to accommodate new universe management section

### Fixed
- **Test Compatibility**: Fixed 2 failing scheduler tests that expected 2 jobs (now correctly expect 3)
- **Smoke Test Integration**: Updated smoke test to include UniverseUpdateJob in mocked scheduler setup

### Testing
- **74/74 agent tests passing** (100% pass rate)
- **5/5 smoke tests passing** (100% pass rate)
- **No regressions introduced** - all existing functionality preserved
- **6 new UniverseUpdateJob tests**: Initialization, execution, success/failure logging, integration with ingestion_log

### Documentation
- Added **Universe Scheduling Implementation** report (`UNIVERSE_SCHEDULING_IMPLEMENTATION.md`) with full technical details
- Added **Universe Scheduling Documentation Update** report (`docs/reports/2026-01-24-universe-scheduling-documentation-update.md`)
- Updated **Project Status** with universe scheduling as 100% complete
- Updated **Specification** with corrected daily schedule (6:05 PM ET for universe updates)
- Updated **Cookbook** with comprehensive universe management recipes and examples

## [1.1.0] - 2026-01-24

### Added
- **Automatic Universe Scheduling (Phase 4 Enhancement)**: UniverseUpdateJob now runs daily at 6:05 PM ET, automatically fetching S&P 500 constituent changes from Wikipedia, applying exclusion rules, and tracking all changes in lineage. Full integration with scheduled ingestion pipeline (Prices → Universe → Features).
- **MCP Server for Claude Integration (Phase 6)**: Full Model Context Protocol server (`hrp/mcp/research_server.py`) with 22 tools covering hypothesis management, data access, backtesting, ML training, quality checks, and lineage tracking
- **Historical Data Backfill Automation**: Complete backfill system (`hrp/data/backfill.py`) with progress tracking, rate limiting, validation, and resume capability for large-scale historical data ingestion
- **Automated Backup/Restore System**: Production-ready backup utilities (`hrp/data/backup.py`) with verification, rotation, scheduled execution, and disaster recovery support
- **Point-in-Time Fundamentals Query**: New `get_fundamentals_as_of()` API method that enforces temporal correctness by only returning fundamentals where report_date <= as_of_date, preventing look-ahead bias in backtests
- **Statistical Overfitting Guards**: Comprehensive validation framework in `hrp/risk/` with significance testing, bootstrap confidence intervals, multiple testing correction (Bonferroni, Benjamini-Hochberg), and robustness checks (parameter sensitivity, time stability, regime stability)
- **Walk-Forward Validation Framework (Phase 5 Complete)**: Full walk-forward cross-validation with expanding/rolling windows, feature selection, stability scoring, and per-fold metrics for ML model validation
- **ML Strategy Signal Generation**: New `hrp/research/strategies.py` module supporting both multi-factor signals (weighted feature combinations) and ML-predicted signals (Ridge, Lasso, Random Forest, LightGBM, XGBoost) with rank/threshold/zscore methods
- **Enhanced Documentation**: Comprehensive cookbook (`docs/cookbook.md` - 1,374 lines), operations guides for backup/restore and data backfill (`docs/operations/`), and functionality discovery E2E testing report documenting all 1,048+ tests

### Changed
- Updated all documentation to reflect Phase 4 and Phase 5 completion (overfitting guards, walk-forward validation)
- Clarified architecture in CLAUDE.md: hypothesis and lineage modules use function-based APIs (not class-based)
- Renamed `docs/plans/Roadmap.md` to `docs/plans/Project-Status.md` for clarity

### Fixed
- DuckDB INTERVAL clause now uses f-string formatting to prevent SQL syntax errors in health trend queries
- Event type constraint violations in test fixtures resolved

## [1.0.1] - 2026-01-22

### Added
- NYSE exchange calendar integration that automatically identifies valid trading days, ensuring backtests and feature calculations skip weekends and market holidays for accurate results

### Fixed
- Prevented incorrect signals and returns from being generated on non-trading days, which was causing research accuracy issues

## [1.0.0] - 2026-01-22

### Added
- Automatic handling of stock splits and dividends to ensure accurate price data in backtests
- Corporate actions tracking to maintain historical adjustment factors for all securities

### Fixed
- Price data now correctly adjusted for past corporate actions, eliminating distortions in historical analysis and backtest results