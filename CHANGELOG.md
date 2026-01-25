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