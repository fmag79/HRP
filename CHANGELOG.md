## [1.1.0] - 2026-01-24

### Added
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