# Version 1 Completion Plan

**Created:** 2026-01-22  
**Status:** Planning  
**Goal:** Complete remaining 5% of Version 1 features to achieve 100% MVP completion

---

## Overview

Version 1 is currently **95% complete**. This plan addresses the remaining features to achieve full MVP status:

1. **Point-in-Time Fundamentals Query Helper** (Not Started)
2. **Corporate Action Dividend Handling** (Partially Complete - splits done)
3. **Automated Backup/Restore Scripts** (Not Started)
4. **Historical Data Backfill Automation** (Not Started)

---

## Task 1: Point-in-Time Fundamentals Query Helper

**Priority:** High
**Effort:** Medium (2-3 days)
**Status:** ✅ COMPLETE

### Problem Statement

The `fundamentals` table exists with `report_date` tracking, but there's no helper function to ensure only fundamentals available on or before a trade date are used in backtests. This is critical for avoiding look-ahead bias.

### Requirements

1. Add `get_fundamentals_as_of()` method to Platform API
2. Query logic that filters by `report_date <= trade_date`
3. Handle multiple metrics for same symbol
4. Support batch queries for multiple symbols
5. Comprehensive input validation
6. Full test coverage

### Implementation Steps

#### Step 1.1: Add Platform API Method

**File:** `hrp/api/platform.py`

Add new method after `get_features()`:

```python
def get_fundamentals_as_of(
    self,
    symbols: List[str],
    metrics: List[str],
    as_of_date: date,
) -> pd.DataFrame:
    """
    Get fundamental metrics for symbols as of a specific date (point-in-time).
    
    Only returns fundamentals where report_date <= as_of_date to prevent
    look-ahead bias in backtests.
    
    Args:
        symbols: List of ticker symbols
        metrics: List of fundamental metric names (e.g., 'revenue', 'eps')
        as_of_date: Date to get fundamentals for (only data available by this date)
    
    Returns:
        DataFrame with columns: symbol, metric, value, report_date, period_end
        
    Example:
        # Get fundamentals as they would have been known on 2023-01-15
        df = api.get_fundamentals_as_of(
            symbols=['AAPL', 'MSFT'],
            metrics=['revenue', 'eps', 'book_value'],
            as_of_date=date(2023, 1, 15)
        )
    """
```

**Query Logic:**
- For each symbol/metric combination, get the most recent fundamental where `report_date <= as_of_date`
- Use window functions to get latest value per symbol/metric
- Return empty DataFrame if no data available (not an error)

#### Step 1.2: Add Validation

**Validations needed:**
- `as_of_date` not in future
- `symbols` list not empty
- `metrics` list not empty
- Symbols exist in universe (optional - may want fundamentals for delisted stocks)

#### Step 1.3: Add Helper for Backtest Integration

**File:** `hrp/research/backtest.py`

Add function to load fundamentals for backtest date range:

```python
def get_fundamentals_for_backtest(
    symbols: list[str],
    metrics: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Load point-in-time fundamentals for backtest.
    
    For each date in the backtest, gets fundamentals as they would
    have been known on that date.
    
    Returns:
        DataFrame with MultiIndex (date, symbol) and metric columns
    """
```

#### Step 1.4: Add Tests

**File:** `tests/test_api/test_fundamentals_api.py` (new file)

Test cases:
1. Empty inputs (symbols, metrics)
2. No fundamentals available
3. Single symbol, single metric
4. Multiple symbols, multiple metrics
5. Point-in-time correctness (only returns data available by as_of_date)
6. Latest value selection (when multiple reports available)
7. Missing data handling (some symbols have data, others don't)
8. Integration test with backtest

**Minimum coverage:** 90%

#### Step 1.5: Documentation

Update:
- `docs/cookbook.md` - Add example of using fundamentals in strategy
- Platform API docstring - Add to method list
- Roadmap - Mark as complete

### Acceptance Criteria

- [x] `get_fundamentals_as_of()` method added to Platform API ✅
- [x] Point-in-time logic prevents look-ahead bias ✅
- [x] Comprehensive input validation ✅
- [x] 90%+ test coverage (30 tests) ✅
- [x] Integration with backtest workflow documented ✅
- [x] All tests passing ✅

**Status: ✅ COMPLETE** (January 22, 2026)

---

## Task 2: Corporate Action Dividend Handling

**Priority:** Medium
**Effort:** Small (1-2 days)
**Status:** ✅ COMPLETE

### Current Status

✅ **Completed:**
- Corporate actions table with splits and dividends
- Split adjustment in `adjust_prices_for_splits()`
- Backtest integration with split adjustment
- 65 tests covering split handling
- All tests passing

❌ **Missing:**
- Dividend adjustment logic
- Total return calculation option
- Dividend reinvestment simulation

### Requirements

1. Add dividend adjustment to price data (optional, for total return)
2. Support both price return and total return backtests
3. Maintain backward compatibility (default: price return only)
4. Document methodology

### Implementation Steps

#### Step 2.1: Add Dividend Adjustment Method

**File:** `hrp/api/platform.py`

Add new method:

```python
def adjust_prices_for_dividends(
    self,
    prices: pd.DataFrame,
    reinvest: bool = True,
) -> pd.DataFrame:
    """
    Adjust historical prices for dividends (total return calculation).
    
    Args:
        prices: DataFrame with columns [symbol, date, close, ...]
        reinvest: If True, assumes dividend reinvestment (default)
    
    Returns:
        DataFrame with adjusted prices (total return basis)
        
    Note:
        This is typically used for total return analysis. For price-only
        backtests, use unadjusted prices.
    """
```

**Logic:**
- For each dividend, adjust all prior prices by factor: `1 + (dividend / price_on_ex_date)`
- Work backward from most recent to oldest
- Only adjust `close` and `adj_close` columns

#### Step 2.2: Add Total Return Option to Backtest

**File:** `hrp/research/config.py`

Update `BacktestConfig`:

```python
@dataclass
class BacktestConfig:
    # ... existing fields ...
    
    total_return: bool = False  # If True, include dividend reinvestment
```

**File:** `hrp/research/backtest.py`

Update `get_price_data()`:

```python
def get_price_data(
    symbols: list[str],
    start: date,
    end: date,
    adjust_splits: bool = True,
    adjust_dividends: bool = False,  # NEW
) -> pd.DataFrame:
```

#### Step 2.3: Add Tests

**File:** `tests/test_api/test_corporate_actions_api.py`

Add test class:

```python
class TestAdjustPricesForDividends:
    """Tests for dividend adjustment logic."""
    
    def test_no_dividends_returns_unchanged(self):
        """No adjustment when no dividends exist."""
    
    def test_single_dividend_adjusts_prior_prices(self):
        """Single dividend adjusts all prices before ex-date."""
    
    def test_multiple_dividends_compound(self):
        """Multiple dividends compound correctly."""
    
    def test_preserves_columns(self):
        """All original columns preserved."""
```

**File:** `tests/test_research/test_backtest_dividends.py` (new file)

Integration tests:
1. Total return backtest vs price return backtest
2. Dividend reinvestment increases returns
3. Consistency with adj_close from data source

**Minimum coverage:** 85%

#### Step 2.4: Documentation

Update:
- `docs/cookbook.md` - Add total return backtest example
- `hrp/research/config.py` - Document total_return flag
- Roadmap - Mark dividend handling as complete

### Acceptance Criteria

- [x] `adjust_prices_for_dividends()` method added ✅
- [x] Total return option in BacktestConfig ✅
- [x] Backward compatible (default: price return only) ✅
- [x] 85%+ test coverage for dividend logic (23 tests) ✅
- [x] Integration tests with backtest ✅
- [x] Documentation updated ✅
- [x] All tests passing ✅

**Status: ✅ COMPLETE** (January 24, 2026)

---

## Task 3: Automated Backup/Restore Scripts

**Priority:** Medium
**Effort:** Medium (2-3 days)
**Status:** ✅ COMPLETE

### Problem Statement

No automated backup strategy exists. Manual backups are error-prone and inconsistent. Need automated daily backups with verification and documented restore procedures.

### Requirements

1. Automated backup script for DuckDB + MLflow
2. Backup verification (checksums, integrity checks)
3. Restore script with validation
4. Backup rotation (keep last N days)
5. Documentation for disaster recovery
6. Integration with scheduled jobs

### Implementation Steps

#### Step 3.1: Create Backup Script

**File:** `hrp/data/backup.py` (new file)

```python
"""
Database and MLflow backup utilities.

Provides automated backup with verification and restore capabilities.
"""

from pathlib import Path
from datetime import datetime
import shutil
import hashlib

def create_backup(
    backup_dir: Path = None,
    include_mlflow: bool = True,
) -> dict:
    """
    Create a backup of the database and MLflow artifacts.
    
    Returns:
        Dictionary with backup metadata (path, size, checksum, timestamp)
    """

def verify_backup(backup_path: Path) -> bool:
    """
    Verify backup integrity using checksums.
    
    Returns:
        True if backup is valid, False otherwise
    """

def restore_backup(
    backup_path: Path,
    target_dir: Path = None,
) -> bool:
    """
    Restore database and MLflow from backup.
    
    Returns:
        True if restore successful, False otherwise
    """

def rotate_backups(
    backup_dir: Path,
    keep_days: int = 30,
) -> int:
    """
    Remove backups older than keep_days.
    
    Returns:
        Number of backups deleted
    """
```

**Backup Strategy:**
- Create timestamped backup directory: `backup_YYYYMMDD_HHMMSS/`
- Copy DuckDB file: `hrp.duckdb` → `backup_*/hrp.duckdb`
- Copy MLflow directory: `mlruns/` → `backup_*/mlruns/`
- Generate checksums: `backup_*/checksums.txt`
- Create metadata: `backup_*/metadata.json` (timestamp, size, version)

#### Step 3.2: Add CLI Interface

**File:** `hrp/data/backup.py`

```python
def main():
    """CLI entry point for backup operations."""
    parser = argparse.ArgumentParser(description="HRP Backup Management")
    parser.add_argument("--backup", action="store_true", help="Create backup")
    parser.add_argument("--restore", type=str, help="Restore from backup path")
    parser.add_argument("--verify", type=str, help="Verify backup integrity")
    parser.add_argument("--rotate", action="store_true", help="Rotate old backups")
    parser.add_argument("--list", action="store_true", help="List available backups")
```

#### Step 3.3: Integrate with Scheduled Jobs

**File:** `hrp/agents/jobs.py`

Add new job:

```python
class BackupJob(IngestionJob):
    """Daily backup job."""
    
    def run(self) -> dict:
        """Create daily backup with verification."""
        from hrp.data.backup import create_backup, verify_backup, rotate_backups
        
        # Create backup
        backup_info = create_backup()
        
        # Verify
        if not verify_backup(backup_info['path']):
            raise RuntimeError("Backup verification failed")
        
        # Rotate old backups
        deleted = rotate_backups(keep_days=30)
        
        return {
            'backup_path': str(backup_info['path']),
            'size_mb': backup_info['size_mb'],
            'deleted_old': deleted,
        }
```

**File:** `hrp/agents/scheduler.py`

Add to schedule:

```python
# Daily backup at 2 AM
scheduler.add_job(
    BackupJob().run,
    trigger='cron',
    hour=2,
    minute=0,
    id='daily_backup',
)
```

#### Step 3.4: Add Tests

**File:** `tests/test_data/test_backup.py` (new file)

Test cases:
1. Create backup (files created, checksums correct)
2. Verify backup (valid vs corrupted)
3. Restore backup (data restored correctly)
4. Rotate backups (old backups deleted)
5. Backup with missing MLflow directory
6. Restore to different location
7. Integration test (backup → corrupt → restore → verify)

**Minimum coverage:** 85%

#### Step 3.5: Documentation

**File:** `docs/operations/backup-restore.md` (new file)

Document:
- Backup schedule and location
- Manual backup procedure
- Restore procedure (step-by-step)
- Disaster recovery scenarios
- Backup verification
- Troubleshooting

### Acceptance Criteria

- [x] Backup script creates timestamped backups ✅
- [x] Checksums verify backup integrity ✅
- [x] Restore script with validation ✅
- [x] Backup rotation (keep last 30 days) ✅
- [x] Integrated with scheduled jobs ✅
- [x] CLI interface for manual operations ✅
- [x] 85%+ test coverage (87% achieved) ✅
- [x] Disaster recovery documentation ✅
- [x] All tests passing ✅

**Status: ✅ COMPLETE** (January 24, 2026)

---

## Task 4: Historical Data Backfill Automation

**Priority:** Medium
**Effort:** Medium (2-3 days)
**Status:** ✅ COMPLETE

### Problem Statement

No automated way to backfill historical data for new symbols or date ranges. Manual backfilling is time-consuming and error-prone. Need resumable, rate-limited backfill with progress tracking.

### Requirements

1. Backfill script for prices, features, corporate actions
2. Progress tracking (resume from failure)
3. Rate limiting (respect API limits)
4. Batch processing (avoid memory issues)
5. Validation (check completeness)
6. CLI interface

### Implementation Steps

#### Step 4.1: Create Backfill Script

**File:** `hrp/data/backfill.py` (new file)

```python
"""
Historical data backfill utilities.

Provides automated backfill with progress tracking and rate limiting.
"""

from datetime import date
from typing import List, Optional
import json
from pathlib import Path

class BackfillProgress:
    """Track backfill progress for resumability."""
    
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.completed_symbols = set()
        self.failed_symbols = set()
        self.load()
    
    def load(self):
        """Load progress from file."""
    
    def save(self):
        """Save progress to file."""
    
    def mark_completed(self, symbol: str):
        """Mark symbol as completed."""
    
    def mark_failed(self, symbol: str):
        """Mark symbol as failed."""

def backfill_prices(
    symbols: List[str],
    start: date,
    end: date,
    source: str = "yfinance",
    batch_size: int = 10,
    progress_file: Path = None,
) -> dict:
    """
    Backfill historical price data.
    
    Args:
        symbols: List of tickers to backfill
        start: Start date
        end: End date
        source: Data source ('yfinance' or 'polygon')
        batch_size: Number of symbols per batch
        progress_file: Path to progress tracking file (for resumability)
    
    Returns:
        Dictionary with backfill statistics
    """

def backfill_features(
    symbols: List[str],
    start: date,
    end: date,
    batch_size: int = 10,
) -> dict:
    """
    Backfill computed features for historical dates.
    """

def backfill_corporate_actions(
    symbols: List[str],
    start: date,
    end: date,
    source: str = "yfinance",
) -> dict:
    """
    Backfill corporate actions (splits, dividends).
    """

def validate_backfill(
    symbols: List[str],
    start: date,
    end: date,
) -> dict:
    """
    Validate backfill completeness.
    
    Checks:
    - All symbols have data
    - No gaps in date range (trading days only)
    - Features computed for all dates with prices
    
    Returns:
        Dictionary with validation results and any gaps found
    """
```

#### Step 4.2: Add Rate Limiting

Use existing `RateLimiter` from `hrp/utils/rate_limiter.py`:

```python
from hrp.utils.rate_limiter import RateLimiter

# Yahoo Finance: 2000 requests/hour
rate_limiter = RateLimiter(max_requests=2000, time_window=3600)

for symbol in symbols:
    rate_limiter.wait_if_needed()
    # Fetch data...
```

#### Step 4.3: Add CLI Interface

**File:** `hrp/data/backfill.py`

```python
def main():
    """CLI entry point for backfill operations."""
    parser = argparse.ArgumentParser(description="HRP Historical Data Backfill")
    parser.add_argument("--symbols", nargs="+", help="Symbols to backfill")
    parser.add_argument("--universe", action="store_true", help="Backfill entire universe")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--prices", action="store_true", help="Backfill prices")
    parser.add_argument("--features", action="store_true", help="Backfill features")
    parser.add_argument("--corporate-actions", action="store_true", help="Backfill corporate actions")
    parser.add_argument("--all", action="store_true", help="Backfill all data types")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--validate", action="store_true", help="Validate completeness")
    parser.add_argument("--resume", type=str, help="Resume from progress file")
```

**Usage Examples:**

```bash
# Backfill S&P 500 prices for 2020-2023
python -m hrp.data.backfill \
    --universe \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --prices

# Backfill specific symbols (all data types)
python -m hrp.data.backfill \
    --symbols AAPL MSFT GOOGL \
    --start 2019-01-01 \
    --all

# Resume failed backfill
python -m hrp.data.backfill \
    --resume backfill_progress_20260122.json

# Validate completeness
python -m hrp.data.backfill \
    --symbols AAPL MSFT \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --validate
```

#### Step 4.4: Add Tests

**File:** `tests/test_data/test_backfill.py` (new file)

Test cases:
1. Backfill prices (small batch)
2. Progress tracking (save/load)
3. Resume from failure
4. Rate limiting (verify delays)
5. Batch processing (multiple batches)
6. Validation (detect gaps)
7. Integration test (full backfill workflow)

**Minimum coverage:** 80%

#### Step 4.5: Documentation

**File:** `docs/operations/data-backfill.md` (new file)

Document:
- When to use backfill vs regular ingestion
- Backfill workflow
- Rate limiting considerations
- Progress tracking and resumability
- Validation procedures
- Troubleshooting common issues

### Acceptance Criteria

- [x] Backfill script for prices, features, corporate actions ✅
- [x] Progress tracking (resumable from failure) ✅
- [x] Rate limiting (respects API limits) ✅
- [x] Batch processing (memory efficient) ✅
- [x] Validation (detects gaps) ✅
- [x] CLI interface with examples ✅
- [x] 80%+ test coverage (88% achieved) ✅
- [x] Documentation for operations ✅
- [x] All tests passing ✅

**Status: ✅ COMPLETE** (January 24, 2026)

---

## Implementation Order

**Recommended sequence:**

1. **Task 1: Point-in-Time Fundamentals** (2-3 days)
   - Critical for research accuracy
   - Blocks hypothesis testing with fundamentals
   - High priority, medium effort

2. **Task 2: Dividend Handling** (1-2 days)
   - Completes corporate actions feature
   - Builds on existing split logic
   - Medium priority, small effort

3. **Task 3: Backup/Restore** (2-3 days)
   - Critical for production safety
   - Independent of other tasks
   - Medium priority, medium effort

4. **Task 4: Historical Backfill** (2-3 days)
   - Operational convenience
   - Not blocking research workflow
   - Medium priority, medium effort

**Total estimated effort:** 7-11 days

---

## Testing Strategy

### Unit Tests
- Each new method/function has dedicated tests
- Edge cases covered (empty inputs, missing data)
- Error handling verified
- Minimum 85% coverage per module

### Integration Tests
- End-to-end workflows tested
- Interaction with existing features verified
- Database state validated
- Backward compatibility ensured

### Manual Verification
- Run full test suite after each task
- Verify dashboard displays new data correctly
- Test CLI interfaces manually
- Check documentation accuracy

---

## Success Criteria

Version 1 is **100% complete** when:

- [ ] All 4 tasks completed
- [ ] All tests passing (373+ tests)
- [ ] Code coverage maintained (>85% for new code)
- [ ] Documentation updated
- [ ] Roadmap updated to reflect 100% completion
- [ ] No regressions in existing functionality
- [ ] Manual verification completed

---

## Risks and Mitigations

### Risk 1: Fundamentals Data Availability
**Risk:** May not have fundamentals data to test with  
**Mitigation:** Use synthetic test data, document real data source integration for later

### Risk 2: Backup Script Complexity
**Risk:** DuckDB file locking during backup  
**Mitigation:** Use DuckDB's EXPORT DATABASE or copy while database is idle

### Risk 3: Backfill Rate Limits
**Risk:** API rate limits slow down backfill  
**Mitigation:** Implement exponential backoff, support multiple data sources

### Risk 4: Test Coverage
**Risk:** Hard to test backup/restore without actual data loss  
**Mitigation:** Use temporary test databases, simulate corruption scenarios

---

## Next Steps

1. Review this plan
2. Create task branches for each task
3. Implement Task 1 (Point-in-Time Fundamentals)
4. Run full test suite
5. Proceed to Task 2
6. Update Roadmap after each completion

---

## Notes

- All tasks maintain backward compatibility
- Existing tests must continue passing
- Documentation is part of "done" for each task
- Code review before marking complete
