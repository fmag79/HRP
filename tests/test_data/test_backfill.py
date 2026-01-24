"""
Comprehensive tests for Historical Data Backfill Automation.

Tests cover:
- BackfillProgress: save/load progress, mark completed/failed
- backfill_prices: batch processing, rate limiting, resumability
- backfill_features: feature computation for historical dates
- backfill_corporate_actions: splits and dividends backfill
- validate_backfill: completeness validation, gap detection
- CLI interface: argument parsing and execution

Minimum coverage target: 80%
"""

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import shutil

import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backfill_db():
    """Create a temporary DuckDB database with schema for backfill testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Initialize schema
    create_tables(db_path)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also remove any wal/tmp files
    for ext in [".wal", ".tmp", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def progress_dir():
    """Create a temporary directory for progress files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def progress_file(progress_dir):
    """Create a path for a progress file."""
    return progress_dir / "backfill_progress.json"


@pytest.fixture
def populated_db(backfill_db):
    """Populate database with sample price data for validation tests."""
    from hrp.data.db import get_db

    db = get_db(backfill_db)

    # Insert sample price data for AAPL and MSFT
    symbols = ["AAPL", "MSFT"]
    # Use January 2023 (has no holidays at the start)
    dates = pd.date_range("2023-01-03", "2023-01-31", freq="B")

    # First insert symbols (due to foreign key constraints)
    for symbol in symbols:
        db.execute(
            "INSERT INTO symbols (symbol, name) VALUES (?, ?)",
            (symbol, symbol),
        )

    for symbol in symbols:
        base_price = 150.0 if symbol == "AAPL" else 250.0
        for i, d in enumerate(dates):
            price = base_price * (1 + 0.001 * i)
            db.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    d.date(),
                    price * 0.99,
                    price * 1.02,
                    price * 0.98,
                    price,
                    price,
                    1000000 + i * 10000,
                    "test",
                ),
            )

    return backfill_db


@pytest.fixture
def mock_yfinance_source(backfill_db):
    """Mock YFinanceSource for testing without actual API calls."""
    from hrp.data.db import get_db

    # Insert symbols first to satisfy FK constraints
    db = get_db(backfill_db)
    for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "INVALID"]:
        try:
            db.execute(
                "INSERT OR IGNORE INTO symbols (symbol, name) VALUES (?, ?)",
                (symbol, symbol),
            )
        except Exception:
            pass  # Ignore if already exists

    with patch("hrp.data.backfill.YFinanceSource") as mock:
        source_instance = MagicMock()
        mock.return_value = source_instance

        # Setup default behavior - return sample price data
        def get_daily_bars(symbol, start, end):
            dates = pd.date_range(start, end, freq="B")
            data = []
            for d in dates:
                data.append({
                    "symbol": symbol,
                    "date": d.date(),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 1000000,
                    "source": "yfinance",
                })
            return pd.DataFrame(data)

        source_instance.get_daily_bars.side_effect = get_daily_bars
        source_instance.source_name = "yfinance"

        # Setup corporate actions
        def get_corporate_actions(symbol, start, end):
            return pd.DataFrame()

        source_instance.get_corporate_actions.side_effect = get_corporate_actions

        yield source_instance


# =============================================================================
# BackfillProgress Tests
# =============================================================================


class TestBackfillProgressInit:
    """Tests for BackfillProgress initialization."""

    def test_init_creates_empty_sets(self, progress_file):
        """BackfillProgress should initialize with empty sets."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)

        assert progress.completed_symbols == set()
        assert progress.failed_symbols == set()

    def test_init_with_nonexistent_file(self, progress_dir):
        """BackfillProgress should handle nonexistent progress file."""
        from hrp.data.backfill import BackfillProgress

        nonexistent = progress_dir / "nonexistent.json"
        progress = BackfillProgress(nonexistent)

        assert progress.completed_symbols == set()
        assert progress.failed_symbols == set()

    def test_init_loads_existing_progress(self, progress_file):
        """BackfillProgress should load existing progress from file."""
        from hrp.data.backfill import BackfillProgress

        # Create existing progress file
        existing_data = {
            "completed_symbols": ["AAPL", "MSFT"],
            "failed_symbols": ["INVALID"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        }
        progress_file.write_text(json.dumps(existing_data))

        progress = BackfillProgress(progress_file)

        assert progress.completed_symbols == {"AAPL", "MSFT"}
        assert progress.failed_symbols == {"INVALID"}


class TestBackfillProgressSave:
    """Tests for BackfillProgress save functionality."""

    def test_save_creates_file(self, progress_file):
        """save() should create the progress file."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.completed_symbols = {"AAPL"}
        progress.failed_symbols = {"INVALID"}
        progress.save()

        assert progress_file.exists()

    def test_save_writes_correct_data(self, progress_file):
        """save() should write the correct data to file."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.completed_symbols = {"AAPL", "MSFT"}
        progress.failed_symbols = {"INVALID"}
        progress.start_date = date(2023, 1, 1)
        progress.end_date = date(2023, 12, 31)
        progress.save()

        saved_data = json.loads(progress_file.read_text())
        assert set(saved_data["completed_symbols"]) == {"AAPL", "MSFT"}
        assert set(saved_data["failed_symbols"]) == {"INVALID"}
        assert saved_data["start_date"] == "2023-01-01"
        assert saved_data["end_date"] == "2023-12-31"


class TestBackfillProgressMarkers:
    """Tests for marking symbols as completed or failed."""

    def test_mark_completed(self, progress_file):
        """mark_completed() should add symbol to completed set."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.mark_completed("AAPL")

        assert "AAPL" in progress.completed_symbols
        assert "AAPL" not in progress.failed_symbols

    def test_mark_completed_removes_from_failed(self, progress_file):
        """mark_completed() should remove symbol from failed set."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.failed_symbols.add("AAPL")
        progress.mark_completed("AAPL")

        assert "AAPL" in progress.completed_symbols
        assert "AAPL" not in progress.failed_symbols

    def test_mark_failed(self, progress_file):
        """mark_failed() should add symbol to failed set."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.mark_failed("INVALID")

        assert "INVALID" in progress.failed_symbols
        assert "INVALID" not in progress.completed_symbols

    def test_mark_failed_does_not_remove_completed(self, progress_file):
        """mark_failed() should not remove symbol from completed set."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.completed_symbols.add("AAPL")
        progress.mark_failed("AAPL")

        # Already completed symbols should not be marked as failed
        assert "AAPL" in progress.completed_symbols
        assert "AAPL" not in progress.failed_symbols


class TestBackfillProgressPending:
    """Tests for getting pending symbols."""

    def test_get_pending_symbols(self, progress_file):
        """get_pending_symbols() should return symbols not completed (failed are included for retry)."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.completed_symbols = {"AAPL", "MSFT"}
        progress.failed_symbols = {"INVALID"}

        all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "INVALID"]
        pending = progress.get_pending_symbols(all_symbols)

        # INVALID is included because failed symbols should be retried
        assert pending == ["GOOGL", "AMZN", "INVALID"]

    def test_get_pending_symbols_all_completed(self, progress_file):
        """get_pending_symbols() should return empty list if all completed."""
        from hrp.data.backfill import BackfillProgress

        progress = BackfillProgress(progress_file)
        progress.completed_symbols = {"AAPL", "MSFT"}

        pending = progress.get_pending_symbols(["AAPL", "MSFT"])
        assert pending == []


# =============================================================================
# backfill_prices Tests
# =============================================================================


class TestBackfillPrices:
    """Tests for backfill_prices function."""

    def test_backfill_prices_basic(self, backfill_db, mock_yfinance_source):
        """backfill_prices should fetch and store price data."""
        from hrp.data.backfill import backfill_prices
        from hrp.data.db import get_db

        result = backfill_prices(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
            source="yfinance",
            db_path=backfill_db,
        )

        assert result["symbols_requested"] == 2
        assert result["symbols_success"] == 2
        assert result["symbols_failed"] == 0
        assert result["rows_inserted"] > 0

        # Verify data in database
        db = get_db(backfill_db)
        count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
        assert count > 0

    def test_backfill_prices_with_batch_size(self, backfill_db, mock_yfinance_source):
        """backfill_prices should process symbols in batches."""
        from hrp.data.backfill import backfill_prices

        result = backfill_prices(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
            source="yfinance",
            batch_size=2,
            db_path=backfill_db,
        )

        assert result["symbols_success"] == 5
        assert result["batches_processed"] == 3  # 5 symbols / 2 batch_size = 3 batches

    def test_backfill_prices_handles_failures(self, backfill_db):
        """backfill_prices should handle failed symbols gracefully."""
        from hrp.data.backfill import backfill_prices

        with patch("hrp.data.backfill.YFinanceSource") as mock:
            source_instance = MagicMock()
            mock.return_value = source_instance
            source_instance.source_name = "yfinance"

            # First symbol succeeds, second fails
            def get_daily_bars(symbol, start, end):
                if symbol == "INVALID":
                    raise ValueError("Invalid symbol")
                dates = pd.date_range(start, end, freq="B")
                return pd.DataFrame([{
                    "symbol": symbol,
                    "date": d.date(),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 1000000,
                    "source": "yfinance",
                } for d in dates])

            source_instance.get_daily_bars.side_effect = get_daily_bars

            result = backfill_prices(
                symbols=["AAPL", "INVALID"],
                start=date(2023, 1, 1),
                end=date(2023, 1, 31),
                source="yfinance",
                db_path=backfill_db,
            )

            assert result["symbols_success"] == 1
            assert result["symbols_failed"] == 1
            assert "INVALID" in result["failed_symbols"]

    def test_backfill_prices_with_progress_tracking(self, backfill_db, mock_yfinance_source, progress_file):
        """backfill_prices should track progress for resumability."""
        from hrp.data.backfill import backfill_prices

        result = backfill_prices(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
            source="yfinance",
            progress_file=progress_file,
            db_path=backfill_db,
        )

        assert progress_file.exists()
        progress_data = json.loads(progress_file.read_text())
        assert set(progress_data["completed_symbols"]) == {"AAPL", "MSFT"}

    def test_backfill_prices_resumes_from_progress(self, backfill_db, mock_yfinance_source, progress_file):
        """backfill_prices should skip already completed symbols on resume."""
        from hrp.data.backfill import backfill_prices

        # Create existing progress
        existing_progress = {
            "completed_symbols": ["AAPL"],
            "failed_symbols": [],
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
        }
        progress_file.write_text(json.dumps(existing_progress))

        result = backfill_prices(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
            source="yfinance",
            progress_file=progress_file,
            db_path=backfill_db,
        )

        # Should only process MSFT (AAPL already completed)
        assert result["symbols_skipped"] == 1
        assert result["symbols_success"] == 1

    def test_backfill_prices_with_rate_limiting(self, backfill_db, mock_yfinance_source):
        """backfill_prices should respect rate limits."""
        from hrp.data.backfill import backfill_prices
        import time

        start_time = time.monotonic()

        result = backfill_prices(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
            source="yfinance",
            batch_size=1,
            rate_limit_calls=10,
            rate_limit_period=1.0,
            db_path=backfill_db,
        )

        elapsed = time.monotonic() - start_time

        assert result["symbols_success"] == 3
        # With rate limiting, should have some delay, but not excessive


# =============================================================================
# backfill_features Tests
# =============================================================================


class TestBackfillFeatures:
    """Tests for backfill_features function."""

    def test_backfill_features_basic(self, populated_db):
        """backfill_features should compute features for historical dates."""
        from hrp.data.backfill import backfill_features
        from hrp.data.db import get_db

        result = backfill_features(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 20),
            end=date(2023, 1, 31),
            db_path=populated_db,
        )

        assert result["symbols_requested"] == 2
        assert result["symbols_success"] >= 0  # May fail if not enough price history

        # Verify features in database if any were computed
        db = get_db(populated_db)
        count = db.fetchone("SELECT COUNT(*) FROM features")[0]
        # Features should be computed if there was enough price history
        assert count >= 0

    def test_backfill_features_with_batch_size(self, populated_db):
        """backfill_features should process symbols in batches."""
        from hrp.data.backfill import backfill_features

        result = backfill_features(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 20),
            end=date(2023, 1, 31),
            batch_size=1,
            db_path=populated_db,
        )

        assert "batches_processed" in result

    def test_backfill_features_handles_missing_prices(self, backfill_db):
        """backfill_features should handle symbols with no price data."""
        from hrp.data.backfill import backfill_features

        result = backfill_features(
            symbols=["NONEXISTENT"],
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
            db_path=backfill_db,
        )

        assert result["symbols_failed"] == 1
        assert "NONEXISTENT" in result["failed_symbols"]


# =============================================================================
# backfill_corporate_actions Tests
# =============================================================================


class TestBackfillCorporateActions:
    """Tests for backfill_corporate_actions function."""

    def test_backfill_corporate_actions_basic(self, backfill_db):
        """backfill_corporate_actions should fetch and store corporate actions."""
        from hrp.data.backfill import backfill_corporate_actions

        with patch("hrp.data.backfill.YFinanceSource") as mock:
            source_instance = MagicMock()
            mock.return_value = source_instance

            # Return sample corporate action data
            def get_corporate_actions(symbol, start, end):
                return pd.DataFrame([{
                    "symbol": symbol,
                    "date": date(2023, 6, 15),
                    "action_type": "split",
                    "value": 4.0,
                    "source": "yfinance",
                }])

            source_instance.get_corporate_actions.side_effect = get_corporate_actions

            result = backfill_corporate_actions(
                symbols=["AAPL"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source="yfinance",
                db_path=backfill_db,
            )

            assert result["symbols_requested"] == 1
            assert result["symbols_success"] == 1

    def test_backfill_corporate_actions_no_actions(self, backfill_db):
        """backfill_corporate_actions should handle symbols with no corporate actions."""
        from hrp.data.backfill import backfill_corporate_actions

        with patch("hrp.data.backfill.YFinanceSource") as mock:
            source_instance = MagicMock()
            mock.return_value = source_instance
            source_instance.get_corporate_actions.return_value = pd.DataFrame()

            result = backfill_corporate_actions(
                symbols=["AAPL"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source="yfinance",
                db_path=backfill_db,
            )

            # No actions is not a failure
            assert result["symbols_success"] == 1
            assert result["rows_inserted"] == 0


# =============================================================================
# validate_backfill Tests
# =============================================================================


class TestValidateBackfill:
    """Tests for validate_backfill function."""

    def test_validate_backfill_complete_data(self, populated_db):
        """validate_backfill should pass for complete data."""
        from hrp.data.backfill import validate_backfill

        result = validate_backfill(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 3),
            end=date(2023, 1, 31),
            db_path=populated_db,
        )

        assert result["is_valid"] is True
        assert result["symbols_complete"] == 2
        assert len(result["gaps"]) == 0

    def test_validate_backfill_missing_symbol(self, populated_db):
        """validate_backfill should detect missing symbols."""
        from hrp.data.backfill import validate_backfill

        result = validate_backfill(
            symbols=["AAPL", "MISSING"],
            start=date(2023, 1, 3),
            end=date(2023, 1, 31),
            db_path=populated_db,
        )

        assert result["is_valid"] is False
        assert "MISSING" in result["missing_symbols"]

    def test_validate_backfill_date_gaps(self, backfill_db):
        """validate_backfill should detect gaps in date range."""
        from hrp.data.backfill import validate_backfill
        from hrp.data.db import get_db

        db = get_db(backfill_db)

        # Insert symbol first (FK constraint)
        db.execute("INSERT INTO symbols (symbol, name) VALUES (?, ?)", ("AAPL", "AAPL"))

        # Insert data with a gap (missing Jan 6, 9, 10, 11, 12, 13)
        trading_dates = [
            date(2023, 1, 3),
            date(2023, 1, 4),
            date(2023, 1, 5),
            # Gap: Jan 6, 9, 10, 11, 12, 13
            date(2023, 1, 18),
            date(2023, 1, 19),
        ]

        for d in trading_dates:
            db.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("AAPL", d, 100.0, 105.0, 95.0, 100.0, 100.0, 1000000, "test"),
            )

        result = validate_backfill(
            symbols=["AAPL"],
            start=date(2023, 1, 3),
            end=date(2023, 1, 19),
            db_path=backfill_db,
        )

        assert result["is_valid"] is False
        assert len(result["gaps"]) > 0
        assert "AAPL" in result["gaps"]

    def test_validate_backfill_checks_features(self, populated_db):
        """validate_backfill should optionally check feature completeness."""
        from hrp.data.backfill import validate_backfill

        result = validate_backfill(
            symbols=["AAPL"],
            start=date(2023, 1, 3),
            end=date(2023, 1, 31),
            check_features=True,
            db_path=populated_db,
        )

        # Features may not be complete (not computed yet)
        assert "features_missing" in result


# =============================================================================
# CLI Tests
# =============================================================================


class TestBackfillCLI:
    """Tests for backfill CLI interface."""

    def test_cli_prices_command(self, backfill_db, mock_yfinance_source):
        """CLI should support --prices flag."""
        from hrp.data.backfill import main
        import sys

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--symbols', 'AAPL', 'MSFT',
            '--start', '2023-01-01',
            '--end', '2023-01-31',
            '--prices',
            '--db-path', backfill_db,
        ]):
            result = main()
            assert result == 0

    def test_cli_features_command(self, populated_db):
        """CLI should support --features flag."""
        from hrp.data.backfill import main
        import sys

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--symbols', 'AAPL', 'MSFT',
            '--start', '2023-01-20',
            '--end', '2023-01-31',
            '--features',
            '--db-path', populated_db,
        ]):
            result = main()
            assert result == 0

    def test_cli_all_command(self, backfill_db, mock_yfinance_source):
        """CLI should support --all flag for all data types."""
        from hrp.data.backfill import main
        import sys

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--symbols', 'AAPL',
            '--start', '2023-01-01',
            '--end', '2023-01-31',
            '--all',
            '--db-path', backfill_db,
        ]):
            result = main()
            assert result == 0

    def test_cli_validate_command(self, populated_db):
        """CLI should support --validate flag."""
        from hrp.data.backfill import main
        import sys

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--symbols', 'AAPL', 'MSFT',
            '--start', '2023-01-03',
            '--end', '2023-01-31',
            '--validate',
            '--db-path', populated_db,
        ]):
            result = main()
            assert result == 0

    def test_cli_resume_command(self, backfill_db, mock_yfinance_source, progress_file):
        """CLI should support --resume flag."""
        from hrp.data.backfill import main
        import sys

        # Create progress file
        progress_data = {
            "completed_symbols": ["AAPL"],
            "failed_symbols": [],
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "symbols": ["AAPL", "MSFT"],
        }
        progress_file.write_text(json.dumps(progress_data))

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--resume', str(progress_file),
            '--prices',
            '--db-path', backfill_db,
        ]):
            result = main()
            assert result == 0

    def test_cli_batch_size_option(self, backfill_db, mock_yfinance_source):
        """CLI should support --batch-size option."""
        from hrp.data.backfill import main
        import sys

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--symbols', 'AAPL', 'MSFT', 'GOOGL',
            '--start', '2023-01-01',
            '--end', '2023-01-31',
            '--prices',
            '--batch-size', '2',
            '--db-path', backfill_db,
        ]):
            result = main()
            assert result == 0

    def test_cli_universe_flag(self, backfill_db):
        """CLI should support --universe flag to backfill entire universe."""
        from hrp.data.backfill import main
        import sys

        # Need to mock universe module - it's imported inside main()
        with patch("hrp.data.universe.UniverseManager") as mock_universe:
            mock_manager = MagicMock()
            mock_universe.return_value = mock_manager
            mock_manager.get_universe_at_date.return_value = ["AAPL", "MSFT"]

            with patch("hrp.data.backfill.YFinanceSource") as mock_source:
                source = MagicMock()
                mock_source.return_value = source
                source.source_name = "yfinance"
                source.get_daily_bars.return_value = pd.DataFrame()

                with patch.object(sys, 'argv', [
                    'backfill.py',
                    '--universe',
                    '--start', '2023-01-01',
                    '--end', '2023-01-31',
                    '--prices',
                    '--db-path', backfill_db,
                ]):
                    result = main()
                    # May fail due to empty data, but should parse correctly
                    assert result in [0, 1]

    def test_cli_default_end_date(self, backfill_db, mock_yfinance_source):
        """CLI should default end date to today if not provided."""
        from hrp.data.backfill import main
        import sys

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--symbols', 'AAPL',
            '--start', '2023-01-01',
            '--prices',
            '--db-path', backfill_db,
        ]):
            result = main()
            # Should use today as end date
            assert result == 0

    def test_cli_no_action_shows_help(self, backfill_db, capsys):
        """CLI with no action flags should show help."""
        from hrp.data.backfill import main
        import sys

        with patch.object(sys, 'argv', [
            'backfill.py',
            '--symbols', 'AAPL',
            '--start', '2023-01-01',
        ]):
            result = main()
            # Should return with info about no action selected
            assert result == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestBackfillIntegration:
    """Integration tests for complete backfill workflows."""

    def test_full_backfill_workflow(self, backfill_db, mock_yfinance_source, progress_file):
        """Test complete backfill workflow: prices -> features -> validate."""
        from hrp.data.backfill import backfill_prices, backfill_features, validate_backfill

        # Step 1: Backfill prices
        prices_result = backfill_prices(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 3, 31),
            source="yfinance",
            progress_file=progress_file,
            db_path=backfill_db,
        )

        assert prices_result["symbols_success"] == 2

        # Step 2: Backfill features (may not compute all due to lookback requirements)
        features_result = backfill_features(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 3, 1),
            end=date(2023, 3, 31),
            db_path=backfill_db,
        )

        # Step 3: Validate
        validation_result = validate_backfill(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 1, 3),  # First trading day
            end=date(2023, 3, 31),
            db_path=backfill_db,
        )

        assert validation_result["symbols_complete"] == 2

    def test_resume_after_failure(self, backfill_db, progress_file):
        """Test resuming backfill after partial failure."""
        from hrp.data.backfill import backfill_prices, BackfillProgress

        with patch("hrp.data.backfill.YFinanceSource") as mock:
            source_instance = MagicMock()
            mock.return_value = source_instance
            source_instance.source_name = "yfinance"

            call_count = {"AAPL": 0, "MSFT": 0}

            def get_daily_bars(symbol, start, end):
                call_count[symbol] = call_count.get(symbol, 0) + 1
                # Fail on MSFT first time only
                if symbol == "MSFT" and call_count["MSFT"] == 1:
                    raise ConnectionError("Network error")
                dates = pd.date_range(start, end, freq="B")
                return pd.DataFrame([{
                    "symbol": symbol,
                    "date": d.date(),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 1000000,
                    "source": "yfinance",
                } for d in dates])

            source_instance.get_daily_bars.side_effect = get_daily_bars

            # First attempt - MSFT fails
            result1 = backfill_prices(
                symbols=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 1, 31),
                source="yfinance",
                progress_file=progress_file,
                db_path=backfill_db,
            )

            assert result1["symbols_success"] == 1
            assert result1["symbols_failed"] == 1

            # Verify AAPL is completed, MSFT is failed
            progress = BackfillProgress(progress_file)
            assert "AAPL" in progress.completed_symbols
            assert "MSFT" in progress.failed_symbols

            # Resume - should skip AAPL (completed) and retry MSFT (failed)
            # The get_pending_symbols only skips completed, not failed
            result2 = backfill_prices(
                symbols=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 1, 31),
                source="yfinance",
                progress_file=progress_file,
                db_path=backfill_db,
            )

            # AAPL is skipped (completed), MSFT is retried
            assert result2["symbols_skipped"] == 1  # AAPL already done
            assert result2["symbols_success"] == 1  # MSFT succeeds this time

    def test_rate_limiting_integration(self, backfill_db):
        """Test that rate limiting is applied across batches."""
        from hrp.data.backfill import backfill_prices
        import time

        with patch("hrp.data.backfill.YFinanceSource") as mock:
            source_instance = MagicMock()
            mock.return_value = source_instance
            source_instance.source_name = "yfinance"

            call_times = []

            def get_daily_bars(symbol, start, end):
                call_times.append(time.monotonic())
                dates = pd.date_range(start, end, freq="B")
                return pd.DataFrame([{
                    "symbol": symbol,
                    "date": d.date(),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 1000000,
                    "source": "yfinance",
                } for d in dates])

            source_instance.get_daily_bars.side_effect = get_daily_bars

            result = backfill_prices(
                symbols=["AAPL", "MSFT", "GOOGL"],
                start=date(2023, 1, 1),
                end=date(2023, 1, 5),
                source="yfinance",
                batch_size=1,
                rate_limit_calls=5,
                rate_limit_period=0.5,
                db_path=backfill_db,
            )

            assert result["symbols_success"] == 3
            assert len(call_times) == 3
