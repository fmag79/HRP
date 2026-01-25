"""
Integration tests for backtest trading calendar integration.
"""

import os
import tempfile
from datetime import date

import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables
from hrp.research.backtest import get_price_data
from hrp.research.config import BacktestConfig


@pytest.fixture
def test_db():
    """Create a temporary DuckDB database with schema for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Set environment variable so get_db() uses the test database
    os.environ["HRP_DB_PATH"] = db_path

    # Create database and schema
    create_tables(db_path)
    db = DatabaseManager(db_path)

    # Insert symbols first to satisfy FK constraints
    with db.connection() as conn:
        conn.execute("""
            INSERT INTO symbols (symbol, name, exchange)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ')
            ON CONFLICT DO NOTHING
        """)

    # Insert test price data
    test_data = []
    
    # Add data for July 2022 (includes July 4th holiday)
    symbols = ["AAPL", "MSFT"]
    dates = [
        date(2022, 7, 1),  # Friday - trading day
        # July 2-3 are weekend - no data
        # July 4 is holiday - no data
        date(2022, 7, 5),  # Tuesday - trading day
        date(2022, 7, 6),  # Wednesday - trading day
        date(2022, 7, 7),  # Thursday - trading day
        date(2022, 7, 8),  # Friday - trading day
    ]
    
    for symbol in symbols:
        for i, dt in enumerate(dates):
            test_data.append({
                "symbol": symbol,
                "date": dt,
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 100.0 + i,
                "adj_close": 100.0 + i,
                "volume": 1000000,
            })
    
    # Add Thanksgiving week data (Nov 21-25, 2022)
    thanksgiving_dates = [
        date(2022, 11, 21),  # Monday - trading day
        date(2022, 11, 22),  # Tuesday - trading day
        date(2022, 11, 23),  # Wednesday - trading day
        # Nov 24 is Thanksgiving - no data
        date(2022, 11, 25),  # Friday - trading day (early close)
    ]
    
    for symbol in symbols:
        for i, dt in enumerate(thanksgiving_dates):
            test_data.append({
                "symbol": symbol,
                "date": dt,
                "open": 150.0 + i,
                "high": 155.0 + i,
                "low": 145.0 + i,
                "close": 150.0 + i,
                "adj_close": 150.0 + i,
                "volume": 1000000,
            })
    
    # Insert data
    df = pd.DataFrame(test_data)
    with db.connection() as conn:
        conn.execute("""
            INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
            SELECT * FROM df
        """)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    try:
        os.remove(db_path)
    except Exception:
        pass


class TestBacktestCalendarIntegration:
    """Tests for backtest trading calendar integration."""

    def test_get_price_data_filters_to_trading_days(self, test_db):
        """Price data should only include trading days."""
        prices = get_price_data(
            symbols=["AAPL", "MSFT"],
            start=date(2022, 7, 1),
            end=date(2022, 7, 8),
            adjust_splits=False,
        )
        
        # Should have 5 trading days (July 1, 5, 6, 7, 8)
        assert len(prices) == 5
        
        # Check dates are correct
        dates = prices.index.date
        assert date(2022, 7, 1) in dates
        assert date(2022, 7, 5) in dates
        assert date(2022, 7, 6) in dates
        assert date(2022, 7, 7) in dates
        assert date(2022, 7, 8) in dates

    def test_get_price_data_excludes_july_4th(self, test_db):
        """Price data should exclude July 4th (Independence Day)."""
        prices = get_price_data(
            symbols=["AAPL"],
            start=date(2022, 7, 1),
            end=date(2022, 7, 8),
            adjust_splits=False,
        )
        
        dates = prices.index.date
        # July 4 should not be in the data
        assert date(2022, 7, 4) not in dates
        # July 2-3 (weekend) should not be in the data
        assert date(2022, 7, 2) not in dates
        assert date(2022, 7, 3) not in dates

    def test_get_price_data_excludes_thanksgiving(self, test_db):
        """Price data should exclude Thanksgiving but include day after."""
        prices = get_price_data(
            symbols=["AAPL"],
            start=date(2022, 11, 21),
            end=date(2022, 11, 25),
            adjust_splits=False,
        )
        
        dates = prices.index.date
        
        # Should include Mon-Wed and Friday
        assert date(2022, 11, 21) in dates
        assert date(2022, 11, 22) in dates
        assert date(2022, 11, 23) in dates
        assert date(2022, 11, 25) in dates  # Day after Thanksgiving (early close)
        
        # Should exclude Thursday (Thanksgiving)
        assert date(2022, 11, 24) not in dates
        
        # Should have 4 trading days
        assert len(prices) == 4

    def test_get_price_data_no_weekend_dates(self, test_db):
        """Price data should never include weekend dates."""
        prices = get_price_data(
            symbols=["AAPL", "MSFT"],
            start=date(2022, 7, 1),
            end=date(2022, 7, 8),
            adjust_splits=False,
        )
        
        # Check that no dates are Saturday or Sunday
        for dt in prices.index:
            weekday = dt.weekday()
            assert weekday < 5, f"Found weekend date: {dt} (weekday={weekday})"

    def test_get_price_data_empty_range_raises(self, test_db):
        """Should raise error if no trading days in range."""
        with pytest.raises(ValueError, match="No trading days found"):
            get_price_data(
                symbols=["AAPL"],
                start=date(2022, 7, 2),  # Saturday
                end=date(2022, 7, 3),    # Sunday
                adjust_splits=False,
            )

    def test_get_price_data_chronological_order(self, test_db):
        """Price data should be in chronological order."""
        prices = get_price_data(
            symbols=["AAPL"],
            start=date(2022, 7, 1),
            end=date(2022, 7, 8),
            adjust_splits=False,
        )
        
        dates = prices.index.date
        assert list(dates) == sorted(dates)


class TestBacktestConfigCalendar:
    """Tests for backtest configuration with calendar."""

    def test_backtest_config_with_holiday_range(self, test_db):
        """Backtest config spanning holidays should filter correctly."""
        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2022, 7, 1),
            end_date=date(2022, 7, 8),
        )

        prices = get_price_data(
            symbols=config.symbols,
            start=config.start_date,
            end=config.end_date,
            adjust_splits=False,
        )

        # Should only have trading days
        assert len(prices) == 5

        # Verify no holidays or weekends
        dates = prices.index.date
        assert date(2022, 7, 4) not in dates  # Holiday
        assert date(2022, 7, 2) not in dates  # Saturday
        assert date(2022, 7, 3) not in dates  # Sunday


@pytest.fixture
def fundamentals_db():
    """Create a temporary DuckDB database with fundamentals data for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    os.environ["HRP_DB_PATH"] = db_path
    create_tables(db_path)
    db = DatabaseManager(db_path)

    # Insert symbols first (needed for FK constraints)
    with db.connection() as conn:
        conn.execute("""
            INSERT INTO symbols (symbol, name, exchange)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ')
        """)

    # Insert fundamentals test data
    with db.connection() as conn:
        conn.execute("""
            INSERT INTO fundamentals (symbol, report_date, period_end, metric, value)
            VALUES
                -- AAPL Q4 2022 (reported Jan 10, 2023)
                ('AAPL', '2023-01-10', '2022-12-31', 'revenue', 117154000000),
                ('AAPL', '2023-01-10', '2022-12-31', 'eps', 1.88),
                -- AAPL Q1 2023 (reported Apr 10, 2023)
                ('AAPL', '2023-04-10', '2023-03-31', 'revenue', 94836000000),
                ('AAPL', '2023-04-10', '2023-03-31', 'eps', 1.52),
                -- MSFT Q4 2022 (reported Jan 15, 2023)
                ('MSFT', '2023-01-15', '2022-12-31', 'revenue', 52747000000),
                ('MSFT', '2023-01-15', '2022-12-31', 'eps', 2.32)
        """)

    yield db_path

    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    try:
        os.remove(db_path)
    except Exception:
        pass


class TestGetFundamentalsForBacktest:
    """Tests for the backtest helper function get_fundamentals_for_backtest."""

    def test_returns_multiindex_dataframe(self, fundamentals_db):
        """Returns DataFrame with MultiIndex (date, symbol)."""
        from hrp.research.backtest import get_fundamentals_for_backtest

        dates = pd.DatetimeIndex([
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-01-20"),
        ])

        result = get_fundamentals_for_backtest(
            symbols=["AAPL"],
            metrics=["revenue"],
            dates=dates,
            db_path=fundamentals_db,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.index.names == ["date", "symbol"]

    def test_metrics_as_columns(self, fundamentals_db):
        """Metrics are returned as columns."""
        from hrp.research.backtest import get_fundamentals_for_backtest

        dates = pd.DatetimeIndex([pd.Timestamp("2023-01-15")])

        result = get_fundamentals_for_backtest(
            symbols=["AAPL"],
            metrics=["revenue", "eps"],
            dates=dates,
            db_path=fundamentals_db,
        )

        assert "revenue" in result.columns
        assert "eps" in result.columns

    def test_point_in_time_correctness(self, fundamentals_db):
        """Verify point-in-time correctness in backtest helper."""
        from hrp.research.backtest import get_fundamentals_for_backtest

        # Dates before and after Q1 2023 report (Apr 10)
        dates = pd.DatetimeIndex([
            pd.Timestamp("2023-03-01"),  # Before Q1 2023 report
            pd.Timestamp("2023-04-15"),  # After Q1 2023 report
        ])

        result = get_fundamentals_for_backtest(
            symbols=["AAPL"],
            metrics=["revenue"],
            dates=dates,
            db_path=fundamentals_db,
        )

        # March 1 should have Q4 2022 data
        march_rev = result.loc[(pd.Timestamp("2023-03-01"), "AAPL"), "revenue"]
        assert march_rev == 117154000000

        # April 15 should have Q1 2023 data
        april_rev = result.loc[(pd.Timestamp("2023-04-15"), "AAPL"), "revenue"]
        assert april_rev == 94836000000

    def test_multiple_symbols(self, fundamentals_db):
        """Handles multiple symbols correctly."""
        from hrp.research.backtest import get_fundamentals_for_backtest

        dates = pd.DatetimeIndex([pd.Timestamp("2023-01-20")])

        result = get_fundamentals_for_backtest(
            symbols=["AAPL", "MSFT"],
            metrics=["revenue"],
            dates=dates,
            db_path=fundamentals_db,
        )

        # Both symbols should have data
        symbols_in_result = result.index.get_level_values("symbol").unique()
        assert "AAPL" in symbols_in_result
        assert "MSFT" in symbols_in_result

    def test_empty_result_structure(self, fundamentals_db):
        """Returns properly structured empty DataFrame when no data."""
        from hrp.research.backtest import get_fundamentals_for_backtest

        # Query before any data is available
        dates = pd.DatetimeIndex([pd.Timestamp("2023-01-01")])

        result = get_fundamentals_for_backtest(
            symbols=["AAPL"],
            metrics=["revenue", "eps"],
            dates=dates,
            db_path=fundamentals_db,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert result.index.names == ["date", "symbol"]

    def test_sorted_by_date_and_symbol(self, fundamentals_db):
        """Result is sorted by date and symbol."""
        from hrp.research.backtest import get_fundamentals_for_backtest

        dates = pd.DatetimeIndex([
            pd.Timestamp("2023-01-20"),
            pd.Timestamp("2023-01-15"),  # Out of order
        ])

        result = get_fundamentals_for_backtest(
            symbols=["MSFT", "AAPL"],  # Out of order
            metrics=["revenue"],
            dates=dates,
            db_path=fundamentals_db,
        )

        # Should be sorted
        assert result.index.is_monotonic_increasing
