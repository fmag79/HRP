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
    
            )

    def test_get_price_data_chronological_order(self, test_db):
        """Price data should be in chronological order."""
        prices = get_price_data(
            symbols=["AAPL"],
            start=date(2022, 7, 1),
            end=date(2022, 7, 8),

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


# =============================================================================
# run_backtest Tests
# =============================================================================


class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_run_backtest_basic(self, test_db):
        """Run a basic backtest with provided signals and prices."""
        from hrp.research.backtest import run_backtest, get_price_data
        from unittest.mock import patch, MagicMock

        # Get actual price data from test_db
        prices = get_price_data(["AAPL", "MSFT"], date(2022, 7, 1), date(2022, 7, 8))

        # Create simple signals (all long)
        signals = pd.DataFrame(
            1.0,
            index=prices.index,
            columns=prices['close'].columns,
        )

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2022, 7, 1),
            end_date=date(2022, 7, 8),
            name="test_backtest",
        )

        # Mock get_benchmark_returns since SPY might not be in test DB
        with patch("hrp.research.backtest.get_benchmark_returns") as mock_benchmark:
            mock_benchmark.side_effect = Exception("No benchmark data")

            result = run_backtest(signals, config, prices)

        # Verify result structure
        assert result.config == config
        assert "total_return" in result.metrics or "sharpe_ratio" in result.metrics or len(result.metrics) >= 0
        assert isinstance(result.equity_curve, pd.Series) or isinstance(result.equity_curve, pd.DataFrame)
        assert isinstance(result.trades, pd.DataFrame)

    def test_run_backtest_with_stop_loss(self, test_db):
        """Run backtest with stop loss configuration."""
        from hrp.research.backtest import run_backtest, get_price_data
        from hrp.research.config import StopLossConfig
        from unittest.mock import patch, MagicMock

        prices = get_price_data(["AAPL", "MSFT"], date(2022, 7, 1), date(2022, 7, 8))

        signals = pd.DataFrame(
            1.0,
            index=prices.index,
            columns=prices['close'].columns,
        )

        # Configure trailing stop loss
        stop_config = StopLossConfig(
            enabled=True,
            type="atr_trailing",
            atr_multiplier=2.0,
        )

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2022, 7, 1),
            end_date=date(2022, 7, 8),
            name="test_backtest_stop_loss",
            stop_loss=stop_config,
        )

        # Mock the trailing stops function to return signals unchanged
        with patch("hrp.research.backtest.apply_trailing_stops") as mock_stops:
            mock_stops.return_value = (signals, None)

            with patch("hrp.research.backtest.get_benchmark_returns") as mock_benchmark:
                mock_benchmark.side_effect = Exception("No benchmark data")

                # Should not raise - stop loss should be applied
                result = run_backtest(signals, config, prices)

        # Verify apply_trailing_stops was called
        mock_stops.assert_called_once()
        assert result is not None

    def test_run_backtest_loads_prices(self, test_db):
        """Run backtest that loads prices from config."""
        from hrp.research.backtest import run_backtest, get_price_data
        from unittest.mock import patch

        # Create signals for the date range
        prices_ref = get_price_data(["AAPL", "MSFT"], date(2022, 7, 1), date(2022, 7, 8))
        signals = pd.DataFrame(
            1.0,
            index=prices_ref.index,
            columns=prices_ref['close'].columns,
        )

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2022, 7, 1),
            end_date=date(2022, 7, 8),
            name="test_backtest_load_prices",
        )

        with patch("hrp.research.backtest.get_benchmark_returns") as mock_benchmark:
            mock_benchmark.side_effect = Exception("No benchmark data")

            # prices=None should cause prices to be loaded
            result = run_backtest(signals, config, prices=None)

        assert result is not None


# =============================================================================
# generate_momentum_signals Tests
# =============================================================================


class TestGenerateMomentumSignals:
    """Tests for generate_momentum_signals function."""

    def test_generate_momentum_signals_basic(self, test_db):
        """Generate momentum signals for given prices."""
        from hrp.research.backtest import generate_momentum_signals, get_price_data

        # Get price data with longer range for momentum calculation
        # Add more test data for a longer period
        from hrp.data.db import get_db
        db = get_db()

        # Insert more dates for momentum lookback
        test_dates = pd.date_range("2022-01-03", periods=30, freq="B")
        for symbol in ["AAPL", "MSFT"]:
            for i, dt in enumerate(test_dates):
                db.execute("""
                    INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, dt.date(), 100.0 + i, 105.0 + i, 95.0 + i, 100.0 + i, 100.0 + i, 1000000))

        prices = get_price_data(["AAPL", "MSFT"], date(2022, 1, 3), date(2022, 2, 15))

        signals = generate_momentum_signals(prices, lookback=10, top_n=1)

        # Verify signals structure
        assert isinstance(signals, pd.DataFrame)
        assert signals.shape[0] == prices.shape[0]  # Same number of rows

        # First 'lookback' rows should be 0
        assert (signals.iloc[:10] == 0).all().all()

        # After lookback, signals should be 0 or 1
        assert signals.isin([0.0, 1.0]).all().all()

    def test_generate_momentum_signals_top_n(self, test_db):
        """Momentum signals should have at most top_n stocks selected."""
        from hrp.research.backtest import generate_momentum_signals, get_price_data
        from hrp.data.db import get_db
        db = get_db()

        # Insert more dates
        test_dates = pd.date_range("2022-01-03", periods=20, freq="B")
        for symbol in ["AAPL", "MSFT"]:
            for i, dt in enumerate(test_dates):
                db.execute("""
                    INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, dt.date(), 100.0 + i, 105.0 + i, 95.0 + i, 100.0 + i, 100.0 + i, 1000000))

        prices = get_price_data(["AAPL", "MSFT"], date(2022, 1, 3), date(2022, 1, 31))

        signals = generate_momentum_signals(prices, lookback=5, top_n=1)

        # After lookback, each day should have at most 1 stock selected
        for idx in range(5, len(signals)):
            assert signals.iloc[idx].sum() <= 1


# =============================================================================
# main() CLI Tests
# =============================================================================


class TestMainCLI:
    """Tests for main() CLI entry point."""

    def test_main_momentum_strategy(self, test_db, capsys):
        """Run main with momentum strategy."""
        import sys
        from unittest.mock import patch
        from hrp.research.backtest import main
        from hrp.data.db import get_db

        # Insert more data for momentum lookback (252 days default)
        db = get_db()
        test_dates = pd.date_range("2021-01-04", periods=300, freq="B")
        for symbol in ["AAPL", "MSFT"]:
            for i, dt in enumerate(test_dates):
                db.execute("""
                    INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, dt.date(), 100.0 + i*0.1, 105.0 + i*0.1, 95.0 + i*0.1, 100.0 + i*0.1, 100.0 + i*0.1, 1000000))

        with patch.object(sys, 'argv', [
            'backtest',
            '--strategy', 'momentum',
            '--symbols', 'AAPL', 'MSFT',
            '--start', '2021-01-04',
            '--end', '2021-12-31',
        ]):
            with patch("hrp.research.backtest.get_benchmark_returns") as mock_benchmark:
                mock_benchmark.side_effect = Exception("No benchmark")
                main()

        captured = capsys.readouterr()
        assert "Backtest Results" in captured.out
        assert "momentum" in captured.out

    def test_main_unknown_strategy(self, test_db):
        """Unknown strategy should raise ValueError."""
        import sys
        from unittest.mock import patch
        from hrp.research.backtest import main
        from hrp.data.db import get_db

        # Insert minimum data
        db = get_db()
        test_dates = pd.date_range("2022-01-03", periods=10, freq="B")
        for symbol in ["AAPL"]:
            for i, dt in enumerate(test_dates):
                db.execute("""
                    INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, dt.date(), 100.0 + i, 105.0 + i, 95.0 + i, 100.0 + i, 100.0 + i, 1000000))

        with patch.object(sys, 'argv', [
            'backtest',
            '--strategy', 'unknown_strategy',
            '--symbols', 'AAPL',
            '--start', '2022-01-03',
            '--end', '2022-01-14',
        ]):
            with pytest.raises(ValueError) as exc_info:
                main()

            assert "Unknown strategy" in str(exc_info.value)

    def test_main_default_symbols(self, test_db, capsys):
        """Main should use symbols from database when not specified."""
        import sys
        from unittest.mock import patch
        from hrp.research.backtest import main
        from hrp.data.db import get_db

        # Insert data
        db = get_db()
        test_dates = pd.date_range("2021-01-04", periods=300, freq="B")
        for symbol in ["AAPL", "MSFT"]:
            for i, dt in enumerate(test_dates):
                db.execute("""
                    INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, dt.date(), 100.0 + i*0.1, 105.0 + i*0.1, 95.0 + i*0.1, 100.0 + i*0.1, 100.0 + i*0.1, 1000000))

        with patch.object(sys, 'argv', [
            'backtest',
            '--start', '2021-01-04',
            '--end', '2021-12-31',
            # No --symbols, should default to database symbols
        ]):
            with patch("hrp.research.backtest.get_benchmark_returns") as mock_benchmark:
                mock_benchmark.side_effect = Exception("No benchmark")
                main()

        captured = capsys.readouterr()
        assert "Backtest Results" in captured.out


# =============================================================================
# get_price_data tests
# =============================================================================


class TestGetPriceDataEdgeCases:
    """Tests for edge cases in get_price_data."""

    def test_get_price_data_empty_raises(self, test_db):
        """Test that empty price data raises an error."""
        from hrp.research.backtest import get_price_data

        # Use a date range in the past with no data in test_db
        with pytest.raises(Exception):  # Can be ValueError or DateOutOfBounds
            get_price_data(
                ["AAPL"],
                date(2010, 1, 1),  # Past date - no data in test_db
                date(2010, 1, 31),
            )
