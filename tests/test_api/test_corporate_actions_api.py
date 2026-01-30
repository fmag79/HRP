"""
Tests for corporate actions API methods in PlatformAPI.

Tests cover:
- get_corporate_actions() method
- Edge cases and error handling
"""

import tempfile
import os
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from hrp.api.platform import PlatformAPI
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_db():
    """Create a temporary DuckDB database with schema for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Initialize schema
    create_tables(db_path)

    # Insert common test symbols to satisfy FK constraints
    db = DatabaseManager(db_path)
    with db.connection() as conn:
        conn.execute("""
            INSERT INTO symbols (symbol, name, exchange)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ'),
                ('GOOGL', 'Alphabet Inc.', 'NASDAQ'),
                ('TSLA', 'Tesla Inc.', 'NASDAQ'),
                ('NVDA', 'NVIDIA Corporation', 'NASDAQ')
            ON CONFLICT DO NOTHING
        """)
        # Also insert into universe for get_prices validation
        conn.execute("""
            INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
            VALUES
                ('AAPL', '2023-01-01', TRUE, 'Technology', 3000000000000),
                ('MSFT', '2023-01-01', TRUE, 'Technology', 2800000000000),
                ('GOOGL', '2023-01-01', TRUE, 'Technology', 1800000000000),
                ('TSLA', '2023-01-01', TRUE, 'Consumer', 800000000000),
                ('NVDA', '2023-01-01', TRUE, 'Technology', 1000000000000)
            ON CONFLICT DO NOTHING
        """)

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
def test_api(test_db):
    """Create a PlatformAPI instance with a test database."""
    return PlatformAPI(db_path=test_db)


@pytest.fixture
def populated_actions(test_api):
    """
    Populate the test database with sample corporate actions.

    Returns the API instance for convenience.
    """
    # Insert sample corporate actions
    test_api._db.execute(
        """
        INSERT INTO corporate_actions (symbol, date, action_type, factor, source)
        VALUES
            ('AAPL', '2023-03-15', 'dividend', 0.23, 'yfinance'),
            ('AAPL', '2023-06-15', 'dividend', 0.24, 'yfinance'),
            ('AAPL', '2023-08-28', 'split', 4.0, 'yfinance'),
            ('MSFT', '2023-05-20', 'dividend', 0.68, 'yfinance'),
            ('MSFT', '2023-08-15', 'dividend', 0.68, 'yfinance'),
            ('TSLA', '2023-07-01', 'split', 3.0, 'yfinance'),
            ('GOOGL', '2023-04-10', 'split', 20.0, 'yfinance')
        """
    )

    return test_api


# =============================================================================
# Test Classes
# =============================================================================


class TestGetCorporateActions:
    """Tests for get_corporate_actions() method."""

    def test_get_corporate_actions_empty_symbols_raises(self, test_api):
        """Test that empty symbols list raises ValueError."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            test_api.get_corporate_actions([], date(2023, 1, 1), date(2023, 12, 31))

    def test_get_corporate_actions_no_data(self, test_api):
        """Test getting corporate actions when no data exists."""
        result = test_api.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_corporate_actions_single_symbol(self, populated_actions):
        """Test getting corporate actions for a single symbol."""
        result = populated_actions.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 12, 31)
        )
        assert not result.empty
        assert "symbol" in result.columns
        assert "date" in result.columns
        assert "action_type" in result.columns
        assert "factor" in result.columns
        assert "source" in result.columns
        assert all(result["symbol"] == "AAPL")
        assert len(result) == 3  # 2 dividends + 1 split

    def test_get_corporate_actions_multiple_symbols(self, populated_actions):
        """Test getting corporate actions for multiple symbols."""
        result = populated_actions.get_corporate_actions(
            ["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 12, 31)
        )
        assert not result.empty
        symbols = result["symbol"].unique()
        assert len(symbols) == 2
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert len(result) == 5  # 3 AAPL + 2 MSFT

    def test_get_corporate_actions_date_range(self, populated_actions):
        """Test that date range filtering works correctly."""
        # Get actions only in Q2 2023
        result = populated_actions.get_corporate_actions(
            ["AAPL", "MSFT"], date(2023, 4, 1), date(2023, 6, 30)
        )
        assert not result.empty
        dates = result["date"].unique()
        for d in dates:
            # Convert to date if it's a Timestamp
            d_date = d.date() if hasattr(d, "date") else d
            assert d_date >= date(2023, 4, 1)
            assert d_date <= date(2023, 6, 30)
        # Should have AAPL dividend (06-15) and MSFT dividend (05-20)
        assert len(result) == 2

    def test_get_corporate_actions_all_columns_present(self, populated_actions):
        """Test that all expected columns are returned."""
        result = populated_actions.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 12, 31)
        )
        expected_columns = ["symbol", "date", "action_type", "factor", "source"]
        for col in expected_columns:
            assert col in result.columns

    def test_get_corporate_actions_filter_by_type(self, populated_actions):
        """Test getting only dividends or only splits."""
        result = populated_actions.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 12, 31)
        )

        # Filter dividends
        dividends = result[result["action_type"] == "dividend"]
        assert len(dividends) == 2

        # Filter splits
        splits = result[result["action_type"] == "split"]
        assert len(splits) == 1
        assert splits.iloc[0]["factor"] == 4.0

    def test_get_corporate_actions_ordered_correctly(self, populated_actions):
        """Test that results are ordered by date, symbol, action_type."""
        result = populated_actions.get_corporate_actions(
            ["AAPL", "MSFT", "TSLA", "GOOGL"], date(2023, 1, 1), date(2023, 12, 31)
        )

        # Verify data is sorted
        for i in range(len(result) - 1):
            current = result.iloc[i]
            next_row = result.iloc[i + 1]

            current_date = current["date"].date() if hasattr(current["date"], "date") else current["date"]
            next_date = next_row["date"].date() if hasattr(next_row["date"], "date") else next_row["date"]

            # Should be ordered by date first
            assert current_date <= next_date

    def test_get_corporate_actions_no_match_in_range(self, populated_actions):
        """Test getting corporate actions with date range that has no matches."""
        result = populated_actions.get_corporate_actions(
            ["AAPL"], date(2022, 1, 1), date(2022, 12, 31)
        )
        assert result.empty

    def test_get_corporate_actions_nonexistent_symbol(self, populated_actions):
        """Test getting corporate actions for symbol with no data."""
        result = populated_actions.get_corporate_actions(
            ["NONEXISTENT"], date(2023, 1, 1), date(2023, 12, 31)
        )
        assert result.empty
