"""
Comprehensive tests for price data ingestion.

Tests cover:
- ingest_prices function with various sources
- Source fallback behavior
- _upsert_prices database function
- get_price_stats function
- Error handling and edge cases
"""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from hrp.data.constants import TEST_SYMBOLS
from hrp.data.ingestion.prices import (
    ingest_prices,
    _upsert_prices,
    get_price_stats,
)
from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def prices_test_db():
    """Create a temporary database for price ingestion tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    # Insert symbols for FK constraints
    db = get_db(db_path)
    with db.connection() as conn:
        conn.execute("""
            INSERT INTO symbols (symbol, name, exchange, asset_type)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ', 'equity'),
                ('MSFT', 'Microsoft Corp.', 'NASDAQ', 'equity'),
                ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'equity'),
                ('TEST', 'Test Corp.', 'NYSE', 'equity')
        """)
        # Insert universe entries
        conn.execute("""
            INSERT INTO universe (symbol, date, in_universe, sector)
            VALUES
                ('AAPL', '2023-06-01', TRUE, 'Technology'),
                ('AAPL', '2023-06-02', TRUE, 'Technology'),
                ('MSFT', '2023-06-01', TRUE, 'Technology'),
                ('MSFT', '2023-06-02', TRUE, 'Technology'),
                ('GOOGL', '2023-06-01', TRUE, 'Technology'),
                ('GOOGL', '2023-06-02', TRUE, 'Technology'),
                ('TEST', '2023-06-01', TRUE, 'Technology'),
                ('TEST', '2023-06-02', TRUE, 'Technology')
        """)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def mock_price_data():
    """Create mock price data DataFrame."""
    return pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [date(2023, 6, 1), date(2023, 6, 2)],
        "open": [180.0, 181.0],
        "high": [182.0, 183.0],
        "low": [179.0, 180.0],
        "close": [181.0, 182.0],
        "adj_close": [181.0, 182.0],
        "volume": [50000000, 55000000],
        "source": ["test", "test"],
    })


# =============================================================================
# TEST_SYMBOLS Constant Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_test_symbols_not_empty(self):
        """TEST_SYMBOLS should contain symbols."""
        assert len(TEST_SYMBOLS) > 0

    def test_test_symbols_are_strings(self):
        """TEST_SYMBOLS should be list of strings."""
        assert all(isinstance(s, str) for s in TEST_SYMBOLS)

    def test_test_symbols_contains_common_tickers(self):
        """TEST_SYMBOLS should contain common large-cap tickers."""
        common_tickers = ["AAPL", "MSFT", "GOOGL"]
        for ticker in common_tickers:
            assert ticker in TEST_SYMBOLS


# =============================================================================
# ingest_prices Tests
# =============================================================================


class TestIngestPrices:
    """Tests for ingest_prices function."""

    def test_ingest_prices_unknown_source_raises(self, prices_test_db):
        """ingest_prices should raise ValueError for unknown source."""
        with pytest.raises(ValueError) as exc_info:
            ingest_prices(
                symbols=["AAPL"],
                start=date(2023, 6, 1),
                end=date(2023, 6, 2),
                source="unknown_source",
            )
        assert "Unknown source" in str(exc_info.value)

    def test_ingest_prices_yfinance_source(self, prices_test_db, mock_price_data):
        """ingest_prices should work with yfinance source."""
        with patch("hrp.data.ingestion.prices.YFinanceSource") as MockYF:
            mock_instance = MagicMock()
            mock_instance.source_name = "yfinance"
            mock_instance.get_daily_bars.return_value = mock_price_data
            MockYF.return_value = mock_instance

            stats = ingest_prices(
                symbols=["AAPL"],
                start=date(2023, 6, 1),
                end=date(2023, 6, 2),
                source="yfinance",
            )

        assert stats["symbols_requested"] == 1
        assert stats["symbols_success"] == 1
        assert stats["rows_inserted"] == 2

    def test_ingest_prices_polygon_fallback_to_yfinance(self, prices_test_db, mock_price_data):
        """ingest_prices should fall back to yfinance if polygon fails."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as MockPolygon:
            # Polygon raises ValueError (missing API key)
            MockPolygon.side_effect = ValueError("No API key")

            with patch("hrp.data.ingestion.prices.YFinanceSource") as MockYF:
                mock_instance = MagicMock()
                mock_instance.source_name = "yfinance"
                mock_instance.get_daily_bars.return_value = mock_price_data
                MockYF.return_value = mock_instance

                stats = ingest_prices(
                    symbols=["AAPL"],
                    start=date(2023, 6, 1),
                    end=date(2023, 6, 2),
                    source="polygon",
                )

        assert stats["symbols_success"] == 1
        assert stats["rows_inserted"] == 2

    def test_ingest_prices_polygon_with_yfinance_fallback_per_symbol(
        self, prices_test_db, mock_price_data
    ):
        """ingest_prices should try yfinance fallback for failed symbols."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as MockPolygon:
            mock_polygon = MagicMock()
            mock_polygon.source_name = "polygon"
            mock_polygon.get_daily_bars.return_value = pd.DataFrame()  # Empty = failure
            MockPolygon.return_value = mock_polygon

            with patch("hrp.data.ingestion.prices.YFinanceSource") as MockYF:
                mock_yf = MagicMock()
                mock_yf.source_name = "yfinance"
                mock_yf.get_daily_bars.return_value = mock_price_data
                MockYF.return_value = mock_yf

                stats = ingest_prices(
                    symbols=["AAPL"],
                    start=date(2023, 6, 1),
                    end=date(2023, 6, 2),
                    source="polygon",
                )

        assert stats["symbols_success"] == 1
        assert stats["fallback_used"] == 1

    def test_ingest_prices_all_sources_fail(self, prices_test_db):
        """ingest_prices should track failed symbols when all sources fail."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as MockPolygon:
            mock_polygon = MagicMock()
            mock_polygon.source_name = "polygon"
            mock_polygon.get_daily_bars.return_value = pd.DataFrame()
            MockPolygon.return_value = mock_polygon

            with patch("hrp.data.ingestion.prices.YFinanceSource") as MockYF:
                mock_yf = MagicMock()
                mock_yf.source_name = "yfinance"
                mock_yf.get_daily_bars.return_value = pd.DataFrame()
                MockYF.return_value = mock_yf

                stats = ingest_prices(
                    symbols=["AAPL"],
                    start=date(2023, 6, 1),
                    end=date(2023, 6, 2),
                    source="polygon",
                )

        assert stats["symbols_success"] == 0
        assert stats["symbols_failed"] == 1
        assert "AAPL" in stats["failed_symbols"]

    def test_ingest_prices_multiple_symbols(self, prices_test_db):
        """ingest_prices should handle multiple symbols."""
        mock_data_aapl = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": [date(2023, 6, 1)],
            "open": [180.0], "high": [182.0], "low": [179.0],
            "close": [181.0], "adj_close": [181.0],
            "volume": [50000000], "source": ["test"],
        })
        mock_data_msft = pd.DataFrame({
            "symbol": ["MSFT"],
            "date": [date(2023, 6, 1)],
            "open": [330.0], "high": [335.0], "low": [328.0],
            "close": [333.0], "adj_close": [333.0],
            "volume": [30000000], "source": ["test"],
        })

        with patch("hrp.data.ingestion.prices.YFinanceSource") as MockYF:
            mock_instance = MagicMock()
            mock_instance.source_name = "yfinance"
            mock_instance.get_daily_bars.side_effect = [mock_data_aapl, mock_data_msft]
            MockYF.return_value = mock_instance

            stats = ingest_prices(
                symbols=["AAPL", "MSFT"],
                start=date(2023, 6, 1),
                end=date(2023, 6, 1),
                source="yfinance",
            )

        assert stats["symbols_requested"] == 2
        assert stats["symbols_success"] == 2
        assert stats["rows_inserted"] == 2

    def test_ingest_prices_exception_handling(self, prices_test_db):
        """ingest_prices should handle exceptions from data source."""
        with patch("hrp.data.ingestion.prices.YFinanceSource") as MockYF:
            mock_instance = MagicMock()
            mock_instance.source_name = "yfinance"
            mock_instance.get_daily_bars.side_effect = Exception("API Error")
            MockYF.return_value = mock_instance

            stats = ingest_prices(
                symbols=["AAPL"],
                start=date(2023, 6, 1),
                end=date(2023, 6, 2),
                source="yfinance",
            )

        assert stats["symbols_failed"] == 1


# =============================================================================
# _upsert_prices Tests
# =============================================================================


class TestUpsertPrices:
    """Tests for _upsert_prices function."""

    def test_upsert_prices_empty_df(self, prices_test_db):
        """_upsert_prices should return 0 for empty DataFrame."""
        db = get_db(prices_test_db)
        result = _upsert_prices(db, pd.DataFrame())
        assert result == 0

    def test_upsert_prices_inserts_records(self, prices_test_db, mock_price_data):
        """_upsert_prices should insert records into database."""
        db = get_db(prices_test_db)
        result = _upsert_prices(db, mock_price_data)

        assert result == 2

        # Verify in database
        count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
        assert count == 2

    def test_upsert_prices_upsert_behavior(self, prices_test_db, mock_price_data):
        """_upsert_prices should update existing records (upsert)."""
        db = get_db(prices_test_db)

        # First insert
        _upsert_prices(db, mock_price_data)

        # Modify and insert again
        mock_price_data["close"] = [185.0, 186.0]
        _upsert_prices(db, mock_price_data)

        # Should still have 2 records (upserted, not duplicated)
        count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
        assert count == 2

        # Verify values updated
        row = db.fetchone(
            "SELECT close FROM prices WHERE symbol = 'AAPL' AND date = '2023-06-01'"
        )
        assert row[0] == 185.0

    def test_upsert_prices_handles_nulls(self, prices_test_db):
        """_upsert_prices should handle null values in optional columns."""
        db = get_db(prices_test_db)

        data = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": [date(2023, 6, 1)],
            "close": [181.0],
            # Missing open, high, low, adj_close, volume
        })

        result = _upsert_prices(db, data)
        assert result == 1


# =============================================================================
# get_price_stats Tests
# =============================================================================


class TestGetPriceStats:
    """Tests for get_price_stats function."""

    def test_get_price_stats_empty_db(self, prices_test_db):
        """get_price_stats should handle empty database."""
        stats = get_price_stats()

        assert stats["total_rows"] == 0
        assert stats["unique_symbols"] == 0
        assert stats["per_symbol"] == []

    def test_get_price_stats_with_data(self, prices_test_db, mock_price_data):
        """get_price_stats should return correct statistics."""
        db = get_db(prices_test_db)
        _upsert_prices(db, mock_price_data)

        stats = get_price_stats()

        assert stats["total_rows"] == 2
        assert stats["unique_symbols"] == 1
        assert len(stats["per_symbol"]) == 1
        assert stats["per_symbol"][0]["symbol"] == "AAPL"
        assert stats["per_symbol"][0]["rows"] == 2

    def test_get_price_stats_date_range(self, prices_test_db, mock_price_data):
        """get_price_stats should return correct date range."""
        db = get_db(prices_test_db)
        _upsert_prices(db, mock_price_data)

        stats = get_price_stats()

        assert stats["date_range"]["start"] == date(2023, 6, 1)
        assert stats["date_range"]["end"] == date(2023, 6, 2)

    def test_get_price_stats_multiple_symbols(self, prices_test_db):
        """get_price_stats should handle multiple symbols."""
        db = get_db(prices_test_db)

        data = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "date": [date(2023, 6, 1), date(2023, 6, 2), date(2023, 6, 1)],
            "close": [181.0, 182.0, 333.0],
            "source": ["test", "test", "test"],
        })
        _upsert_prices(db, data)

        stats = get_price_stats()

        assert stats["total_rows"] == 3
        assert stats["unique_symbols"] == 2
        assert len(stats["per_symbol"]) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestPriceIngestionIntegration:
    """Integration tests for price ingestion."""

    def test_full_ingestion_workflow(self, prices_test_db, mock_price_data):
        """Test complete ingestion and stats workflow."""
        # Mock the data source
        with patch("hrp.data.ingestion.prices.YFinanceSource") as MockYF:
            mock_instance = MagicMock()
            mock_instance.source_name = "yfinance"
            mock_instance.get_daily_bars.return_value = mock_price_data
            MockYF.return_value = mock_instance

            # Ingest
            ingest_stats = ingest_prices(
                symbols=["AAPL"],
                start=date(2023, 6, 1),
                end=date(2023, 6, 2),
                source="yfinance",
            )

        # Verify ingestion stats
        assert ingest_stats["symbols_success"] == 1
        assert ingest_stats["rows_inserted"] == 2

        # Verify database stats
        db_stats = get_price_stats()
        assert db_stats["total_rows"] == 2
        assert db_stats["unique_symbols"] == 1
