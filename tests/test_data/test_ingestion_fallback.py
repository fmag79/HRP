"""
Integration tests for data ingestion fallback mechanism.

Tests cover:
- Polygon initialization failure -> YFinance fallback
- Per-symbol Polygon failure -> YFinance fallback
- Successful Polygon ingestion (no fallback needed)
- Explicit YFinance source selection
- Stats tracking (especially fallback_used counter)
- Corporate actions ingestion
- Database upsert functionality
"""

import os
from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from polygon.exceptions import BadResponse

from hrp.data.ingestion.corporate_actions import (
    _upsert_corporate_actions,
    ingest_corporate_actions,
)
from hrp.data.ingestion.prices import _upsert_prices, ingest_prices


class TestPriceIngestionFallback:
    """Tests for price ingestion fallback mechanism."""

    def test_polygon_init_failure_falls_back_to_yfinance(self, test_db):
        """Test that Polygon initialization failure falls back to YFinance."""
        # Mock PolygonSource to raise ValueError on initialization
        with patch("hrp.data.ingestion.prices.PolygonSource") as mock_polygon, \
             patch("hrp.data.ingestion.prices.YFinanceSource") as mock_yfinance:

            # Polygon initialization fails
            mock_polygon.side_effect = ValueError("POLYGON_API_KEY not found")

            # YFinance succeeds
            mock_yf_instance = Mock()
            mock_yf_instance.source_name = "yfinance"
            mock_yf_instance.get_daily_bars.return_value = pd.DataFrame({
                "symbol": ["AAPL"],
                "date": [date(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [103.0],
                "adj_close": [103.0],
                "volume": [1000000],
                "source": ["yfinance"],
            })
            mock_yfinance.return_value = mock_yf_instance

            # Run ingestion
            stats = ingest_prices(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="polygon",
            )

            # Verify fallback occurred
            assert stats["symbols_success"] == 1
            assert stats["symbols_failed"] == 0
            assert stats["rows_inserted"] == 1
            assert stats["fallback_used"] == 0  # Fallback was at initialization level

            # Verify YFinance was used
            mock_yf_instance.get_daily_bars.assert_called_once()

    def test_per_symbol_polygon_failure_falls_back_to_yfinance(self, test_db):
        """Test that per-symbol Polygon failure falls back to YFinance."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as mock_polygon, \
             patch("hrp.data.ingestion.prices.YFinanceSource") as mock_yfinance:

            # Setup Polygon instance that fails for specific symbol
            mock_pg_instance = Mock()
            mock_pg_instance.source_name = "polygon"
            mock_pg_instance.get_daily_bars.side_effect = BadResponse("Not found")
            mock_polygon.return_value = mock_pg_instance

            # Setup YFinance instance that succeeds
            mock_yf_instance = Mock()
            mock_yf_instance.source_name = "yfinance"
            mock_yf_instance.get_daily_bars.return_value = pd.DataFrame({
                "symbol": ["AAPL"],
                "date": [date(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [103.0],
                "adj_close": [103.0],
                "volume": [1000000],
                "source": ["yfinance"],
            })
            mock_yfinance.return_value = mock_yf_instance

            # Run ingestion
            stats = ingest_prices(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="polygon",
            )

            # Verify fallback occurred
            assert stats["symbols_success"] == 1
            assert stats["symbols_failed"] == 0
            assert stats["rows_inserted"] == 1
            assert stats["fallback_used"] == 1  # Fallback was used for this symbol

            # Verify both sources were called
            mock_pg_instance.get_daily_bars.assert_called_once()
            mock_yf_instance.get_daily_bars.assert_called_once()

    def test_successful_polygon_ingestion_no_fallback(self, test_db):
        """Test successful Polygon ingestion without fallback."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as mock_polygon, \
             patch("hrp.data.ingestion.prices.YFinanceSource") as mock_yfinance:

            # Setup Polygon instance that succeeds
            mock_pg_instance = Mock()
            mock_pg_instance.source_name = "polygon"
            mock_pg_instance.get_daily_bars.return_value = pd.DataFrame({
                "symbol": ["AAPL", "AAPL"],
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [103.0, 104.0],
                "adj_close": [103.0, 104.0],
                "volume": [1000000, 1100000],
                "source": ["polygon", "polygon"],
            })
            mock_polygon.return_value = mock_pg_instance

            # Setup YFinance instance (should not be called)
            mock_yf_instance = Mock()
            mock_yf_instance.source_name = "yfinance"
            mock_yfinance.return_value = mock_yf_instance

            # Run ingestion
            stats = ingest_prices(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="polygon",
            )

            # Verify no fallback occurred
            assert stats["symbols_success"] == 1
            assert stats["symbols_failed"] == 0
            assert stats["rows_inserted"] == 2
            assert stats["fallback_used"] == 0

            # Verify only Polygon was called
            mock_pg_instance.get_daily_bars.assert_called_once()
            mock_yf_instance.get_daily_bars.assert_not_called()

    def test_explicit_yfinance_source_no_polygon(self, test_db):
        """Test explicit YFinance source selection (no Polygon involved)."""
        with patch("hrp.data.ingestion.prices.YFinanceSource") as mock_yfinance:

            # Setup YFinance instance
            mock_yf_instance = Mock()
            mock_yf_instance.source_name = "yfinance"
            mock_yf_instance.get_daily_bars.return_value = pd.DataFrame({
                "symbol": ["MSFT"],
                "date": [date(2024, 1, 1)],
                "open": [200.0],
                "high": [210.0],
                "low": [199.0],
                "close": [205.0],
                "adj_close": [205.0],
                "volume": [2000000],
                "source": ["yfinance"],
            })
            mock_yfinance.return_value = mock_yf_instance

            # Run ingestion with explicit yfinance source
            stats = ingest_prices(
                symbols=["MSFT"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="yfinance",
            )

            # Verify YFinance was used
            assert stats["symbols_success"] == 1
            assert stats["symbols_failed"] == 0
            assert stats["rows_inserted"] == 1
            assert stats["fallback_used"] == 0  # No fallback with explicit source

            mock_yf_instance.get_daily_bars.assert_called_once()

    def test_both_sources_fail(self, test_db):
        """Test that symbol fails when both Polygon and YFinance fail."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as mock_polygon, \
             patch("hrp.data.ingestion.prices.YFinanceSource") as mock_yfinance:

            # Both sources fail
            mock_pg_instance = Mock()
            mock_pg_instance.source_name = "polygon"
            mock_pg_instance.get_daily_bars.side_effect = BadResponse("Not found")
            mock_polygon.return_value = mock_pg_instance

            mock_yf_instance = Mock()
            mock_yf_instance.source_name = "yfinance"
            mock_yf_instance.get_daily_bars.side_effect = Exception("Network error")
            mock_yfinance.return_value = mock_yf_instance

            # Run ingestion
            stats = ingest_prices(
                symbols=["INVALID"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="polygon",
            )

            # Verify symbol failed
            assert stats["symbols_success"] == 0
            assert stats["symbols_failed"] == 1
            assert stats["rows_inserted"] == 0
            assert stats["failed_symbols"] == ["INVALID"]

            # Verify both sources were attempted
            mock_pg_instance.get_daily_bars.assert_called_once()
            mock_yf_instance.get_daily_bars.assert_called_once()

    def test_polygon_empty_data_falls_back_to_yfinance(self, test_db):
        """Test that empty Polygon data triggers fallback to YFinance."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as mock_polygon, \
             patch("hrp.data.ingestion.prices.YFinanceSource") as mock_yfinance:

            # Polygon returns empty DataFrame
            mock_pg_instance = Mock()
            mock_pg_instance.source_name = "polygon"
            mock_pg_instance.get_daily_bars.return_value = pd.DataFrame()
            mock_polygon.return_value = mock_pg_instance

            # YFinance succeeds
            mock_yf_instance = Mock()
            mock_yf_instance.source_name = "yfinance"
            mock_yf_instance.get_daily_bars.return_value = pd.DataFrame({
                "symbol": ["AAPL"],
                "date": [date(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [103.0],
                "adj_close": [103.0],
                "volume": [1000000],
                "source": ["yfinance"],
            })
            mock_yfinance.return_value = mock_yf_instance

            # Run ingestion
            stats = ingest_prices(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="polygon",
            )

            # Verify fallback occurred
            assert stats["symbols_success"] == 1
            assert stats["fallback_used"] == 1
            assert stats["rows_inserted"] == 1

    def test_multiple_symbols_mixed_sources(self, test_db):
        """Test ingestion with multiple symbols using different sources."""
        with patch("hrp.data.ingestion.prices.PolygonSource") as mock_polygon, \
             patch("hrp.data.ingestion.prices.YFinanceSource") as mock_yfinance:

            # Polygon succeeds for AAPL, fails for MSFT
            def polygon_side_effect(symbol, start, end):
                if symbol == "AAPL":
                    return pd.DataFrame({
                        "symbol": ["AAPL"],
                        "date": [date(2024, 1, 1)],
                        "open": [100.0],
                        "high": [105.0],
                        "low": [99.0],
                        "close": [103.0],
                        "adj_close": [103.0],
                        "volume": [1000000],
                        "source": ["polygon"],
                    })
                else:
                    raise BadResponse("Not found")

            mock_pg_instance = Mock()
            mock_pg_instance.source_name = "polygon"
            mock_pg_instance.get_daily_bars.side_effect = polygon_side_effect
            mock_polygon.return_value = mock_pg_instance

            # YFinance used for MSFT
            def yfinance_side_effect(symbol, start, end):
                if symbol == "MSFT":
                    return pd.DataFrame({
                        "symbol": ["MSFT"],
                        "date": [date(2024, 1, 1)],
                        "open": [200.0],
                        "high": [210.0],
                        "low": [199.0],
                        "close": [205.0],
                        "adj_close": [205.0],
                        "volume": [2000000],
                        "source": ["yfinance"],
                    })
                return pd.DataFrame()

            mock_yf_instance = Mock()
            mock_yf_instance.source_name = "yfinance"
            mock_yf_instance.get_daily_bars.side_effect = yfinance_side_effect
            mock_yfinance.return_value = mock_yf_instance

            # Run ingestion
            stats = ingest_prices(
                symbols=["AAPL", "MSFT"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="polygon",
            )

            # Verify stats
            assert stats["symbols_success"] == 2
            assert stats["symbols_failed"] == 0
            assert stats["rows_inserted"] == 2
            assert stats["fallback_used"] == 1  # MSFT used fallback

    def test_invalid_source_raises_error(self, test_db):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            ingest_prices(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                source="invalid_source",
            )


class TestPriceUpsert:
    """Tests for _upsert_prices database function."""

    def test_upsert_prices_inserts_new_data(self, test_db):
        """Test that upsert inserts new price data."""
        from hrp.data.db import get_db

        db = get_db()

        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "adj_close": [103.0, 104.0],
            "volume": [1000000, 1100000],
            "source": ["polygon", "polygon"],
        })

        rows_inserted = _upsert_prices(db, df)

        assert rows_inserted == 2

        # Verify data in database
        result = db.execute("SELECT * FROM prices WHERE symbol = 'AAPL' ORDER BY date").fetchall()
        assert len(result) == 2
        assert result[0][2] == date(2024, 1, 1)  # date column
        assert result[0][6] == 103.0  # close column

    def test_upsert_prices_replaces_existing_data(self, test_db):
        """Test that upsert replaces existing price data."""
        from hrp.data.db import get_db

        db = get_db()

        # Insert initial data
        df1 = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "adj_close": [103.0],
            "volume": [1000000],
            "source": ["polygon"],
        })
        _upsert_prices(db, df1)

        # Upsert with updated data
        df2 = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": [date(2024, 1, 1)],
            "open": [101.0],
            "high": [106.0],
            "low": [100.0],
            "close": [104.0],
            "adj_close": [104.0],
            "volume": [1100000],
            "source": ["yfinance"],
        })
        rows_inserted = _upsert_prices(db, df2)

        assert rows_inserted == 1

        # Verify only one row exists with updated data
        result = db.execute("SELECT * FROM prices WHERE symbol = 'AAPL'").fetchall()
        assert len(result) == 1
        assert result[0][6] == 104.0  # Updated close price
        assert result[0][9] == "yfinance"  # Updated source

    def test_upsert_prices_empty_dataframe(self, test_db):
        """Test that upsert handles empty DataFrame."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame()

        rows_inserted = _upsert_prices(db, df)

        assert rows_inserted == 0


class TestCorporateActionsIngestion:
    """Tests for corporate actions ingestion."""

    def test_successful_corporate_actions_ingestion(self, test_db):
        """Test successful corporate actions ingestion."""
        with patch("hrp.data.ingestion.corporate_actions.PolygonSource") as mock_polygon:

            mock_pg_instance = Mock()
            mock_pg_instance.get_corporate_actions.return_value = pd.DataFrame({
                "symbol": ["AAPL", "AAPL"],
                "date": [date(2024, 1, 15), date(2024, 2, 15)],
                "action_type": ["dividend", "dividend"],
                "factor": [0.25, 0.25],
                "source": ["polygon", "polygon"],
            })
            mock_polygon.return_value = mock_pg_instance

            stats = ingest_corporate_actions(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                source="polygon",
            )

            assert stats["symbols_success"] == 1
            assert stats["symbols_failed"] == 0
            assert stats["actions_fetched"] == 2
            assert stats["actions_inserted"] == 2
            assert stats["action_types"] == ["split", "dividend"]

    def test_corporate_actions_no_data(self, test_db):
        """Test corporate actions ingestion with no data."""
        with patch("hrp.data.ingestion.corporate_actions.PolygonSource") as mock_polygon:

            mock_pg_instance = Mock()
            mock_pg_instance.get_corporate_actions.return_value = pd.DataFrame()
            mock_polygon.return_value = mock_pg_instance

            stats = ingest_corporate_actions(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                source="polygon",
            )

            assert stats["symbols_success"] == 1
            assert stats["symbols_failed"] == 0
            assert stats["actions_fetched"] == 0
            assert stats["actions_inserted"] == 0

    def test_corporate_actions_api_error(self, test_db):
        """Test corporate actions ingestion with API error."""
        with patch("hrp.data.ingestion.corporate_actions.PolygonSource") as mock_polygon:

            mock_pg_instance = Mock()
            mock_pg_instance.get_corporate_actions.side_effect = BadResponse("Not found")
            mock_polygon.return_value = mock_pg_instance

            stats = ingest_corporate_actions(
                symbols=["INVALID"],
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                source="polygon",
            )

            assert stats["symbols_success"] == 0
            assert stats["symbols_failed"] == 1
            assert stats["failed_symbols"] == ["INVALID"]
            assert stats["actions_fetched"] == 0
            assert stats["actions_inserted"] == 0

    def test_corporate_actions_filter_by_type(self, test_db):
        """Test corporate actions ingestion with action type filter."""
        with patch("hrp.data.ingestion.corporate_actions.PolygonSource") as mock_polygon:

            mock_pg_instance = Mock()
            mock_pg_instance.get_corporate_actions.return_value = pd.DataFrame({
                "symbol": ["AAPL"],
                "date": [date(2024, 1, 15)],
                "action_type": ["split"],
                "factor": [2.0],
                "source": ["polygon"],
            })
            mock_polygon.return_value = mock_pg_instance

            stats = ingest_corporate_actions(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                action_types=["split"],
                source="polygon",
            )

            assert stats["symbols_success"] == 1
            assert stats["action_types"] == ["split"]
            assert stats["actions_fetched"] == 1

    def test_corporate_actions_polygon_init_failure(self, test_db):
        """Test corporate actions ingestion when Polygon init fails."""
        with patch("hrp.data.ingestion.corporate_actions.PolygonSource") as mock_polygon:

            mock_polygon.side_effect = ValueError("POLYGON_API_KEY not found")

            with pytest.raises(ValueError, match="POLYGON_API_KEY not found"):
                ingest_corporate_actions(
                    symbols=["AAPL"],
                    start=date(2024, 1, 1),
                    end=date(2024, 12, 31),
                    source="polygon",
                )

    def test_corporate_actions_invalid_source(self, test_db):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            ingest_corporate_actions(
                symbols=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                source="yfinance",  # Not yet supported
            )


class TestCorporateActionsUpsert:
    """Tests for _upsert_corporate_actions database function."""

    def test_upsert_corporate_actions_inserts_new_data(self, test_db):
        """Test that upsert inserts new corporate actions data."""
        from hrp.data.db import get_db

        db = get_db()

        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": [date(2024, 1, 15), date(2024, 2, 15)],
            "action_type": ["dividend", "split"],
            "factor": [0.25, 2.0],
            "source": ["polygon", "polygon"],
        })

        rows_inserted = _upsert_corporate_actions(db, df)

        assert rows_inserted == 2

        # Verify data in database
        result = db.execute(
            "SELECT * FROM corporate_actions WHERE symbol = 'AAPL' ORDER BY date"
        ).fetchall()
        assert len(result) == 2
        assert result[0][2] == date(2024, 1, 15)
        assert result[0][3] == "dividend"
        assert result[1][3] == "split"

    def test_upsert_corporate_actions_replaces_existing_data(self, test_db):
        """Test that upsert replaces existing corporate actions data."""
        from hrp.data.db import get_db

        db = get_db()

        # Insert initial data
        df1 = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": [date(2024, 1, 15)],
            "action_type": ["dividend"],
            "factor": [0.25],
            "source": ["polygon"],
        })
        _upsert_corporate_actions(db, df1)

        # Upsert with updated data
        df2 = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": [date(2024, 1, 15)],
            "action_type": ["dividend"],
            "factor": [0.30],  # Updated factor
            "source": ["polygon"],
        })
        rows_inserted = _upsert_corporate_actions(db, df2)

        assert rows_inserted == 1

        # Verify only one row exists with updated data
        result = db.execute(
            "SELECT * FROM corporate_actions WHERE symbol = 'AAPL'"
        ).fetchall()
        assert len(result) == 1
        assert result[0][4] == 0.30  # Updated factor

    def test_upsert_corporate_actions_empty_dataframe(self, test_db):
        """Test that upsert handles empty DataFrame."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame()

        rows_inserted = _upsert_corporate_actions(db, df)

        assert rows_inserted == 0
