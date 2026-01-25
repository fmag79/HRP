"""
Tests for the fundamentals data ingestion module.

Tests cover:
- Point-in-time validation (valid pass, invalid filtered)
- Database upsert (insert new, update existing)
- YFinanceFundamentalsAdapter behavior
- ingest_fundamentals function
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.ingestion.fundamentals import (
    DEFAULT_METRICS,
    YFINANCE_PIT_BUFFER_DAYS,
    YFinanceFundamentalsAdapter,
    _upsert_fundamentals,
    _validate_point_in_time,
    get_fundamentals_stats,
    ingest_fundamentals,
)


class TestValidatePointInTime:
    """Tests for the _validate_point_in_time function."""

    def test_valid_records_pass(self):
        """Test that valid point-in-time records pass validation."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "report_date": [date(2023, 5, 1), date(2023, 8, 1)],
            "period_end": [date(2023, 3, 31), date(2023, 6, 30)],
            "metric": ["revenue", "revenue"],
            "value": [100.0, 110.0],
            "source": ["test", "test"],
        })

        result = _validate_point_in_time(df)

        assert len(result) == 2
        assert list(result["value"]) == [100.0, 110.0]

    def test_invalid_records_filtered(self):
        """Test that records with period_end > report_date are filtered."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "report_date": [date(2023, 5, 1), date(2023, 3, 15), date(2023, 8, 1)],
            "period_end": [date(2023, 3, 31), date(2023, 3, 31), date(2023, 6, 30)],
            "metric": ["revenue", "revenue", "revenue"],
            "value": [100.0, 90.0, 110.0],  # Second has future data
            "source": ["test", "test", "test"],
        })

        result = _validate_point_in_time(df)

        # Second record should be filtered (period_end > report_date)
        assert len(result) == 2
        assert list(result["value"]) == [100.0, 110.0]

    def test_same_date_is_valid(self):
        """Test that period_end == report_date is valid (edge case)."""
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": [date(2023, 3, 31)],
            "period_end": [date(2023, 3, 31)],
            "metric": ["revenue"],
            "value": [100.0],
            "source": ["test"],
        })

        result = _validate_point_in_time(df)

        assert len(result) == 1
        assert result.iloc[0]["value"] == 100.0

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = _validate_point_in_time(df)
        assert len(result) == 0

    def test_string_dates_converted(self):
        """Test that string dates are properly converted."""
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": ["2023-05-01"],
            "period_end": ["2023-03-31"],
            "metric": ["revenue"],
            "value": [100.0],
            "source": ["test"],
        })

        result = _validate_point_in_time(df)
        assert len(result) == 1


class TestUpsertFundamentals:
    """Tests for the _upsert_fundamentals function."""

    def test_insert_new_records(self, test_db_with_sources):
        """Test inserting new fundamental records."""
        from hrp.data.db import get_db

        db = get_db(test_db_with_sources)

        # Add data source for fundamentals
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO data_sources (source_id, source_type, status) "
                "VALUES ('test_fundamentals', 'test', 'active') ON CONFLICT DO NOTHING"
            )

        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "report_date": [date(2023, 5, 1), date(2023, 8, 1)],
            "period_end": [date(2023, 3, 31), date(2023, 6, 30)],
            "metric": ["revenue", "revenue"],
            "value": [100.0, 110.0],
            "source": ["test_fundamentals", "test_fundamentals"],
        })

        rows_inserted = _upsert_fundamentals(db, df)

        assert rows_inserted == 2

        # Verify data was inserted
        with db.connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM fundamentals WHERE symbol = 'AAPL'"
            ).fetchone()
            assert result[0] == 2

    def test_update_existing_records(self, test_db_with_sources):
        """Test updating existing fundamental records (upsert)."""
        from hrp.data.db import get_db

        db = get_db(test_db_with_sources)

        # Add data source
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO data_sources (source_id, source_type, status) "
                "VALUES ('test_fundamentals', 'test', 'active') ON CONFLICT DO NOTHING"
            )

        # Insert initial data
        df1 = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": [date(2023, 5, 1)],
            "period_end": [date(2023, 3, 31)],
            "metric": ["revenue"],
            "value": [100.0],
            "source": ["test_fundamentals"],
        })
        _upsert_fundamentals(db, df1)

        # Update with new value
        df2 = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": [date(2023, 5, 1)],
            "period_end": [date(2023, 3, 31)],
            "metric": ["revenue"],
            "value": [105.0],  # Updated value
            "source": ["test_fundamentals"],
        })
        _upsert_fundamentals(db, df2)

        # Verify only one record exists with updated value
        with db.connection() as conn:
            result = conn.execute(
                "SELECT value FROM fundamentals "
                "WHERE symbol = 'AAPL' AND report_date = '2023-05-01' AND metric = 'revenue'"
            ).fetchone()
            assert result[0] == 105.0

    def test_empty_dataframe(self, test_db_with_sources):
        """Test handling of empty DataFrame."""
        from hrp.data.db import get_db

        db = get_db(test_db_with_sources)
        df = pd.DataFrame()

        rows_inserted = _upsert_fundamentals(db, df)
        assert rows_inserted == 0


class TestYFinanceFundamentalsAdapter:
    """Tests for the YFinanceFundamentalsAdapter class."""

    def test_get_fundamentals_returns_dataframe(self):
        """Test that get_fundamentals returns proper DataFrame structure."""
        adapter = YFinanceFundamentalsAdapter()

        # Mock the yfinance Ticker
        mock_income = pd.DataFrame(
            {"2023-03-31": [100e9, 25e9, 1.50]},
            index=["Total Revenue", "Net Income", "Diluted EPS"],
        )
        mock_income.columns = pd.to_datetime(mock_income.columns)

        mock_balance = pd.DataFrame(
            {"2023-03-31": [350e9, 250e9, 100e9]},
            index=["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity"],
        )
        mock_balance.columns = pd.to_datetime(mock_balance.columns)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.quarterly_income_stmt = mock_income
            mock_instance.quarterly_balance_sheet = mock_balance
            mock_ticker.return_value = mock_instance

            result = adapter.get_fundamentals("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert list(result.columns) == [
            "symbol", "report_date", "period_end", "metric", "value", "source"
        ]
        assert all(result["symbol"] == "AAPL")
        assert all(result["source"] == "yfinance")

    def test_point_in_time_buffer_applied(self):
        """Test that 45-day buffer is applied to report_date."""
        adapter = YFinanceFundamentalsAdapter()

        period_end = date(2023, 3, 31)
        expected_report_date = period_end + timedelta(days=YFINANCE_PIT_BUFFER_DAYS)

        mock_income = pd.DataFrame(
            {"2023-03-31": [100e9]},
            index=["Total Revenue"],
        )
        mock_income.columns = pd.to_datetime(mock_income.columns)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.quarterly_income_stmt = mock_income
            mock_instance.quarterly_balance_sheet = pd.DataFrame()
            mock_ticker.return_value = mock_instance

            result = adapter.get_fundamentals("AAPL", metrics=["revenue"])

        assert len(result) == 1
        assert result.iloc[0]["period_end"] == period_end
        assert result.iloc[0]["report_date"] == expected_report_date

    def test_empty_data_returns_empty_dataframe(self):
        """Test that missing data returns empty DataFrame."""
        adapter = YFinanceFundamentalsAdapter()

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.quarterly_income_stmt = pd.DataFrame()
            mock_instance.quarterly_balance_sheet = pd.DataFrame()
            mock_ticker.return_value = mock_instance

            result = adapter.get_fundamentals("INVALID")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_date_filtering(self):
        """Test that date filtering works correctly."""
        adapter = YFinanceFundamentalsAdapter()

        mock_income = pd.DataFrame(
            {
                "2022-12-31": [90e9],  # Before start_date
                "2023-03-31": [100e9],  # In range
                "2023-06-30": [110e9],  # In range
            },
            index=["Total Revenue"],
        )
        mock_income.columns = pd.to_datetime(mock_income.columns)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.quarterly_income_stmt = mock_income
            mock_instance.quarterly_balance_sheet = pd.DataFrame()
            mock_ticker.return_value = mock_instance

            # Note: report_date = period_end + 45 days
            # 2022-12-31 + 45 = 2023-02-14 (before start_date 2023-05-01)
            # 2023-03-31 + 45 = 2023-05-15 (in range)
            # 2023-06-30 + 45 = 2023-08-14 (in range)
            result = adapter.get_fundamentals(
                "AAPL",
                metrics=["revenue"],
                start_date=date(2023, 5, 1),
                end_date=date(2023, 12, 31),
            )

        # First period should be filtered out
        assert len(result) == 2

    def test_batch_fetch(self):
        """Test batch fetching for multiple symbols."""
        adapter = YFinanceFundamentalsAdapter()

        mock_income = pd.DataFrame(
            {"2023-03-31": [100e9]},
            index=["Total Revenue"],
        )
        mock_income.columns = pd.to_datetime(mock_income.columns)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.quarterly_income_stmt = mock_income
            mock_instance.quarterly_balance_sheet = pd.DataFrame()
            mock_ticker.return_value = mock_instance

            result = adapter.get_fundamentals_batch(
                ["AAPL", "MSFT"],
                metrics=["revenue"],
            )

        assert isinstance(result, pd.DataFrame)
        # Each symbol should have one record
        assert len(result) == 2
        assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}


class TestIngestFundamentals:
    """Tests for the ingest_fundamentals function."""

    def test_successful_ingestion(self, test_db_with_sources):
        """Test successful fundamentals ingestion."""
        from hrp.data.db import get_db

        db = get_db(test_db_with_sources)

        # Add yfinance data source
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO data_sources (source_id, source_type, status) "
                "VALUES ('yfinance', 'api', 'active') ON CONFLICT DO NOTHING"
            )

        mock_income = pd.DataFrame(
            {"2023-03-31": [100e9, 25e9, 1.50]},
            index=["Total Revenue", "Net Income", "Diluted EPS"],
        )
        mock_income.columns = pd.to_datetime(mock_income.columns)

        mock_balance = pd.DataFrame(
            {"2023-03-31": [350e9, 250e9, 100e9]},
            index=["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity"],
        )
        mock_balance.columns = pd.to_datetime(mock_balance.columns)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.quarterly_income_stmt = mock_income
            mock_instance.quarterly_balance_sheet = mock_balance
            mock_ticker.return_value = mock_instance

            stats = ingest_fundamentals(
                symbols=["AAPL"],
                source="yfinance",
            )

        assert stats["symbols_success"] == 1
        assert stats["symbols_failed"] == 0
        assert stats["records_inserted"] > 0

    def test_failed_symbol_continues(self, test_db_with_sources):
        """Test that failure for one symbol doesn't stop others."""
        from hrp.data.db import get_db

        db = get_db(test_db_with_sources)

        # Add yfinance data source
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO data_sources (source_id, source_type, status) "
                "VALUES ('yfinance', 'api', 'active') ON CONFLICT DO NOTHING"
            )

        mock_income = pd.DataFrame(
            {"2023-03-31": [100e9]},
            index=["Total Revenue"],
        )
        mock_income.columns = pd.to_datetime(mock_income.columns)

        call_count = [0]

        def mock_ticker_side_effect(symbol):
            call_count[0] += 1
            mock_instance = MagicMock()
            if symbol == "INVALID":
                mock_instance.quarterly_income_stmt = pd.DataFrame()
                mock_instance.quarterly_balance_sheet = pd.DataFrame()
            else:
                mock_instance.quarterly_income_stmt = mock_income
                mock_instance.quarterly_balance_sheet = pd.DataFrame()
            return mock_instance

        with patch("yfinance.Ticker", side_effect=mock_ticker_side_effect):
            stats = ingest_fundamentals(
                symbols=["AAPL", "INVALID", "MSFT"],
                metrics=["revenue"],
                source="yfinance",
            )

        assert stats["symbols_success"] == 2
        assert stats["symbols_failed"] == 1
        assert "INVALID" in stats["failed_symbols"]

    def test_invalid_source_raises(self, test_db_with_sources):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            ingest_fundamentals(
                symbols=["AAPL"],
                source="invalid_source",
            )


class TestGetFundamentalsStats:
    """Tests for the get_fundamentals_stats function."""

    def test_stats_with_data(self, test_db_with_sources):
        """Test statistics with existing data."""
        from hrp.data.db import get_db

        db = get_db(test_db_with_sources)

        # Add data source
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO data_sources (source_id, source_type, status) "
                "VALUES ('test_fundamentals', 'test', 'active') ON CONFLICT DO NOTHING"
            )

        # Insert some test data
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "report_date": [date(2023, 5, 1), date(2023, 8, 1), date(2023, 5, 1)],
            "period_end": [date(2023, 3, 31), date(2023, 6, 30), date(2023, 3, 31)],
            "metric": ["revenue", "revenue", "eps"],
            "value": [100.0, 110.0, 5.5],
            "source": ["test_fundamentals", "test_fundamentals", "test_fundamentals"],
        })
        _upsert_fundamentals(db, df)

        stats = get_fundamentals_stats()

        assert stats["total_records"] == 3
        assert stats["unique_symbols"] == 2
        assert stats["date_range"]["start"] is not None
        assert stats["date_range"]["end"] is not None
        assert len(stats["metrics"]) == 2  # revenue and eps
        assert len(stats["per_symbol"]) == 2

    def test_stats_empty_database(self, test_db_with_sources):
        """Test statistics with empty fundamentals table."""
        stats = get_fundamentals_stats()

        assert stats["total_records"] == 0
        assert stats["unique_symbols"] == 0
