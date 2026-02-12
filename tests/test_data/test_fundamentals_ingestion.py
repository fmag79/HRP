"""
Comprehensive tests for fundamentals data ingestion.

Tests cover:
- YFinanceFundamentalsAdapter class
- _validate_point_in_time function
- _upsert_fundamentals function
- ingest_fundamentals function
- get_fundamentals_stats function
- Snapshot fundamentals functions
- Error handling and edge cases
"""

import os
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from hrp.data.ingestion.fundamentals import (
    DEFAULT_METRICS,
    YFINANCE_PIT_BUFFER_DAYS,
    SNAPSHOT_METRICS,
    YFinanceFundamentalsAdapter,
    _validate_point_in_time,
    _upsert_fundamentals,
    ingest_fundamentals,
    get_fundamentals_stats,
    ingest_snapshot_fundamentals,
    _upsert_snapshot_fundamentals,
    get_latest_fundamentals,
)
from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fundamentals_test_db():
    """Create a temporary database for fundamentals ingestion tests."""
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
        # Insert data sources for FK constraint on fundamentals.source
        conn.execute("""
            INSERT INTO data_sources (source_id, source_type, api_name, status)
            VALUES
                ('test', 'manual', 'test', 'active'),
                ('yfinance', 'api', 'yfinance', 'active'),
                ('simfin', 'api', 'simfin', 'active')
            ON CONFLICT DO NOTHING
        """)
        # Insert universe entries
        conn.execute("""
            INSERT INTO universe (symbol, date, in_universe, sector)
            VALUES
                ('AAPL', '2023-06-01', TRUE, 'Technology'),
                ('MSFT', '2023-06-01', TRUE, 'Technology'),
                ('GOOGL', '2023-06-01', TRUE, 'Technology'),
                ('TEST', '2023-06-01', TRUE, 'Technology')
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
def mock_fundamentals_df():
    """Create mock fundamentals DataFrame."""
    return pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "report_date": [date(2023, 5, 15), date(2023, 5, 15), date(2023, 5, 15)],
        "period_end": [date(2023, 3, 31), date(2023, 3, 31), date(2023, 3, 31)],
        "metric": ["revenue", "eps", "book_value"],
        "value": [94836000000.0, 1.52, 50672000000.0],
        "source": ["yfinance", "yfinance", "yfinance"],  # Use valid source_id
    })


@pytest.fixture
def mock_snapshot_df():
    """Create mock snapshot fundamentals DataFrame."""
    return pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "market_cap": [2800000000000.0, 2500000000000.0],
        "pe_ratio": [28.5, 32.1],
        "pb_ratio": [45.2, 12.3],
        "dividend_yield": [0.005, 0.008],
        "ev_ebitda": [22.5, 25.3],
    })


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_metrics_not_empty(self):
        """DEFAULT_METRICS should contain metrics."""
        assert len(DEFAULT_METRICS) > 0

    def test_default_metrics_contains_expected(self):
        """DEFAULT_METRICS should contain expected metrics."""
        expected = ["revenue", "eps", "book_value"]
        for metric in expected:
            assert metric in DEFAULT_METRICS

    def test_yfinance_pit_buffer_reasonable(self):
        """YFINANCE_PIT_BUFFER_DAYS should be reasonable."""
        # 10-Q must be filed within 40-45 days of quarter end
        assert 30 <= YFINANCE_PIT_BUFFER_DAYS <= 60

    def test_snapshot_metrics_not_empty(self):
        """SNAPSHOT_METRICS should contain metrics."""
        assert len(SNAPSHOT_METRICS) > 0

    def test_snapshot_metrics_contains_expected(self):
        """SNAPSHOT_METRICS should contain expected metrics."""
        expected = ["market_cap", "pe_ratio", "pb_ratio"]
        for metric in expected:
            assert metric in SNAPSHOT_METRICS


# =============================================================================
# YFinanceFundamentalsAdapter Tests
# =============================================================================


class TestYFinanceFundamentalsAdapter:
    """Tests for YFinanceFundamentalsAdapter class."""

    def test_init(self):
        """YFinanceFundamentalsAdapter should initialize."""
        adapter = YFinanceFundamentalsAdapter()
        assert adapter.source_name == "yfinance"

    def test_get_fundamentals_returns_dataframe(self):
        """get_fundamentals should return a DataFrame."""
        adapter = YFinanceFundamentalsAdapter()

        # Mock yfinance ticker
        with patch("hrp.data.ingestion.fundamentals.yf.Ticker") as MockTicker:
            mock_ticker = MagicMock()

            # Create mock income statement (rows = metrics, columns = dates)
            mock_income = pd.DataFrame(
                {pd.Timestamp("2023-03-31"): [94836000000, 1.52]},
                index=pd.Index(["Total Revenue", "Diluted EPS"]),
            )
            mock_ticker.quarterly_income_stmt = mock_income

            # Create mock balance sheet
            mock_balance = pd.DataFrame(
                {pd.Timestamp("2023-03-31"): [50672000000]},
                index=pd.Index(["Stockholders Equity"]),
            )
            mock_ticker.quarterly_balance_sheet = mock_balance

            MockTicker.return_value = mock_ticker

            result = adapter.get_fundamentals("AAPL", metrics=["revenue", "eps"])

        assert isinstance(result, pd.DataFrame)

    def test_get_fundamentals_empty_data(self):
        """get_fundamentals should return empty DataFrame for no data."""
        adapter = YFinanceFundamentalsAdapter()

        with patch("hrp.data.ingestion.fundamentals.yf.Ticker") as MockTicker:
            mock_ticker = MagicMock()
            mock_ticker.quarterly_income_stmt = pd.DataFrame()
            mock_ticker.quarterly_balance_sheet = pd.DataFrame()
            MockTicker.return_value = mock_ticker

            result = adapter.get_fundamentals("INVALID")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_fundamentals_date_filtering(self):
        """get_fundamentals should filter by date range."""
        adapter = YFinanceFundamentalsAdapter()

        with patch("hrp.data.ingestion.fundamentals.yf.Ticker") as MockTicker:
            mock_ticker = MagicMock()

            # Create data with multiple periods (rows = metrics, columns = dates)
            mock_income = pd.DataFrame(
                {
                    pd.Timestamp("2022-12-31"): [90000000000],
                    pd.Timestamp("2023-03-31"): [95000000000],
                },
                index=pd.Index(["Total Revenue"]),
            )
            mock_ticker.quarterly_income_stmt = mock_income
            mock_ticker.quarterly_balance_sheet = pd.DataFrame()

            MockTicker.return_value = mock_ticker

            # Filter to only include Q1 2023 (report_date = period_end + 45 days)
            # Q1 2023 ends 2023-03-31, report_date ~= 2023-05-15
            result = adapter.get_fundamentals(
                "AAPL",
                metrics=["revenue"],
                start_date=date(2023, 5, 1),
                end_date=date(2023, 6, 30),
            )

        # Should only include the Q1 2023 data
        assert len(result) == 1
        assert result.iloc[0]["value"] == 95000000000

    def test_get_fundamentals_batch(self):
        """get_fundamentals_batch should fetch data for multiple symbols."""
        adapter = YFinanceFundamentalsAdapter()

        with patch.object(adapter, "get_fundamentals") as mock_get:
            mock_get.side_effect = [
                pd.DataFrame({"symbol": ["AAPL"], "metric": ["revenue"], "value": [100]}),
                pd.DataFrame({"symbol": ["MSFT"], "metric": ["revenue"], "value": [200]}),
            ]

            result = adapter.get_fundamentals_batch(
                symbols=["AAPL", "MSFT"],
                metrics=["revenue"],
            )

        assert len(result) == 2
        assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}

    def test_get_fundamentals_batch_partial_failure(self):
        """get_fundamentals_batch should continue on partial failures."""
        adapter = YFinanceFundamentalsAdapter()

        with patch.object(adapter, "get_fundamentals") as mock_get:
            mock_get.side_effect = [
                Exception("API Error"),
                pd.DataFrame({"symbol": ["MSFT"], "metric": ["revenue"], "value": [200]}),
            ]

            result = adapter.get_fundamentals_batch(
                symbols=["AAPL", "MSFT"],
                metrics=["revenue"],
            )

        # Should still return MSFT data
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "MSFT"

    def test_get_fundamentals_batch_all_fail(self):
        """get_fundamentals_batch should return empty DataFrame if all fail."""
        adapter = YFinanceFundamentalsAdapter()

        with patch.object(adapter, "get_fundamentals") as mock_get:
            mock_get.side_effect = [Exception("Error 1"), Exception("Error 2")]

            result = adapter.get_fundamentals_batch(
                symbols=["AAPL", "MSFT"],
                metrics=["revenue"],
            )

        assert result.empty


# =============================================================================
# _validate_point_in_time Tests
# =============================================================================


class TestValidatePointInTime:
    """Tests for _validate_point_in_time function."""

    def test_validate_empty_df(self):
        """_validate_point_in_time should handle empty DataFrame."""
        result = _validate_point_in_time(pd.DataFrame())
        assert result.empty

    def test_validate_valid_records(self):
        """_validate_point_in_time should keep valid records."""
        df = pd.DataFrame({
            "report_date": [date(2023, 5, 15)],
            "period_end": [date(2023, 3, 31)],
            "value": [100],
        })
        result = _validate_point_in_time(df)
        assert len(result) == 1

    def test_validate_filters_lookahead(self):
        """_validate_point_in_time should filter look-ahead bias records."""
        df = pd.DataFrame({
            "report_date": [date(2023, 3, 15)],  # Before period_end!
            "period_end": [date(2023, 3, 31)],
            "value": [100],
        })
        result = _validate_point_in_time(df)
        assert len(result) == 0

    def test_validate_mixed_records(self):
        """_validate_point_in_time should filter only invalid records."""
        df = pd.DataFrame({
            "report_date": [date(2023, 5, 15), date(2023, 3, 15)],
            "period_end": [date(2023, 3, 31), date(2023, 3, 31)],
            "value": [100, 200],
        })
        result = _validate_point_in_time(df)
        assert len(result) == 1
        assert result.iloc[0]["value"] == 100

    def test_validate_string_dates(self):
        """_validate_point_in_time should handle string date columns."""
        df = pd.DataFrame({
            "report_date": ["2023-05-15"],
            "period_end": ["2023-03-31"],
            "value": [100],
        })
        result = _validate_point_in_time(df)
        assert len(result) == 1


# =============================================================================
# _upsert_fundamentals Tests
# =============================================================================


class TestUpsertFundamentals:
    """Tests for _upsert_fundamentals function."""

    def test_upsert_empty_df(self, fundamentals_test_db):
        """_upsert_fundamentals should return 0 for empty DataFrame."""
        db = get_db(fundamentals_test_db)
        result = _upsert_fundamentals(db, pd.DataFrame())
        assert result == 0

    def test_upsert_inserts_records(self, fundamentals_test_db, mock_fundamentals_df):
        """_upsert_fundamentals should insert records into database."""
        db = get_db(fundamentals_test_db)
        result = _upsert_fundamentals(db, mock_fundamentals_df)

        assert result == 3

        # Verify in database
        count = db.fetchone("SELECT COUNT(*) FROM fundamentals")[0]
        assert count == 3

    def test_upsert_upsert_behavior(self, fundamentals_test_db, mock_fundamentals_df):
        """_upsert_fundamentals should update existing records."""
        db = get_db(fundamentals_test_db)

        # First insert
        _upsert_fundamentals(db, mock_fundamentals_df)

        # Modify and insert again
        mock_fundamentals_df["value"] = [100000000000.0, 2.0, 60000000000.0]
        _upsert_fundamentals(db, mock_fundamentals_df)

        # Should still have 3 records
        count = db.fetchone("SELECT COUNT(*) FROM fundamentals")[0]
        assert count == 3


# =============================================================================
# ingest_fundamentals Tests
# =============================================================================


class TestIngestFundamentals:
    """Tests for ingest_fundamentals function."""

    def test_ingest_unknown_source_raises(self, fundamentals_test_db):
        """ingest_fundamentals should raise ValueError for unknown source."""
        with pytest.raises(ValueError) as exc_info:
            ingest_fundamentals(
                symbols=["AAPL"],
                source="unknown_source",
            )
        assert "Unknown source" in str(exc_info.value)

    def test_ingest_yfinance_source(self, fundamentals_test_db, mock_fundamentals_df):
        """ingest_fundamentals should work with yfinance source."""
        with patch.object(
            YFinanceFundamentalsAdapter, "get_fundamentals"
        ) as mock_get:
            mock_get.return_value = mock_fundamentals_df

            stats = ingest_fundamentals(
                symbols=["AAPL"],
                source="yfinance",
            )

        assert stats["symbols_requested"] == 1
        assert stats["symbols_success"] == 1
        assert stats["records_inserted"] == 3

    def test_ingest_simfin_fallback(self, fundamentals_test_db, mock_fundamentals_df):
        """ingest_fundamentals should fall back to yfinance if simfin unavailable."""
        # SimFinSource is imported inside the function, so we patch the import
        with patch.dict("sys.modules", {"hrp.data.sources.simfin_source": MagicMock()}):
            with patch(
                "hrp.data.sources.simfin_source.SimFinSource",
                side_effect=ValueError("No API key"),
            ):
                with patch.object(
                    YFinanceFundamentalsAdapter, "get_fundamentals"
                ) as mock_get:
                    mock_get.return_value = mock_fundamentals_df

                    stats = ingest_fundamentals(
                        symbols=["AAPL"],
                        source="simfin",
                    )

        assert stats["symbols_success"] == 1

    def test_ingest_tracks_pit_violations(self, fundamentals_test_db):
        """ingest_fundamentals should track point-in-time violations."""
        # Create data with PIT violation
        invalid_df = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": [date(2023, 3, 15)],  # Before period_end!
            "period_end": [date(2023, 3, 31)],
            "metric": ["revenue"],
            "value": [100000000000.0],
            "source": ["test"],
        })

        with patch.object(
            YFinanceFundamentalsAdapter, "get_fundamentals"
        ) as mock_get:
            mock_get.return_value = invalid_df

            stats = ingest_fundamentals(
                symbols=["AAPL"],
                source="yfinance",
            )

        assert stats["pit_violations_filtered"] == 1
        assert stats["records_inserted"] == 0

    def test_ingest_multiple_symbols(self, fundamentals_test_db):
        """ingest_fundamentals should handle multiple symbols."""
        df1 = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": [date(2023, 5, 15)],
            "period_end": [date(2023, 3, 31)],
            "metric": ["revenue"],
            "value": [100],
            "source": ["test"],
        })
        df2 = pd.DataFrame({
            "symbol": ["MSFT"],
            "report_date": [date(2023, 5, 15)],
            "period_end": [date(2023, 3, 31)],
            "metric": ["revenue"],
            "value": [200],
            "source": ["test"],
        })

        with patch.object(
            YFinanceFundamentalsAdapter, "get_fundamentals"
        ) as mock_get:
            mock_get.side_effect = [df1, df2]

            stats = ingest_fundamentals(
                symbols=["AAPL", "MSFT"],
                source="yfinance",
            )

        assert stats["symbols_requested"] == 2
        assert stats["symbols_success"] == 2


# =============================================================================
# get_fundamentals_stats Tests
# =============================================================================


class TestGetFundamentalsStats:
    """Tests for get_fundamentals_stats function."""

    def test_stats_empty_db(self, fundamentals_test_db):
        """get_fundamentals_stats should handle empty database."""
        stats = get_fundamentals_stats()

        assert stats["total_records"] == 0
        assert stats["unique_symbols"] == 0

    def test_stats_with_data(self, fundamentals_test_db, mock_fundamentals_df):
        """get_fundamentals_stats should return correct statistics."""
        db = get_db(fundamentals_test_db)
        _upsert_fundamentals(db, mock_fundamentals_df)

        stats = get_fundamentals_stats()

        assert stats["total_records"] == 3
        assert stats["unique_symbols"] == 1
        assert len(stats["metrics"]) == 3


# =============================================================================
# Snapshot Fundamentals Tests
# =============================================================================


class TestSnapshotFundamentals:
    """Tests for snapshot fundamentals functions."""

    def test_upsert_snapshot_empty(self, fundamentals_test_db):
        """_upsert_snapshot_fundamentals should return 0 for empty DataFrame."""
        db = get_db(fundamentals_test_db)
        result = _upsert_snapshot_fundamentals(db, pd.DataFrame())
        assert result == 0

    def test_upsert_snapshot_inserts(self, fundamentals_test_db):
        """_upsert_snapshot_fundamentals should insert records."""
        db = get_db(fundamentals_test_db)

        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": [date(2023, 6, 1), date(2023, 6, 1)],
            "feature_name": ["market_cap", "pe_ratio"],
            "value": [2800000000000.0, 28.5],
            "version": ["v1", "v1"],
        })

        result = _upsert_snapshot_fundamentals(db, df)
        assert result == 2

    def test_ingest_snapshot_fundamentals(self, fundamentals_test_db, mock_snapshot_df):
        """ingest_snapshot_fundamentals should ingest snapshot data."""
        # FundamentalSource is imported inside the function
        with patch(
            "hrp.data.sources.fundamental_source.FundamentalSource"
        ) as MockSource:
            mock_source = MagicMock()
            mock_source.get_fundamentals_batch.return_value = mock_snapshot_df
            MockSource.return_value = mock_source

            stats = ingest_snapshot_fundamentals(
                symbols=["AAPL", "MSFT"],
                as_of_date=date(2023, 6, 1),
            )

        assert stats["symbols_requested"] == 2
        assert stats["symbols_success"] == 2
        # 2 symbols * 5 metrics = 10 records
        assert stats["records_inserted"] == 10

    def test_ingest_snapshot_empty_data(self, fundamentals_test_db):
        """ingest_snapshot_fundamentals should handle empty data."""
        with patch(
            "hrp.data.sources.fundamental_source.FundamentalSource"
        ) as MockSource:
            mock_source = MagicMock()
            mock_source.get_fundamentals_batch.return_value = pd.DataFrame()
            MockSource.return_value = mock_source

            stats = ingest_snapshot_fundamentals(
                symbols=["INVALID"],
                as_of_date=date(2023, 6, 1),
            )

        assert stats["symbols_failed"] == 1

    def test_get_latest_fundamentals(self, fundamentals_test_db):
        """get_latest_fundamentals should retrieve latest values."""
        db = get_db(fundamentals_test_db)

        # Insert snapshot data with same date for all metrics (typical use case)
        with db.connection() as conn:
            conn.execute("""
                INSERT INTO features (symbol, date, feature_name, value, version)
                VALUES
                    ('AAPL', '2023-06-01', 'market_cap', 2700000000000.0, 'v1'),
                    ('AAPL', '2023-06-01', 'pe_ratio', 28.5, 'v1')
            """)

        result = get_latest_fundamentals(["AAPL"], ["market_cap", "pe_ratio"])

        assert not result.empty
        # Result is pivoted, so market_cap and pe_ratio are columns
        aapl_row = result[result["symbol"] == "AAPL"].iloc[0]
        assert aapl_row["market_cap"] == 2700000000000.0
        assert aapl_row["pe_ratio"] == 28.5

    def test_get_latest_fundamentals_empty(self, fundamentals_test_db):
        """get_latest_fundamentals should return empty for no data."""
        result = get_latest_fundamentals(["INVALID"], ["market_cap"])
        assert result.empty


# =============================================================================
# Integration Tests
# =============================================================================


class TestFundamentalsIngestionIntegration:
    """Integration tests for fundamentals ingestion."""

    def test_full_workflow(self, fundamentals_test_db, mock_fundamentals_df):
        """Test complete ingestion and stats workflow."""
        with patch.object(
            YFinanceFundamentalsAdapter, "get_fundamentals"
        ) as mock_get:
            mock_get.return_value = mock_fundamentals_df

            # Ingest
            ingest_stats = ingest_fundamentals(
                symbols=["AAPL"],
                source="yfinance",
            )

        assert ingest_stats["symbols_success"] == 1
        assert ingest_stats["records_inserted"] == 3

        # Check stats
        db_stats = get_fundamentals_stats()
        assert db_stats["total_records"] == 3
