"""
Tests for fundamental analysis features added in Phase 2.

Tests cover:
- FundamentalSource class (get_fundamentals, get_fundamentals_batch)
- Snapshot fundamentals ingestion
- Fundamental passthrough features in FEATURE_FUNCTIONS
- SnapshotFundamentalsJob
"""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.features.computation import (
    FEATURE_FUNCTIONS,
    compute_dividend_yield,
    compute_ev_ebitda,
    compute_market_cap,
    compute_pb_ratio,
    compute_pe_ratio,
    compute_shares_outstanding,
)
from hrp.data.schema import create_tables
from hrp.data.sources.fundamental_source import FundamentalSource, FUNDAMENTAL_METRICS


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fundamental_db():
    """Create a temporary DuckDB database with schema for testing fundamentals."""
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
                ('GOOGL', 'Alphabet Inc.', 'NASDAQ')
            ON CONFLICT DO NOTHING
        """)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", ".tmp", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def sample_prices():
    """Create minimal sample price data for passthrough feature testing."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    symbols = ["AAPL", "MSFT"]

    data = []
    for symbol in symbols:
        for dt in dates:
            data.append({
                "date": dt,
                "symbol": symbol,
                "open": 150.0,
                "high": 152.0,
                "low": 148.0,
                "close": 151.0,
                "adj_close": 151.0,
                "volume": 1000000,
            })

    df = pd.DataFrame(data)
    df = df.set_index(["date", "symbol"])
    return df


@pytest.fixture
def mock_yfinance_info():
    """Mock yfinance ticker info for testing."""
    return {
        "marketCap": 3000000000000,  # $3T
        "trailingPE": 28.5,
        "priceToBook": 45.2,
        "dividendYield": 0.005,  # 0.5%
        "enterpriseToEbitda": 22.1,
        "sharesOutstanding": 15700000000,  # 15.7B shares
        "longName": "Apple Inc.",
        "sector": "Technology",
    }


# =============================================================================
# Test: Fundamental Feature Registration
# =============================================================================


class TestFundamentalFeatureRegistration:
    """Test that fundamental features are properly registered."""

    def test_market_cap_registered(self):
        """market_cap should be in FEATURE_FUNCTIONS."""
        assert "market_cap" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["market_cap"] == compute_market_cap

    def test_pe_ratio_registered(self):
        """pe_ratio should be in FEATURE_FUNCTIONS."""
        assert "pe_ratio" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["pe_ratio"] == compute_pe_ratio

    def test_pb_ratio_registered(self):
        """pb_ratio should be in FEATURE_FUNCTIONS."""
        assert "pb_ratio" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["pb_ratio"] == compute_pb_ratio

    def test_dividend_yield_registered(self):
        """dividend_yield should be in FEATURE_FUNCTIONS."""
        assert "dividend_yield" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["dividend_yield"] == compute_dividend_yield

    def test_ev_ebitda_registered(self):
        """ev_ebitda should be in FEATURE_FUNCTIONS."""
        assert "ev_ebitda" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["ev_ebitda"] == compute_ev_ebitda

    def test_shares_outstanding_registered(self):
        """shares_outstanding should be in FEATURE_FUNCTIONS."""
        assert "shares_outstanding" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["shares_outstanding"] == compute_shares_outstanding

    def test_total_features_count(self):
        """Should have 45 total features (39 technical + 6 fundamental)."""
        assert len(FEATURE_FUNCTIONS) == 45


# =============================================================================
# Test: Passthrough Features
# =============================================================================


class TestPassthroughFeatures:
    """Test that passthrough features return NaN (values come from ingestion)."""

    def test_market_cap_returns_nan(self, sample_prices):
        """market_cap compute function should return NaN values."""
        result = compute_market_cap(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "market_cap" in result.columns
        assert len(result) == len(sample_prices)
        # All values should be NaN (passthrough)
        assert result["market_cap"].isna().all()

    def test_pe_ratio_returns_nan(self, sample_prices):
        """pe_ratio compute function should return NaN values."""
        result = compute_pe_ratio(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "pe_ratio" in result.columns
        assert result["pe_ratio"].isna().all()

    def test_pb_ratio_returns_nan(self, sample_prices):
        """pb_ratio compute function should return NaN values."""
        result = compute_pb_ratio(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "pb_ratio" in result.columns
        assert result["pb_ratio"].isna().all()

    def test_dividend_yield_returns_nan(self, sample_prices):
        """dividend_yield compute function should return NaN values."""
        result = compute_dividend_yield(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "dividend_yield" in result.columns
        assert result["dividend_yield"].isna().all()

    def test_ev_ebitda_returns_nan(self, sample_prices):
        """ev_ebitda compute function should return NaN values."""
        result = compute_ev_ebitda(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ev_ebitda" in result.columns
        assert result["ev_ebitda"].isna().all()

    def test_shares_outstanding_returns_nan(self, sample_prices):
        """shares_outstanding compute function should return NaN values."""
        result = compute_shares_outstanding(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "shares_outstanding" in result.columns
        assert result["shares_outstanding"].isna().all()


# =============================================================================
# Test: FundamentalSource
# =============================================================================


class TestFundamentalSource:
    """Tests for FundamentalSource class."""

    def test_fundamental_metrics_mapping(self):
        """FUNDAMENTAL_METRICS should have correct yfinance mappings."""
        assert "market_cap" in FUNDAMENTAL_METRICS
        assert FUNDAMENTAL_METRICS["market_cap"] == "marketCap"
        assert FUNDAMENTAL_METRICS["pe_ratio"] == "trailingPE"
        assert FUNDAMENTAL_METRICS["pb_ratio"] == "priceToBook"
        assert FUNDAMENTAL_METRICS["dividend_yield"] == "dividendYield"
        assert FUNDAMENTAL_METRICS["ev_ebitda"] == "enterpriseToEbitda"
        assert FUNDAMENTAL_METRICS["shares_outstanding"] == "sharesOutstanding"

    def test_source_name(self):
        """FundamentalSource should have correct source_name."""
        source = FundamentalSource()
        assert source.source_name == "yfinance_fundamentals"

    @patch("hrp.data.sources.fundamental_source.yf.Ticker")
    def test_get_fundamentals(self, mock_ticker_class, mock_yfinance_info):
        """get_fundamentals should extract metrics from yfinance info."""
        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        source = FundamentalSource()
        result = source.get_fundamentals("AAPL", as_of_date=date(2024, 1, 15))

        assert result["symbol"] == "AAPL"
        assert result["date"] == date(2024, 1, 15)
        assert result["market_cap"] == 3000000000000
        assert result["pe_ratio"] == 28.5
        assert result["pb_ratio"] == 45.2
        assert result["dividend_yield"] == 0.005
        assert result["ev_ebitda"] == 22.1
        assert result["source"] == "yfinance_fundamentals"

    @patch("hrp.data.sources.fundamental_source.yf.Ticker")
    def test_get_fundamentals_handles_missing(self, mock_ticker_class):
        """get_fundamentals should handle missing metrics gracefully."""
        # Setup mock with partial data
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "marketCap": 1000000000,
            # Missing: trailingPE, priceToBook, dividendYield, enterpriseToEbitda
        }
        mock_ticker_class.return_value = mock_ticker

        source = FundamentalSource()
        result = source.get_fundamentals("TEST")

        assert result["market_cap"] == 1000000000
        assert result["pe_ratio"] is None
        assert result["pb_ratio"] is None
        assert result["dividend_yield"] is None
        assert result["ev_ebitda"] is None

    @patch("hrp.data.sources.fundamental_source.yf.Ticker")
    def test_get_fundamentals_batch(self, mock_ticker_class, mock_yfinance_info):
        """get_fundamentals_batch should return DataFrame for multiple symbols."""
        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        source = FundamentalSource()
        result = source.get_fundamentals_batch(
            ["AAPL", "MSFT"],
            as_of_date=date(2024, 1, 15)
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result["symbol"].values) == {"AAPL", "MSFT"}
        assert "market_cap" in result.columns
        assert "pe_ratio" in result.columns

    @patch("hrp.data.sources.fundamental_source.yf.Ticker")
    def test_validate_symbol(self, mock_ticker_class, mock_yfinance_info):
        """validate_symbol should return True for valid symbols."""
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        source = FundamentalSource()
        assert source.validate_symbol("AAPL") is True

    @patch("hrp.data.sources.fundamental_source.yf.Ticker")
    def test_validate_symbol_invalid(self, mock_ticker_class):
        """validate_symbol should return False for invalid symbols."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"marketCap": None}  # No market cap = invalid
        mock_ticker_class.return_value = mock_ticker

        source = FundamentalSource()
        assert source.validate_symbol("INVALID") is False


# =============================================================================
# Test: Snapshot Fundamentals Ingestion
# =============================================================================


class TestSnapshotFundamentalsIngestion:
    """Tests for snapshot fundamentals ingestion."""

    def test_ingest_snapshot_fundamentals(self, fundamental_db):
        """ingest_snapshot_fundamentals should store data in features table."""
        from hrp.data.ingestion.fundamentals import ingest_snapshot_fundamentals

        # Setup mock
        mock_source = MagicMock()
        mock_source.get_fundamentals_batch.return_value = pd.DataFrame([
            {
                "symbol": "AAPL",
                "date": date(2024, 1, 15),
                "market_cap": 3e12,
                "pe_ratio": 28.5,
                "pb_ratio": 45.2,
                "dividend_yield": 0.005,
                "ev_ebitda": 22.1,
                "source": "yfinance_fundamentals",
            }
        ])

        # Run ingestion with test database
        with patch("hrp.data.ingestion.fundamentals.get_db") as mock_get_db, \
             patch("hrp.data.sources.fundamental_source.yf.Ticker") as mock_ticker:
            mock_get_db.return_value = DatabaseManager(fundamental_db)

            # Setup yfinance mock (use smaller market cap to fit DECIMAL(18,6))
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {
                "marketCap": 300000000000,  # 300B fits in DECIMAL(18,6)
                "trailingPE": 28.5,
                "priceToBook": 45.2,
                "dividendYield": 0.005,
                "enterpriseToEbitda": 22.1,
                "sharesOutstanding": 15700000000,  # 15.7B shares
            }
            mock_ticker.return_value = mock_ticker_instance

            result = ingest_snapshot_fundamentals(
                symbols=["AAPL"],
                as_of_date=date(2024, 1, 15)
            )

        assert result["symbols_success"] == 1
        assert result["records_fetched"] == 6  # 6 metrics for 1 symbol
        assert result["records_inserted"] == 6

    def test_ingest_handles_partial_data(self, fundamental_db):
        """Ingestion should handle symbols with missing data."""
        from hrp.data.ingestion.fundamentals import ingest_snapshot_fundamentals

        with patch("hrp.data.ingestion.fundamentals.get_db") as mock_get_db, \
             patch("hrp.data.sources.fundamental_source.yf.Ticker") as mock_ticker:
            mock_get_db.return_value = DatabaseManager(fundamental_db)

            # Setup yfinance mock with partial data (use smaller market cap)
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {
                "marketCap": 300000000000,  # 300B fits in DECIMAL(18,6)
                "trailingPE": None,  # Missing
                "priceToBook": None,  # Missing
                "dividendYield": 0.005,
                "enterpriseToEbitda": None,  # Missing
            }
            mock_ticker.return_value = mock_ticker_instance

            result = ingest_snapshot_fundamentals(
                symbols=["AAPL"],
                as_of_date=date(2024, 1, 15)
            )

        # Only 2 non-None metrics should be stored
        assert result["records_fetched"] == 2
        assert result["records_inserted"] == 2


# =============================================================================
# Test: SnapshotFundamentalsJob
# =============================================================================


class TestSnapshotFundamentalsJob:
    """Tests for SnapshotFundamentalsJob."""

    def test_job_initialization(self):
        """SnapshotFundamentalsJob should initialize with correct defaults."""
        from hrp.agents.jobs import SnapshotFundamentalsJob

        job = SnapshotFundamentalsJob()

        assert job.job_id == "snapshot_fundamentals"
        assert job.symbols is None
        assert job.as_of_date == date.today()

    def test_job_custom_parameters(self):
        """SnapshotFundamentalsJob should accept custom parameters."""
        from hrp.agents.jobs import SnapshotFundamentalsJob

        job = SnapshotFundamentalsJob(
            symbols=["AAPL", "MSFT"],
            as_of_date=date(2024, 6, 1),
            job_id="custom_snapshot",
        )

        assert job.symbols == ["AAPL", "MSFT"]
        assert job.as_of_date == date(2024, 6, 1)
        assert job.job_id == "custom_snapshot"
