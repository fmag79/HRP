"""
Comprehensive tests for Polygon.io data source adapter.

Tests cover:
- Initialization with and without API key
- get_daily_bars() with various scenarios
- get_multiple_symbols() batch processing
- get_ticker_details() ticker information
- validate_symbol() validation logic
- get_splits() stock split data
- get_dividends() dividend data
- get_corporate_actions() unified interface
- Rate limiting integration
- Error handling and retry logic
"""

import os
from datetime import date, datetime
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
import pytest
from polygon.exceptions import BadResponse

from hrp.data.sources.polygon_source import PolygonSource


class MockAgg:
    """Mock aggregates bar from Polygon API."""

    def __init__(
        self,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class MockSplit:
    """Mock stock split from Polygon API."""

    def __init__(self, execution_date: str, split_from: float, split_to: float):
        self.execution_date = execution_date
        self.split_from = split_from
        self.split_to = split_to


class MockDividend:
    """Mock dividend from Polygon API."""

    def __init__(self, ex_dividend_date: str, cash_amount: float):
        self.ex_dividend_date = ex_dividend_date
        self.cash_amount = cash_amount


class MockTickerDetails:
    """Mock ticker details from Polygon API."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Test Company")
        self.market = kwargs.get("market", "stocks")
        self.locale = kwargs.get("locale", "us")
        self.primary_exchange = kwargs.get("primary_exchange", "XNAS")
        self.type = kwargs.get("type", "CS")
        self.currency_name = kwargs.get("currency_name", "usd")
        self.market_cap = kwargs.get("market_cap", 1000000000)
        self.active = kwargs.get("active", True)


class TestPolygonSourceInitialization:
    """Tests for PolygonSource initialization."""

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        source = PolygonSource(api_key="test_key_123")
        assert source.source_name == "polygon"
        assert source.rate_limiter is not None
        assert source.rate_limiter.max_calls == 5
        assert source.rate_limiter.period == 60

    @patch.dict(os.environ, {"POLYGON_API_KEY": "env_test_key"})
    def test_initialization_from_environment(self):
        """Test initialization with API key from environment."""
        with patch("hrp.data.sources.polygon_source.get_config") as mock_config:
            mock_config.return_value.api.polygon_api_key = "env_test_key"
            source = PolygonSource()
            assert source.source_name == "polygon"

    def test_initialization_without_api_key_raises_error(self):
        """Test initialization without API key raises ValueError."""
        with patch("hrp.data.sources.polygon_source.get_config") as mock_config:
            mock_config.return_value.api.polygon_api_key = None
            with pytest.raises(ValueError, match="POLYGON_API_KEY not found"):
                PolygonSource()

    def test_rate_limiter_configured_correctly(self):
        """Test rate limiter is configured for Basic tier (5 calls/min)."""
        source = PolygonSource(api_key="test_key")
        assert source.rate_limiter.max_calls == 5
        assert source.rate_limiter.period == 60


class TestGetDailyBars:
    """Tests for get_daily_bars() method."""

    def test_get_daily_bars_success(self):
        """Test successful daily bars fetch."""
        source = PolygonSource(api_key="test_key")

        # Mock aggregates data
        mock_aggs = [
            MockAgg(
                timestamp=1609459200000,  # 2021-01-01
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            ),
            MockAgg(
                timestamp=1609545600000,  # 2021-01-02
                open=103.0,
                high=108.0,
                low=102.0,
                close=107.0,
                volume=1200000,
            ),
        ]

        with patch.object(source.client, "get_aggs", return_value=mock_aggs):
            df = source.get_daily_bars("AAPL", date(2021, 1, 1), date(2021, 1, 2))

        assert len(df) == 2
        assert list(df.columns) == [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "source",
        ]
        assert df["symbol"].iloc[0] == "AAPL"
        assert df["date"].iloc[0] == date(2021, 1, 1)
        assert df["open"].iloc[0] == 100.0
        assert df["close"].iloc[0] == 103.0
        assert df["adj_close"].iloc[0] == 103.0
        assert df["volume"].iloc[0] == 1000000
        assert df["source"].iloc[0] == "polygon"

    def test_get_daily_bars_no_data(self):
        """Test get_daily_bars with no data returned."""
        source = PolygonSource(api_key="test_key")

        with patch.object(source.client, "get_aggs", return_value=[]):
            df = source.get_daily_bars("INVALID", date(2021, 1, 1), date(2021, 1, 2))

        assert df.empty

    def test_get_daily_bars_bad_response_error(self):
        """Test get_daily_bars with BadResponse error."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "get_aggs", side_effect=BadResponse("API error")
        ):
            with pytest.raises(BadResponse):
                source.get_daily_bars("AAPL", date(2021, 1, 1), date(2021, 1, 2))

    def test_get_daily_bars_uses_rate_limiter(self):
        """Test that get_daily_bars acquires rate limit token."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.rate_limiter, "acquire"
        ) as mock_acquire, patch.object(source.client, "get_aggs", return_value=[]):
            source.get_daily_bars("AAPL", date(2021, 1, 1), date(2021, 1, 2))
            mock_acquire.assert_called_once()

    def test_get_daily_bars_date_formatting(self):
        """Test that dates are formatted correctly for API."""
        source = PolygonSource(api_key="test_key")

        with patch.object(source.client, "get_aggs", return_value=[]) as mock_get_aggs:
            source.get_daily_bars("AAPL", date(2021, 1, 15), date(2021, 2, 28))

            # Verify API was called with correct date format
            call_kwargs = mock_get_aggs.call_args.kwargs
            assert call_kwargs["from_"] == "2021-01-15"
            assert call_kwargs["to"] == "2021-02-28"
            assert call_kwargs["adjusted"] is True
            assert call_kwargs["timespan"] == "day"
            assert call_kwargs["multiplier"] == 1


class TestGetMultipleSymbols:
    """Tests for get_multiple_symbols() method."""

    def test_get_multiple_symbols_success(self):
        """Test successful fetch for multiple symbols."""
        source = PolygonSource(api_key="test_key")

        mock_agg_aapl = [
            MockAgg(
                timestamp=1609459200000, open=100.0, high=105.0, low=99.0, close=103.0, volume=1000000
            )
        ]
        mock_agg_msft = [
            MockAgg(
                timestamp=1609459200000, open=200.0, high=205.0, low=199.0, close=203.0, volume=2000000
            )
        ]

        with patch.object(source.client, "get_aggs") as mock_get_aggs:
            mock_get_aggs.side_effect = [mock_agg_aapl, mock_agg_msft]

            df = source.get_multiple_symbols(
                ["AAPL", "MSFT"], date(2021, 1, 1), date(2021, 1, 1)
            )

        assert len(df) == 2
        assert "AAPL" in df["symbol"].values
        assert "MSFT" in df["symbol"].values

    def test_get_multiple_symbols_skip_failed(self):
        """Test that failed symbols are skipped."""
        source = PolygonSource(api_key="test_key")

        mock_agg_aapl = [
            MockAgg(
                timestamp=1609459200000, open=100.0, high=105.0, low=99.0, close=103.0, volume=1000000
            )
        ]

        with patch.object(source.client, "get_aggs") as mock_get_aggs:
            # First symbol succeeds, second fails
            mock_get_aggs.side_effect = [mock_agg_aapl, BadResponse("API error")]

            df = source.get_multiple_symbols(
                ["AAPL", "INVALID"], date(2021, 1, 1), date(2021, 1, 1)
            )

        assert len(df) == 1
        assert df["symbol"].iloc[0] == "AAPL"

    def test_get_multiple_symbols_all_failed(self):
        """Test when all symbols fail."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "get_aggs", side_effect=BadResponse("API error")
        ):
            df = source.get_multiple_symbols(
                ["INVALID1", "INVALID2"], date(2021, 1, 1), date(2021, 1, 1)
            )

        assert df.empty


class TestGetTickerDetails:
    """Tests for get_ticker_details() method."""

    def test_get_ticker_details_success(self):
        """Test successful ticker details fetch."""
        source = PolygonSource(api_key="test_key")

        mock_details = MockTickerDetails(
            name="Apple Inc.",
            market="stocks",
            locale="us",
            primary_exchange="XNAS",
            type="CS",
            currency_name="usd",
            market_cap=2500000000000,
        )

        with patch.object(source.client, "get_ticker_details", return_value=mock_details):
            details = source.get_ticker_details("AAPL")

        assert details["symbol"] == "AAPL"
        assert details["name"] == "Apple Inc."
        assert details["market"] == "stocks"
        assert details["locale"] == "us"
        assert details["primary_exchange"] == "XNAS"
        assert details["type"] == "CS"
        assert details["currency_name"] == "usd"
        assert details["market_cap"] == 2500000000000

    def test_get_ticker_details_error(self):
        """Test ticker details with error returns minimal info."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "get_ticker_details", side_effect=Exception("API error")
        ):
            details = source.get_ticker_details("AAPL")

        assert details == {"symbol": "AAPL"}

    def test_get_ticker_details_uses_rate_limiter(self):
        """Test that get_ticker_details acquires rate limit token."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.rate_limiter, "acquire"
        ) as mock_acquire, patch.object(
            source.client, "get_ticker_details", return_value=MockTickerDetails()
        ):
            source.get_ticker_details("AAPL")
            mock_acquire.assert_called_once()


class TestValidateSymbol:
    """Tests for validate_symbol() method."""

    def test_validate_symbol_valid_active(self):
        """Test validation of valid, active symbol."""
        source = PolygonSource(api_key="test_key")

        mock_details = MockTickerDetails(active=True)

        with patch.object(source.client, "get_ticker_details", return_value=mock_details):
            is_valid = source.validate_symbol("AAPL")

        assert is_valid is True

    def test_validate_symbol_inactive(self):
        """Test validation of inactive symbol."""
        source = PolygonSource(api_key="test_key")

        mock_details = MockTickerDetails(active=False)

        with patch.object(source.client, "get_ticker_details", return_value=mock_details):
            is_valid = source.validate_symbol("DELISTED")

        assert is_valid is False

    def test_validate_symbol_not_found(self):
        """Test validation of non-existent symbol."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "get_ticker_details", side_effect=BadResponse("Not found")
        ):
            is_valid = source.validate_symbol("INVALID")

        assert is_valid is False

    def test_validate_symbol_error(self):
        """Test validation with unexpected error."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "get_ticker_details", side_effect=Exception("Network error")
        ):
            is_valid = source.validate_symbol("AAPL")

        assert is_valid is False

    def test_validate_symbol_uses_rate_limiter(self):
        """Test that validate_symbol acquires rate limit token."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.rate_limiter, "acquire"
        ) as mock_acquire, patch.object(
            source.client, "get_ticker_details", return_value=MockTickerDetails()
        ):
            source.validate_symbol("AAPL")
            mock_acquire.assert_called_once()


class TestGetSplits:
    """Tests for get_splits() method."""

    def test_get_splits_success(self):
        """Test successful splits fetch."""
        source = PolygonSource(api_key="test_key")

        mock_splits = [
            MockSplit(execution_date="2020-08-31", split_from=1.0, split_to=4.0),
            MockSplit(execution_date="2014-06-09", split_from=1.0, split_to=7.0),
        ]

        with patch.object(source.client, "list_splits", return_value=mock_splits):
            df = source.get_splits("AAPL", date(2014, 1, 1), date(2020, 12, 31))

        assert len(df) == 2
        assert list(df.columns) == ["symbol", "date", "action_type", "factor", "source"]
        assert df["symbol"].iloc[0] == "AAPL"
        assert df["date"].iloc[0] == date(2020, 8, 31)
        assert df["action_type"].iloc[0] == "split"
        assert df["factor"].iloc[0] == 4.0
        assert df["source"].iloc[0] == "polygon"

    def test_get_splits_no_data(self):
        """Test get_splits with no splits found."""
        source = PolygonSource(api_key="test_key")

        with patch.object(source.client, "list_splits", return_value=[]):
            df = source.get_splits("MSFT", date(2020, 1, 1), date(2020, 12, 31))

        assert df.empty

    def test_get_splits_bad_response_error(self):
        """Test get_splits with BadResponse error."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "list_splits", side_effect=BadResponse("API error")
        ):
            with pytest.raises(BadResponse):
                source.get_splits("AAPL", date(2020, 1, 1), date(2020, 12, 31))

    def test_get_splits_uses_rate_limiter(self):
        """Test that get_splits acquires rate limit token."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.rate_limiter, "acquire"
        ) as mock_acquire, patch.object(source.client, "list_splits", return_value=[]):
            source.get_splits("AAPL", date(2020, 1, 1), date(2020, 12, 31))
            mock_acquire.assert_called_once()

    def test_get_splits_date_range(self):
        """Test that splits API is called with correct date range."""
        source = PolygonSource(api_key="test_key")

        with patch.object(source.client, "list_splits", return_value=[]) as mock_list:
            source.get_splits("AAPL", date(2020, 1, 15), date(2020, 12, 31))

            call_kwargs = mock_list.call_args.kwargs
            assert call_kwargs["execution_date_gte"] == "2020-01-15"
            assert call_kwargs["execution_date_lte"] == "2020-12-31"


class TestGetDividends:
    """Tests for get_dividends() method."""

    def test_get_dividends_success(self):
        """Test successful dividends fetch."""
        source = PolygonSource(api_key="test_key")

        mock_dividends = [
            MockDividend(ex_dividend_date="2021-02-05", cash_amount=0.205),
            MockDividend(ex_dividend_date="2021-05-07", cash_amount=0.22),
        ]

        with patch.object(source.client, "list_dividends", return_value=mock_dividends):
            df = source.get_dividends("AAPL", date(2021, 1, 1), date(2021, 12, 31))

        assert len(df) == 2
        assert list(df.columns) == ["symbol", "date", "action_type", "factor", "source"]
        assert df["symbol"].iloc[0] == "AAPL"
        assert df["date"].iloc[0] == date(2021, 2, 5)
        assert df["action_type"].iloc[0] == "dividend"
        assert df["factor"].iloc[0] == 0.205
        assert df["source"].iloc[0] == "polygon"

    def test_get_dividends_no_data(self):
        """Test get_dividends with no dividends found."""
        source = PolygonSource(api_key="test_key")

        with patch.object(source.client, "list_dividends", return_value=[]):
            df = source.get_dividends("MSFT", date(2021, 1, 1), date(2021, 12, 31))

        assert df.empty

    def test_get_dividends_bad_response_error(self):
        """Test get_dividends with BadResponse error."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "list_dividends", side_effect=BadResponse("API error")
        ):
            with pytest.raises(BadResponse):
                source.get_dividends("AAPL", date(2021, 1, 1), date(2021, 12, 31))

    def test_get_dividends_uses_rate_limiter(self):
        """Test that get_dividends acquires rate limit token."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.rate_limiter, "acquire"
        ) as mock_acquire, patch.object(
            source.client, "list_dividends", return_value=[]
        ):
            source.get_dividends("AAPL", date(2021, 1, 1), date(2021, 12, 31))
            mock_acquire.assert_called_once()

    def test_get_dividends_date_range(self):
        """Test that dividends API is called with correct date range."""
        source = PolygonSource(api_key="test_key")

        with patch.object(
            source.client, "list_dividends", return_value=[]
        ) as mock_list:
            source.get_dividends("AAPL", date(2021, 1, 15), date(2021, 12, 31))

            call_kwargs = mock_list.call_args.kwargs
            assert call_kwargs["ex_dividend_date_gte"] == "2021-01-15"
            assert call_kwargs["ex_dividend_date_lte"] == "2021-12-31"


class TestGetCorporateActions:
    """Tests for get_corporate_actions() unified interface."""

    def test_get_corporate_actions_all_types(self):
        """Test fetching all corporate action types."""
        source = PolygonSource(api_key="test_key")

        mock_splits = [
            MockSplit(execution_date="2020-08-31", split_from=1.0, split_to=4.0)
        ]
        mock_dividends = [
            MockDividend(ex_dividend_date="2021-02-05", cash_amount=0.205)
        ]

        with patch.object(
            source.client, "list_splits", return_value=mock_splits
        ), patch.object(source.client, "list_dividends", return_value=mock_dividends):
            df = source.get_corporate_actions("AAPL", date(2020, 1, 1), date(2021, 12, 31))

        assert len(df) == 2
        assert "split" in df["action_type"].values
        assert "dividend" in df["action_type"].values

    def test_get_corporate_actions_splits_only(self):
        """Test fetching splits only."""
        source = PolygonSource(api_key="test_key")

        mock_splits = [
            MockSplit(execution_date="2020-08-31", split_from=1.0, split_to=4.0)
        ]

        with patch.object(
            source.client, "list_splits", return_value=mock_splits
        ), patch.object(
            source.client, "list_dividends", return_value=[]
        ) as mock_div:
            df = source.get_corporate_actions(
                "AAPL", date(2020, 1, 1), date(2021, 12, 31), action_types=["split"]
            )

        assert len(df) == 1
        assert df["action_type"].iloc[0] == "split"
        # Dividends should not be called
        mock_div.assert_not_called()

    def test_get_corporate_actions_dividends_only(self):
        """Test fetching dividends only."""
        source = PolygonSource(api_key="test_key")

        mock_dividends = [
            MockDividend(ex_dividend_date="2021-02-05", cash_amount=0.205)
        ]

        with patch.object(
            source.client, "list_splits", return_value=[]
        ) as mock_splits, patch.object(
            source.client, "list_dividends", return_value=mock_dividends
        ):
            df = source.get_corporate_actions(
                "AAPL", date(2020, 1, 1), date(2021, 12, 31), action_types=["dividend"]
            )

        assert len(df) == 1
        assert df["action_type"].iloc[0] == "dividend"
        # Splits should not be called
        mock_splits.assert_not_called()

    def test_get_corporate_actions_sorted_by_date(self):
        """Test that corporate actions are sorted by date."""
        source = PolygonSource(api_key="test_key")

        mock_splits = [
            MockSplit(execution_date="2021-03-01", split_from=1.0, split_to=2.0)
        ]
        mock_dividends = [
            MockDividend(ex_dividend_date="2021-01-05", cash_amount=0.20),
            MockDividend(ex_dividend_date="2021-04-05", cash_amount=0.22),
        ]

        with patch.object(
            source.client, "list_splits", return_value=mock_splits
        ), patch.object(source.client, "list_dividends", return_value=mock_dividends):
            df = source.get_corporate_actions("AAPL", date(2021, 1, 1), date(2021, 12, 31))

        # Should be sorted by date
        assert df["date"].iloc[0] == date(2021, 1, 5)
        assert df["date"].iloc[1] == date(2021, 3, 1)
        assert df["date"].iloc[2] == date(2021, 4, 5)

    def test_get_corporate_actions_no_data(self):
        """Test get_corporate_actions with no data."""
        source = PolygonSource(api_key="test_key")

        with patch.object(source.client, "list_splits", return_value=[]), patch.object(
            source.client, "list_dividends", return_value=[]
        ):
            df = source.get_corporate_actions("AAPL", date(2021, 1, 1), date(2021, 12, 31))

        assert df.empty
        assert list(df.columns) == ["symbol", "date", "action_type", "factor", "source"]

    def test_get_corporate_actions_error_handling(self):
        """Test that errors in one action type don't prevent others."""
        source = PolygonSource(api_key="test_key")

        mock_dividends = [
            MockDividend(ex_dividend_date="2021-02-05", cash_amount=0.205)
        ]

        with patch.object(
            source.client, "list_splits", side_effect=Exception("API error")
        ), patch.object(source.client, "list_dividends", return_value=mock_dividends):
            df = source.get_corporate_actions("AAPL", date(2021, 1, 1), date(2021, 12, 31))

        # Should still get dividends even though splits failed
        assert len(df) == 1
        assert df["action_type"].iloc[0] == "dividend"


class TestRetryIntegration:
    """Tests for retry integration with Polygon adapter."""

    def test_get_daily_bars_retries_on_transient_error(self):
        """Test that get_daily_bars retries on transient errors."""
        source = PolygonSource(api_key="test_key")

        call_count = [0]

        def mock_get_aggs(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Transient error")
            return []

        with patch.object(source.client, "get_aggs", side_effect=mock_get_aggs):
            df = source.get_daily_bars("AAPL", date(2021, 1, 1), date(2021, 1, 2))

        # Should have retried once
        assert call_count[0] == 2
        assert df.empty

    def test_get_splits_retries_on_timeout(self):
        """Test that get_splits retries on timeout."""
        source = PolygonSource(api_key="test_key")

        call_count = [0]
        mock_splits = [
            MockSplit(execution_date="2020-08-31", split_from=1.0, split_to=4.0)
        ]

        def mock_list_splits(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("Request timeout")
            return mock_splits

        with patch.object(source.client, "list_splits", side_effect=mock_list_splits):
            df = source.get_splits("AAPL", date(2020, 1, 1), date(2020, 12, 31))

        # Should have retried twice
        assert call_count[0] == 3
        assert len(df) == 1

    def test_validate_symbol_no_retry_on_bad_response(self):
        """Test that validate_symbol doesn't retry on BadResponse."""
        source = PolygonSource(api_key="test_key")

        call_count = [0]

        def mock_get_ticker(*args, **kwargs):
            call_count[0] += 1
            raise BadResponse("Symbol not found")

        with patch.object(source.client, "get_ticker_details", side_effect=mock_get_ticker):
            is_valid = source.validate_symbol("INVALID")

        # Should not retry BadResponse (non-transient)
        assert call_count[0] == 1
        assert is_valid is False


class TestRateLimiting:
    """Tests for rate limiting integration."""

    def test_concurrent_calls_respect_rate_limit(self):
        """Test that multiple calls respect rate limiting."""
        source = PolygonSource(api_key="test_key")

        with patch.object(source.client, "get_aggs", return_value=[]):
            # Make 6 calls (exceeds 5 calls/min limit)
            for i in range(6):
                source.get_daily_bars("AAPL", date(2021, 1, 1), date(2021, 1, 1))

        # Rate limiter should have been called 6 times
        # (implementation detail, but good to verify integration)
        assert source.rate_limiter.available_tokens() < 5

    def test_validate_symbol_acquires_token_before_api_call(self):
        """Test that rate limiter is called before API calls."""
        source = PolygonSource(api_key="test_key")

        acquire_called = [False]
        api_called = [False]

        def mock_acquire():
            acquire_called[0] = True
            # Ensure API hasn't been called yet
            assert not api_called[0], "API called before rate limiter acquire"

        def mock_get_ticker(*args, **kwargs):
            api_called[0] = True
            return MockTickerDetails()

        with patch.object(
            source.rate_limiter, "acquire", side_effect=mock_acquire
        ), patch.object(source.client, "get_ticker_details", side_effect=mock_get_ticker):
            source.validate_symbol("AAPL")

        assert acquire_called[0]
        assert api_called[0]


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_get_daily_bars_with_single_day(self):
        """Test fetching data for a single day."""
        source = PolygonSource(api_key="test_key")

        mock_aggs = [
            MockAgg(
                timestamp=1609459200000, open=100.0, high=105.0, low=99.0, close=103.0, volume=1000000
            )
        ]

        with patch.object(source.client, "get_aggs", return_value=mock_aggs):
            df = source.get_daily_bars("AAPL", date(2021, 1, 1), date(2021, 1, 1))

        assert len(df) == 1
        assert df["date"].iloc[0] == date(2021, 1, 1)

    def test_get_multiple_symbols_empty_list(self):
        """Test get_multiple_symbols with empty symbol list."""
        source = PolygonSource(api_key="test_key")

        df = source.get_multiple_symbols([], date(2021, 1, 1), date(2021, 1, 2))

        assert df.empty

    def test_get_corporate_actions_empty_action_types(self):
        """Test get_corporate_actions with empty action types list."""
        source = PolygonSource(api_key="test_key")

        df = source.get_corporate_actions(
            "AAPL", date(2021, 1, 1), date(2021, 12, 31), action_types=[]
        )

        assert df.empty

    def test_split_factor_calculation(self):
        """Test correct calculation of split factors."""
        source = PolygonSource(api_key="test_key")

        # 2-for-1 split: split_from=1, split_to=2 → factor=2.0
        # 3-for-1 split: split_from=1, split_to=3 → factor=3.0
        # Reverse 1-for-5 split: split_from=5, split_to=1 → factor=0.2
        mock_splits = [
            MockSplit(execution_date="2020-08-31", split_from=1.0, split_to=2.0),
            MockSplit(execution_date="2020-09-01", split_from=1.0, split_to=3.0),
            MockSplit(execution_date="2020-09-02", split_from=5.0, split_to=1.0),
        ]

        with patch.object(source.client, "list_splits", return_value=mock_splits):
            df = source.get_splits("TEST", date(2020, 1, 1), date(2020, 12, 31))

        assert df["factor"].iloc[0] == 2.0
        assert df["factor"].iloc[1] == 3.0
        assert df["factor"].iloc[2] == 0.2

    def test_ticker_details_missing_attributes(self):
        """Test ticker details with missing attributes."""
        source = PolygonSource(api_key="test_key")

        # Create mock with minimal attributes
        mock_details = Mock()
        mock_details.name = "Test Company"
        # Don't set other attributes

        with patch.object(source.client, "get_ticker_details", return_value=mock_details):
            details = source.get_ticker_details("TEST")

        # Should handle missing attributes gracefully
        assert details["symbol"] == "TEST"
        assert details["name"] == "Test Company"
        assert details["market"] == ""
        assert details["market_cap"] is None
