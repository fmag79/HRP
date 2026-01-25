"""
Comprehensive tests for the SimFin data source.

Tests cover:
- RateLimiter class functionality
- SimFinSource initialization with/without API key
- get_daily_bars raises NotImplementedError
- validate_symbol method
- get_fundamentals for single symbol
- get_fundamentals_batch for multiple symbols
- Date filtering
- Error handling
"""

import time
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from hrp.data.sources.simfin_source import (
    RateLimiter,
    METRIC_MAPPING,
    DEFAULT_METRICS,
    SIMFIN_TO_METRIC,
)


# =============================================================================
# RateLimiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_init_defaults(self):
        """RateLimiter should initialize with default limits."""
        limiter = RateLimiter()
        assert limiter.max_requests == 60
        assert limiter.window_seconds == 3600
        assert limiter._request_times == []

    def test_init_custom_limits(self):
        """RateLimiter should accept custom limits."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60

    def test_remaining_requests_initial(self):
        """remaining_requests should return max_requests initially."""
        limiter = RateLimiter(max_requests=10)
        assert limiter.remaining_requests == 10

    def test_remaining_requests_after_request(self):
        """remaining_requests should decrease after wait_if_needed."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        limiter.wait_if_needed()
        assert limiter.remaining_requests == 9

    def test_remaining_requests_multiple(self):
        """remaining_requests should track multiple requests."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        for _ in range(5):
            limiter.wait_if_needed()
        assert limiter.remaining_requests == 5

    def test_wait_if_needed_no_block(self):
        """wait_if_needed should not block when under limit."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        # Should complete almost instantly
        assert elapsed < 0.1

    def test_wait_if_needed_records_time(self):
        """wait_if_needed should record request time."""
        limiter = RateLimiter()
        assert len(limiter._request_times) == 0
        limiter.wait_if_needed()
        assert len(limiter._request_times) == 1

    def test_remaining_requests_expires_old(self):
        """remaining_requests should not count expired requests."""
        limiter = RateLimiter(max_requests=10, window_seconds=1)
        limiter.wait_if_needed()
        assert limiter.remaining_requests == 9
        # Wait for window to expire
        time.sleep(1.1)
        assert limiter.remaining_requests == 10

    def test_wait_if_needed_cleans_old_requests(self):
        """wait_if_needed should remove requests outside window."""
        limiter = RateLimiter(max_requests=10, window_seconds=1)
        limiter.wait_if_needed()
        assert len(limiter._request_times) == 1
        time.sleep(1.1)
        limiter.wait_if_needed()
        # Should have cleaned old request, only 1 recent
        assert len(limiter._request_times) == 1


# =============================================================================
# SimFinSource Initialization Tests
# =============================================================================


class TestSimFinSourceInit:
    """Tests for SimFinSource initialization."""

    def test_init_without_api_key_raises(self):
        """SimFinSource should raise ValueError without API key."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = None

            with pytest.raises(ValueError) as exc_info:
                from hrp.data.sources.simfin_source import SimFinSource
                SimFinSource()

            assert "SimFin API key required" in str(exc_info.value)

    def test_init_with_api_key_param(self):
        """SimFinSource should accept API key as parameter."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = None

            mock_sf = MagicMock()
            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource(api_key="test_key_123")

            assert source.api_key == "test_key_123"
            mock_sf.set_api_key.assert_called_with("test_key_123")

    def test_init_with_env_api_key(self):
        """SimFinSource should use API key from config."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "env_key_456"

            mock_sf = MagicMock()
            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource()

            assert source.api_key == "env_key_456"

    def test_init_simfin_import_error(self):
        """SimFinSource should raise ImportError if simfin not installed."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            with patch.dict("sys.modules", {"simfin": None}):
                # Force import error
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "simfin":
                        raise ImportError("No module named 'simfin'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    # Need to reimport to trigger the import
                    with pytest.raises(ImportError) as exc_info:
                        from hrp.data.sources.simfin_source import SimFinSource
                        SimFinSource(api_key="test_key")

                    assert "simfin package required" in str(exc_info.value)

    def test_init_sets_data_dir(self):
        """SimFinSource should set simfin data directory."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            mock_sf = MagicMock()
            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                SimFinSource(api_key="test_key")

            mock_sf.set_data_dir.assert_called_with("~/hrp-data/simfin/")

    def test_source_name(self):
        """SimFinSource should have correct source_name."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            mock_sf = MagicMock()
            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource(api_key="test_key")

            assert source.source_name == "simfin"


# =============================================================================
# get_daily_bars Tests
# =============================================================================


class TestGetDailyBars:
    """Tests for get_daily_bars method."""

    def test_get_daily_bars_raises_not_implemented(self):
        """get_daily_bars should raise NotImplementedError."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            mock_sf = MagicMock()
            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource(api_key="test_key")

                with pytest.raises(NotImplementedError) as exc_info:
                    source.get_daily_bars("AAPL", date(2023, 1, 1), date(2023, 12, 31))

                assert "SimFin does not provide price data" in str(exc_info.value)


# =============================================================================
# validate_symbol Tests
# =============================================================================


class TestValidateSymbol:
    """Tests for validate_symbol method."""

    def test_validate_symbol_exists(self):
        """validate_symbol should return True for valid symbol."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            mock_sf = MagicMock()
            # Create mock income DataFrame with AAPL in index
            mock_income = pd.DataFrame(
                {"Revenue": [1000]},
                index=pd.MultiIndex.from_tuples(
                    [("AAPL", date(2023, 3, 31))],
                    names=["Ticker", "Report Date"]
                )
            )
            mock_sf.load_income.return_value = mock_income

            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource(api_key="test_key")
                result = source.validate_symbol("AAPL")

            assert result is True

    def test_validate_symbol_not_exists(self):
        """validate_symbol should return False for invalid symbol."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            mock_sf = MagicMock()
            # Create mock income DataFrame without INVALID
            mock_income = pd.DataFrame(
                {"Revenue": [1000]},
                index=pd.MultiIndex.from_tuples(
                    [("AAPL", date(2023, 3, 31))],
                    names=["Ticker", "Report Date"]
                )
            )
            mock_sf.load_income.return_value = mock_income

            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource(api_key="test_key")
                result = source.validate_symbol("INVALID")

            assert result is False

    def test_validate_symbol_error_returns_false(self):
        """validate_symbol should return False on error."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            mock_sf = MagicMock()
            mock_sf.load_income.side_effect = Exception("API Error")

            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource(api_key="test_key")
                result = source.validate_symbol("AAPL")

            assert result is False


# =============================================================================
# get_fundamentals Tests
# =============================================================================


class TestGetFundamentals:
    """Tests for get_fundamentals method."""

    def _create_mock_source(self, mock_sf):
        """Helper to create a SimFinSource with mocked simfin."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                return SimFinSource(api_key="test_key")

    def test_get_fundamentals_returns_dataframe(self):
        """get_fundamentals should return a DataFrame."""
        mock_sf = MagicMock()

        # Create mock income DataFrame
        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000],
                "Net Income": [20000000],
                "Diluted EPS": [2.50],
                "Publish Date": [pd.Timestamp("2023-04-15")],
                "Report Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )

        # Create mock balance DataFrame
        mock_balance = pd.DataFrame(
            {
                "Book Value": [50000000],
                "Total Assets": [300000000],
                "Total Liabilities": [250000000],
                "Publish Date": [pd.Timestamp("2023-04-15")],
                "Report Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_get_fundamentals_correct_columns(self):
        """get_fundamentals should return correct columns."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000],
                "Publish Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )
        # Empty DataFrame with proper MultiIndex structure
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals("AAPL", metrics=["revenue"])

        expected_columns = ["symbol", "report_date", "period_end", "metric", "value", "source"]
        assert list(result.columns) == expected_columns

    def test_get_fundamentals_symbol_not_found(self):
        """get_fundamentals should return empty DataFrame for unknown symbol."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {"Revenue": [100000000]},
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals("INVALID")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_fundamentals_date_filtering_start(self):
        """get_fundamentals should filter by start_date."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000, 110000000],
                "Publish Date": [
                    pd.Timestamp("2023-01-15"),
                    pd.Timestamp("2023-04-15"),
                ],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("AAPL", pd.Timestamp("2022-12-31")),
                    ("AAPL", pd.Timestamp("2023-03-31")),
                ],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals(
            "AAPL",
            metrics=["revenue"],
            start_date=date(2023, 2, 1),
        )

        # Should only include the April record
        assert len(result) == 1
        assert result.iloc[0]["value"] == 110000000.0

    def test_get_fundamentals_date_filtering_end(self):
        """get_fundamentals should filter by end_date."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000, 110000000],
                "Publish Date": [
                    pd.Timestamp("2023-01-15"),
                    pd.Timestamp("2023-04-15"),
                ],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("AAPL", pd.Timestamp("2022-12-31")),
                    ("AAPL", pd.Timestamp("2023-03-31")),
                ],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals(
            "AAPL",
            metrics=["revenue"],
            end_date=date(2023, 2, 1),
        )

        # Should only include the January record
        assert len(result) == 1
        assert result.iloc[0]["value"] == 100000000.0

    def test_get_fundamentals_specific_metrics(self):
        """get_fundamentals should fetch only requested metrics."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000],
                "Net Income": [20000000],
                "Diluted EPS": [2.50],
                "Publish Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals("AAPL", metrics=["revenue", "eps"])

        metrics_returned = result["metric"].unique().tolist()
        assert "revenue" in metrics_returned
        assert "eps" in metrics_returned
        assert "net_income" not in metrics_returned

    def test_get_fundamentals_unknown_metric_skipped(self):
        """get_fundamentals should skip unknown metrics."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000],
                "Publish Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        # Should not raise, just skip unknown metric
        result = source.get_fundamentals("AAPL", metrics=["revenue", "unknown_metric"])

        assert len(result) == 1
        assert result.iloc[0]["metric"] == "revenue"

    def test_get_fundamentals_source_column(self):
        """get_fundamentals should set source column to 'simfin'."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000],
                "Publish Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals("AAPL", metrics=["revenue"])

        assert all(result["source"] == "simfin")

    def test_get_fundamentals_raises_on_error(self):
        """get_fundamentals should raise on API error."""
        mock_sf = MagicMock()
        mock_sf.load_income.side_effect = Exception("API Error")

        source = self._create_mock_source(mock_sf)

        with pytest.raises(Exception) as exc_info:
            source.get_fundamentals("AAPL")

        assert "API Error" in str(exc_info.value)

    def test_get_fundamentals_fallback_to_report_date(self):
        """get_fundamentals should fall back to Report Date if Publish Date is NA."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000],
                "Publish Date": [pd.NaT],  # Missing publish date
                "Report Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals("AAPL", metrics=["revenue"])

        assert len(result) == 1
        assert result.iloc[0]["report_date"] == date(2023, 4, 15)


# =============================================================================
# get_fundamentals_batch Tests
# =============================================================================


class TestGetFundamentalsBatch:
    """Tests for get_fundamentals_batch method."""

    def _create_mock_source(self, mock_sf):
        """Helper to create a SimFinSource with mocked simfin."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                return SimFinSource(api_key="test_key")

    def test_get_fundamentals_batch_multiple_symbols(self):
        """get_fundamentals_batch should fetch data for multiple symbols."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000, 200000000],
                "Publish Date": [
                    pd.Timestamp("2023-04-15"),
                    pd.Timestamp("2023-04-20"),
                ],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("AAPL", pd.Timestamp("2023-03-31")),
                    ("MSFT", pd.Timestamp("2023-03-31")),
                ],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals_batch(["AAPL", "MSFT"], metrics=["revenue"])

        assert len(result) == 2
        symbols = result["symbol"].unique().tolist()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_get_fundamentals_batch_partial_failure(self):
        """get_fundamentals_batch should continue on partial failures."""
        mock_sf = MagicMock()

        # Only AAPL has data
        mock_income = pd.DataFrame(
            {
                "Revenue": [100000000],
                "Publish Date": [pd.Timestamp("2023-04-15")],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2023-03-31"))],
                names=["Ticker", "Report Date"]
            )
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals_batch(
            ["AAPL", "INVALID"],
            metrics=["revenue"],
        )

        # Should still return AAPL data
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"

    def test_get_fundamentals_batch_all_fail(self):
        """get_fundamentals_batch should return empty DataFrame if all fail."""
        mock_sf = MagicMock()

        mock_income = pd.DataFrame(
            {"Revenue": []},
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )
        mock_balance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Report Date"])
        )

        mock_sf.load_income.return_value = mock_income
        mock_sf.load_balance.return_value = mock_balance

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals_batch(["INVALID1", "INVALID2"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_fundamentals_batch_empty_list(self):
        """get_fundamentals_batch should return empty DataFrame for empty list."""
        mock_sf = MagicMock()
        mock_sf.load_income.return_value = pd.DataFrame()
        mock_sf.load_balance.return_value = pd.DataFrame()

        source = self._create_mock_source(mock_sf)
        result = source.get_fundamentals_batch([])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# =============================================================================
# remaining_requests Property Tests
# =============================================================================


class TestRemainingRequestsProperty:
    """Tests for remaining_requests property."""

    def test_remaining_requests_delegates_to_limiter(self):
        """remaining_requests should delegate to rate limiter."""
        with patch("hrp.data.sources.simfin_source.get_config") as mock_config:
            mock_config.return_value.api.simfin_api_key = "test_key"

            mock_sf = MagicMock()
            with patch.dict("sys.modules", {"simfin": mock_sf}):
                from hrp.data.sources.simfin_source import SimFinSource
                source = SimFinSource(api_key="test_key")

                # Initial value should be 60 (default max)
                assert source.remaining_requests == 60


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_metric_mapping_has_expected_keys(self):
        """METRIC_MAPPING should have expected metric keys."""
        expected_metrics = [
            "revenue",
            "net_income",
            "eps",
            "book_value",
            "total_assets",
            "total_liabilities",
        ]
        for metric in expected_metrics:
            assert metric in METRIC_MAPPING

    def test_metric_mapping_has_simfin_names(self):
        """METRIC_MAPPING should map to SimFin indicator names."""
        assert METRIC_MAPPING["revenue"] == "Revenue"
        assert METRIC_MAPPING["eps"] == "Diluted EPS"
        assert METRIC_MAPPING["book_value"] == "Book Value"

    def test_simfin_to_metric_reverse_mapping(self):
        """SIMFIN_TO_METRIC should be reverse of METRIC_MAPPING."""
        for our_name, simfin_name in METRIC_MAPPING.items():
            assert SIMFIN_TO_METRIC[simfin_name] == our_name

    def test_default_metrics_is_all_metrics(self):
        """DEFAULT_METRICS should include all mapped metrics."""
        assert set(DEFAULT_METRICS) == set(METRIC_MAPPING.keys())
