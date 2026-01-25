"""Tests for hrp/research/benchmark.py."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.research.benchmark import (
    BENCHMARKS,
    compare_to_benchmark,
    ensure_benchmark_data,
    get_benchmark_prices,
    get_benchmark_returns,
)


class TestGetBenchmarkPrices:
    """Tests for get_benchmark_prices function."""

    def test_returns_data_from_database(self):
        """Test that data is loaded from database when available."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "close": [100 + i for i in range(10)],
            "adj_close": [100 + i for i in range(10)],
        })

        with patch("hrp.research.benchmark.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = mock_df
            mock_get_db.return_value = mock_db

            result = get_benchmark_prices("SPY")

        assert len(result) == 10
        assert "close" in result.columns
        assert "adj_close" in result.columns

    def test_fallback_to_yfinance(self):
        """Test that yfinance is used when database is empty."""
        mock_yf_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "close": [100 + i for i in range(5)],
            "adj_close": [100 + i for i in range(5)],
            "open": [99 + i for i in range(5)],
        })

        with patch("hrp.research.benchmark.get_db") as mock_get_db, \
             patch("hrp.research.benchmark.YFinanceSource") as mock_yf:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = pd.DataFrame()  # Empty DB
            mock_get_db.return_value = mock_db

            mock_source = MagicMock()
            mock_source.get_daily_bars.return_value = mock_yf_df
            mock_yf.return_value = mock_source

            result = get_benchmark_prices("SPY")

        assert len(result) == 5
        mock_source.get_daily_bars.assert_called_once()

    def test_empty_raises_valueerror(self):
        """Test that ValueError is raised when no data is available anywhere."""
        with patch("hrp.research.benchmark.get_db") as mock_get_db, \
             patch("hrp.research.benchmark.YFinanceSource") as mock_yf:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = pd.DataFrame()  # Empty DB
            mock_get_db.return_value = mock_db

            mock_source = MagicMock()
            mock_source.get_daily_bars.return_value = pd.DataFrame()  # Empty yfinance
            mock_yf.return_value = mock_source

            with pytest.raises(ValueError, match="No data found"):
                get_benchmark_prices("SPY")

    def test_date_filtering(self):
        """Test that date parameters are passed to query."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "close": [100 + i for i in range(10)],
            "adj_close": [100 + i for i in range(10)],
        })

        with patch("hrp.research.benchmark.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = mock_df
            mock_get_db.return_value = mock_db

            get_benchmark_prices(
                "SPY",
                start=date(2020, 1, 1),
                end=date(2020, 12, 31),
            )

            # Check that query params include dates
            call_args = mock_db.fetchdf.call_args
            assert date(2020, 1, 1) in call_args[0][1]
            assert date(2020, 12, 31) in call_args[0][1]

    def test_different_benchmark(self):
        """Test that different benchmarks can be queried."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "close": [300 + i for i in range(5)],
            "adj_close": [300 + i for i in range(5)],
        })

        with patch("hrp.research.benchmark.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = mock_df
            mock_get_db.return_value = mock_db

            result = get_benchmark_prices("QQQ")

        call_args = mock_db.fetchdf.call_args
        assert "QQQ" in call_args[0][1]


class TestGetBenchmarkReturns:
    """Tests for get_benchmark_returns function."""

    def test_returns_series(self):
        """Test that a Series of returns is returned."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "close": [100 + i for i in range(10)],
            "adj_close": [100 + i for i in range(10)],
        })

        with patch("hrp.research.benchmark.get_benchmark_prices") as mock_prices:
            mock_prices.return_value = mock_df

            result = get_benchmark_returns("SPY")

        assert isinstance(result, pd.Series)
        assert result.name == "SPY"
        assert len(result) == 9  # One less due to pct_change

    def test_uses_adjusted_close(self):
        """Test that adjusted close is used by default."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "close": [100, 101, 102, 103, 104],
            "adj_close": [90, 91.8, 93.6, 95.4, 97.2],  # Different from close
        })

        with patch("hrp.research.benchmark.get_benchmark_prices") as mock_prices:
            mock_prices.return_value = mock_df

            result = get_benchmark_returns("SPY", use_adjusted=True)

        # Should use adj_close returns
        assert abs(result.iloc[0] - 0.02) < 0.001  # 91.8/90 - 1

    def test_uses_close_fallback(self):
        """Test that close is used when adj_close missing or use_adjusted=False."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "close": [100, 101, 102, 103, 104],
            "adj_close": [90, 91.8, 93.6, 95.4, 97.2],
        })

        with patch("hrp.research.benchmark.get_benchmark_prices") as mock_prices:
            mock_prices.return_value = mock_df

            result = get_benchmark_returns("SPY", use_adjusted=False)

        # Should use close returns
        assert abs(result.iloc[0] - 0.01) < 0.001  # 101/100 - 1

    def test_date_index(self):
        """Test that result is indexed by date."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "close": [100 + i for i in range(5)],
            "adj_close": [100 + i for i in range(5)],
        })

        with patch("hrp.research.benchmark.get_benchmark_prices") as mock_prices:
            mock_prices.return_value = mock_df

            result = get_benchmark_returns("SPY")

        assert result.index.name == "date"


class TestCompareToBenchmark:
    """Tests for compare_to_benchmark function."""

    def test_basic(self):
        """Test that comparison returns strategy, benchmark, excess metrics."""
        strategy_returns = pd.Series(
            [0.01, -0.005, 0.02, 0.015, -0.01],
            index=pd.date_range("2020-01-01", periods=5),
            name="strategy",
        )

        benchmark_returns = pd.Series(
            [0.005, -0.002, 0.01, 0.008, -0.005],
            index=pd.date_range("2020-01-01", periods=5),
            name="SPY",
        )

        with patch("hrp.research.benchmark.get_benchmark_returns") as mock_bench:
            mock_bench.return_value = benchmark_returns

            result = compare_to_benchmark(strategy_returns, "SPY")

        assert "strategy" in result
        assert "benchmark" in result
        assert "excess_return" in result
        assert "excess_sharpe" in result

    def test_different_benchmark(self):
        """Test comparison works with different benchmarks."""
        strategy_returns = pd.Series(
            [0.01, 0.02, 0.03],
            index=pd.date_range("2020-01-01", periods=3),
        )

        benchmark_returns = pd.Series(
            [0.005, 0.01, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )

        with patch("hrp.research.benchmark.get_benchmark_returns") as mock_bench:
            mock_bench.return_value = benchmark_returns

            result = compare_to_benchmark(strategy_returns, "QQQ")

        mock_bench.assert_called_once()
        assert mock_bench.call_args[0][0] == "QQQ"

    def test_excess_return_calculation(self):
        """Test that excess_return is strategy - benchmark."""
        strategy_returns = pd.Series(
            [0.01, 0.02, 0.03],
            index=pd.date_range("2020-01-01", periods=3),
        )

        benchmark_returns = pd.Series(
            [0.005, 0.01, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )

        with patch("hrp.research.benchmark.get_benchmark_returns") as mock_bench, \
             patch("hrp.research.metrics.calculate_metrics") as mock_metrics:
            mock_bench.return_value = benchmark_returns

            # Mock metrics to return known values
            mock_metrics.side_effect = [
                {"total_return": 0.10, "sharpe_ratio": 1.5},  # strategy
                {"total_return": 0.05, "sharpe_ratio": 1.0},  # benchmark
            ]

            result = compare_to_benchmark(strategy_returns, "SPY")

        assert result["excess_return"] == 0.05  # 0.10 - 0.05
        assert result["excess_sharpe"] == 0.5   # 1.5 - 1.0

    def test_date_alignment(self):
        """Test that dates are properly aligned between strategy and benchmark."""
        # Strategy has more dates
        strategy_returns = pd.Series(
            [0.01, 0.02, 0.03, 0.04, 0.05],
            index=pd.date_range("2020-01-01", periods=5),
        )

        # Benchmark has fewer dates (missing some)
        benchmark_returns = pd.Series(
            [0.005, 0.01, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )

        with patch("hrp.research.benchmark.get_benchmark_returns") as mock_bench:
            mock_bench.return_value = benchmark_returns

            result = compare_to_benchmark(strategy_returns, "SPY")

        # Should complete without error - dates aligned via dropna
        assert "strategy" in result

    def test_handles_datetime_index(self):
        """Test that datetime indices are handled correctly."""
        # Use datetime index (not just date)
        strategy_returns = pd.Series(
            [0.01, 0.02, 0.03],
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        )

        benchmark_returns = pd.Series(
            [0.005, 0.01, 0.015],
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        )

        with patch("hrp.research.benchmark.get_benchmark_returns") as mock_bench:
            mock_bench.return_value = benchmark_returns

            result = compare_to_benchmark(strategy_returns, "SPY")

        assert "strategy" in result


class TestEnsureBenchmarkData:
    """Tests for ensure_benchmark_data function."""

    def test_loads_when_empty(self):
        """Test that data is loaded when database is empty."""
        with patch("hrp.research.benchmark.get_db") as mock_get_db, \
             patch("hrp.data.ingestion.prices.ingest_prices") as mock_ingest:
            mock_db = MagicMock()
            mock_db.fetchone.return_value = (0,)  # No rows
            mock_get_db.return_value = mock_db

            ensure_benchmark_data("SPY", years=10)

        mock_ingest.assert_called_once()
        call_args = mock_ingest.call_args[0]
        assert "SPY" in call_args[0]

    def test_skips_when_exists(self):
        """Test that ingestion is skipped when data exists."""
        with patch("hrp.research.benchmark.get_db") as mock_get_db, \
             patch("hrp.data.ingestion.prices.ingest_prices") as mock_ingest:
            mock_db = MagicMock()
            mock_db.fetchone.return_value = (2500,)  # Has rows
            mock_get_db.return_value = mock_db

            ensure_benchmark_data("SPY", years=10)

        mock_ingest.assert_not_called()

    def test_custom_years(self):
        """Test that years parameter affects date range."""
        with patch("hrp.research.benchmark.get_db") as mock_get_db, \
             patch("hrp.data.ingestion.prices.ingest_prices") as mock_ingest:
            mock_db = MagicMock()
            mock_db.fetchone.return_value = (0,)  # No rows
            mock_get_db.return_value = mock_db

            ensure_benchmark_data("SPY", years=5)

        call_args = mock_ingest.call_args[0]
        start_date = call_args[1]
        # Should start 5 years ago
        expected_year = date.today().year - 5
        assert start_date.year == expected_year

    def test_different_benchmark(self):
        """Test that different benchmarks can be loaded."""
        with patch("hrp.research.benchmark.get_db") as mock_get_db, \
             patch("hrp.data.ingestion.prices.ingest_prices") as mock_ingest:
            mock_db = MagicMock()
            mock_db.fetchone.return_value = (0,)  # No rows
            mock_get_db.return_value = mock_db

            ensure_benchmark_data("QQQ", years=10)

        call_args = mock_ingest.call_args[0]
        assert "QQQ" in call_args[0]


class TestBenchmarksConstant:
    """Tests for BENCHMARKS constant."""

    def test_contains_standard_benchmarks(self):
        """Test that standard benchmarks are defined."""
        assert "SPY" in BENCHMARKS
        assert "QQQ" in BENCHMARKS
        assert "IWM" in BENCHMARKS
        assert "DIA" in BENCHMARKS

    def test_has_descriptions(self):
        """Test that all benchmarks have descriptions."""
        for ticker, description in BENCHMARKS.items():
            assert isinstance(description, str)
            assert len(description) > 0
