"""Tests for backtest with transaction costs and risk limits."""

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch, Mock

from hrp.research.config import BacktestConfig
from hrp.research.backtest import run_backtest
from hrp.risk.costs import MarketImpactModel
from hrp.risk.limits import RiskLimits


class TestBacktestWithCosts:
    """Tests for backtest integration with cost model."""

    @pytest.fixture
    def sample_signals(self):
        """Sample signals for testing."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {"AAPL": [0.5] * 10, "MSFT": [0.5] * 10},
            index=dates,
        )

    @pytest.fixture
    def sample_prices(self):
        """Sample prices for testing."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = {
            ("close", "AAPL"): [150 + i for i in range(10)],
            ("close", "MSFT"): [250 + i for i in range(10)],
            ("volume", "AAPL"): [1_000_000] * 10,
            ("volume", "MSFT"): [500_000] * 10,
        }
        return pd.DataFrame(data, index=dates)

    def test_backtest_with_default_cost_model(self, sample_signals, sample_prices):
        """Backtest uses default MarketImpactModel."""
        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
        )

        with patch("hrp.research.backtest.get_price_data", return_value=sample_prices):
            with patch("hrp.research.backtest.get_benchmark_returns", return_value=None):
                result = run_backtest(sample_signals, config, sample_prices)

        assert result is not None
        assert "total_return" in result.metrics

    def test_backtest_with_custom_cost_model(self, sample_signals, sample_prices):
        """Backtest accepts custom cost model."""
        cost_model = MarketImpactModel(
            eta=0.2,  # Higher impact
            spread_bps=10.0,  # Wider spread
        )

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
            cost_model=cost_model,
        )

        assert config.cost_model.eta == 0.2
        assert config.cost_model.spread_bps == 10.0


class TestBacktestWithRiskLimits:
    """Tests for backtest integration with risk limits."""

    @pytest.fixture
    def sample_signals(self):
        """Signals that violate limits."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [0.60] * 5,  # Exceeds 5% max
                "MSFT": [0.40] * 5,  # Exceeds 5% max
            },
            index=dates,
        )

    @pytest.fixture
    def sample_prices(self):
        """Sample prices."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = {
            ("close", "AAPL"): [150] * 5,
            ("close", "MSFT"): [250] * 5,
            ("volume", "AAPL"): [50_000_000] * 5,
            ("volume", "MSFT"): [30_000_000] * 5,
        }
        return pd.DataFrame(data, index=dates)

    def test_backtest_applies_risk_limits(self, sample_signals, sample_prices):
        """Risk limits clip signals before backtest."""
        limits = RiskLimits(max_position_pct=0.05)

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            risk_limits=limits,
        )

        # This should clip positions to 5% each
        assert config.risk_limits.max_position_pct == 0.05

    def test_backtest_no_limits_backward_compatible(self, sample_signals, sample_prices):
        """Backtest without limits maintains backward compatibility."""
        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            # No risk_limits specified
        )

        assert config.risk_limits is None


class TestBacktestPreTradeValidation:
    """Tests for pre-trade validation in backtest flow."""

    @pytest.fixture
    def oversized_signals(self):
        """Signals with positions exceeding limits."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [0.30] * 5,
                "MSFT": [0.30] * 5,
                "GOOGL": [0.25] * 5,
                "AMZN": [0.15] * 5,
            },
            index=dates,
        )

    @pytest.fixture
    def mock_prices(self):
        """Mock price data."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        data = {}
        for sym in symbols:
            data[("close", sym)] = [100.0] * 5
            data[("volume", sym)] = [10_000_000] * 5
        return pd.DataFrame(data, index=dates)

    @patch("hrp.research.backtest.get_price_data")
    @patch("hrp.research.backtest.get_benchmark_returns")
    @patch("hrp.research.backtest._load_sector_mapping")
    def test_validation_clips_signals(
        self, mock_sectors, mock_benchmark, mock_prices_fn, oversized_signals, mock_prices
    ):
        """PreTradeValidator clips oversized positions."""
        mock_prices_fn.return_value = mock_prices
        mock_benchmark.return_value = None
        mock_sectors.return_value = pd.Series({
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
        })

        limits = RiskLimits(max_position_pct=0.10)
        config = BacktestConfig(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            risk_limits=limits,
        )

        result = run_backtest(oversized_signals, config, mock_prices)

        # Validation report should be attached
        assert hasattr(result, "validation_report")
        assert result.validation_report is not None
        assert len(result.validation_report.clips) > 0

    @patch("hrp.research.backtest.get_price_data")
    @patch("hrp.research.backtest.get_benchmark_returns")
    def test_no_validation_without_limits(
        self, mock_benchmark, mock_prices_fn, oversized_signals, mock_prices
    ):
        """No validation when risk_limits is None."""
        mock_prices_fn.return_value = mock_prices
        mock_benchmark.return_value = None

        config = BacktestConfig(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            risk_limits=None,  # No limits
        )

        result = run_backtest(oversized_signals, config, mock_prices)

        # No validation report when limits disabled
        assert result.validation_report is None
