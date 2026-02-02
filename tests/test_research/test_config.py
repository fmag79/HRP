"""Tests for research configuration classes."""

import pytest

from hrp.research.config import (
    StopLossConfig,
    CostModel,
    BacktestConfig,
)


class TestStopLossConfig:
    """Tests for StopLossConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = StopLossConfig()
        assert config.enabled is False
        assert config.type == "atr_trailing"
        assert config.atr_multiplier == 2.0
        assert config.atr_period == 14
        assert config.fixed_pct == 0.05
        assert config.lookback_for_high == 1

    def test_config_creation_with_enabled(self):
        """Test creating enabled stop-loss config."""
        config = StopLossConfig(enabled=True)
        assert config.enabled is True

    def test_config_fixed_pct_type(self):
        """Test creating fixed percentage stop-loss config."""
        config = StopLossConfig(
            enabled=True,
            type="fixed_pct",
            fixed_pct=0.10,
        )
        assert config.type == "fixed_pct"
        assert config.fixed_pct == 0.10

    def test_config_atr_trailing_type(self):
        """Test creating ATR trailing stop-loss config."""
        config = StopLossConfig(
            enabled=True,
            type="atr_trailing",
            atr_multiplier=3.0,
            atr_period=20,
        )
        assert config.type == "atr_trailing"
        assert config.atr_multiplier == 3.0
        assert config.atr_period == 20

    def test_config_volatility_scaled_type(self):
        """Test creating volatility-scaled stop-loss config."""
        config = StopLossConfig(
            enabled=True,
            type="volatility_scaled",
        )
        assert config.type == "volatility_scaled"

    def test_config_invalid_type(self):
        """Test config rejects invalid stop-loss type."""
        with pytest.raises(ValueError, match="Invalid stop-loss type"):
            StopLossConfig(type="invalid_type")

    def test_config_invalid_atr_multiplier(self):
        """Test config rejects non-positive ATR multiplier."""
        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            StopLossConfig(atr_multiplier=0)

        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            StopLossConfig(atr_multiplier=-1.0)

    def test_config_invalid_atr_period(self):
        """Test config rejects ATR period < 1."""
        with pytest.raises(ValueError, match="atr_period must be at least 1"):
            StopLossConfig(atr_period=0)

    def test_config_invalid_fixed_pct(self):
        """Test config rejects invalid fixed_pct values."""
        with pytest.raises(ValueError, match="fixed_pct must be between 0 and 1"):
            StopLossConfig(fixed_pct=0)

        with pytest.raises(ValueError, match="fixed_pct must be between 0 and 1"):
            StopLossConfig(fixed_pct=1.0)

        with pytest.raises(ValueError, match="fixed_pct must be between 0 and 1"):
            StopLossConfig(fixed_pct=1.5)

    def test_config_invalid_lookback_for_high(self):
        """Test config rejects lookback_for_high < 1."""
        with pytest.raises(ValueError, match="lookback_for_high must be at least 1"):
            StopLossConfig(lookback_for_high=0)


class TestCostModel:
    """Tests for CostModel dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating CostModel with default values."""
        config = CostModel()
        assert config.commission_per_share == 0.005
        assert config.commission_min == 1.00
        assert config.commission_max_pct == 0.01
        assert config.spread_bps == 5.0
        assert config.slippage_bps == 5.0

    def test_total_cost_pct(self):
        """Test total_cost_pct includes spread, slippage, and commission."""
        config = CostModel(spread_bps=10.0, slippage_bps=5.0)
        cost = config.total_cost_pct()
        # Spread + slippage = (10 + 5) / 10000 = 0.0015
        # Commission = 0.005 / 50.0 = 0.0001
        # Total = 0.0016
        assert cost > 0.0015  # Must be higher than spread+slippage alone
        assert cost == 0.0015 + 0.005 / 50.0  # 0.0016

    def test_total_cost_pct_with_custom_price(self):
        """Test total_cost_pct with explicit share price."""
        config = CostModel(spread_bps=10.0, slippage_bps=5.0)
        # With a $100 share price, commission = 0.005/100 = 0.00005
        cost = config.total_cost_pct(avg_share_price=100.0)
        assert cost == 0.0015 + 0.005 / 100.0


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating BacktestConfig with default values."""
        config = BacktestConfig()
        assert config.symbols == []
        assert config.sizing_method == "equal"
        assert config.max_position_pct == 0.10
        assert config.max_positions == 20
        assert isinstance(config.costs, CostModel)
        assert isinstance(config.stop_loss, StopLossConfig)

    def test_config_with_stop_loss(self):
        """Test creating BacktestConfig with custom stop-loss."""
        stop_loss = StopLossConfig(enabled=True, atr_multiplier=2.5)
        config = BacktestConfig(stop_loss=stop_loss)

        assert config.stop_loss.enabled is True
        assert config.stop_loss.atr_multiplier == 2.5

    def test_config_stop_loss_default_not_enabled(self):
        """Test BacktestConfig has stop-loss disabled by default."""
        config = BacktestConfig()
        assert config.stop_loss.enabled is False

    def test_config_with_all_options(self):
        """Test creating BacktestConfig with all options specified."""
        from datetime import date

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            sizing_method="volatility",
            max_position_pct=0.05,
            max_positions=10,
            costs=CostModel(spread_bps=3.0),
            stop_loss=StopLossConfig(enabled=True, type="fixed_pct", fixed_pct=0.08),
            name="Test Backtest",
            description="A test backtest",
        )

        assert config.symbols == ["AAPL", "MSFT"]
        assert config.sizing_method == "volatility"
        assert config.max_position_pct == 0.05
        assert config.costs.spread_bps == 3.0
        assert config.stop_loss.enabled is True
        assert config.stop_loss.type == "fixed_pct"
        assert config.stop_loss.fixed_pct == 0.08


class TestModuleExports:
    """Test that config module classes are properly exported."""

    def test_import_from_research_config(self):
        """Test importing from hrp.research.config."""
        from hrp.research.config import (
            StopLossConfig,
            CostModel,
            BacktestConfig,
            BacktestResult,
        )

        assert StopLossConfig is not None
        assert CostModel is not None
        assert BacktestConfig is not None
        assert BacktestResult is not None
