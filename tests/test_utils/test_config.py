"""Tests for utility configuration classes."""

import pytest

from hrp.utils.config import DefaultBacktestConfig


class TestDefaultBacktestConfig:
    """Tests for DefaultBacktestConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating DefaultBacktestConfig with default values."""
        config = DefaultBacktestConfig()
        assert config.max_position_pct == 0.10
        assert config.max_positions == 20
        assert config.min_position_pct == 0.02
        assert config.commission_pct == 0.0005
        assert config.slippage_pct == 0.001
        assert config.max_gross_exposure == 1.0
        assert config.strategy_stop_loss == 0.15
        assert config.portfolio_stop_loss == 0.20

    def test_config_with_custom_values(self):
        """Test creating DefaultBacktestConfig with custom values."""
        config = DefaultBacktestConfig(
            max_position_pct=0.05,
            max_positions=10,
            min_position_pct=0.01,
            commission_pct=0.001,
            slippage_pct=0.002,
            max_gross_exposure=0.8,
            strategy_stop_loss=0.10,
            portfolio_stop_loss=0.15,
        )

        assert config.max_position_pct == 0.05
        assert config.max_positions == 10
        assert config.min_position_pct == 0.01
        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.002
        assert config.max_gross_exposure == 0.8
        assert config.strategy_stop_loss == 0.10
        assert config.portfolio_stop_loss == 0.15
