"""Tests for signal-to-order conversion."""
from decimal import Decimal

import pandas as pd
import pytest

from hrp.execution.orders import OrderSide, OrderType
from hrp.execution.signal_converter import ConversionConfig, SignalConverter


def test_signal_converter_config_validation():
    """Test conversion config validation."""
    with pytest.raises(ValueError, match="max_position_pct must be between 0 and 1"):
        ConversionConfig(portfolio_value=Decimal("100000"), max_position_pct=1.5)

    with pytest.raises(ValueError, match="max_positions must be positive"):
        ConversionConfig(portfolio_value=Decimal("100000"), max_positions=0)


def test_signal_converter_valid_config():
    """Test creating a valid conversion config."""
    config = ConversionConfig(
        portfolio_value=Decimal("100000"),
        max_positions=20,
        max_position_pct=0.10,
    )

    assert config.portfolio_value == Decimal("100000")
    assert config.max_positions == 20
    assert config.max_position_pct == 0.10


def test_signal_converter_rank_signals_to_orders():
    """Test converting rank-based signals to orders."""
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "signal": [1.0, 1.0, 0.0, 0.0],  # Long AAPL and MSFT
        "prediction": [0.05, 0.04, 0.01, 0.00],
    })

    config = ConversionConfig(
        portfolio_value=Decimal("100000"),
        max_positions=20,
        max_position_pct=0.10,
    )

    converter = SignalConverter(config)
    orders = converter.signals_to_orders(signals, method="rank")

    assert len(orders) == 2
    assert orders[0].symbol == "AAPL"  # Highest prediction first
    assert orders[0].side == OrderSide.BUY
    assert orders[0].order_type == OrderType.MARKET


def test_signal_converter_respects_position_limits():
    """Test signal converter respects position count limits."""
    # Create more signals than max_positions
    symbols = [f"SYM{i}" for i in range(25)]
    signals = pd.DataFrame({
        "symbol": symbols,
        "signal": [1.0] * 25,
        "prediction": [0.05 - i * 0.001 for i in range(25)],
    })

    config = ConversionConfig(
        portfolio_value=Decimal("100000"),
        max_positions=20,  # Hard limit
        max_position_pct=0.05,
    )

    converter = SignalConverter(config)
    orders = converter.signals_to_orders(signals, method="rank")

    # Should only create 20 orders (max_positions limit)
    assert len(orders) <= 20


def test_signal_converter_no_buy_signals():
    """Test converter handles no buy signals."""
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "signal": [0.0, 0.0],  # No long signals
        "prediction": [0.01, 0.02],
    })

    config = ConversionConfig(portfolio_value=Decimal("100000"))
    converter = SignalConverter(config)
    orders = converter.signals_to_orders(signals, method="rank")

    assert len(orders) == 0


def test_signal_converter_with_custom_prices():
    """Test converter uses provided prices."""
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "signal": [1.0, 1.0],
        "prediction": [0.05, 0.04],
    })

    prices = {
        "AAPL": Decimal("175.00"),
        "MSFT": Decimal("350.00"),
    }

    config = ConversionConfig(
        portfolio_value=Decimal("100000"),
        max_position_pct=0.10,  # $10,000 per position
    )

    converter = SignalConverter(config)
    orders = converter.signals_to_orders(
        signals, method="rank", current_prices=prices
    )

    assert len(orders) == 2
    # At $175 per share, $10,000 = ~57 shares
    assert orders[0].quantity == 57  # AAPL at $175


def test_signal_converter_skips_small_orders():
    """Test converter skips orders below minimum value."""
    signals = pd.DataFrame({
        "symbol": ["AAPL"],
        "signal": [1.0],
        "prediction": [0.05],
    })

    config = ConversionConfig(
        portfolio_value=Decimal("1000"),  # Very small portfolio
        max_position_pct=0.01,  # 1% = $10 position
        min_order_value=Decimal("100.00"),  # But min order is $100
    )

    converter = SignalConverter(config)
    orders = converter.signals_to_orders(signals, method="rank")

    # Should skip because position value < min_order_value
    assert len(orders) == 0


def test_signal_converter_rebalance():
    """Test rebalancing from current to target positions."""
    current_positions = {"AAPL": 50, "MSFT": 30, "IBM": 20}  # Exit IBM

    target_signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "signal": [1.0, 1.0, 1.0],  # Add GOOGL
        "prediction": [0.05, 0.04, 0.03],
    })

    current_prices = {
        "AAPL": Decimal("175.00"),
        "MSFT": Decimal("350.00"),
        "GOOGL": Decimal("140.00"),
        "IBM": Decimal("120.00"),
    }

    config = ConversionConfig(
        portfolio_value=Decimal("100000"),
        max_positions=20,
        max_position_pct=0.10,
    )

    converter = SignalConverter(config)
    orders = converter.rebalance_to_orders(
        current_positions, target_signals, current_prices
    )

    # Should have: sell IBM (exit), buy GOOGL (new), adjust AAPL/MSFT
    symbols_in_orders = {o.symbol for o in orders}
    assert "IBM" in symbols_in_orders  # Should exit
    assert "GOOGL" in symbols_in_orders  # Should add

    # Find IBM order
    ibm_orders = [o for o in orders if o.symbol == "IBM"]
    assert len(ibm_orders) == 1
    assert ibm_orders[0].side == OrderSide.SELL
    assert ibm_orders[0].quantity == 20
