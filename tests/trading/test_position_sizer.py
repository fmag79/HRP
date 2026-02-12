"""Tests for VaR-based position sizing.

Tests cover:
- Budget allocation across positions
- Sizing edge cases (min/max limits)
- VaR threshold enforcement
- CVaR integration
- Fallback behavior with insufficient data
"""

import pytest
import numpy as np
from decimal import Decimal
from hrp.trading.position_sizer import (
    VaRPositionSizer,
    PositionSizingConfig,
    PositionSizeResult,
)


@pytest.fixture
def config():
    """Default position sizing config."""
    return PositionSizingConfig(
        max_position_var=Decimal("0.02"),
        max_portfolio_var=Decimal("0.05"),
        min_position_size=Decimal("1"),
        max_position_size=Decimal("0.20"),
        risk_target=Decimal("0.01"),
        confidence_level=0.95,
        time_horizon=1,
        use_cvar=False,
    )


@pytest.fixture
def returns():
    """Sample return data for VaR calculation."""
    np.random.seed(42)
    return list(np.random.normal(0.001, 0.02, 100))


def test_config_validation(config):
    """Test configuration validation."""
    # Valid config should work
    assert config.max_position_var > 0
    assert config.max_portfolio_var > 0
    assert config.min_position_size > 0
    assert config.max_position_size <= 1
    assert config.max_position_var <= config.max_portfolio_var


def test_config_invalid_values():
    """Test configuration rejects invalid values."""
    with pytest.raises(ValueError):
        PositionSizingConfig(max_position_var=Decimal("0"))

    with pytest.raises(ValueError):
        PositionSizingConfig(max_portfolio_var=Decimal("0"))

    with pytest.raises(ValueError):
        PositionSizingConfig(min_position_size=Decimal("0"))

    with pytest.raises(ValueError):
        PositionSizingConfig(max_position_size=Decimal("1.5"))

    with pytest.raises(ValueError):
        PositionSizingConfig(
            max_position_var=Decimal("0.10"),
            max_portfolio_var=Decimal("0.05"),
        )


def test_max_position_size(config):
    """Test maximum position size calculation."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")

    # Should respect max position size (20% of portfolio)
    max_size = sizer.max_position_size(price, portfolio_value)

    # Max 20% of $10,000 at $100/share = 20 shares max
    assert max_size == 20


def test_max_position_size_minimum(config):
    """Test minimum position size is respected."""
    sizer = VaRPositionSizer(config)

    price = Decimal("1000")
    portfolio_value = Decimal("10000")

    # Even with high price, should return minimum of 1 share
    max_size = sizer.max_position_size(price, portfolio_value)

    assert max_size >= 1


def test_adjust_for_var_within_limits(config, returns):
    """Test position adjustment when within VaR limits."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")
    proposed_size = 10

    result = sizer.adjust_for_var(
        proposed_size=proposed_size,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # Should not limit if within VaR budget
    assert result.recommended_size == proposed_size
    assert result.limit_reason is None
    assert result.position_value == price * proposed_size


def test_adjust_for_var_exceeds_position_limit(config, returns):
    """Test position adjustment when VaR exceeds single position limit."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")

    # Large position that would exceed VaR limit
    proposed_size = 50

    result = sizer.adjust_for_var(
        proposed_size=proposed_size,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # Should reduce size to fit within VaR budget
    assert result.recommended_size < proposed_size
    assert result.limit_reason is not None
    assert "VaR limit" in result.limit_reason.lower()

    # Should still respect minimum
    assert result.recommended_size >= 1


def test_adjust_for_var_portfolio_limit(config, returns):
    """Test position adjustment when portfolio VaR budget exceeded."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")
    proposed_size = 10

    # Simulate existing positions using most of budget
    existing_var = Decimal("400")  # Close to 5% of $10,000

    result = sizer.adjust_for_var(
        proposed_size=proposed_size,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
        existing_positions_var=existing_var,
    )

    # Should reduce size to fit within portfolio budget
    if result.limit_reason:
        assert "portfolio" in result.limit_reason.lower()


def test_adjust_for_var_insufficient_data(config):
    """Test fallback behavior with insufficient return data."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")
    proposed_size = 10

    # Not enough data for VaR
    result = sizer.adjust_for_var(
        proposed_size=proposed_size,
        price=price,
        returns=[0.01] * 10,  # Too few data points
        portfolio_value=portfolio_value,
    )

    # Should fall back to percentage-based sizing
    assert result.recommended_size <= proposed_size
    assert result.limit_reason is not None


def test_portfolio_var_budget_empty(config):
    """Test portfolio VaR budget with no existing positions."""
    sizer = VaRPositionSizer(config)

    portfolio_value = Decimal("10000")

    budget = sizer.portfolio_var_budget(portfolio_value)

    # Should have full budget available
    assert budget["total_budget"] == portfolio_value * config.max_portfolio_var
    assert budget["used_budget"] == Decimal("0")
    assert budget["available_budget"] == budget["total_budget"]
    assert budget["utilization"] == Decimal("0")


def test_portfolio_var_budget_with_positions(config):
    """Test portfolio VaR budget with existing positions."""
    sizer = VaRPositionSizer(config)

    portfolio_value = Decimal("10000")

    # Simulate existing positions
    existing_positions = [
        {"symbol": "AAPL", "value": 2000, "var": 150},
        {"symbol": "MSFT", "value": 1500, "var": 100},
    ]

    budget = sizer.portfolio_var_budget(portfolio_value, existing_positions)

    # Should account for existing VaR usage
    assert budget["used_budget"] == Decimal("250")
    assert budget["available_budget"] == budget["total_budget"] - Decimal("250")
    assert budget["utilization"] > 0


def test_portfolio_var_budget_utilization(config):
    """Test portfolio VaR budget utilization calculation."""
    sizer = VaRPositionSizer(config)

    portfolio_value = Decimal("10000")

    # Fill 80% of budget
    existing_positions = [{"var": Decimal("400")}]

    budget = sizer.portfolio_var_budget(portfolio_value, existing_positions)

    total_budget = portfolio_value * config.max_portfolio_var
    expected_utilization = Decimal("400") / total_budget

    assert abs(budget["utilization"] - expected_utilization) < Decimal("0.01")


def test_calculate_optimal_size(config, returns):
    """Test optimal position size calculation."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")

    result = sizer.calculate_optimal_size(
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # Should return a valid size
    assert result.recommended_size >= config.min_position_size
    assert result.position_value is not None
    assert result.var_ratio is not None


def test_calculate_optimal_size_with_target(config, returns):
    """Test optimal size with custom VaR target."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")
    target_var_ratio = Decimal("0.005")  # More aggressive 0.5%

    result = sizer.calculate_optimal_size(
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
        target_var_ratio=target_var_ratio,
    )

    # Should use custom target
    assert result.recommended_size >= config.min_position_size
    # VaR ratio should be close to target
    assert result.var_ratio is not None


def test_calculate_optimal_size_with_existing_positions(config, returns):
    """Test optimal size with existing positions consuming budget."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")

    # Existing position using some budget
    existing_positions = [{"var": Decimal("200")}]

    result = sizer.calculate_optimal_size(
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
        existing_positions=existing_positions,
    )

    # Should account for existing VaR usage
    assert result.portfolio_var_used is not None
    assert result.portfolio_var_used >= Decimal("200")


def test_cvar_enabled(config, returns):
    """Test CVaR integration when enabled."""
    config_cvar = PositionSizingConfig(
        **{
            **config.__dict__,
            "use_cvar": True,
        }
    )

    sizer = VaRPositionSizer(config_cvar)

    price = Decimal("100")
    portfolio_value = Decimal("10000")
    proposed_size = 10

    result = sizer.adjust_for_var(
        proposed_size=proposed_size,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # CVaR values should be present when enabled
    assert result.cvar_at_size is not None


def test_cvar_disabled(config, returns):
    """Test CVaR not used when disabled."""
    assert not config.use_cvar

    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")
    proposed_size = 10

    result = sizer.adjust_for_var(
        proposed_size=proposed_size,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # CVaR should be None when disabled
    assert result.cvar_at_size is None


def test_cache_functionality(config, returns):
    """Test VaR calculation caching."""
    sizer = VaRPositionSizer(config)

    price = Decimal("100")
    portfolio_value = Decimal("10000")

    # First calculation
    result1 = sizer.adjust_for_var(
        proposed_size=10,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # Second calculation with same parameters (should use cache)
    result2 = sizer.adjust_for_var(
        proposed_size=10,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # Results should be consistent
    assert result1.recommended_size == result2.recommended_size

    # Clear cache
    sizer.clear_cache()

    # Third calculation (cache cleared)
    result3 = sizer.adjust_for_var(
        proposed_size=10,
        price=price,
        returns=returns,
        portfolio_value=portfolio_value,
    )

    # Still should produce same result
    assert result1.recommended_size == result3.recommended_size


def test_invalid_inputs(config):
    """Test error handling for invalid inputs."""
    sizer = VaRPositionSizer(config)

    with pytest.raises(ValueError):
        sizer.max_position_size(
            price=Decimal("0"),
            portfolio_value=Decimal("10000"),
        )

    with pytest.raises(ValueError):
        sizer.max_position_size(
            price=Decimal("100"),
            portfolio_value=Decimal("0"),
        )

    with pytest.raises(ValueError):
        sizer.adjust_for_var(
            proposed_size=0,
            price=Decimal("100"),
            returns=[],
            portfolio_value=Decimal("10000"),
        )

    with pytest.raises(ValueError):
        sizer.adjust_for_var(
            proposed_size=10,
            price=Decimal("0"),
            returns=[],
            portfolio_value=Decimal("10000"),
        )


def test_edge_case_zero_portfolio_value(config):
    """Test behavior with zero portfolio value."""
    sizer = VaRPositionSizer(config)

    with pytest.raises(ValueError):
        sizer.max_position_size(
            price=Decimal("100"),
            portfolio_value=Decimal("0"),
        )


def test_edge_case_very_high_price(config):
    """Test behavior with very high share price."""
    sizer = VaRPositionSizer(config)

    price = Decimal("10000")
    portfolio_value = Decimal("10000")

    max_size = sizer.max_position_size(price, portfolio_value)

    # Should return minimum 1 share
    assert max_size == 1
