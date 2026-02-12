"""Trading configuration for Robinhood execution.

Configuration constants and defaults for trading operations.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass
class TradingConfig:
    """Main trading configuration."""

    # Robinhood API settings
    robinhood_username: Optional[str] = None
    robinhood_password: Optional[str] = None
    paper_trading: bool = True

    # Order defaults
    default_order_type: str = "market"  # market, limit, stop
    default_time_in_force: str = "gfd"  # Good for day
    default_timeout: int = 30

    # Risk management
    max_position_var: Decimal = Decimal("0.02")  # 2% per position
    max_portfolio_var: Decimal = Decimal("0.05")  # 5% total portfolio
    use_cvar: bool = False  # Use CVaR for sizing
    enable_risk_checks: bool = True

    # Position sizing
    min_position_size: int = 1
    max_position_size: Decimal = Decimal("0.20")  # 20% max
    risk_target: Decimal = Decimal("0.01")  # Target 1% VaR

    # VaR calculation
    var_confidence: float = 0.95
    var_time_horizon: int = 1  # 1-day VaR
    var_method: str = "parametric"  # parametric, historical, monte_carlo

    # Rate limiting
    api_rate_limit_delay: float = 0.2  # Seconds between requests
    max_retries: int = 3

    # Logging
    log_orders: bool = True
    log_performance: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_position_var <= 0:
            raise ValueError("max_position_var must be positive")

        if self.max_portfolio_var <= 0:
            raise ValueError("max_portfolio_var must be positive")

        if self.max_position_var > self.max_portfolio_var:
            raise ValueError("max_position_var cannot exceed max_portfolio_var")

        if self.max_position_size > 1:
            raise ValueError("max_position_size cannot exceed 1.0")

        if self.min_position_size < 1:
            raise ValueError("min_position_size must be at least 1")


def get_default_config() -> TradingConfig:
    """Get default trading configuration.

    Returns:
        TradingConfig with safe defaults
    """
    return TradingConfig()


def get_paper_trading_config() -> TradingConfig:
    """Get configuration for paper trading.

    Returns:
        TradingConfig with paper trading enabled
    """
    return TradingConfig(paper_trading=True)


def get_live_trading_config() -> TradingConfig:
    """Get configuration for live trading.

    Returns:
        TradingConfig with paper trading disabled
    """
    return TradingConfig(paper_trading=False)
