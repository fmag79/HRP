"""Trading module for Robinhood order execution.

This module provides:
- Trading configuration (config.py)
- Robinhood API client (robinhood_client.py)
- Order lifecycle management (order_manager.py)
- Risk-based position sizing (position_sizer.py)
- Risk engine integration (risk_engine.py)
"""

from hrp.trading.config import (
    TradingConfig,
    get_default_config,
    get_paper_trading_config,
    get_live_trading_config,
)
from hrp.trading.robinhood_client import (
    RobinhoodClient,
    RobinhoodConfig,
    RobinhoodError,
    RateLimitError,
    AuthenticationError,
)
from hrp.trading.order_manager import (
    OrderManager,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderSubmission,
)
from hrp.trading.position_sizer import (
    VaRPositionSizer,
    PositionSizingConfig,
    PositionSizeResult,
)
from hrp.trading.risk_engine import (
    RiskEngine,
    RiskConfig,
    RiskResult,
)

__all__ = [
    # Configuration
    "TradingConfig",
    "get_default_config",
    "get_paper_trading_config",
    "get_live_trading_config",
    # Robinhood Client
    "RobinhoodClient",
    "RobinhoodConfig",
    "RobinhoodError",
    "RateLimitError",
    "AuthenticationError",
    # Order Manager
    "OrderManager",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "OrderSubmission",
    # Position Sizer
    "VaRPositionSizer",
    "PositionSizingConfig",
    "PositionSizeResult",
    # Risk Engine
    "RiskEngine",
    "RiskConfig",
    "RiskResult",
]
