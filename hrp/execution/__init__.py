"""Live trading, order management, broker integration.

This module contains:
- Broker connections (Interactive Brokers, Robinhood)
- Order management system
- Position tracking
- Signal-to-order conversion
- VaR-aware position sizing
- Trade execution logic

Status: Tier 4 - Implemented
"""

from hrp.execution.broker import BrokerConfig, IBKRBroker
from hrp.execution.orders import Order, OrderManager, OrderSide, OrderStatus, OrderType
from hrp.execution.position_sizer import PositionSizer, PositionSizingConfig
from hrp.execution.positions import Position, PositionTracker
from hrp.execution.signal_converter import ConversionConfig, SignalConverter

__all__ = [
    "BrokerConfig",
    "ConversionConfig",
    "IBKRBroker",
    "Order",
    "OrderManager",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "PositionSizer",
    "PositionSizingConfig",
    "PositionTracker",
    "SignalConverter",
]
