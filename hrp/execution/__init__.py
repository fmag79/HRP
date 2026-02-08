"""Live trading, order management, broker integration.

This module contains:
- Broker connections (Interactive Brokers)
- Order management system
- Position tracking
- Signal-to-order conversion
- Trade execution logic

Status: Tier 4 - Implemented
"""

from hrp.execution.broker import BrokerConfig, IBKRBroker
from hrp.execution.orders import Order, OrderManager, OrderSide, OrderStatus, OrderType
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
    "PositionTracker",
    "SignalConverter",
]
