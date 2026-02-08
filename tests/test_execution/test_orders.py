"""Tests for order management system."""
import pytest
from datetime import datetime
from decimal import Decimal
from hrp.execution.orders import Order, OrderType, OrderSide, OrderStatus, OrderManager


def test_order_creation():
    """Test creating a market order."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.quantity == 10
    assert order.order_type == OrderType.MARKET
    assert order.status == OrderStatus.PENDING
    assert order.limit_price is None


def test_limit_order_requires_price():
    """Test limit order requires limit_price."""
    with pytest.raises(ValueError, match="Limit orders require limit_price"):
        Order(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=None,
        )


def test_limit_order_with_price():
    """Test creating a valid limit order."""
    order = Order(
        symbol="MSFT",
        side=OrderSide.SELL,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00"),
    )

    assert order.order_type == OrderType.LIMIT
    assert order.limit_price == Decimal("150.00")


def test_order_quantity_must_be_positive():
    """Test order quantity must be positive."""
    with pytest.raises(ValueError, match="quantity must be positive"):
        Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,
            order_type=OrderType.MARKET,
        )


def test_limit_price_must_be_positive():
    """Test limit price must be positive."""
    with pytest.raises(ValueError, match="limit_price must be positive"):
        Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("-10.00"),
        )


@pytest.fixture
def mock_broker():
    """Mock IBKR broker for testing."""
    from unittest.mock import Mock

    broker = Mock()
    broker.is_connected.return_value = True

    # Mock IB instance
    broker.ib = Mock()

    # Mock trade result
    mock_trade = Mock()
    mock_trade.order.orderId = 12345
    broker.ib.placeOrder.return_value = mock_trade

    return broker


def test_order_manager_submit_market_order(mock_broker):
    """Test submitting a market order via broker."""
    manager = OrderManager(mock_broker)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    submitted = manager.submit_order(order)

    assert submitted.status == OrderStatus.SUBMITTED
    assert submitted.broker_order_id is not None
    assert submitted.submitted_at is not None


def test_order_manager_requires_connection(mock_broker):
    """Test order manager requires broker connection."""
    mock_broker.is_connected.return_value = False
    manager = OrderManager(mock_broker)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    with pytest.raises(ValueError, match="Broker not connected"):
        manager.submit_order(order)


def test_order_manager_get_order(mock_broker):
    """Test getting order by ID."""
    manager = OrderManager(mock_broker)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    submitted = manager.submit_order(order)

    retrieved = manager.get_order(order.order_id)
    assert retrieved is not None
    assert retrieved.order_id == order.order_id


def test_order_manager_get_all_orders(mock_broker):
    """Test getting all tracked orders."""
    manager = OrderManager(mock_broker)

    order1 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
    order2 = Order(symbol="MSFT", side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET)

    manager.submit_order(order1)
    manager.submit_order(order2)

    all_orders = manager.get_all_orders()
    assert len(all_orders) == 2
