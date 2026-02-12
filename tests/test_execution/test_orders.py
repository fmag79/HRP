"""Tests for order management system."""
from decimal import Decimal

import pytest

from hrp.execution.orders import Order, OrderManager, OrderSide, OrderStatus, OrderType


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

    manager.submit_order(order)

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


# Tests for new order types (STOP_LOSS, STOP_LIMIT, TRAILING_STOP)


def test_stop_loss_order_requires_stop_price():
    """Test STOP_LOSS order requires stop_price."""
    with pytest.raises(ValueError, match="STOP_LOSS orders require stop_price"):
        Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.STOP_LOSS,
            stop_price=None,
        )


def test_stop_loss_order_valid():
    """Test creating a valid STOP_LOSS order."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.STOP_LOSS,
        stop_price=Decimal("140.00"),
    )

    assert order.order_type == OrderType.STOP_LOSS
    assert order.stop_price == Decimal("140.00")
    assert order.limit_price is None


def test_stop_loss_stop_price_must_be_positive():
    """Test STOP_LOSS stop_price must be positive."""
    with pytest.raises(ValueError, match="stop_price must be positive"):
        Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.STOP_LOSS,
            stop_price=Decimal("-10.00"),
        )


def test_stop_limit_order_requires_both_prices():
    """Test STOP_LIMIT order requires both stop_price and limit_price."""
    with pytest.raises(ValueError, match="STOP_LIMIT orders require both"):
        Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.STOP_LIMIT,
            stop_price=Decimal("145.00"),
            limit_price=None,
        )

    with pytest.raises(ValueError, match="STOP_LIMIT orders require both"):
        Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.STOP_LIMIT,
            stop_price=None,
            limit_price=Decimal("150.00"),
        )


def test_stop_limit_order_valid():
    """Test creating a valid STOP_LIMIT order."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.STOP_LIMIT,
        stop_price=Decimal("145.00"),
        limit_price=Decimal("150.00"),
    )

    assert order.order_type == OrderType.STOP_LIMIT
    assert order.stop_price == Decimal("145.00")
    assert order.limit_price == Decimal("150.00")


def test_trailing_stop_order_requires_trail_amount():
    """Test TRAILING_STOP order requires trail_amount."""
    with pytest.raises(ValueError, match="TRAILING_STOP orders require trail_amount"):
        Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.TRAILING_STOP,
            trail_amount=None,
        )


def test_trailing_stop_order_valid_percentage():
    """Test creating a valid TRAILING_STOP order with percentage."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.TRAILING_STOP,
        trail_amount=5.0,
        trail_type="percentage",
    )

    assert order.order_type == OrderType.TRAILING_STOP
    assert order.trail_amount == 5.0
    assert order.trail_type == "percentage"


def test_trailing_stop_order_valid_amount():
    """Test creating a valid TRAILING_STOP order with amount."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.TRAILING_STOP,
        trail_amount=10.0,
        trail_type="amount",
    )

    assert order.order_type == OrderType.TRAILING_STOP
    assert order.trail_amount == 10.0
    assert order.trail_type == "amount"


def test_trailing_stop_trail_amount_must_be_positive():
    """Test TRAILING_STOP trail_amount must be positive."""
    with pytest.raises(ValueError, match="trail_amount must be positive"):
        Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.TRAILING_STOP,
            trail_amount=-5.0,
        )


def test_trailing_stop_trail_type_validation():
    """Test TRAILING_STOP trail_type must be valid."""
    with pytest.raises(ValueError, match="trail_type must be"):
        Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.TRAILING_STOP,
            trail_amount=5.0,
            trail_type="invalid",
        )
