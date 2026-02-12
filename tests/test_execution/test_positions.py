"""Tests for position tracking."""
from datetime import date
from decimal import Decimal
from unittest.mock import Mock

import pytest

from hrp.execution.positions import Position, PositionTracker


def test_position_creation():
    """Test creating a position."""
    position = Position(
        symbol="AAPL",
        quantity=10,
        entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
        as_of_date=date(2026, 2, 8),
    )

    assert position.symbol == "AAPL"
    assert position.quantity == 10
    assert position.market_value == Decimal("1550.00")
    assert position.cost_basis == Decimal("1500.00")
    assert position.unrealized_pnl == Decimal("50.00")
    assert position.unrealized_pnl_pct == pytest.approx(0.0333, rel=1e-2)


def test_position_with_commission():
    """Test position cost basis includes commission."""
    position = Position(
        symbol="AAPL",
        quantity=10,
        entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
        as_of_date=date(2026, 2, 8),
        commission_paid=Decimal("5.00"),
    )

    assert position.cost_basis == Decimal("1505.00")
    assert position.unrealized_pnl == Decimal("45.00")


def test_position_update_price():
    """Test updating position price."""
    position = Position(
        symbol="MSFT",
        quantity=5,
        entry_price=Decimal("300.00"),
        current_price=Decimal("300.00"),
        as_of_date=date(2026, 2, 8),
    )

    position.update_price(Decimal("310.00"))

    assert position.current_price == Decimal("310.00")
    assert position.unrealized_pnl == Decimal("50.00")


def test_position_zero_cost_basis():
    """Test position with zero entry price."""
    position = Position(
        symbol="FREE",
        quantity=10,
        entry_price=Decimal("0.00"),
        current_price=Decimal("10.00"),
        as_of_date=date(2026, 2, 8),
    )

    # Should not divide by zero
    assert position.unrealized_pnl_pct == 0.0


@pytest.fixture
def mock_api():
    """Mock PlatformAPI for testing."""
    api = Mock()
    api.get_db.return_value = Mock()
    return api


@pytest.fixture
def mock_broker():
    """Mock IBKR broker with positions."""
    broker = Mock()
    broker.is_connected.return_value = True
    broker.config.account = "DU123456"

    # Mock IB instance
    broker.ib = Mock()

    # Mock positions
    mock_pos1 = Mock()
    mock_pos1.contract.symbol = "AAPL"
    mock_pos1.position = 10
    mock_pos1.avgCost = 150.0
    mock_pos1.account = "DU123456"

    mock_pos2 = Mock()
    mock_pos2.contract.symbol = "MSFT"
    mock_pos2.position = 5
    mock_pos2.avgCost = 300.0
    mock_pos2.account = "DU123456"

    broker.ib.positions.return_value = [mock_pos1, mock_pos2]

    # Mock market data
    mock_ticker = Mock()
    mock_ticker.last = 155.0
    broker.ib.reqMktData.return_value = mock_ticker
    broker.ib.sleep = Mock()

    return broker


def test_position_tracker_sync(mock_broker, mock_api):
    """Test syncing positions from broker."""
    tracker = PositionTracker(mock_broker, mock_api)

    positions = tracker.sync_positions()

    assert len(positions) == 2
    assert positions[0].symbol == "AAPL"
    assert positions[1].symbol == "MSFT"


def test_position_tracker_get_position(mock_broker, mock_api):
    """Test getting position by symbol."""
    tracker = PositionTracker(mock_broker, mock_api)
    tracker.sync_positions()

    position = tracker.get_position("AAPL")
    assert position is not None
    assert position.symbol == "AAPL"


def test_position_tracker_get_all(mock_broker, mock_api):
    """Test getting all positions."""
    tracker = PositionTracker(mock_broker, mock_api)
    tracker.sync_positions()

    all_positions = tracker.get_all_positions()
    assert len(all_positions) == 2


def test_position_tracker_requires_connection(mock_broker, mock_api):
    """Test position tracker requires broker connection."""
    mock_broker.is_connected.return_value = False
    tracker = PositionTracker(mock_broker, mock_api)

    with pytest.raises(ValueError, match="Broker not connected"):
        tracker.sync_positions()
