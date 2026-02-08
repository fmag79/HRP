"""Tests for trading API methods."""
import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


@pytest.fixture
def mock_db():
    """Mock database connection."""
    db = Mock()
    db.fetchdf = Mock(return_value=pd.DataFrame())
    db.fetchone = Mock(return_value=None)
    db.execute = Mock()
    db.commit = Mock()
    return db


@pytest.fixture
def api_with_mock_db(mock_db):
    """Create PlatformAPI with mocked database."""
    with patch("hrp.api.platform.get_db", return_value=mock_db):
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()
        return api


def test_get_live_positions(api_with_mock_db, mock_db):
    """Test getting live positions."""
    mock_db.fetchdf.return_value = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "quantity": [10, 5],
    })

    positions = api_with_mock_db.get_live_positions()

    assert len(positions) == 2
    mock_db.fetchdf.assert_called_once()


def test_get_live_positions_with_date(api_with_mock_db, mock_db):
    """Test getting positions for specific date."""
    api_with_mock_db.get_live_positions(as_of_date=date(2026, 2, 8))

    call_args = mock_db.fetchdf.call_args
    assert "as_of_date = ?" in call_args[0][0]


def test_get_executed_trades(api_with_mock_db, mock_db):
    """Test getting executed trades."""
    mock_db.fetchdf.return_value = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["buy", "buy", "sell"],
    })

    trades = api_with_mock_db.get_executed_trades(limit=5)

    assert len(trades) == 3


def test_get_executed_trades_with_filters(api_with_mock_db, mock_db):
    """Test getting trades with filters."""
    api_with_mock_db.get_executed_trades(
        symbol="AAPL",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 8),
    )

    call_args = mock_db.fetchdf.call_args
    query = call_args[0][0]
    assert "symbol = ?" in query
    assert "filled_at >= ?" in query
    assert "filled_at <= ?" in query


def test_record_trade(api_with_mock_db, mock_db):
    """Test recording executed trade."""
    from hrp.execution.orders import Order, OrderSide, OrderType

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    trade_id = api_with_mock_db.record_trade(
        order=order,
        filled_price=Decimal("150.00"),
        commission=Decimal("1.00"),
    )

    assert trade_id is not None
    mock_db.execute.assert_called_once()
    mock_db.commit.assert_called_once()


def test_get_portfolio_value(api_with_mock_db, mock_db):
    """Test getting portfolio value."""
    mock_db.fetchone.return_value = (100000.0,)

    value = api_with_mock_db.get_portfolio_value()

    assert value == Decimal("100000.0")


def test_get_portfolio_value_empty(api_with_mock_db, mock_db):
    """Test portfolio value when no positions."""
    mock_db.fetchone.return_value = (None,)

    value = api_with_mock_db.get_portfolio_value()

    assert value == Decimal("0")
