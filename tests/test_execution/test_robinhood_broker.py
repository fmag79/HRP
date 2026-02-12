"""Tests for Robinhood broker."""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from hrp.execution.orders import Order, OrderSide, OrderStatus, OrderType
from hrp.execution.robinhood_broker import RobinhoodBroker, RobinhoodConfig


class TestRobinhoodConfig:
    """Tests for RobinhoodConfig."""

    def test_default_config(self):
        """Test default broker configuration."""
        config = RobinhoodConfig(username="test@example.com", password="test_password")

        assert config.username == "test@example.com"
        assert config.password == "test_password"
        assert config.totp_secret is None
        assert config.account_number is None
        assert config.paper_trading is True  # Safety default
        assert config.rate_limit is None

    def test_custom_config(self):
        """Test custom broker configuration."""
        config = RobinhoodConfig(
            username="test@example.com",
            password="test_password",
            totp_secret="ABC123",
            account_number="12345",
            paper_trading=False,
        )

        assert config.totp_secret == "ABC123"
        assert config.account_number == "12345"
        assert config.paper_trading is False


@patch("hrp.execution.robinhood_broker.RobinhoodSession")
@patch("hrp.execution.robinhood_broker.RateLimiter")
class TestRobinhoodBroker:
    """Tests for RobinhoodBroker."""

    def test_initialization(self, mock_rate_limiter_cls, mock_session_cls):
        """Test broker initialization."""
        config = RobinhoodConfig(username="test@example.com", password="test_password")
        broker = RobinhoodBroker(config)

        assert broker.config == config
        assert not broker._connected
        mock_rate_limiter_cls.assert_called_once()
        mock_session_cls.assert_called_once()

    def test_connect_success(self, mock_rate_limiter_cls, mock_session_cls):
        """Test successful connection."""
        mock_session = MagicMock()
        mock_session.login.return_value = True
        mock_session_cls.return_value = mock_session

        config = RobinhoodConfig(username="test@example.com", password="test_password")
        broker = RobinhoodBroker(config)

        broker.connect()

        assert broker._connected
        mock_session.login.assert_called_once()

    def test_connect_failure(self, mock_rate_limiter_cls, mock_session_cls):
        """Test connection failure."""
        mock_session = MagicMock()
        mock_session.login.return_value = False
        mock_session_cls.return_value = mock_session

        config = RobinhoodConfig(username="test@example.com", password="test_password")
        broker = RobinhoodBroker(config)

        with pytest.raises(ConnectionError, match="authentication failed"):
            broker.connect()

        assert not broker._connected

    def test_disconnect(self, mock_rate_limiter_cls, mock_session_cls):
        """Test disconnect."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        config = RobinhoodConfig(username="test@example.com", password="test_password")
        broker = RobinhoodBroker(config)
        broker._connected = True

        broker.disconnect()

        assert not broker._connected
        mock_session.logout.assert_called_once()

    def test_is_connected(self, mock_rate_limiter_cls, mock_session_cls):
        """Test connection status check."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        config = RobinhoodConfig(username="test@example.com", password="test_password")
        broker = RobinhoodBroker(config)

        broker._connected = False
        assert not broker.is_connected()

        broker._connected = True
        assert broker.is_connected()

    def test_place_order_paper_trading(self, mock_rate_limiter_cls, mock_session_cls):
        """Test placing order in paper trading mode."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=True
        )
        broker = RobinhoodBroker(config)
        broker._connected = True

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
        )

        result = broker.place_order(order)

        # Paper trading: no actual API call
        assert result.status == OrderStatus.SUBMITTED
        assert result.broker_order_id is not None
        assert result.broker_order_id.startswith("PAPER-")
        mock_limiter.acquire.assert_called_once_with(is_order=True)

    def test_place_order_market(self, mock_rate_limiter_cls, mock_session_cls):
        """Test placing market order."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.order_buy_market.return_value = {
            "id": "order-123",
            "state": "confirmed",
        }

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
        )

        result = broker.place_order(order)

        assert result.status == OrderStatus.SUBMITTED
        assert result.broker_order_id == "order-123"
        mock_rh.order_buy_market.assert_called_once_with("AAPL", 10)
        mock_limiter.reset.assert_called_once()

    def test_place_order_limit(self, mock_rate_limiter_cls, mock_session_cls):
        """Test placing limit order."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.order_sell_limit.return_value = {
            "id": "order-456",
            "state": "confirmed",
        }

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
        )

        result = broker.place_order(order)

        assert result.status == OrderStatus.SUBMITTED
        assert result.broker_order_id == "order-456"
        mock_rh.order_sell_limit.assert_called_once_with("AAPL", 10, 150.0)

    def test_place_order_stop_loss(self, mock_rate_limiter_cls, mock_session_cls):
        """Test placing stop-loss order."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.order_sell_stop_loss.return_value = {
            "id": "order-789",
            "state": "confirmed",
        }

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.STOP_LOSS,
            stop_price=Decimal("140.00"),
        )

        result = broker.place_order(order)

        assert result.status == OrderStatus.SUBMITTED
        mock_rh.order_sell_stop_loss.assert_called_once_with("AAPL", 10, 140.0)

    def test_place_order_stop_limit(self, mock_rate_limiter_cls, mock_session_cls):
        """Test placing stop-limit order."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.order_buy_stop_limit.return_value = {
            "id": "order-sl1",
            "state": "confirmed",
        }

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.STOP_LIMIT,
            stop_price=Decimal("145.00"),
            limit_price=Decimal("150.00"),
        )

        result = broker.place_order(order)

        assert result.status == OrderStatus.SUBMITTED
        mock_rh.order_buy_stop_limit.assert_called_once_with("AAPL", 10, 150.0, 145.0)

    def test_place_order_trailing_stop(self, mock_rate_limiter_cls, mock_session_cls):
        """Test placing trailing stop order."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.order_sell_trailing_stop.return_value = {
            "id": "order-trail1",
            "state": "confirmed",
        }

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.TRAILING_STOP,
            trail_amount=5.0,
            trail_type="percentage",
        )

        result = broker.place_order(order)

        assert result.status == OrderStatus.SUBMITTED
        mock_rh.order_sell_trailing_stop.assert_called_once_with(
            "AAPL", 10, 5.0, "percentage"
        )

    def test_place_order_none_response(self, mock_rate_limiter_cls, mock_session_cls):
        """Test handling None response (silent failure)."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.order_buy_market.return_value = None  # Silent failure

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
        )

        result = broker.place_order(order)

        assert result.status == OrderStatus.REJECTED

    def test_place_order_error_response(self, mock_rate_limiter_cls, mock_session_cls):
        """Test handling error response."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.order_buy_market.return_value = {"detail": "Insufficient buying power"}

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
        )

        result = broker.place_order(order)

        assert result.status == OrderStatus.REJECTED

    def test_place_order_not_connected(self, mock_rate_limiter_cls, mock_session_cls):
        """Test placing order when not connected."""
        config = RobinhoodConfig(username="test@example.com", password="test_password")
        broker = RobinhoodBroker(config)

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
        )

        with pytest.raises(ValueError, match="not connected"):
            broker.place_order(order)

    def test_cancel_order_paper_trading(self, mock_rate_limiter_cls, mock_session_cls):
        """Test cancelling order in paper trading."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=True
        )
        broker = RobinhoodBroker(config)
        broker._connected = True

        result = broker.cancel_order("PAPER-123")

        assert result is True

    def test_cancel_order_success(self, mock_rate_limiter_cls, mock_session_cls):
        """Test cancelling order successfully."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.cancel_stock_order.return_value = {"status": "cancelled"}

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        result = broker.cancel_order("order-123")

        assert result is True
        mock_rh.cancel_stock_order.assert_called_once_with("order-123")

    def test_get_order_status(self, mock_rate_limiter_cls, mock_session_cls):
        """Test getting order status."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.get_stock_order_info.return_value = {"state": "filled"}

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        status = broker.get_order_status("order-123")

        assert status == OrderStatus.FILLED
        mock_rh.get_stock_order_info.assert_called_once_with("order-123")

    def test_get_positions(self, mock_rate_limiter_cls, mock_session_cls):
        """Test getting positions."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.build_holdings.return_value = {
            "AAPL": {
                "quantity": "10.0",
                "average_buy_price": "150.00",
                "equity": "1500.00",
            }
        }

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        positions = broker.get_positions()

        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["quantity"] == 10.0
        assert positions[0]["avg_price"] == Decimal("150.00")

    def test_get_portfolio_value(self, mock_rate_limiter_cls, mock_session_cls):
        """Test getting portfolio value."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.load_portfolio_profile.return_value = {"equity": "50000.00"}

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        value = broker.get_portfolio_value()

        assert value == Decimal("50000.00")

    def test_get_quote(self, mock_rate_limiter_cls, mock_session_cls):
        """Test getting quote."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.get_latest_price.return_value = ["152.50"]

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        price = broker.get_quote("AAPL")

        assert price == Decimal("152.50")
        mock_rh.get_latest_price.assert_called_once_with("AAPL")

    def test_get_quotes_multiple(self, mock_rate_limiter_cls, mock_session_cls):
        """Test getting multiple quotes."""
        mock_session = MagicMock()
        mock_session.is_authenticated.return_value = True
        mock_session_cls.return_value = mock_session

        mock_limiter = MagicMock()
        mock_rate_limiter_cls.return_value = mock_limiter

        mock_rh = MagicMock()
        mock_rh.get_latest_price.return_value = ["152.50", "275.30"]

        config = RobinhoodConfig(
            username="test@example.com", password="test_password", paper_trading=False
        )
        broker = RobinhoodBroker(config)
        broker._connected = True
        broker._rh = mock_rh

        quotes = broker.get_quotes(["AAPL", "MSFT"])

        assert quotes["AAPL"] == Decimal("152.50")
        assert quotes["MSFT"] == Decimal("275.30")

    def test_context_manager(self, mock_rate_limiter_cls, mock_session_cls):
        """Test context manager usage."""
        mock_session = MagicMock()
        mock_session.login.return_value = True
        mock_session_cls.return_value = mock_session

        config = RobinhoodConfig(username="test@example.com", password="test_password")

        with RobinhoodBroker(config) as broker:
            assert broker._connected

        mock_session.logout.assert_called_once()

    def test_map_robinhood_state(self, mock_rate_limiter_cls, mock_session_cls):
        """Test state mapping."""
        config = RobinhoodConfig(username="test@example.com", password="test_password")
        broker = RobinhoodBroker(config)

        assert broker._map_robinhood_state("queued") == OrderStatus.SUBMITTED
        assert broker._map_robinhood_state("confirmed") == OrderStatus.SUBMITTED
        assert broker._map_robinhood_state("partially_filled") == OrderStatus.PARTIALLY_FILLED
        assert broker._map_robinhood_state("filled") == OrderStatus.FILLED
        assert broker._map_robinhood_state("cancelled") == OrderStatus.CANCELLED
        assert broker._map_robinhood_state("rejected") == OrderStatus.REJECTED
        assert broker._map_robinhood_state("failed") == OrderStatus.REJECTED
        assert broker._map_robinhood_state(None) == OrderStatus.REJECTED
        assert broker._map_robinhood_state("unknown_state") == OrderStatus.PENDING
