"""Robinhood broker implementation."""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from hrp.execution.orders import Order, OrderStatus, OrderType
from hrp.execution.rate_limiter import RateLimitConfig, RateLimiter
from hrp.execution.robinhood_auth import RobinhoodAuthConfig, RobinhoodSession

logger = logging.getLogger(__name__)


@dataclass
class RobinhoodConfig:
    """Robinhood broker configuration.

    Security notes:
        - password and totp_secret should come from environment variables
        - paper_trading defaults to True for safety
        - Set ROBINHOOD_PAPER_TRADING=false to enable live trading
    """

    username: str
    password: str
    totp_secret: str | None = None
    account_number: str | None = None  # Multi-account support
    paper_trading: bool = True  # Safety flag (logs but blocks real orders)
    rate_limit: RateLimitConfig | None = None


class RobinhoodBroker:
    """Robinhood broker implementation.

    Implements BaseBroker protocol for Robinhood API using robin_stocks library.
    Supports all order types: market, limit, stop-loss, stop-limit, trailing stop.

    Example:
        >>> config = RobinhoodConfig(
        ...     username=os.getenv("ROBINHOOD_USERNAME"),
        ...     password=os.getenv("ROBINHOOD_PASSWORD"),
        ...     totp_secret=os.getenv("ROBINHOOD_TOTP_SECRET"),
        ...     paper_trading=True,
        ... )
        >>> broker = RobinhoodBroker(config)
        >>> broker.connect()
        >>> order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
        >>> filled_order = broker.place_order(order)
        >>> broker.disconnect()
    """

    def __init__(self, config: RobinhoodConfig) -> None:
        """Initialize Robinhood broker.

        Args:
            config: Broker configuration.

        Raises:
            ImportError: If robin_stocks not installed.
        """
        self.config = config
        self._connected = False

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(config.rate_limit)

        # Initialize auth session
        auth_config = RobinhoodAuthConfig(
            username=config.username,
            password=config.password,
            totp_secret=config.totp_secret,
        )
        self.session = RobinhoodSession(auth_config)

        # Import robin_stocks
        try:
            import robin_stocks.robinhood as rh

            self._rh = rh
        except ImportError:
            raise ImportError(
                "robin_stocks not installed. Install with: pip install robin-stocks"
            )

        if config.paper_trading:
            logger.warning(
                "⚠️  PAPER TRADING MODE ENABLED - Orders will be logged but NOT executed"
            )

    def connect(self) -> None:
        """Connect to Robinhood API.

        Authenticates with Robinhood and validates session.

        Raises:
            ConnectionError: If authentication fails.
        """
        try:
            logger.info("Connecting to Robinhood API...")
            if not self.session.login():
                raise ConnectionError("Robinhood authentication failed")

            self._connected = True
            logger.info("✅ Connected to Robinhood (paper_trading=%s)", self.config.paper_trading)

        except Exception as e:
            logger.error("Failed to connect to Robinhood: %s", e)
            raise ConnectionError(f"Robinhood connection failed: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Robinhood API."""
        if self._connected:
            self.session.logout()
            self._connected = False
            logger.info("Disconnected from Robinhood")

    def is_connected(self) -> bool:
        """Check if connected to Robinhood.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected and self.session.is_authenticated()

    def place_order(self, order: Order) -> Order:
        """Place an order with Robinhood.

        Args:
            order: Order to place.

        Returns:
            Order with updated status and broker_order_id.

        Raises:
            ValueError: If broker not connected or order invalid.
            RuntimeError: If order submission fails.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        # Validate order
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        # Ensure session valid
        self.session.ensure_authenticated()

        # Apply rate limiting (orders have stricter cooldown)
        self.rate_limiter.acquire(is_order=True)

        # Paper trading: log but don't execute
        if self.config.paper_trading:
            logger.info(
                "📄 PAPER TRADE: %s %d %s @ %s (order_type=%s)",
                order.side.value.upper(),
                order.quantity,
                order.symbol,
                order.limit_price or "MARKET",
                order.order_type.value,
            )
            order.status = OrderStatus.SUBMITTED
            order.broker_order_id = f"PAPER-{uuid.uuid4().hex[:8]}"
            order.submitted_at = datetime.now()
            return order

        # Execute live order
        try:
            response = self._submit_order_to_robinhood(order)

            # Check for None (silent failure)
            if response is None:
                logger.error("Robinhood order returned None (silent failure)")
                order.status = OrderStatus.REJECTED
                return order

            # Check for error response
            if isinstance(response, dict) and "detail" in response:
                error_detail = response["detail"]
                logger.error("Robinhood order rejected: %s", error_detail)
                order.status = OrderStatus.REJECTED
                return order

            # Success - extract order ID and update status
            order.broker_order_id = response.get("id")
            order.status = self._map_robinhood_state(response.get("state"))
            order.submitted_at = datetime.now()

            logger.info(
                "✅ Order submitted: %s %d %s (id=%s, status=%s)",
                order.side.value.upper(),
                order.quantity,
                order.symbol,
                order.broker_order_id,
                order.status.value,
            )

            # Reset rate limiter retry counter on success
            self.rate_limiter.reset()

            return order

        except Exception as e:
            logger.exception("Order submission failed: %s", e)
            order.status = OrderStatus.REJECTED
            raise RuntimeError(f"Order submission failed: {e}") from e

    def _submit_order_to_robinhood(self, order: Order) -> dict | None:
        """Submit order to Robinhood API based on order type.

        Args:
            order: Order to submit.

        Returns:
            Robinhood API response dict or None on failure.
        """
        symbol = order.symbol
        quantity = order.quantity
        side = order.side.value.lower()  # "buy" or "sell"

        # Map order type to robin_stocks function
        if order.order_type == OrderType.MARKET:
            if side == "buy":
                return self._rh.order_buy_market(symbol, quantity)
            else:
                return self._rh.order_sell_market(symbol, quantity)

        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            price = float(order.limit_price)
            if side == "buy":
                return self._rh.order_buy_limit(symbol, quantity, price)
            else:
                return self._rh.order_sell_limit(symbol, quantity, price)

        elif order.order_type == OrderType.STOP_LOSS:
            if order.stop_price is None:
                raise ValueError("STOP_LOSS order requires stop_price")
            stop_price = float(order.stop_price)
            if side == "buy":
                return self._rh.order_buy_stop_loss(symbol, quantity, stop_price)
            else:
                return self._rh.order_sell_stop_loss(symbol, quantity, stop_price)

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                raise ValueError("STOP_LIMIT order requires stop_price and limit_price")
            stop_price = float(order.stop_price)
            limit_price = float(order.limit_price)
            if side == "buy":
                return self._rh.order_buy_stop_limit(
                    symbol, quantity, limit_price, stop_price
                )
            else:
                return self._rh.order_sell_stop_limit(
                    symbol, quantity, limit_price, stop_price
                )

        elif order.order_type == OrderType.TRAILING_STOP:
            if order.trail_amount is None:
                raise ValueError("TRAILING_STOP order requires trail_amount")

            trail_amount = order.trail_amount
            trail_type = order.trail_type  # "percentage" or "amount"

            if side == "buy":
                return self._rh.order_buy_trailing_stop(
                    symbol, quantity, trail_amount, trail_type
                )
            else:
                return self._rh.order_sell_trailing_stop(
                    symbol, quantity, trail_amount, trail_type
                )

        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.

        Args:
            broker_order_id: Robinhood order ID.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        # Paper trading: simulate cancel
        if self.config.paper_trading:
            logger.info("📄 PAPER TRADE: Cancelled order %s", broker_order_id)
            return True

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            response = self._rh.cancel_stock_order(broker_order_id)

            # None response = failure
            if response is None:
                logger.error("Cancel order returned None for %s", broker_order_id)
                return False

            logger.info("✅ Cancelled order %s", broker_order_id)
            return True

        except Exception as e:
            logger.error("Cancel order failed for %s: %s", broker_order_id, e)
            return False

    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get order status from Robinhood.

        Args:
            broker_order_id: Robinhood order ID.

        Returns:
            Current order status.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        # Paper trading: simulate filled
        if self.config.paper_trading:
            return OrderStatus.FILLED

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            response = self._rh.get_stock_order_info(broker_order_id)

            if response is None:
                logger.warning("Order status returned None for %s", broker_order_id)
                return OrderStatus.REJECTED

            state = response.get("state")
            return self._map_robinhood_state(state)

        except Exception as e:
            logger.error("Get order status failed for %s: %s", broker_order_id, e)
            return OrderStatus.REJECTED

    def get_open_orders(self) -> list[dict]:
        """Get all open orders.

        Returns:
            List of open order dicts.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        if self.config.paper_trading:
            return []

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            response = self._rh.get_all_open_stock_orders()
            return response if response is not None else []
        except Exception as e:
            logger.error("Get open orders failed: %s", e)
            return []

    def get_positions(self) -> list[dict]:
        """Get current positions.

        Returns:
            List of position dicts with keys: symbol, quantity, avg_price, market_value.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        if self.config.paper_trading:
            return []

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            positions = self._rh.build_holdings()

            if positions is None:
                logger.warning("Get positions returned None")
                return []

            # Convert to standardized format
            result = []
            for symbol, data in positions.items():
                result.append(
                    {
                        "symbol": symbol,
                        "quantity": float(data.get("quantity", 0)),
                        "avg_price": Decimal(data.get("average_buy_price", "0")),
                        "market_value": Decimal(data.get("equity", "0")),
                    }
                )

            return result

        except Exception as e:
            logger.error("Get positions failed: %s", e)
            return []

    def get_account_info(self) -> dict:
        """Get account information.

        Returns:
            Account info dict.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        if self.config.paper_trading:
            return {"buying_power": "100000.00", "portfolio_value": "100000.00"}

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            account = self._rh.load_account_profile()
            return account if account is not None else {}
        except Exception as e:
            logger.error("Get account info failed: %s", e)
            return {}

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value.

        Returns:
            Total portfolio value in USD.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        if self.config.paper_trading:
            return Decimal("100000.00")

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            profile = self._rh.load_portfolio_profile()

            if profile is None:
                logger.warning("Get portfolio value returned None")
                return Decimal("0.00")

            equity = profile.get("equity", "0")
            return Decimal(equity)

        except Exception as e:
            logger.error("Get portfolio value failed: %s", e)
            return Decimal("0.00")

    def get_quote(self, symbol: str) -> Decimal:
        """Get current market price for symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Current market price.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            quote = self._rh.get_latest_price(symbol)

            if quote is None or not quote:
                logger.warning("Get quote returned None for %s", symbol)
                return Decimal("0.00")

            # get_latest_price returns a list
            price = quote[0] if isinstance(quote, list) else quote
            return Decimal(str(price))

        except Exception as e:
            logger.error("Get quote failed for %s: %s", symbol, e)
            return Decimal("0.00")

    def get_quotes(self, symbols: list[str]) -> dict[str, Decimal]:
        """Get current market prices for multiple symbols.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dict mapping symbol to current market price.
        """
        if not self.is_connected():
            raise ValueError("Broker not connected")

        self.session.ensure_authenticated()
        self.rate_limiter.acquire()

        try:
            quotes = self._rh.get_latest_price(symbols)

            if quotes is None:
                logger.warning("Get quotes returned None")
                return {}

            # Map symbols to prices
            result = {}
            for symbol, price in zip(symbols, quotes):
                result[symbol] = Decimal(str(price)) if price else Decimal("0.00")

            return result

        except Exception as e:
            logger.error("Get quotes failed: %s", e)
            return {}

    def _map_robinhood_state(self, state: str | None) -> OrderStatus:
        """Map Robinhood order state to OrderStatus.

        Args:
            state: Robinhood order state string.

        Returns:
            Mapped OrderStatus.
        """
        if state is None:
            return OrderStatus.REJECTED

        state = state.lower()

        if state in ("queued", "unconfirmed", "confirmed"):
            return OrderStatus.SUBMITTED
        elif state == "partially_filled":
            return OrderStatus.PARTIALLY_FILLED
        elif state == "filled":
            return OrderStatus.FILLED
        elif state == "cancelled":
            return OrderStatus.CANCELLED
        elif state in ("rejected", "failed"):
            return OrderStatus.REJECTED
        else:
            logger.warning("Unknown Robinhood state: %s", state)
            return OrderStatus.PENDING

    def __enter__(self) -> "RobinhoodBroker":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.disconnect()
