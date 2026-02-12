"""Broker connection and account management.

Defines BaseBroker protocol for broker abstraction and IBKRBroker implementation.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hrp.execution.orders import Order, OrderStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseBroker(Protocol):
    """Protocol for broker implementations.

    All brokers (IBKR, Robinhood, etc.) must implement this interface
    for interchangeable usage in trading agents.
    """

    def connect(self) -> None:
        """Connect to broker API.

        Raises:
            ConnectionError: If connection fails.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from broker API."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to broker.

        Returns:
            True if connected, False otherwise.
        """
        ...

    def place_order(self, order: "Order") -> "Order":
        """Place an order with the broker.

        Args:
            order: Order to place.

        Returns:
            Order with updated status and broker_order_id.

        Raises:
            ValueError: If broker not connected or order invalid.
        """
        ...

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.

        Args:
            broker_order_id: Broker's order ID.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        ...

    def get_order_status(self, broker_order_id: str) -> "OrderStatus":
        """Get order status from broker.

        Args:
            broker_order_id: Broker's order ID.

        Returns:
            Current order status.
        """
        ...

    def get_positions(self) -> list[dict]:
        """Get current positions.

        Returns:
            List of position dicts with keys: symbol, quantity, avg_price, market_value.
        """
        ...

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value.

        Returns:
            Total portfolio value in USD.
        """
        ...

    def get_quote(self, symbol: str) -> Decimal:
        """Get current market price for symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Current market price.
        """
        ...

    def get_quotes(self, symbols: list[str]) -> dict[str, Decimal]:
        """Get current market prices for multiple symbols.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dict mapping symbol to current market price.
        """
        ...

    def __enter__(self) -> "BaseBroker":
        """Context manager entry."""
        ...

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        ...


@dataclass
class BrokerConfig:
    """IBKR broker connection configuration."""

    host: str
    port: int
    client_id: int
    account: str
    paper_trading: bool = True
    timeout: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.host:
            raise ValueError("host is required")
        if not self.account:
            raise ValueError("account is required")
        if self.paper_trading and self.port != 7497:
            logger.warning(f"Paper trading typically uses port 7497, got {self.port}")


class IBKRBroker:
    """Interactive Brokers connection manager."""

    def __init__(self, config: BrokerConfig) -> None:
        """Initialize broker with configuration.

        Args:
            config: Broker connection configuration
        """
        from ib_insync import IB

        self.config = config
        self.ib: IB = IB()
        self._connected = False

    def connect(self) -> None:
        """Connect to IBKR TWS/Gateway.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                readonly=False,
                timeout=self.config.timeout,
            )
            self._connected = True
            logger.info(
                f"Connected to IBKR (paper={self.config.paper_trading}, "
                f"account={self.config.account})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise ConnectionError(f"IBKR connection failed: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        """Check if connected to IBKR.

        Returns:
            True if connected, False otherwise
        """
        return self._connected and self.ib.isConnected()

    def __enter__(self) -> "IBKRBroker":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
