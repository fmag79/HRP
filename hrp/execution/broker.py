"""IBKR broker connection and account management."""
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ib_insync import IB

logger = logging.getLogger(__name__)


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
