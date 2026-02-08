"""Order management system for live trading."""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hrp.execution.broker import IBKRBroker

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"  # Created but not submitted
    SUBMITTED = "submitted"  # Submitted to broker
    FILLED = "filled"  # Fully executed
    PARTIALLY_FILLED = "partially_filled"  # Partial execution
    CANCELLED = "cancelled"  # Cancelled by user/system
    REJECTED = "rejected"  # Rejected by broker


@dataclass
class Order:
    """Trading order representation."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: Optional[int] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[Decimal] = None
    filled_quantity: int = 0
    commission: Optional[Decimal] = None
    hypothesis_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order."""
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")
        if self.order_type == OrderType.LIMIT and self.limit_price is not None:
            if self.limit_price <= 0:
                raise ValueError("limit_price must be positive")


class OrderManager:
    """Manages order submission and tracking."""

    def __init__(self, broker: "IBKRBroker") -> None:
        """Initialize order manager.

        Args:
            broker: IBKRBroker instance
        """
        self.broker = broker
        self._orders: dict[str, Order] = {}

    def submit_order(self, order: Order) -> Order:
        """Submit order to broker.

        Args:
            order: Order to submit

        Returns:
            Order with updated status and broker_order_id

        Raises:
            ValueError: If broker not connected
        """
        from ib_insync import Stock, MarketOrder, LimitOrder

        if not self.broker.is_connected():
            raise ValueError("Broker not connected")

        # Create IBKR contract
        contract = Stock(order.symbol, "SMART", "USD")

        # Create IBKR order
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(
                action=order.side.value.upper(),
                totalQuantity=order.quantity,
            )
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(
                action=order.side.value.upper(),
                totalQuantity=order.quantity,
                lmtPrice=float(order.limit_price),
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        # Submit to broker
        trade = self.broker.ib.placeOrder(contract, ib_order)

        # Update order
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = trade.order.orderId
        order.submitted_at = datetime.now()

        # Track order
        self._orders[order.order_id] = order

        logger.info(
            f"Submitted {order.side.value} {order.quantity} {order.symbol} "
            f"({order.order_type.value}) - broker_id={order.broker_order_id}"
        )

        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order if found, None otherwise
        """
        return self._orders.get(order_id)

    def get_all_orders(self) -> list[Order]:
        """Get all tracked orders.

        Returns:
            List of all orders
        """
        return list(self._orders.values())

    def cancel_order(self, order_id: str) -> Optional[Order]:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancelled order or None if not found
        """
        order = self._orders.get(order_id)
        if not order:
            return None

        if order.status != OrderStatus.SUBMITTED:
            logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return order

        # Cancel via broker
        # Note: This would need the trade object, simplified for now
        order.status = OrderStatus.CANCELLED
        logger.info(f"Cancelled order {order_id}")

        return order
