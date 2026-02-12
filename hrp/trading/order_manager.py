"""Order lifecycle management for Robinhood trading.

This module provides high-level order management including order submission,
status tracking, cancellation, and order history. It integrates with
the RobinhoodClient for API communication.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration.

    Maps Robinhood status codes to internal status.
    """

    PENDING = "pending"  # Created but not submitted
    SUBMITTED = "submitted"  # Submitted to broker
    CONFIRMED = "confirmed"  # Confirmed by broker
    FILLED = "filled"  # Fully executed
    PARTIALLY_FILLED = "partially_filled"  # Partial execution
    CANCELLED = "cancelled"  # Cancelled by user/system
    REJECTED = "rejected"  # Rejected by broker/risk checks
    FAILED = "failed"  # Failed due to error
    UNKNOWN = "unknown"  # Status unknown


# Robinhood status mapping to internal status
RH_STATUS_MAP: Dict[str, OrderStatus] = {
    "queued": OrderStatus.PENDING,
    "unconfirmed": OrderStatus.SUBMITTED,
    "confirmed": OrderStatus.CONFIRMED,
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "failed": OrderStatus.FAILED,
}


@dataclass
class Order:
    """Trading order representation.

    This is the internal order representation used by HRP,
    independent of broker-specific data structures.
    """

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[Decimal] = None
    filled_quantity: int = 0
    commission: Optional[Decimal] = None
    rejection_reason: Optional[str] = None
    hypothesis_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require stop_price")

        if self.order_type == OrderType.STOP_LIMIT and (
            self.limit_price is None or self.stop_price is None
        ):
            raise ValueError("Stop-limit orders require both limit_price and stop_price")

        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive")

        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary.

        Returns:
            Order as dictionary
        """
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": float(self.limit_price) if self.limit_price else None,
            "stop_price": float(self.stop_price) if self.stop_price else None,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat()
            if self.submitted_at
            else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": float(self.filled_price) if self.filled_price else None,
            "filled_quantity": self.filled_quantity,
            "commission": float(self.commission) if self.commission else None,
            "rejection_reason": self.rejection_reason,
            "hypothesis_id": self.hypothesis_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """Create Order from dictionary.

        Args:
            data: Order dictionary

        Returns:
            Order instance
        """
        return cls(
            order_id=data["order_id"],
            broker_order_id=data.get("broker_order_id"),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=data["quantity"],
            limit_price=Decimal(str(data["limit_price"]))
            if data.get("limit_price")
            else None,
            stop_price=Decimal(str(data["stop_price"]))
            if data.get("stop_price")
            else None,
            status=OrderStatus(data["status"]),
            submitted_at=datetime.fromisoformat(data["submitted_at"])
            if data.get("submitted_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            filled_at=datetime.fromisoformat(data["filled_at"])
            if data.get("filled_at")
            else None,
            filled_price=Decimal(str(data["filled_price"]))
            if data.get("filled_price")
            else None,
            filled_quantity=data.get("filled_quantity", 0),
            commission=Decimal(str(data["commission"]))
            if data.get("commission")
            else None,
            rejection_reason=data.get("rejection_reason"),
            hypothesis_id=data.get("hypothesis_id"),
        )


@dataclass
class OrderSubmission:
    """Result of order submission."""

    success: bool
    order: Optional[Order] = None
    error_message: Optional[str] = None


class OrderManager:
    """Manages order submission, tracking, and lifecycle.

    This provides the main interface for order operations in HRP.
    """

    def __init__(self, robinhood_client) -> None:
        """Initialize order manager.

        Args:
            robinhood_client: RobinhoodClient instance
        """
        self.client = robinhood_client
        self._orders: Dict[str, Order] = {}

    def submit_order(self, order: Order) -> OrderSubmission:
        """Submit order to Robinhood.

        Args:
            order: Order to submit

        Returns:
            OrderSubmission with result

        Raises:
            ValueError: If order is invalid
        """
        # Validate order
        try:
            # This will raise ValueError if invalid
            Order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )
        except ValueError as e:
            logger.error(f"Invalid order: {e}")
            return OrderSubmission(
                success=False, error_message=f"Invalid order: {e}"
            )

        # Update status to submitted
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        order.updated_at = datetime.now()

        # Submit to Robinhood
        try:
            response = self.client.place_order(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.side.value,
                order_type=order.order_type.value,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )

            # Store broker order ID
            order.broker_order_id = response.get("id")
            order.status = OrderStatus.CONFIRMED
            order.updated_at = datetime.now()

            # Store order
            self._orders[order.order_id] = order

            logger.info(
                f"Order submitted: {order.order_id} -> {order.broker_order_id}"
            )

            return OrderSubmission(success=True, order=order)

        except Exception as e:
            logger.error(f"Order submission failed: {e}")

            # Update order status
            order.status = OrderStatus.FAILED
            order.updated_at = datetime.now()

            return OrderSubmission(
                success=False, error_message=f"Submission failed: {e}"
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Internal order ID

        Returns:
            True if cancellation succeeded, False otherwise

        Raises:
            ValueError: If order not found
        """
        order = self._orders.get(order_id)

        if order is None:
            raise ValueError(f"Order {order_id} not found")

        # Check if order can be cancelled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            logger.warning(
                f"Order {order_id} cannot be cancelled (status={order.status})"
            )
            return False

        try:
            # Cancel via broker
            if order.broker_order_id:
                self.client.cancel_order(order.broker_order_id)

            # Update order status
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()

            logger.info(f"Order cancelled: {order_id}")

            return True

        except Exception as e:
            logger.error(f"Cancellation failed for order {order_id}: {e}")
            return False

    def update_order_status(self, order_id: str) -> Order:
        """Update order status from Robinhood.

        Args:
            order_id: Internal order ID

        Returns:
            Updated Order

        Raises:
            ValueError: If order not found
        """
        order = self._orders.get(order_id)

        if order is None:
            raise ValueError(f"Order {order_id} not found")

        # If no broker order ID, can't update
        if not order.broker_order_id:
            return order

        try:
            # Fetch latest status from broker
            broker_order = self.client.get_order(order.broker_order_id)

            # Map Robinhood status to internal status
            rh_state = broker_order.get("state")
            new_status = RH_STATUS_MAP.get(rh_state, OrderStatus.UNKNOWN)

            # Update if status changed
            if new_status != order.status:
                logger.info(
                    f"Order {order_id} status: {order.status.value} -> {new_status.value}"
                )
                order.status = new_status
                order.updated_at = datetime.now()

                # If filled, update fill details
                if new_status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    order.filled_at = datetime.now()
                    order.filled_price = Decimal(
                        str(broker_order.get("average_price", 0))
                    )
                    order.filled_quantity = broker_order.get("filled_quantity", 0)
                    order.commission = Decimal(
                        str(broker_order.get("total_commission", 0))
                    )

                # If rejected, store reason
                if new_status == OrderStatus.REJECTED:
                    order.rejection_reason = broker_order.get("cancel_reason")

            return order

        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.

        Args:
            order_id: Internal order ID

        Returns:
            Order if found, None otherwise
        """
        return self._orders.get(order_id)

    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> list[Order]:
        """Get filtered orders.

        Args:
            symbol: Filter by symbol
            status: Filter by status

        Returns:
            List of matching orders
        """
        orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]

        if status:
            orders = [o for o in orders if o.status == status]

        # Sort by submitted_at descending
        orders.sort(key=lambda o: o.submitted_at or datetime.min, reverse=True)

        return orders

    def sync_orders(self) -> int:
        """Sync all open orders from Robinhood.

        Updates status of all orders in tracking.

        Returns:
            Number of orders updated
        """
        updated = 0

        for order_id, order in self._orders.items():
            # Only sync open orders
            if order.status in [
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.CONFIRMED,
                OrderStatus.PARTIALLY_FILLED,
            ]:
                try:
                    self.update_order_status(order_id)
                    updated += 1
                except Exception as e:
                    logger.error(f"Failed to sync order {order_id}: {e}")

        logger.info(f"Synced {updated} orders")
        return updated

    def load_order_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> list[Order]:
        """Load order history from Robinhood.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of orders to load

        Returns:
            List of orders
        """
        try:
            # Fetch from Robinhood
            rh_orders = self.client.get_orders(symbol=symbol, limit=limit)

            # Convert to internal Order format
            orders = []
            for rh_order in rh_orders:
                order = self._rh_order_to_order(rh_order)
                if order:
                    # Update tracking
                    if order.order_id not in self._orders:
                        self._orders[order.order_id] = order
                    else:
                        # Update existing order
                        self._orders[order.order_id].status = order.status
                        self._orders[order.order_id].updated_at = order.updated_at
                        if order.filled_at:
                            self._orders[order.order_id].filled_at = order.filled_at
                            self._orders[order.order_id].filled_price = (
                                order.filled_price
                            )
                            self._orders[order.order_id].filled_quantity = (
                                order.filled_quantity
                            )

                    orders.append(order)

            logger.info(f"Loaded {len(orders)} orders from history")
            return orders

        except Exception as e:
            logger.error(f"Failed to load order history: {e}")
            return []

    def _rh_order_to_order(self, rh_order: Dict[str, Any]) -> Optional[Order]:
        """Convert Robinhood order to internal Order.

        Args:
            rh_order: Robinhood order dictionary

        Returns:
            Order instance or None if conversion fails
        """
        try:
            # Extract relevant fields
            symbol = rh_order.get("instrument", {}).get("symbol")
            if not symbol:
                return None

            # Map Robinhood order type
            rh_order_type = rh_order.get("type")
            if rh_order_type == "market":
                order_type = OrderType.MARKET
            elif rh_order_type == "limit":
                order_type = OrderType.LIMIT
            elif rh_order_type == "stop":
                order_type = OrderType.STOP
            elif rh_order_type == "stop_limit":
                order_type = OrderType.STOP_LIMIT
            else:
                order_type = OrderType.MARKET

            # Map Robinhood side
            side_str = rh_order.get("side")
            side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

            # Map status
            rh_state = rh_order.get("state")
            status = RH_STATUS_MAP.get(rh_state, OrderStatus.UNKNOWN)

            # Create order
            order = Order(
                order_id=rh_order.get("id", str(uuid.uuid4())),
                broker_order_id=rh_order.get("id"),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=int(float(rh_order.get("quantity", 0))),
                limit_price=Decimal(str(rh_order.get("price", 0))) if rh_order.get("price") else None,
                stop_price=Decimal(str(rh_order.get("stop_price", 0))) if rh_order.get("stop_price") else None,
                status=status,
                submitted_at=datetime.fromisoformat(
                    rh_order.get("created_at").replace("Z", "+00:00")
                )
                if rh_order.get("created_at")
                else None,
                updated_at=datetime.fromisoformat(
                    rh_order.get("updated_at").replace("Z", "+00:00")
                )
                if rh_order.get("updated_at")
                else None,
                filled_at=datetime.fromisoformat(
                    rh_order.get("last_transaction_at", "").replace("Z", "+00:00")
                )
                if rh_order.get("last_transaction_at")
                else None,
                filled_price=Decimal(str(rh_order.get("average_price", 0))) if rh_order.get("average_price") else None,
                filled_quantity=int(float(rh_order.get("filled_quantity", 0))),
                commission=Decimal(str(rh_order.get("total_commission", 0))),
                rejection_reason=rh_order.get("cancel_reason"),
            )

            return order

        except Exception as e:
            logger.error(f"Failed to convert RH order: {e}")
            return None
