"""Position tracking and synchronization."""
import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrp.api.platform import PlatformAPI
    from hrp.execution.broker import IBKRBroker

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position representation."""

    symbol: str
    quantity: int
    entry_price: Decimal
    current_price: Decimal
    as_of_date: date
    hypothesis_id: str | None = None
    commission_paid: Decimal = Decimal("0.00")

    @property
    def market_value(self) -> Decimal:
        """Current market value of position."""
        return Decimal(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        """Cost basis including commissions."""
        return Decimal(self.quantity) * self.entry_price + self.commission_paid

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return float(self.unrealized_pnl / self.cost_basis)

    def update_price(self, new_price: Decimal) -> None:
        """Update current price.

        Args:
            new_price: New market price
        """
        self.current_price = new_price
        logger.debug(f"{self.symbol}: price updated to {new_price}, PnL={self.unrealized_pnl}")


class PositionTracker:
    """Tracks and synchronizes positions with broker."""

    def __init__(self, broker: "IBKRBroker", api: "PlatformAPI") -> None:
        """Initialize position tracker.

        Args:
            broker: IBKRBroker instance
            api: PlatformAPI instance
        """
        self.broker = broker
        self.api = api
        self._positions: dict[str, Position] = {}

    def sync_positions(self) -> list[Position]:
        """Sync positions from broker.

        Returns:
            List of current positions

        Raises:
            ValueError: If broker not connected
        """
        if not self.broker.is_connected():
            raise ValueError("Broker not connected")

        # Get positions from IBKR
        ib_positions = self.broker.ib.positions()

        positions = []
        for ib_pos in ib_positions:
            if ib_pos.account != self.broker.config.account:
                continue

            symbol = ib_pos.contract.symbol
            quantity = int(ib_pos.position)
            avg_cost = Decimal(str(ib_pos.avgCost))

            # Get current market price
            ticker = self.broker.ib.reqMktData(ib_pos.contract)
            self.broker.ib.sleep(1)  # Wait for price update
            current_price = Decimal(str(ticker.last)) if ticker.last else avg_cost

            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=avg_cost,
                current_price=current_price,
                as_of_date=date.today(),
            )

            positions.append(position)
            self._positions[symbol] = position

        logger.info(f"Synced {len(positions)} positions from broker")
        return positions

    def get_position(self, symbol: str) -> Position | None:
        """Get position by symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position if found, None otherwise
        """
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[Position]:
        """Get all tracked positions.

        Returns:
            List of all positions
        """
        return list(self._positions.values())

    def persist_positions(self) -> None:
        """Persist positions to database."""
        from hrp.data.db import get_db

        conn = get_db()

        for position in self._positions.values():
            conn.execute(
                """
                INSERT OR REPLACE INTO live_positions (
                    symbol, quantity, entry_price, current_price,
                    market_value, cost_basis, unrealized_pnl,
                    unrealized_pnl_pct, hypothesis_id, as_of_date, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.symbol,
                    position.quantity,
                    float(position.entry_price),
                    float(position.current_price),
                    float(position.market_value),
                    float(position.cost_basis),
                    float(position.unrealized_pnl),
                    position.unrealized_pnl_pct,
                    position.hypothesis_id,
                    position.as_of_date,
                    datetime.now(),
                ),
            )

        conn.commit()
        logger.info(f"Persisted {len(self._positions)} positions to database")

    def calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value.

        Returns:
            Total market value of all positions
        """
        return sum(
            (p.market_value for p in self._positions.values()),
            start=Decimal("0.00"),
        )

    def calculate_total_pnl(self) -> Decimal:
        """Calculate total unrealized P&L.

        Returns:
            Total unrealized profit/loss
        """
        return sum(
            (p.unrealized_pnl for p in self._positions.values()),
            start=Decimal("0.00"),
        )
