"""Convert ML predictions/signals to trading orders."""
import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

import pandas as pd

from hrp.execution.orders import Order, OrderSide, OrderType
from hrp.execution.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for signal-to-order conversion."""

    portfolio_value: Decimal
    max_positions: int = 20
    max_position_pct: float = 0.10  # 10% max per position
    min_order_value: Decimal = Decimal("100.00")  # Minimum order size

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_position_pct <= 0 or self.max_position_pct > 1:
            raise ValueError("max_position_pct must be between 0 and 1")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")


class SignalConverter:
    """Converts ML signals to trading orders with risk limits."""

    def __init__(
        self,
        config: ConversionConfig,
        position_sizer: PositionSizer | None = None,
    ) -> None:
        """Initialize signal converter.

        Args:
            config: Conversion configuration
            position_sizer: Optional VaR-aware position sizer for risk-based sizing
        """
        self.config = config
        self.position_sizer = position_sizer

        if position_sizer:
            logger.info("SignalConverter initialized with VaR-aware position sizing")
        else:
            logger.info("SignalConverter initialized with equal-weight sizing")

    def signals_to_orders(
        self,
        signals: pd.DataFrame,
        method: Literal["rank", "threshold", "zscore"] = "rank",
        current_prices: dict[str, Decimal] | None = None,
        use_var_sizing: bool = False,
        as_of_date: date | None = None,
        current_positions: dict[str, int] | None = None,
    ) -> list[Order]:
        """Convert signals to trading orders.

        Args:
            signals: DataFrame with columns [symbol, signal, prediction]
                    signal: 1.0 (long), 0.0 (no position), -1.0 (short, not supported yet)
            method: Signal generation method used
            current_prices: Optional dict of symbol -> current price
                          (fetched from broker if not provided)
            use_var_sizing: If True and position_sizer is available, use VaR-based sizing
            as_of_date: Date for VaR feature lookup (required if use_var_sizing=True)
            current_positions: Current positions for VaR budget calculation

        Returns:
            List of Order objects ready for submission
        """
        # Filter for buy signals only (signal = 1.0)
        buy_signals = signals[signals["signal"] == 1.0].copy()

        if buy_signals.empty:
            logger.info("No buy signals to convert to orders")
            return []

        # Sort by prediction strength (highest first)
        buy_signals = buy_signals.sort_values("prediction", ascending=False)

        # Apply position limit
        buy_signals = buy_signals.head(self.config.max_positions)

        logger.info(
            f"Converting {len(buy_signals)} signals to orders "
            f"(method={method}, max_positions={self.config.max_positions}, "
            f"use_var_sizing={use_var_sizing})"
        )

        # Use VaR-based position sizing if enabled and available
        if use_var_sizing and self.position_sizer and as_of_date:
            if current_prices is None:
                raise ValueError("current_prices required for VaR-based sizing")

            # Use position sizer to determine quantities
            target_positions = self.position_sizer.size_all_positions(
                signals=buy_signals,
                current_positions=current_positions or {},
                current_prices=current_prices,
                as_of_date=as_of_date,
            )

            # Convert to orders
            orders = []
            for symbol, quantity in target_positions.items():
                if quantity > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                    )
                    orders.append(order)

                    price = current_prices[symbol]
                    order_value = Decimal(quantity) * price
                    logger.debug(
                        f"Created VaR-sized order: BUY {quantity} {symbol} @ ${price:.2f} "
                        f"(value=${order_value:,.2f})"
                    )

            logger.info(
                f"Created {len(orders)} VaR-sized orders from {len(buy_signals)} signals"
            )
            return orders

        # Fallback to equal-weight sizing
        # Calculate position size for each signal
        max_position_value = self.config.portfolio_value * Decimal(
            str(self.config.max_position_pct)
        )

        orders = []

        for _, row in buy_signals.iterrows():
            symbol = row["symbol"]

            # Get current price
            if current_prices and symbol in current_prices:
                price = current_prices[symbol]
            else:
                # Default price for testing - in production, would fetch from broker
                price = Decimal("150.00")
                logger.debug(f"Using default price for {symbol}: {price}")

            # Calculate quantity
            quantity = int(max_position_value / price)

            if quantity <= 0:
                logger.debug(f"Skipping {symbol}: calculated quantity is 0")
                continue

            # Skip if order too small
            order_value = Decimal(quantity) * price
            if order_value < self.config.min_order_value:
                logger.debug(
                    f"Skipping {symbol}: order value {order_value} "
                    f"below minimum {self.config.min_order_value}"
                )
                continue

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            orders.append(order)

            logger.debug(
                f"Created order: BUY {quantity} {symbol} @ ~{price} "
                f"(value={order_value:.2f})"
            )

        logger.info(f"Created {len(orders)} orders from {len(buy_signals)} signals")
        return orders

    def rebalance_to_orders(
        self,
        current_positions: dict[str, int],
        target_signals: pd.DataFrame,
        current_prices: dict[str, Decimal],
        use_var_sizing: bool = False,
        as_of_date: date | None = None,
    ) -> list[Order]:
        """Generate orders to rebalance from current to target positions.

        Args:
            current_positions: Dict of symbol -> current quantity
            target_signals: DataFrame with target signals
            current_prices: Dict of symbol -> current price
            use_var_sizing: If True and position_sizer is available, use VaR-based sizing
            as_of_date: Date for VaR feature lookup (required if use_var_sizing=True)

        Returns:
            List of orders (buys and sells) for rebalancing
        """
        orders = []

        # Get target positions from signals
        target_orders = self.signals_to_orders(
            signals=target_signals,
            method="rank",
            current_prices=current_prices,
            use_var_sizing=use_var_sizing,
            as_of_date=as_of_date,
            current_positions=current_positions,
        )
        target_positions = {o.symbol: o.quantity for o in target_orders}

        # Generate sell orders for positions to exit
        for symbol, current_qty in current_positions.items():
            if symbol not in target_positions:
                # Exit position
                orders.append(
                    Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=current_qty,
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug(f"Rebalance: SELL {current_qty} {symbol} (exit)")

        # Generate buy orders for new positions
        for symbol, target_qty in target_positions.items():
            current_qty = current_positions.get(symbol, 0)

            if target_qty > current_qty:
                # Increase position
                orders.append(
                    Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=target_qty - current_qty,
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug(
                    f"Rebalance: BUY {target_qty - current_qty} {symbol} "
                    f"(increase from {current_qty})"
                )
            elif target_qty < current_qty:
                # Decrease position
                orders.append(
                    Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=current_qty - target_qty,
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug(
                    f"Rebalance: SELL {current_qty - target_qty} {symbol} "
                    f"(decrease from {current_qty})"
                )

        logger.info(f"Generated {len(orders)} rebalancing orders")
        return orders
