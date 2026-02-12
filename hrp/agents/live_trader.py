"""Live trading agent for executing signals."""
import logging
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from hrp.agents.jobs import DataRequirement, IngestionJob
from hrp.api.platform import PlatformAPI
from hrp.execution.broker import BaseBroker, BrokerConfig, IBKRBroker
from hrp.execution.orders import Order, OrderManager, OrderSide, OrderStatus, OrderType
from hrp.execution.position_sizer import PositionSizer, PositionSizingConfig
from hrp.execution.positions import PositionTracker
from hrp.execution.robinhood_broker import RobinhoodBroker, RobinhoodConfig
from hrp.execution.signal_converter import ConversionConfig, SignalConverter

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for live trading."""

    portfolio_value: Decimal
    max_positions: int = 20
    max_position_pct: float = 0.10
    min_order_value: Decimal = Decimal("100.00")
    dry_run: bool = False
    broker_type: str = "robinhood"  # "robinhood" or "ibkr"
    use_var_sizing: bool = True  # Enable VaR-based position sizing
    stop_loss_pct: float | None = None  # Auto-attach stop-loss (e.g., 0.05 = 5%)

    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Load trading config from environment."""
        stop_loss = os.getenv("HRP_AUTO_STOP_LOSS_PCT")
        return cls(
            portfolio_value=Decimal(os.getenv("HRP_PORTFOLIO_VALUE", "100000")),
            max_positions=int(os.getenv("HRP_MAX_POSITIONS", "20")),
            max_position_pct=float(os.getenv("HRP_MAX_POSITION_PCT", "0.10")),
            min_order_value=Decimal(os.getenv("HRP_MIN_ORDER_VALUE", "100")),
            dry_run=os.getenv("HRP_TRADING_DRY_RUN", "true").lower() == "true",
            broker_type=os.getenv("HRP_BROKER_TYPE", "robinhood").lower(),
            use_var_sizing=os.getenv("HRP_USE_VAR_SIZING", "true").lower() == "true",
            stop_loss_pct=float(stop_loss) if stop_loss else None,
        )


class LiveTradingAgent(IngestionJob):
    """Agent for live trading execution.

    This agent:
    1. Gets latest predictions for deployed strategies
    2. Converts predictions to signals
    3. Compares against current positions
    4. Generates rebalancing orders
    5. Submits orders to broker (if not dry-run)

    IMPORTANT: This agent is disabled by default for safety.
    Set HRP_TRADING_DRY_RUN=false to enable actual order submission.
    """

    def __init__(
        self,
        trading_config: TradingConfig | None = None,
        broker_config: BrokerConfig | None = None,
        api: PlatformAPI | None = None,
        job_id: str = "live_trader",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        """Initialize live trading agent.

        Args:
            trading_config: Trading configuration (from env if None)
            broker_config: Broker configuration (from env if None)
            api: PlatformAPI instance (creates new if None)
            job_id: Job identifier
            max_retries: Maximum retry attempts
            retry_backoff: Exponential backoff multiplier
        """
        data_requirements = [
            DataRequirement(
                table="features",
                min_rows=100,
                max_age_days=3,
                date_column="date",
                description="Recent feature data",
            ),
        ]

        super().__init__(
            job_id,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            data_requirements=data_requirements,
        )

        if api is not None:
            self.api = api

        self.trading_config = trading_config or TradingConfig.from_env()
        self.broker_config = broker_config or self._broker_config_from_env()

    def _broker_config_from_env(self) -> BrokerConfig | RobinhoodConfig:
        """Create broker config from environment variables.

        Returns appropriate config based on HRP_BROKER_TYPE env var.
        """
        broker_type = os.getenv("HRP_BROKER_TYPE", "robinhood").lower()

        if broker_type == "robinhood":
            return RobinhoodConfig(
                username=os.getenv("ROBINHOOD_USERNAME", ""),
                password=os.getenv("ROBINHOOD_PASSWORD", ""),
                totp_secret=os.getenv("ROBINHOOD_TOTP_SECRET"),
                account_number=os.getenv("ROBINHOOD_ACCOUNT_NUMBER"),
                paper_trading=os.getenv("ROBINHOOD_PAPER_TRADING", "true").lower() == "true",
            )
        elif broker_type == "ibkr":
            return BrokerConfig(
                host=os.getenv("IBKR_HOST", "127.0.0.1"),
                port=int(os.getenv("IBKR_PORT", "7497")),
                client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
                account=os.getenv("IBKR_ACCOUNT", ""),
                paper_trading=os.getenv("IBKR_PAPER_TRADING", "true").lower() == "true",
            )
        else:
            raise ValueError(f"Unknown broker type: {broker_type}. Use 'robinhood' or 'ibkr'")

    def _create_broker(self) -> BaseBroker:
        """Create broker instance based on config type."""
        if isinstance(self.broker_config, RobinhoodConfig):
            return RobinhoodBroker(self.broker_config)
        elif isinstance(self.broker_config, BrokerConfig):
            return IBKRBroker(self.broker_config)
        else:
            raise ValueError(f"Unknown broker config type: {type(self.broker_config)}")

    def execute(self) -> dict[str, Any]:
        """Execute live trading agent.

        Returns:
            Dict with execution stats:
                - status: success, no_predictions, dry_run, or error
                - orders_generated: Count of orders created
                - orders_submitted: Count of orders sent to broker
                - positions_synced: Current position count
        """
        logger.info(f"Starting live trading agent (dry_run={self.trading_config.dry_run})")

        # Get deployed strategies
        deployed = self.api.get_deployed_strategies()
        if not deployed:
            logger.warning("No deployed strategies found")
            return {
                "status": "no_deployed_strategies",
                "orders_generated": 0,
                "orders_submitted": 0,
                "records_fetched": 0,
                "records_inserted": 0,
            }

        # Get latest predictions
        universe = self.api.get_universe(as_of_date=date.today())
        all_predictions = []

        for strategy in deployed:
            hypothesis_id = strategy.get("hypothesis_id") or getattr(
                strategy, "hypothesis_id", None
            )
            metadata = strategy.get("metadata") or getattr(strategy, "metadata", {})
            model_name = metadata.get("model_name") if isinstance(metadata, dict) else None

            if not model_name:
                continue

            try:
                predictions = self.api.predict_model(
                    model_name=model_name,
                    symbols=universe,
                    as_of_date=date.today(),
                )
                if predictions is not None and not predictions.empty:
                    all_predictions.append(predictions)
                    logger.info(
                        f"Got {len(predictions)} predictions from {hypothesis_id}"
                    )
            except Exception as e:
                logger.error(f"Failed to get predictions for {hypothesis_id}: {e}")
                continue

        if not all_predictions:
            logger.warning("No predictions available")
            return {
                "status": "no_predictions",
                "orders_generated": 0,
                "orders_submitted": 0,
                "records_fetched": 0,
                "records_inserted": 0,
            }

        # Combine predictions (simple average for now)
        import pandas as pd

        combined = pd.concat(all_predictions)
        combined = combined.groupby("symbol").agg({
            "prediction": "mean",
            "signal": "max",  # Take the strongest signal
        }).reset_index()

        # Dry-run mode - just generate orders without submitting
        if self.trading_config.dry_run:
            config = ConversionConfig(
                portfolio_value=self.trading_config.portfolio_value,
                max_positions=self.trading_config.max_positions,
                max_position_pct=self.trading_config.max_position_pct,
                min_order_value=self.trading_config.min_order_value,
            )
            converter = SignalConverter(config)
            orders = converter.signals_to_orders(combined, method="rank")

            logger.info(f"[DRY RUN] Would submit {len(orders)} orders")
            for order in orders:
                logger.info(
                    f"[DRY RUN] {order.side.value.upper()} {order.quantity} {order.symbol}"
                )

            return {
                "status": "dry_run",
                "orders_generated": len(orders),
                "orders_submitted": 0,
                "records_fetched": len(combined),
                "records_inserted": 0,
            }

        # Live mode - connect to broker and execute
        with self._create_broker() as broker:
            # Sync positions
            tracker = PositionTracker(broker, self.api)
            positions = tracker.sync_positions()
            tracker.persist_positions()

            current_positions = {p.symbol: p.quantity for p in positions}
            current_prices = {
                p.symbol: p.current_price for p in positions
            }

            # Get portfolio value from positions
            portfolio_value = tracker.calculate_portfolio_value()
            if portfolio_value == 0:
                portfolio_value = self.trading_config.portfolio_value

            # Create position sizer if VaR sizing enabled
            position_sizer = None
            if self.trading_config.use_var_sizing:
                sizing_config = PositionSizingConfig(
                    portfolio_value=portfolio_value,
                    max_portfolio_var_pct=float(
                        os.getenv("HRP_MAX_PORTFOLIO_VAR_PCT", "0.02")
                    ),
                    max_position_var_pct=float(
                        os.getenv("HRP_MAX_POSITION_VAR_PCT", "0.005")
                    ),
                    min_position_value=self.trading_config.min_order_value,
                    max_position_pct=self.trading_config.max_position_pct,
                )
                position_sizer = PositionSizer(sizing_config, self.api)
                logger.info("VaR-based position sizing enabled")

            # Convert signals to orders
            config = ConversionConfig(
                portfolio_value=portfolio_value,
                max_positions=self.trading_config.max_positions,
                max_position_pct=self.trading_config.max_position_pct,
                min_order_value=self.trading_config.min_order_value,
            )
            converter = SignalConverter(config, position_sizer=position_sizer)

            # Generate rebalancing orders
            orders = converter.rebalance_to_orders(
                current_positions=current_positions,
                target_signals=combined,
                current_prices=current_prices,
                use_var_sizing=self.trading_config.use_var_sizing,
                as_of_date=date.today(),
            )

            # Add stop-loss orders for buy orders if configured
            if self.trading_config.stop_loss_pct:
                orders_with_stops = self._add_stop_loss_orders(
                    orders, current_prices, self.trading_config.stop_loss_pct
                )
                logger.info(
                    f"Added {len(orders_with_stops) - len(orders)} stop-loss orders "
                    f"at {self.trading_config.stop_loss_pct:.1%}"
                )
                orders = orders_with_stops

            # Submit orders
            order_manager = OrderManager(broker)
            submitted_orders = []
            submitted_count = 0

            for order in orders:
                try:
                    filled_order = order_manager.submit_order(order)
                    submitted_orders.append(filled_order)
                    submitted_count += 1

                    # Record trade in database
                    self.api.record_trade(
                        order=filled_order,
                        filled_price=current_prices.get(
                            order.symbol, Decimal("0")
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to submit order for {order.symbol}: {e}")

            logger.info(
                f"Submitted {submitted_count}/{len(orders)} orders"
            )

            # Poll order status until settled
            if submitted_orders:
                settled_orders = self._poll_order_status(
                    submitted_orders, order_manager, broker
                )
                logger.info(
                    f"Order status: {sum(1 for o in settled_orders if o.status == OrderStatus.FILLED)} filled, "
                    f"{sum(1 for o in settled_orders if o.status == OrderStatus.CANCELLED)} cancelled, "
                    f"{sum(1 for o in settled_orders if o.status == OrderStatus.REJECTED)} rejected"
                )

            # Log to lineage
            self.api.log_event(
                event_type="agent_run_complete",
                actor="system:live_trader",
                details={
                    "orders_generated": len(orders),
                    "orders_submitted": submitted_count,
                    "positions_synced": len(positions),
                    "dry_run": False,
                    "use_var_sizing": self.trading_config.use_var_sizing,
                    "stop_loss_pct": self.trading_config.stop_loss_pct,
                    "broker_type": self.trading_config.broker_type,
                },
            )

            return {
                "status": "success",
                "orders_generated": len(orders),
                "orders_submitted": submitted_count,
                "positions_synced": len(positions),
                "records_fetched": len(combined),
                "records_inserted": submitted_count,
            }

    def _add_stop_loss_orders(
        self,
        orders: list[Order],
        current_prices: dict[str, Decimal],
        stop_loss_pct: float,
    ) -> list[Order]:
        """Add stop-loss orders for each buy order.

        Args:
            orders: List of orders to process
            current_prices: Current prices for calculating stop prices
            stop_loss_pct: Stop-loss percentage (e.g., 0.05 for 5% stop)

        Returns:
            List of orders with stop-loss orders added after each buy
        """
        orders_with_stops = []

        for order in orders:
            orders_with_stops.append(order)

            # Add stop-loss for buy orders only
            if order.side == OrderSide.BUY:
                # Get entry price
                entry_price = current_prices.get(order.symbol)
                if not entry_price or entry_price <= 0:
                    logger.warning(
                        f"Cannot add stop-loss for {order.symbol}: missing price"
                    )
                    continue

                # Calculate stop price
                stop_price = entry_price * (Decimal("1.0") - Decimal(str(stop_loss_pct)))

                # Create stop-loss order
                stop_order = Order(
                    symbol=order.symbol,
                    side=OrderSide.SELL,
                    quantity=order.quantity,
                    order_type=OrderType.STOP_LOSS,
                    stop_price=stop_price,
                )

                orders_with_stops.append(stop_order)

                logger.debug(
                    f"Added stop-loss for {order.symbol}: "
                    f"SELL {order.quantity} @ ${stop_price:.2f} "
                    f"({stop_loss_pct:.1%} below ${entry_price:.2f})"
                )

        return orders_with_stops

    def _poll_order_status(
        self,
        orders: list[Order],
        order_manager: OrderManager,
        broker: BaseBroker,
        poll_interval: int = 10,
        max_polls: int = 30,
    ) -> list[Order]:
        """Poll order status until all orders are settled.

        Args:
            orders: List of submitted orders to track
            order_manager: Order manager for status queries
            broker: Broker instance for status queries
            poll_interval: Seconds between polls (default: 10)
            max_polls: Maximum number of polls before giving up (default: 30 = 5 minutes)

        Returns:
            List of orders with updated status
        """
        import time

        from hrp.execution.orders import OrderStatus

        # Track orders that need monitoring
        pending_orders = {
            o.broker_order_id: o for o in orders if o.broker_order_id
        }

        if not pending_orders:
            logger.info("No orders to monitor (no broker_order_ids)")
            return orders

        logger.info(
            f"Monitoring {len(pending_orders)} orders "
            f"(poll every {poll_interval}s, max {max_polls} polls)"
        )

        poll_count = 0

        while pending_orders and poll_count < max_polls:
            poll_count += 1
            time.sleep(poll_interval)

            logger.debug(f"Poll {poll_count}/{max_polls}: checking {len(pending_orders)} orders")

            for broker_order_id in list(pending_orders.keys()):
                try:
                    # Query order status from broker
                    status = broker.get_order_status(broker_order_id)

                    # Update order object
                    order = pending_orders[broker_order_id]
                    order.status = status

                    # Remove from pending if settled
                    if status in (
                        OrderStatus.FILLED,
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                    ):
                        logger.info(
                            f"Order {broker_order_id} settled: "
                            f"{order.side.value.upper()} {order.quantity} {order.symbol} - {status.value}"
                        )
                        del pending_orders[broker_order_id]

                except Exception as e:
                    logger.error(
                        f"Error checking status for order {broker_order_id}: {e}"
                    )

        if pending_orders:
            logger.warning(
                f"{len(pending_orders)} orders still pending after {poll_count} polls"
            )

        return orders
