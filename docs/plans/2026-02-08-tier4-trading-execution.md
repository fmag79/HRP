# Tier 4: Trading/Live Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable paper trading with Interactive Brokers by building broker integration, order management, position tracking, and live prediction pipeline.

**Architecture:** Three-layer execution system: (1) Daily prediction job generates signals from production models, (2) Signal-to-order conversion with risk limits, (3) IBKR broker integration for order execution and position sync. All operations logged to lineage for audit trail.

**Tech Stack:** ib_insync (IBKR API), existing VectorBT backtest engine, MLflow model registry, DuckDB for persistence, launchd scheduling

---

## Phase 1: Broker Integration Foundation

### Task 1: IBKR Connection Manager

**Files:**
- Create: `hrp/execution/broker.py`
- Create: `tests/test_execution/test_broker.py`
- Create: `.env.example` (add IBKR vars)

**Step 1: Write the failing test**

Create `tests/test_execution/test_broker.py`:

```python
"""Tests for IBKR broker connection."""
import pytest
from unittest.mock import Mock, patch
from hrp.execution.broker import IBKRBroker, BrokerConfig


def test_broker_config_validation():
    """Test broker config requires all fields."""
    with pytest.raises(ValueError, match="host is required"):
        BrokerConfig(host="", port=7497, client_id=1, account="")


def test_broker_connection_paper_trading():
    """Test broker connects to paper trading account."""
    config = BrokerConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=1,
        account="DU123456",
        paper_trading=True,
    )

    with patch("hrp.execution.broker.IB") as mock_ib:
        mock_ib.return_value.connect.return_value = None
        mock_ib.return_value.isConnected.return_value = True

        broker = IBKRBroker(config)
        broker.connect()

        assert broker.is_connected()
        mock_ib.return_value.connect.assert_called_once_with(
            "127.0.0.1", 7497, clientId=1, readonly=False
        )


def test_broker_disconnect():
    """Test broker disconnects cleanly."""
    config = BrokerConfig(
        host="127.0.0.1", port=7497, client_id=1,
        account="DU123456", paper_trading=True
    )

    with patch("hrp.execution.broker.IB") as mock_ib:
        mock_ib.return_value.isConnected.return_value = True
        broker = IBKRBroker(config)
        broker.connect()
        broker.disconnect()

        mock_ib.return_value.disconnect.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_execution/test_broker.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.execution.broker'"

**Step 3: Install ib_insync dependency**

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing deps ...
    "ib-insync>=0.9.86",
]
```

Run:
```bash
pip install ib-insync
```

**Step 4: Write minimal broker implementation**

Create `hrp/execution/broker.py`:

```python
"""IBKR broker connection and account management."""
import logging
from dataclasses import dataclass
from typing import Optional

from ib_insync import IB, Stock, MarketOrder, LimitOrder, Order, Trade

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

    def __post_init__(self):
        """Validate configuration."""
        if not self.host:
            raise ValueError("host is required")
        if not self.account:
            raise ValueError("account is required")
        if self.paper_trading and self.port != 7497:
            logger.warning(f"Paper trading typically uses port 7497, got {self.port}")


class IBKRBroker:
    """Interactive Brokers connection manager."""

    def __init__(self, config: BrokerConfig):
        """Initialize broker with configuration.

        Args:
            config: Broker connection configuration
        """
        self.config = config
        self.ib = IB()
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
            raise ConnectionError(f"IBKR connection failed: {e}")

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

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_execution/test_broker.py -v
```

Expected: PASS (3 tests)

**Step 6: Add environment variables example**

Add to `.env.example`:

```bash
# Interactive Brokers Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper trading (4002 for live IB Gateway)
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=DU123456  # Paper trading account
IBKR_PAPER_TRADING=true
```

**Step 7: Commit**

```bash
git add hrp/execution/broker.py tests/test_execution/test_broker.py .env.example pyproject.toml
git commit -m "feat(execution): add IBKR broker connection manager

- IBKRBroker class with connect/disconnect
- BrokerConfig dataclass with validation
- Context manager support
- Paper trading port validation
- Environment variable configuration"
```

---

### Task 2: Order Management System

**Files:**
- Create: `hrp/execution/orders.py`
- Create: `tests/test_execution/test_orders.py`
- Modify: `hrp/data/schema.sql` (add executed_trades table)

**Step 1: Write the failing test**

Create `tests/test_execution/test_orders.py`:

```python
"""Tests for order management system."""
import pytest
from datetime import datetime
from decimal import Decimal
from hrp.execution.orders import Order, OrderType, OrderSide, OrderStatus, OrderManager


def test_order_creation():
    """Test creating a market order."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.quantity == 10
    assert order.order_type == OrderType.MARKET
    assert order.status == OrderStatus.PENDING
    assert order.limit_price is None


def test_limit_order_requires_price():
    """Test limit order requires limit_price."""
    with pytest.raises(ValueError, match="Limit orders require limit_price"):
        Order(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=None,
        )


def test_order_manager_submit_market_order(mock_broker):
    """Test submitting a market order via broker."""
    manager = OrderManager(mock_broker)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    submitted = manager.submit_order(order)

    assert submitted.status == OrderStatus.SUBMITTED
    assert submitted.broker_order_id is not None
    assert submitted.submitted_at is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_execution/test_orders.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.execution.orders'"

**Step 3: Write order management implementation**

Create `hrp/execution/orders.py`:

```python
"""Order management system for live trading."""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from ib_insync import Stock, MarketOrder, LimitOrder, Order as IBOrder

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

    def __post_init__(self):
        """Validate order."""
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")
        if self.order_type == OrderType.LIMIT and self.limit_price <= 0:
            raise ValueError("limit_price must be positive")


class OrderManager:
    """Manages order submission and tracking."""

    def __init__(self, broker):
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
```

**Step 4: Add broker mock fixture**

Add to `tests/test_execution/conftest.py`:

```python
"""Test fixtures for execution tests."""
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_broker():
    """Mock IBKR broker for testing."""
    broker = Mock()
    broker.is_connected.return_value = True

    # Mock IB instance
    broker.ib = Mock()

    # Mock trade result
    mock_trade = Mock()
    mock_trade.order.orderId = 12345
    broker.ib.placeOrder.return_value = mock_trade

    return broker
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_execution/test_orders.py -v
```

Expected: PASS (3 tests)

**Step 6: Add executed_trades database table**

Add to `hrp/data/schema.sql`:

```sql
-- Executed trades (live/paper trading)
CREATE TABLE IF NOT EXISTS executed_trades (
    trade_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    broker_order_id INTEGER,
    hypothesis_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit')),
    limit_price DECIMAL(10, 2),
    filled_price DECIMAL(10, 2),
    filled_quantity INTEGER NOT NULL CHECK (filled_quantity >= 0),
    commission DECIMAL(10, 2),
    status TEXT NOT NULL CHECK (status IN ('pending', 'submitted', 'filled', 'partially_filled', 'cancelled', 'rejected')),
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

CREATE INDEX IF NOT EXISTS idx_executed_trades_hypothesis ON executed_trades(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_executed_trades_symbol ON executed_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_executed_trades_filled_at ON executed_trades(filled_at);
```

**Step 7: Commit**

```bash
git add hrp/execution/orders.py tests/test_execution/test_orders.py tests/test_execution/conftest.py hrp/data/schema.sql
git commit -m "feat(execution): add order management system

- Order dataclass with validation
- OrderType, OrderSide, OrderStatus enums
- OrderManager for submission and tracking
- Market and limit order support
- executed_trades database table
- Broker order ID tracking"
```

---

### Task 3: Position Tracking and Sync

**Files:**
- Create: `hrp/execution/positions.py`
- Create: `tests/test_execution/test_positions.py`
- Modify: `hrp/data/schema.sql` (add live_positions table)

**Step 1: Write the failing test**

Create `tests/test_execution/test_positions.py`:

```python
"""Tests for position tracking."""
import pytest
from decimal import Decimal
from datetime import date
from hrp.execution.positions import Position, PositionTracker


def test_position_creation():
    """Test creating a position."""
    position = Position(
        symbol="AAPL",
        quantity=10,
        entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
        as_of_date=date(2026, 2, 8),
    )

    assert position.symbol == "AAPL"
    assert position.quantity == 10
    assert position.market_value == Decimal("1550.00")
    assert position.cost_basis == Decimal("1500.00")
    assert position.unrealized_pnl == Decimal("50.00")
    assert position.unrealized_pnl_pct == pytest.approx(0.0333, rel=1e-3)


def test_position_update_price():
    """Test updating position price."""
    position = Position(
        symbol="MSFT",
        quantity=5,
        entry_price=Decimal("300.00"),
        current_price=Decimal("300.00"),
        as_of_date=date(2026, 2, 8),
    )

    position.update_price(Decimal("310.00"))

    assert position.current_price == Decimal("310.00")
    assert position.unrealized_pnl == Decimal("50.00")


def test_position_tracker_sync(mock_broker, mock_api):
    """Test syncing positions from broker."""
    tracker = PositionTracker(mock_broker, mock_api)

    positions = tracker.sync_positions()

    assert len(positions) == 2
    assert positions[0].symbol == "AAPL"
    assert positions[1].symbol == "MSFT"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_execution/test_positions.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.execution.positions'"

**Step 3: Write position tracking implementation**

Create `hrp/execution/positions.py`:

```python
"""Position tracking and synchronization."""
import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position representation."""
    symbol: str
    quantity: int
    entry_price: Decimal
    current_price: Decimal
    as_of_date: date
    hypothesis_id: Optional[str] = None
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

    def __init__(self, broker, api):
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

    def get_position(self, symbol: str) -> Optional[Position]:
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
        conn = self.api.get_db()

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
```

**Step 4: Add mock fixtures**

Add to `tests/test_execution/conftest.py`:

```python
from decimal import Decimal
from unittest.mock import Mock


@pytest.fixture
def mock_api():
    """Mock PlatformAPI for testing."""
    api = Mock()
    api.get_db.return_value = Mock()
    return api


@pytest.fixture
def mock_broker():
    """Mock IBKR broker with positions."""
    broker = Mock()
    broker.is_connected.return_value = True
    broker.config.account = "DU123456"

    # Mock IB instance
    broker.ib = Mock()

    # Mock positions
    mock_pos1 = Mock()
    mock_pos1.contract.symbol = "AAPL"
    mock_pos1.position = 10
    mock_pos1.avgCost = 150.0
    mock_pos1.account = "DU123456"

    mock_pos2 = Mock()
    mock_pos2.contract.symbol = "MSFT"
    mock_pos2.position = 5
    mock_pos2.avgCost = 300.0
    mock_pos2.account = "DU123456"

    broker.ib.positions.return_value = [mock_pos1, mock_pos2]

    # Mock market data
    mock_ticker = Mock()
    mock_ticker.last = 155.0
    broker.ib.reqMktData.return_value = mock_ticker
    broker.ib.sleep = Mock()

    return broker
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_execution/test_positions.py -v
```

Expected: PASS (3 tests)

**Step 6: Add live_positions table**

Add to `hrp/data/schema.sql`:

```sql
-- Live positions (synced from broker)
CREATE TABLE IF NOT EXISTS live_positions (
    symbol TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10, 2) NOT NULL,
    current_price DECIMAL(10, 2) NOT NULL,
    market_value DECIMAL(15, 2) NOT NULL,
    cost_basis DECIMAL(15, 2) NOT NULL,
    unrealized_pnl DECIMAL(15, 2) NOT NULL,
    unrealized_pnl_pct REAL NOT NULL,
    hypothesis_id TEXT,
    as_of_date DATE NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

CREATE INDEX IF NOT EXISTS idx_live_positions_hypothesis ON live_positions(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_live_positions_date ON live_positions(as_of_date);
```

**Step 7: Commit**

```bash
git add hrp/execution/positions.py tests/test_execution/test_positions.py tests/test_execution/conftest.py hrp/data/schema.sql
git commit -m "feat(execution): add position tracking and sync

- Position dataclass with PnL calculations
- PositionTracker for broker synchronization
- Market data fetching for current prices
- live_positions database table
- Position persistence to database"
```

---

## Phase 2: Signal-to-Order Pipeline

### Task 4: Daily Prediction Job

**Files:**
- Create: `hrp/agents/prediction_job.py`
- Create: `tests/test_agents/test_prediction_job.py`
- Modify: `hrp/agents/run_job.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_prediction_job.py`:

```python
"""Tests for daily prediction job."""
import pytest
from datetime import date
from unittest.mock import Mock, patch
from hrp.agents.prediction_job import DailyPredictionJob


def test_prediction_job_requires_deployed_strategies(mock_api):
    """Test job checks for deployed strategies."""
    mock_api.get_deployed_strategies.return_value = []

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "no_deployed_strategies"
    assert result["predictions_generated"] == 0


def test_prediction_job_generates_predictions(mock_api):
    """Test job generates predictions for deployed strategies."""
    # Mock deployed strategy
    mock_strategy = Mock()
    mock_strategy.hypothesis_id = "HYP-2026-001"
    mock_strategy.metadata = {"model_name": "momentum_v1"}

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.get_universe.return_value = ["AAPL", "MSFT", "GOOGL"]
    mock_api.predict_model.return_value = Mock(shape=(3, 4))  # 3 predictions

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "success"
    assert result["predictions_generated"] == 3
    assert result["strategies_processed"] == 1
    mock_api.predict_model.assert_called_once()


def test_prediction_job_handles_prediction_errors(mock_api):
    """Test job continues on prediction errors."""
    mock_strategy = Mock()
    mock_strategy.hypothesis_id = "HYP-2026-001"
    mock_strategy.metadata = {"model_name": "broken_model"}

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.get_universe.return_value = ["AAPL"]
    mock_api.predict_model.side_effect = ValueError("Model not found")

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "partial_failure"
    assert result["predictions_generated"] == 0
    assert result["errors"] == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_agents/test_prediction_job.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.agents.prediction_job'"

**Step 3: Write prediction job implementation**

Create `hrp/agents/prediction_job.py`:

```python
"""Daily prediction job for deployed strategies."""
import logging
from datetime import date
from typing import Any

from hrp.agents.jobs import IngestionJob, DataRequirement
from hrp.api.platform import PlatformAPI

logger = logging.getLogger(__name__)


class DailyPredictionJob(IngestionJob):
    """Generate daily predictions for all deployed strategies."""

    def __init__(
        self,
        job_id: str = "daily_predictions",
        api: PlatformAPI | None = None,
    ):
        """Initialize daily prediction job.

        Args:
            job_id: Job identifier
            api: PlatformAPI instance (creates new if None)
        """
        data_requirements = [
            DataRequirement(
                table="features",
                min_rows=100,
                max_age_days=3,
                date_column="date",
                description="Recent feature data",
            ),
            DataRequirement(
                table="prices",
                min_rows=1000,
                max_age_days=3,
                date_column="date",
                description="Recent price data",
            ),
        ]

        super().__init__(job_id, data_requirements=data_requirements)
        self.api = api or PlatformAPI()

    def execute(self) -> dict[str, Any]:
        """Execute daily prediction job.

        Returns:
            Dict with execution stats:
                - status: success, partial_failure, or no_deployed_strategies
                - predictions_generated: Count of predictions
                - strategies_processed: Count of strategies
                - errors: Count of errors
        """
        logger.info("Starting daily prediction job")

        # Get all deployed strategies
        deployed = self.api.get_deployed_strategies()

        if not deployed:
            logger.warning("No deployed strategies found")
            return {
                "status": "no_deployed_strategies",
                "predictions_generated": 0,
                "strategies_processed": 0,
                "errors": 0,
            }

        # Get universe for predictions
        universe = self.api.get_universe(as_of_date=date.today())

        total_predictions = 0
        errors = 0

        for strategy in deployed:
            hypothesis_id = strategy.hypothesis_id

            try:
                # Get model name from metadata
                model_name = strategy.metadata.get("model_name")
                if not model_name:
                    logger.warning(
                        f"Strategy {hypothesis_id} has no model_name in metadata, skipping"
                    )
                    errors += 1
                    continue

                # Generate predictions
                predictions = self.api.predict_model(
                    model_name=model_name,
                    symbols=universe,
                    as_of_date=date.today(),
                    model_version=None,  # Use production version
                )

                total_predictions += len(predictions)

                logger.info(
                    f"Generated {len(predictions)} predictions for {hypothesis_id} "
                    f"(model={model_name})"
                )

                # Log to lineage
                self.api.log_event(
                    event_type="prediction_generated",
                    actor="system",
                    hypothesis_id=hypothesis_id,
                    details={
                        "model_name": model_name,
                        "num_predictions": len(predictions),
                        "as_of_date": str(date.today()),
                    },
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate predictions for {hypothesis_id}: {e}"
                )
                errors += 1
                continue

        status = "success" if errors == 0 else "partial_failure"

        return {
            "status": status,
            "predictions_generated": total_predictions,
            "strategies_processed": len(deployed) - errors,
            "errors": errors,
        }
```

**Step 4: Add mock fixture**

Add to `tests/test_agents/conftest.py`:

```python
"""Test fixtures for agent tests."""
import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_api():
    """Mock PlatformAPI for testing."""
    api = Mock()
    api.get_universe.return_value = ["AAPL", "MSFT"]
    api.log_event = Mock()
    return api
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_agents/test_prediction_job.py -v
```

Expected: PASS (3 tests)

**Step 6: Add CLI handler**

Modify `hrp/agents/run_job.py` to add prediction job:

```python
def run_predictions(dry_run: bool = False) -> dict:
    """Run daily prediction job.

    Args:
        dry_run: If True, log actions without executing

    Returns:
        Execution stats dict
    """
    from hrp.agents.prediction_job import DailyPredictionJob

    if dry_run:
        logger.info("[DRY RUN] Would run daily prediction job")
        return {"status": "dry_run", "job": "predictions"}

    job = DailyPredictionJob()
    return job.run()


# In main() function, add to job dispatcher:
    elif args.job == "predictions":
        result = run_predictions(dry_run=args.dry_run)
```

**Step 7: Create launchd plist**

Create `launchd/com.hrp.predictions.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hrp.predictions</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/HRP/.venv/bin/python</string>
        <string>-m</string>
        <string>hrp.agents.run_job</string>
        <string>--job</string>
        <string>predictions</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/HRP</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>18</integer>
        <key>Minute</key>
        <integer>15</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>~/hrp-data/logs/predictions.log</string>
    <key>StandardErrorPath</key>
    <string>~/hrp-data/logs/predictions.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>/path/to/HRP</string>
    </dict>
</dict>
</plist>
```

**Step 8: Commit**

```bash
git add hrp/agents/prediction_job.py tests/test_agents/test_prediction_job.py tests/test_agents/conftest.py hrp/agents/run_job.py launchd/com.hrp.predictions.plist
git commit -m "feat(agents): add daily prediction job

- DailyPredictionJob for deployed strategies
- Automatic prediction generation for universe
- Error handling with partial failure support
- Lineage logging for predictions
- launchd scheduling (daily at 6:15 PM ET)"
```

---

### Task 5: Signal-to-Order Conversion

**Files:**
- Create: `hrp/execution/signal_converter.py`
- Create: `tests/test_execution/test_signal_converter.py`

**Step 1: Write the failing test**

Create `tests/test_execution/test_signal_converter.py`:

```python
"""Tests for signal-to-order conversion."""
import pytest
import pandas as pd
from decimal import Decimal
from hrp.execution.signal_converter import SignalConverter, ConversionConfig
from hrp.execution.orders import Order, OrderSide, OrderType


def test_signal_converter_config_validation():
    """Test conversion config validation."""
    with pytest.raises(ValueError, match="max_position_pct must be between 0 and 1"):
        ConversionConfig(max_position_pct=1.5)

    with pytest.raises(ValueError, match="max_positions must be positive"):
        ConversionConfig(max_positions=0)


def test_signal_converter_rank_signals_to_orders():
    """Test converting rank-based signals to orders."""
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "signal": [1.0, 1.0, 0.0, 0.0],  # Long AAPL and MSFT
        "prediction": [0.05, 0.04, 0.01, 0.00],
    })

    config = ConversionConfig(
        portfolio_value=Decimal("100000"),
        max_positions=20,
        max_position_pct=0.10,
    )

    converter = SignalConverter(config)
    orders = converter.signals_to_orders(signals, method="rank")

    assert len(orders) == 2
    assert orders[0].symbol == "AAPL"
    assert orders[0].side == OrderSide.BUY
    assert orders[0].order_type == OrderType.MARKET
    # Max position = 100000 * 0.10 = 10000
    # At current price ~150, quantity = 10000 / 150 = 66 shares


def test_signal_converter_respects_risk_limits():
    """Test signal converter respects position size limits."""
    signals = pd.DataFrame({
        "symbol": ["AAPL"] * 25,  # More signals than max_positions
        "signal": [1.0] * 25,
        "prediction": [0.05] * 25,
    })

    config = ConversionConfig(
        portfolio_value=Decimal("100000"),
        max_positions=20,  # Hard limit
        max_position_pct=0.05,
    )

    converter = SignalConverter(config)
    orders = converter.signals_to_orders(signals, method="rank")

    # Should only create 20 orders (max_positions limit)
    assert len(orders) <= 20
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_execution/test_signal_converter.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.execution.signal_converter'"

**Step 3: Write signal converter implementation**

Create `hrp/execution/signal_converter.py`:

```python
"""Convert ML predictions/signals to trading orders."""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import pandas as pd

from hrp.execution.orders import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for signal-to-order conversion."""
    portfolio_value: Decimal
    max_positions: int = 20
    max_position_pct: float = 0.10  # 10% max per position
    min_order_value: Decimal = Decimal("100.00")  # Minimum order size

    def __post_init__(self):
        """Validate configuration."""
        if self.max_position_pct <= 0 or self.max_position_pct > 1:
            raise ValueError("max_position_pct must be between 0 and 1")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")


class SignalConverter:
    """Converts ML signals to trading orders with risk limits."""

    def __init__(self, config: ConversionConfig):
        """Initialize signal converter.

        Args:
            config: Conversion configuration
        """
        self.config = config

    def signals_to_orders(
        self,
        signals: pd.DataFrame,
        method: Literal["rank", "threshold", "zscore"] = "rank",
        current_prices: dict[str, Decimal] | None = None,
    ) -> list[Order]:
        """Convert signals to trading orders.

        Args:
            signals: DataFrame with columns [symbol, signal, prediction]
                    signal: 1.0 (long), 0.0 (no position), -1.0 (short, not supported yet)
            method: Signal generation method used
            current_prices: Optional dict of symbol -> current price
                          (fetched from broker if not provided)

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
            f"(method={method}, max_positions={self.config.max_positions})"
        )

        # Calculate position size for each signal
        max_position_value = self.config.portfolio_value * Decimal(
            str(self.config.max_position_pct)
        )

        orders = []

        for _, row in buy_signals.iterrows():
            symbol = row["symbol"]

            # Get current price (mock for now, would fetch from broker)
            if current_prices and symbol in current_prices:
                price = current_prices[symbol]
            else:
                # TODO: Fetch from broker
                price = Decimal("150.00")  # Mock price

            # Calculate quantity
            quantity = int(max_position_value / price)

            # Skip if order too small
            order_value = quantity * price
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_execution/test_signal_converter.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add hrp/execution/signal_converter.py tests/test_execution/test_signal_converter.py
git commit -m "feat(execution): add signal-to-order conversion

- SignalConverter for ML signal translation
- ConversionConfig with risk limits
- Position sizing based on portfolio value
- Max positions and max position % enforcement
- Minimum order value filtering"
```

---

## Phase 3: Live Trading Execution

### Task 6: Live Trading Agent

**Files:**
- Create: `hrp/agents/live_trader.py`
- Create: `tests/test_agents/test_live_trader.py`
- Modify: `hrp/agents/run_job.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_live_trader.py`:

```python
"""Tests for live trading agent."""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from decimal import Decimal
from hrp.agents.live_trader import LiveTradingAgent
from hrp.execution.orders import OrderStatus


def test_live_trader_dry_run_mode(mock_api, mock_broker):
    """Test live trader in dry run mode."""
    agent = LiveTradingAgent(api=mock_api, broker=mock_broker, dry_run=True)

    result = agent.execute()

    assert result["status"] == "dry_run"
    assert result["orders_submitted"] == 0


def test_live_trader_no_predictions(mock_api, mock_broker):
    """Test live trader handles no predictions."""
    mock_api.get_deployed_strategies.return_value = []

    agent = LiveTradingAgent(api=mock_api, broker=mock_broker, dry_run=False)
    result = agent.execute()

    assert result["status"] == "no_strategies"
    assert result["orders_submitted"] == 0


def test_live_trader_submits_orders(mock_api, mock_broker):
    """Test live trader submits orders from signals."""
    # Mock deployed strategy
    mock_strategy = Mock()
    mock_strategy.hypothesis_id = "HYP-2026-001"
    mock_strategy.metadata = {"model_name": "momentum_v1"}

    # Mock predictions
    predictions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "prediction": [0.05, 0.04],
    })

    # Mock signals
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "signal": [1.0, 1.0],
        "prediction": [0.05, 0.04],
    })

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.get_model_predictions.return_value = predictions

    with patch("hrp.agents.live_trader.predictions_to_signals", return_value=signals):
        agent = LiveTradingAgent(
            api=mock_api,
            broker=mock_broker,
            dry_run=False,
            portfolio_value=Decimal("100000"),
        )
        result = agent.execute()

    assert result["status"] == "success"
    assert result["orders_submitted"] == 2
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_agents/test_live_trader.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.agents.live_trader'"

**Step 3: Write live trader implementation**

Create `hrp/agents/live_trader.py`:

```python
"""Live trading execution agent."""
import logging
from datetime import date
from decimal import Decimal
from typing import Any

from hrp.agents.jobs import IngestionJob, DataRequirement
from hrp.api.platform import PlatformAPI
from hrp.execution.broker import IBKRBroker, BrokerConfig
from hrp.execution.orders import OrderManager
from hrp.execution.positions import PositionTracker
from hrp.execution.signal_converter import SignalConverter, ConversionConfig
from hrp.ml.signals import predictions_to_signals

logger = logging.getLogger(__name__)


class LiveTradingAgent(IngestionJob):
    """Execute trades based on daily predictions."""

    def __init__(
        self,
        job_id: str = "live_trader",
        api: PlatformAPI | None = None,
        broker: IBKRBroker | None = None,
        dry_run: bool = True,
        portfolio_value: Decimal = Decimal("100000"),
    ):
        """Initialize live trading agent.

        Args:
            job_id: Job identifier
            api: PlatformAPI instance
            broker: IBKRBroker instance (creates from env if None)
            dry_run: If True, log orders without submitting
            portfolio_value: Current portfolio value for position sizing
        """
        data_requirements = [
            DataRequirement(
                table="model_performance_history",
                min_rows=1,
                max_age_days=1,
                date_column="prediction_date",
                description="Recent predictions",
            ),
        ]

        super().__init__(job_id, data_requirements=data_requirements)
        self.api = api or PlatformAPI()
        self.broker = broker
        self.dry_run = dry_run
        self.portfolio_value = portfolio_value

    def execute(self) -> dict[str, Any]:
        """Execute live trading logic.

        Returns:
            Dict with execution stats:
                - status: dry_run, no_strategies, success, or failure
                - orders_submitted: Count of orders submitted
                - positions_synced: Count of positions synced
        """
        if self.dry_run:
            logger.info("[DRY RUN] Live trading agent - no orders will be submitted")
            return {"status": "dry_run", "orders_submitted": 0}

        logger.info("Starting live trading agent")

        # Get deployed strategies
        deployed = self.api.get_deployed_strategies()

        if not deployed:
            logger.warning("No deployed strategies found")
            return {"status": "no_strategies", "orders_submitted": 0}

        # Connect to broker
        if not self.broker:
            broker_config = self._load_broker_config()
            self.broker = IBKRBroker(broker_config)

        try:
            self.broker.connect()

            # Initialize components
            order_manager = OrderManager(self.broker)
            position_tracker = PositionTracker(self.broker, self.api)

            # Sync current positions
            current_positions = position_tracker.sync_positions()
            logger.info(f"Synced {len(current_positions)} current positions")

            # Process each strategy
            total_orders = 0

            for strategy in deployed:
                hypothesis_id = strategy.hypothesis_id
                model_name = strategy.metadata.get("model_name")

                if not model_name:
                    logger.warning(f"Strategy {hypothesis_id} missing model_name")
                    continue

                # Get latest predictions
                predictions = self.api.get_model_predictions(
                    model_name=model_name,
                    as_of_date=date.today(),
                )

                if predictions.empty:
                    logger.warning(f"No predictions for {model_name}")
                    continue

                # Convert predictions to signals
                signals = predictions_to_signals(
                    predictions,
                    method="rank",
                    top_pct=0.1,  # Top 10% long
                )

                # Convert signals to orders
                converter = SignalConverter(
                    ConversionConfig(
                        portfolio_value=self.portfolio_value,
                        max_positions=20,
                        max_position_pct=0.10,
                    )
                )

                orders = converter.signals_to_orders(signals, method="rank")

                # Submit orders
                for order in orders:
                    try:
                        submitted = order_manager.submit_order(order)
                        total_orders += 1

                        logger.info(
                            f"Submitted order: {submitted.side.value} "
                            f"{submitted.quantity} {submitted.symbol} "
                            f"(broker_id={submitted.broker_order_id})"
                        )

                        # Log to lineage
                        self.api.log_event(
                            event_type="order_submitted",
                            actor="system",
                            hypothesis_id=hypothesis_id,
                            details={
                                "order_id": order.order_id,
                                "broker_order_id": submitted.broker_order_id,
                                "symbol": order.symbol,
                                "side": order.side.value,
                                "quantity": order.quantity,
                            },
                        )

                    except Exception as e:
                        logger.error(f"Failed to submit order for {order.symbol}: {e}")
                        continue

            # Final position sync
            position_tracker.sync_positions()
            position_tracker.persist_positions()

            return {
                "status": "success",
                "orders_submitted": total_orders,
                "positions_synced": len(position_tracker.get_all_positions()),
            }

        finally:
            if self.broker:
                self.broker.disconnect()

    def _load_broker_config(self) -> BrokerConfig:
        """Load broker config from environment.

        Returns:
            BrokerConfig instance
        """
        import os

        return BrokerConfig(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "7497")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
            account=os.getenv("IBKR_ACCOUNT", "DU123456"),
            paper_trading=os.getenv("IBKR_PAPER_TRADING", "true").lower() == "true",
        )
```

**Step 4: Add mock fixtures**

Update `tests/test_agents/conftest.py`:

```python
import pandas as pd
from decimal import Decimal


@pytest.fixture
def mock_api():
    """Mock PlatformAPI for testing."""
    api = Mock()
    api.get_deployed_strategies.return_value = []
    api.get_model_predictions.return_value = pd.DataFrame()
    api.log_event = Mock()
    return api


@pytest.fixture
def mock_broker():
    """Mock IBKR broker for testing."""
    broker = Mock()
    broker.is_connected.return_value = True
    broker.connect = Mock()
    broker.disconnect = Mock()
    broker.ib = Mock()
    broker.ib.positions.return_value = []
    return broker
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_agents/test_live_trader.py -v
```

Expected: PASS (3 tests)

**Step 6: Add CLI handler**

Modify `hrp/agents/run_job.py`:

```python
def run_live_trader(dry_run: bool = False, trading_dry_run: bool = True) -> dict:
    """Run live trading agent.

    Args:
        dry_run: If True, don't run job at all
        trading_dry_run: If True, run job but don't submit orders

    Returns:
        Execution stats dict
    """
    from hrp.agents.live_trader import LiveTradingAgent

    if dry_run:
        logger.info("[DRY RUN] Would run live trading agent")
        return {"status": "dry_run", "job": "live-trader"}

    agent = LiveTradingAgent(dry_run=trading_dry_run)
    return agent.execute()


# In main() function:
    elif args.job == "live-trader":
        # Add --trading-dry-run flag to parser
        result = run_live_trader(
            dry_run=args.dry_run,
            trading_dry_run=args.trading_dry_run,
        )
```

**Step 7: Create launchd plist (disabled by default)**

Create `launchd/com.hrp.live-trader.plist.disabled`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hrp.live-trader</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/HRP/.venv/bin/python</string>
        <string>-m</string>
        <string>hrp.agents.run_job</string>
        <string>--job</string>
        <string>live-trader</string>
        <string>--no-trading-dry-run</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/HRP</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>18</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>~/hrp-data/logs/live-trader.log</string>
    <key>StandardErrorPath</key>
    <string>~/hrp-data/logs/live-trader.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>/path/to/HRP</string>
    </dict>
</dict>
</plist>
```

**Step 8: Commit**

```bash
git add hrp/agents/live_trader.py tests/test_agents/test_live_trader.py tests/test_agents/conftest.py hrp/agents/run_job.py launchd/com.hrp.live-trader.plist.disabled
git commit -m "feat(agents): add live trading agent

- LiveTradingAgent for order execution
- Prediction → signal → order pipeline
- Broker connection management
- Position syncing and persistence
- Lineage logging for orders
- Dry run mode for safety
- Disabled launchd plist (manual enable required)"
```

---

## Phase 4: Monitoring and Dashboard

### Task 7: Model Drift Monitoring Job

**Files:**
- Create: `hrp/agents/drift_monitor_job.py`
- Create: `tests/test_agents/test_drift_monitor_job.py`
- Modify: `hrp/agents/run_job.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_drift_monitor_job.py`:

```python
"""Tests for model drift monitoring job."""
import pytest
from unittest.mock import Mock
from hrp.agents.drift_monitor_job import DriftMonitorJob


def test_drift_monitor_no_deployed_models(mock_api):
    """Test drift monitor handles no deployed models."""
    mock_api.get_deployed_strategies.return_value = []

    job = DriftMonitorJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "no_models"
    assert result["models_checked"] == 0


def test_drift_monitor_detects_drift(mock_api):
    """Test drift monitor detects model drift."""
    mock_strategy = Mock()
    mock_strategy.hypothesis_id = "HYP-2026-001"
    mock_strategy.metadata = {"model_name": "momentum_v1"}

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.check_model_drift.return_value = {
        "summary": {"drift_detected": True},
        "prediction_drift": Mock(is_drift=True, metric_value=0.3),
    }

    job = DriftMonitorJob(api=mock_api, auto_rollback=False)
    result = job.execute()

    assert result["status"] == "drift_detected"
    assert result["models_checked"] == 1
    assert result["models_with_drift"] == 1


def test_drift_monitor_auto_rollback(mock_api):
    """Test drift monitor triggers rollback on drift."""
    mock_strategy = Mock()
    mock_strategy.hypothesis_id = "HYP-2026-001"
    mock_strategy.metadata = {"model_name": "momentum_v1"}

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.check_model_drift.return_value = {
        "summary": {"drift_detected": True},
        "prediction_drift": Mock(is_drift=True, metric_value=0.3),
    }
    mock_api.rollback_deployment.return_value = {"status": "rolled_back"}

    job = DriftMonitorJob(api=mock_api, auto_rollback=True)
    result = job.execute()

    assert result["status"] == "drift_detected"
    assert result["rollbacks_triggered"] == 1
    mock_api.rollback_deployment.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_agents/test_drift_monitor_job.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.agents.drift_monitor_job'"

**Step 3: Write drift monitor implementation**

Create `hrp/agents/drift_monitor_job.py`:

```python
"""Model drift monitoring job."""
import logging
from datetime import date, timedelta
from typing import Any

from hrp.agents.jobs import IngestionJob, DataRequirement
from hrp.api.platform import PlatformAPI

logger = logging.getLogger(__name__)


class DriftMonitorJob(IngestionJob):
    """Monitor deployed models for drift and trigger rollbacks."""

    def __init__(
        self,
        job_id: str = "drift_monitor",
        api: PlatformAPI | None = None,
        auto_rollback: bool = False,
        drift_threshold: float = 0.2,
    ):
        """Initialize drift monitor job.

        Args:
            job_id: Job identifier
            api: PlatformAPI instance
            auto_rollback: If True, automatically rollback on drift
            drift_threshold: Drift threshold (default 0.2 = 20%)
        """
        data_requirements = [
            DataRequirement(
                table="model_performance_history",
                min_rows=10,
                max_age_days=7,
                date_column="prediction_date",
                description="Recent prediction history",
            ),
        ]

        super().__init__(job_id, data_requirements=data_requirements)
        self.api = api or PlatformAPI()
        self.auto_rollback = auto_rollback
        self.drift_threshold = drift_threshold

    def execute(self) -> dict[str, Any]:
        """Execute drift monitoring.

        Returns:
            Dict with execution stats:
                - status: no_models, ok, or drift_detected
                - models_checked: Count of models checked
                - models_with_drift: Count of models with drift
                - rollbacks_triggered: Count of rollbacks (if auto_rollback)
        """
        logger.info("Starting model drift monitoring")

        # Get deployed strategies
        deployed = self.api.get_deployed_strategies()

        if not deployed:
            logger.warning("No deployed models to monitor")
            return {
                "status": "no_models",
                "models_checked": 0,
                "models_with_drift": 0,
                "rollbacks_triggered": 0,
            }

        models_with_drift = 0
        rollbacks = 0

        for strategy in deployed:
            hypothesis_id = strategy.hypothesis_id
            model_name = strategy.metadata.get("model_name")

            if not model_name:
                logger.warning(f"Strategy {hypothesis_id} missing model_name")
                continue

            try:
                # Get recent predictions for current data
                end_date = date.today()
                start_date = end_date - timedelta(days=7)

                current_data = self.api.get_model_predictions(
                    model_name=model_name,
                    start_date=start_date,
                    end_date=end_date,
                )

                if current_data.empty:
                    logger.warning(f"No recent predictions for {model_name}")
                    continue

                # Check for drift
                drift_result = self.api.check_model_drift(
                    model_name=model_name,
                    current_data=current_data,
                    reference_data=None,  # Uses baseline from training
                )

                if drift_result["summary"]["drift_detected"]:
                    models_with_drift += 1

                    logger.warning(
                        f"Drift detected for {model_name} (hypothesis={hypothesis_id})"
                    )

                    # Log to lineage
                    self.api.log_event(
                        event_type="drift_detected",
                        actor="system",
                        hypothesis_id=hypothesis_id,
                        details={
                            "model_name": model_name,
                            "drift_metrics": {
                                k: v.metric_value
                                for k, v in drift_result.items()
                                if k != "summary" and hasattr(v, "metric_value")
                            },
                        },
                    )

                    # Trigger rollback if enabled
                    if self.auto_rollback:
                        logger.warning(f"Triggering automatic rollback for {model_name}")

                        rollback_result = self.api.rollback_deployment(
                            model_name=model_name,
                            to_version=None,  # Rollback to previous version
                            actor="system",
                            reason=f"Automatic rollback due to drift detection",
                        )

                        rollbacks += 1
                        logger.info(f"Rolled back {model_name}: {rollback_result}")
                else:
                    logger.info(f"No drift detected for {model_name}")

            except Exception as e:
                logger.error(f"Failed to check drift for {model_name}: {e}")
                continue

        status = "drift_detected" if models_with_drift > 0 else "ok"

        return {
            "status": status,
            "models_checked": len(deployed),
            "models_with_drift": models_with_drift,
            "rollbacks_triggered": rollbacks,
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agents/test_drift_monitor_job.py -v
```

Expected: PASS (3 tests)

**Step 5: Add CLI handler**

Modify `hrp/agents/run_job.py`:

```python
def run_drift_monitor(
    dry_run: bool = False,
    auto_rollback: bool = False,
) -> dict:
    """Run drift monitoring job.

    Args:
        dry_run: If True, log actions without executing
        auto_rollback: If True, automatically rollback on drift

    Returns:
        Execution stats dict
    """
    from hrp.agents.drift_monitor_job import DriftMonitorJob

    if dry_run:
        logger.info("[DRY RUN] Would run drift monitoring job")
        return {"status": "dry_run", "job": "drift-monitor"}

    job = DriftMonitorJob(auto_rollback=auto_rollback)
    return job.run()


# In main():
    elif args.job == "drift-monitor":
        result = run_drift_monitor(
            dry_run=args.dry_run,
            auto_rollback=args.auto_rollback,
        )
```

**Step 6: Create launchd plist**

Create `launchd/com.hrp.drift-monitor.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hrp.drift-monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/HRP/.venv/bin/python</string>
        <string>-m</string>
        <string>hrp.agents.run_job</string>
        <string>--job</string>
        <string>drift-monitor</string>
        <string>--no-auto-rollback</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/HRP</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>19</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>~/hrp-data/logs/drift-monitor.log</string>
    <key>StandardErrorPath</key>
    <string>~/hrp-data/logs/drift-monitor.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>/path/to/HRP</string>
    </dict>
</dict>
</plist>
```

**Step 7: Commit**

```bash
git add hrp/agents/drift_monitor_job.py tests/test_agents/test_drift_monitor_job.py hrp/agents/run_job.py launchd/com.hrp.drift-monitor.plist
git commit -m "feat(agents): add model drift monitoring job

- DriftMonitorJob for deployed models
- Automatic drift detection (prediction, feature, concept)
- Optional automatic rollback on drift
- Lineage logging for drift events
- Daily scheduled monitoring (7 PM ET)"
```

---

### Task 8: Trading Dashboard Page

**Files:**
- Create: `hrp/dashboard/pages/trading.py`
- Modify: `hrp/dashboard/pages/__init__.py`
- Modify: `hrp/dashboard/app.py`

**Step 1: Write dashboard page**

Create `hrp/dashboard/pages/trading.py`:

```python
"""Trading dashboard page for live/paper trading monitoring."""
import logging
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from hrp.api.platform import PlatformAPI

logger = logging.getLogger(__name__)


def render() -> None:
    """Render trading dashboard page."""
    st.title("📊 Live Trading")

    api = PlatformAPI()

    # Overview metrics
    st.header("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get live positions
        positions = api.query_readonly(
            "SELECT * FROM live_positions ORDER BY unrealized_pnl DESC",
            params=(),
        )

        total_value = positions["market_value"].sum() if not positions.empty else 0
        total_pnl = positions["unrealized_pnl"].sum() if not positions.empty else 0
        num_positions = len(positions)

        col1.metric("Portfolio Value", f"${total_value:,.2f}")
        col2.metric("Unrealized P&L", f"${total_pnl:,.2f}")
        col3.metric("Open Positions", num_positions)
        col4.metric("Cash", "$0.00")  # TODO: Get from broker

    except Exception as e:
        st.error(f"Error loading portfolio metrics: {e}")
        positions = pd.DataFrame()

    # Positions table
    st.header("Current Positions")

    if not positions.empty:
        # Format for display
        display_positions = positions.copy()
        display_positions["market_value"] = display_positions["market_value"].apply(
            lambda x: f"${x:,.2f}"
        )
        display_positions["unrealized_pnl"] = display_positions["unrealized_pnl"].apply(
            lambda x: f"${x:,.2f}"
        )
        display_positions["unrealized_pnl_pct"] = display_positions[
            "unrealized_pnl_pct"
        ].apply(lambda x: f"{x*100:.2f}%")

        st.dataframe(
            display_positions[
                [
                    "symbol",
                    "quantity",
                    "entry_price",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "unrealized_pnl_pct",
                    "hypothesis_id",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No open positions")

    # Recent trades
    st.header("Recent Trades")

    try:
        trades = api.query_readonly(
            """
            SELECT * FROM executed_trades
            ORDER BY filled_at DESC
            LIMIT 50
            """,
            params=(),
        )

        if not trades.empty:
            st.dataframe(
                trades[
                    [
                        "symbol",
                        "side",
                        "filled_quantity",
                        "filled_price",
                        "commission",
                        "status",
                        "filled_at",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.info("No recent trades")

    except Exception as e:
        st.error(f"Error loading trades: {e}")

    # Model performance
    st.header("Model Performance")

    deployed = api.get_deployed_strategies()

    if deployed:
        for strategy in deployed:
            with st.expander(f"🚀 {strategy.hypothesis_id} - {strategy.title}"):
                model_name = strategy.metadata.get("model_name", "Unknown")

                st.write(f"**Model:** {model_name}")
                st.write(f"**Deployed:** {strategy.metadata.get('deployed_at', 'N/A')}")

                # Get recent predictions
                try:
                    predictions = api.get_model_predictions(
                        model_name=model_name,
                        as_of_date=date.today(),
                    )

                    if not predictions.empty:
                        st.metric(
                            "Latest Predictions",
                            len(predictions),
                            f"Mean: {predictions['prediction'].mean():.4f}",
                        )
                    else:
                        st.info("No recent predictions")

                except Exception as e:
                    st.warning(f"Could not load predictions: {e}")

                # Drift status
                try:
                    drift_checks = api.query_readonly(
                        """
                        SELECT * FROM model_drift_checks
                        WHERE model_name = ?
                        ORDER BY check_date DESC
                        LIMIT 1
                        """,
                        params=(model_name,),
                    )

                    if not drift_checks.empty:
                        latest = drift_checks.iloc[0]
                        drift_status = "⚠️ Drift Detected" if latest["is_drift_detected"] else "✅ No Drift"
                        st.write(f"**Drift Status:** {drift_status}")
                    else:
                        st.write("**Drift Status:** Not checked")

                except Exception as e:
                    st.warning(f"Could not load drift status: {e}")
    else:
        st.info("No deployed strategies")

    # Actions
    st.header("Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Sync Positions"):
            st.info("Position sync requires broker connection (not implemented in UI)")

    with col2:
        if st.button("📊 Run Drift Check"):
            st.info("Drift check scheduled daily at 7 PM ET")


def main():
    """Main entry point for testing."""
    render()


if __name__ == "__main__":
    main()
```

**Step 2: Register page**

Modify `hrp/dashboard/pages/__init__.py`:

```python
from hrp.dashboard.pages import (
    # ... existing imports ...
    trading,
)

__all__ = [
    # ... existing exports ...
    "trading",
]
```

**Step 3: Add to app routing**

Modify `hrp/dashboard/app.py`:

```python
# In render_sidebar():
    pages = [
        # ... existing pages ...
        "Trading",
    ]

# In main():
    elif page == "Trading":
        from hrp.dashboard.pages import trading
        trading.render()
```

**Step 4: Test dashboard**

```bash
streamlit run hrp/dashboard/app.py
```

Navigate to Trading page and verify:
- Portfolio metrics display
- Positions table works (empty if no positions)
- Recent trades section works
- Model performance cards display

**Step 5: Commit**

```bash
git add hrp/dashboard/pages/trading.py hrp/dashboard/pages/__init__.py hrp/dashboard/app.py
git commit -m "feat(dashboard): add trading dashboard page

- Portfolio overview metrics (value, P&L, positions)
- Current positions table with P&L
- Recent trades history
- Model performance monitoring
- Drift status display
- Position sync and drift check actions"
```

---

## Phase 5: API Integration and Documentation

### Task 9: Platform API Trading Methods

**Files:**
- Modify: `hrp/api/platform.py`
- Create: `tests/test_api/test_trading_api.py`

**Step 1: Write the failing test**

Create `tests/test_api/test_trading_api.py`:

```python
"""Tests for trading API methods."""
import pytest
from unittest.mock import Mock, patch
from hrp.api.platform import PlatformAPI


def test_get_live_positions(mock_db):
    """Test getting live positions."""
    mock_db.execute.return_value.fetchdf.return_value = Mock(
        __len__=lambda self: 2,
        iterrows=lambda: iter([
            (0, {"symbol": "AAPL", "quantity": 10}),
            (1, {"symbol": "MSFT", "quantity": 5}),
        ]),
    )

    with patch("hrp.api.platform.get_db", return_value=mock_db):
        api = PlatformAPI()
        positions = api.get_live_positions()

    assert len(positions) == 2


def test_get_executed_trades(mock_db):
    """Test getting executed trades."""
    mock_db.execute.return_value.fetchdf.return_value = Mock(__len__=lambda self: 5)

    with patch("hrp.api.platform.get_db", return_value=mock_db):
        api = PlatformAPI()
        trades = api.get_executed_trades(limit=5)

    assert len(trades) == 5


def test_record_trade(mock_db):
    """Test recording executed trade."""
    from hrp.execution.orders import Order, OrderSide, OrderType
    from decimal import Decimal

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    with patch("hrp.api.platform.get_db", return_value=mock_db):
        api = PlatformAPI()
        api.record_trade(
            order=order,
            filled_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

    mock_db.execute.assert_called()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_api/test_trading_api.py -v
```

Expected: FAIL with "AttributeError: 'PlatformAPI' object has no attribute 'get_live_positions'"

**Step 3: Add trading API methods**

Modify `hrp/api/platform.py`, add at end of class:

```python
    # =========================================================================
    # Trading and Execution
    # =========================================================================

    def get_live_positions(
        self,
        as_of_date: date | None = None,
    ) -> pd.DataFrame:
        """Get current live positions.

        Args:
            as_of_date: Date to get positions for (default: today)

        Returns:
            DataFrame with position details
        """
        conn = get_db(read_only=True)

        if as_of_date:
            positions = conn.execute(
                """
                SELECT * FROM live_positions
                WHERE as_of_date = ?
                ORDER BY unrealized_pnl DESC
                """,
                (as_of_date,),
            ).fetchdf()
        else:
            positions = conn.execute(
                """
                SELECT * FROM live_positions
                ORDER BY unrealized_pnl DESC
                """
            ).fetchdf()

        return positions

    def get_executed_trades(
        self,
        symbol: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get executed trades history.

        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            limit: Maximum trades to return

        Returns:
            DataFrame with trade history
        """
        conn = get_db(read_only=True)

        where_clauses = []
        params = []

        if symbol:
            where_clauses.append("symbol = ?")
            params.append(symbol)

        if start_date:
            where_clauses.append("filled_at >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("filled_at <= ?")
            params.append(end_date)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        trades = conn.execute(
            f"""
            SELECT * FROM executed_trades
            {where_sql}
            ORDER BY filled_at DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchdf()

        return trades

    def record_trade(
        self,
        order,
        filled_price: Decimal,
        filled_quantity: int | None = None,
        commission: Decimal | None = None,
        hypothesis_id: str | None = None,
    ) -> str:
        """Record an executed trade.

        Args:
            order: Order object
            filled_price: Execution price
            filled_quantity: Filled quantity (defaults to order quantity)
            commission: Commission paid
            hypothesis_id: Associated hypothesis

        Returns:
            trade_id
        """
        import uuid
        from datetime import datetime

        conn = get_db()

        trade_id = str(uuid.uuid4())
        filled_qty = filled_quantity or order.quantity

        conn.execute(
            """
            INSERT INTO executed_trades (
                trade_id, order_id, broker_order_id, hypothesis_id,
                symbol, side, quantity, order_type, limit_price,
                filled_price, filled_quantity, commission,
                status, submitted_at, filled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                order.order_id,
                order.broker_order_id,
                hypothesis_id,
                order.symbol,
                order.side.value,
                order.quantity,
                order.order_type.value,
                float(order.limit_price) if order.limit_price else None,
                float(filled_price),
                filled_qty,
                float(commission) if commission else None,
                "filled",
                order.submitted_at,
                datetime.now(),
            ),
        )

        conn.commit()

        logger.info(f"Recorded trade {trade_id} for {order.symbol}")

        return trade_id

    def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value from live positions.

        Returns:
            Total portfolio value
        """
        conn = get_db(read_only=True)

        result = conn.execute(
            "SELECT SUM(market_value) as total FROM live_positions"
        ).fetchone()

        total = result[0] if result and result[0] else 0

        return Decimal(str(total))
```

**Step 4: Add mock fixture**

Create `tests/test_api/conftest.py`:

```python
"""Test fixtures for API tests."""
import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_db():
    """Mock database connection."""
    db = Mock()
    db.execute = Mock(return_value=Mock())
    db.commit = Mock()
    return db
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_api/test_trading_api.py -v
```

Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add hrp/api/platform.py tests/test_api/test_trading_api.py tests/test_api/conftest.py
git commit -m "feat(api): add trading and execution API methods

- get_live_positions() for position queries
- get_executed_trades() for trade history
- record_trade() for trade persistence
- get_portfolio_value() for portfolio metrics
- Symbol, date, and limit filtering support"
```

---

### Task 10: Documentation and Migration Guide

**Files:**
- Create: `docs/operations/tier4-trading-setup.md`
- Create: `docs/operations/ibkr-setup-guide.md`
- Modify: `docs/plans/Project-Status.md`
- Modify: `CLAUDE.md`

**Step 1: Write IBKR setup guide**

Create `docs/operations/ibkr-setup-guide.md`:

```markdown
# Interactive Brokers Setup Guide

Guide for setting up Interactive Brokers (IBKR) paper trading for HRP.

## Prerequisites

- Interactive Brokers account (paper trading)
- TWS (Trader Workstation) or IB Gateway installed
- Python 3.11+ with ib_insync library

## Step 1: Create Paper Trading Account

1. Log into [IBKR Account Management](https://www.interactivebrokers.com)
2. Navigate to Settings → Paper Trading
3. Create paper trading account (username format: `DU123456`)
4. Note your paper trading credentials

## Step 2: Install TWS/IB Gateway

**Option A: TWS (Trader Workstation)** - Full GUI
- Download from [IBKR TWS Download](https://www.interactivebrokers.com/en/trading/tws.php)
- Install for your platform

**Option B: IB Gateway** - Headless (recommended for servers)
- Download from [IBKR Gateway Download](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
- Lighter weight, no GUI

## Step 3: Configure API Access

1. Launch TWS/Gateway and log in with paper trading credentials
2. Navigate to: **File → Global Configuration → API → Settings**
3. Enable API:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API (disable for live trading)
   - Socket port: `7497` (paper trading default)
   - Trusted IP: `127.0.0.1`
4. Click **OK** and restart TWS/Gateway

## Step 4: Configure HRP Environment

Add to `.env`:

```bash
# Interactive Brokers Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper trading port (4002 for live IB Gateway)
IBKR_CLIENT_ID=1  # Unique client ID (1-32)
IBKR_ACCOUNT=DU123456  # Your paper trading account
IBKR_PAPER_TRADING=true
```

## Step 5: Test Connection

```bash
# Test broker connection
python -c "
from hrp.execution.broker import IBKRBroker, BrokerConfig

config = BrokerConfig(
    host='127.0.0.1',
    port=7497,
    client_id=1,
    account='DU123456',
    paper_trading=True,
)

with IBKRBroker(config) as broker:
    print(f'Connected: {broker.is_connected()}')
"
```

Expected output: `Connected: True`

## Step 6: Verify Positions and Orders

```python
from hrp.execution.broker import IBKRBroker, BrokerConfig
from hrp.execution.positions import PositionTracker
from hrp.api.platform import PlatformAPI

api = PlatformAPI()
config = BrokerConfig(
    host='127.0.0.1',
    port=7497,
    client_id=1,
    account='DU123456',
    paper_trading=True,
)

with IBKRBroker(config) as broker:
    tracker = PositionTracker(broker, api)
    positions = tracker.sync_positions()
    print(f"Synced {len(positions)} positions")
```

## Troubleshooting

### Connection Refused

**Problem:** `ConnectionError: IBKR connection failed`

**Solutions:**
1. Verify TWS/Gateway is running
2. Check API is enabled in settings
3. Verify port number (7497 for paper, 7496 for live TWS)
4. Check firewall allows localhost connections

### Authentication Failed

**Problem:** `Authentication failed`

**Solutions:**
1. Verify paper trading credentials
2. Log in manually to TWS/Gateway first
3. Check account number matches environment variable

### API Not Enabled

**Problem:** `API connection rejected`

**Solutions:**
1. File → Global Configuration → API → Settings
2. Enable "Enable ActiveX and Socket Clients"
3. Restart TWS/Gateway

### Trusted IP

**Problem:** `Connection rejected from untrusted IP`

**Solutions:**
1. Add `127.0.0.1` to Trusted IPs in API settings
2. Or disable IP restriction (not recommended for production)

## Production Considerations

### Live Trading (DO NOT enable without review)

To switch to live trading:

1. **Update environment:**
   ```bash
   IBKR_PORT=7496  # Live TWS
   IBKR_ACCOUNT=U123456  # Live account (not DU)
   IBKR_PAPER_TRADING=false
   ```

2. **Disable Read-Only API** in TWS settings

3. **Enable live trader job:**
   ```bash
   mv launchd/com.hrp.live-trader.plist.disabled launchd/com.hrp.live-trader.plist
   scripts/manage_launchd.sh reload
   ```

### Security Checklist

- [ ] API credentials stored in `.env` (not committed)
- [ ] TWS/Gateway protected with strong password
- [ ] Firewall restricts API access to localhost only
- [ ] Read-Only API enabled initially
- [ ] Test all operations in paper trading first
- [ ] Position limits configured correctly
- [ ] Drift monitoring enabled before live trading
- [ ] Emergency stop mechanism tested

## Resources

- [IBKR API Documentation](https://interactivebrokers.github.io/tws-api/)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)
- [TWS API Reference](https://www.interactivebrokers.com/en/software/api/api.htm)
```

**Step 2: Write Tier 4 setup guide**

Create `docs/operations/tier4-trading-setup.md`:

```markdown
# Tier 4: Trading Setup Guide

Complete setup guide for enabling live/paper trading in HRP.

## Overview

Tier 4 adds live trading execution capabilities:
- Daily prediction generation for deployed models
- Signal-to-order conversion with risk limits
- IBKR broker integration for order execution
- Position tracking and P&L monitoring
- Model drift detection with automatic rollback

## Prerequisites

- Tier 1-3 complete (data, research, ML, ops)
- At least one deployed strategy (status = 'deployed')
- Interactive Brokers paper trading account
- TWS/IB Gateway installed and configured

See `docs/operations/ibkr-setup-guide.md` for IBKR setup.

## Database Migration

**Run schema updates:**

```bash
# Apply new tables
python -c "
from hrp.data.db import get_db

conn = get_db()
conn.execute(open('hrp/data/schema.sql').read())
conn.commit()
print('Schema updated')
"
```

**Verify tables:**

```bash
python -c "
from hrp.data.db import get_db

conn = get_db(read_only=True)
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print('Tables:', [t[0] for t in tables])
"
```

Expected tables: `executed_trades`, `live_positions`

## Job Configuration

### 1. Daily Predictions

**Schedule:** Daily at 6:15 PM ET (after feature computation)

```bash
# Install job
scripts/manage_launchd.sh install

# Verify
launchctl list | grep com.hrp.predictions

# Test manually
python -m hrp.agents.run_job --job predictions --dry-run
python -m hrp.agents.run_job --job predictions
```

**Check logs:**

```bash
tail -f ~/hrp-data/logs/predictions.log
```

### 2. Drift Monitoring

**Schedule:** Daily at 7:00 PM ET (after predictions)

```bash
# Test drift check
python -m hrp.agents.run_job --job drift-monitor --dry-run

# Enable auto-rollback (optional, not recommended initially)
# Edit launchd/com.hrp.drift-monitor.plist, add:
# <string>--auto-rollback</string>

# Reload
scripts/manage_launchd.sh reload
```

### 3. Live Trading (DISABLED by default)

**IMPORTANT:** Live trader is disabled by default for safety.

**Enable manually:**

```bash
# Rename plist to enable
mv launchd/com.hrp.live-trader.plist.disabled launchd/com.hrp.live-trader.plist

# Test in dry-run mode first
python -m hrp.agents.run_job --job live-trader --trading-dry-run

# When ready, edit plist and remove --trading-dry-run flag
# Then reload
scripts/manage_launchd.sh reload
```

## Verification Checklist

### Pre-Trading

- [ ] IBKR TWS/Gateway running and connected
- [ ] Broker connection test passes
- [ ] At least one strategy deployed
- [ ] Daily predictions job runs successfully
- [ ] Drift monitoring job runs successfully
- [ ] Trading dashboard page displays correctly

### Trading Enabled

- [ ] Live trader runs in dry-run mode without errors
- [ ] Orders generated match expected signals
- [ ] Position limits enforced (max 20 positions, 10% each)
- [ ] Minimum order value check works ($100)
- [ ] Drift monitoring detects test drift
- [ ] Dashboard shows positions and trades

### Production Ready

- [ ] Tested in paper trading for 1+ week
- [ ] No drift detected on deployed models
- [ ] Order execution matches backtest costs
- [ ] Position reconciliation works correctly
- [ ] Rollback mechanism tested
- [ ] Emergency stop procedure documented

## Monitoring

### Dashboard

Access trading dashboard:

```bash
streamlit run hrp/dashboard/app.py
```

Navigate to **Trading** page:
- Portfolio overview (value, P&L, positions)
- Current positions table
- Recent trades history
- Model performance and drift status

### Logs

Monitor execution logs:

```bash
# Predictions
tail -f ~/hrp-data/logs/predictions.log

# Drift monitoring
tail -f ~/hrp-data/logs/drift-monitor.log

# Live trading
tail -f ~/hrp-data/logs/live-trader.log
```

### Alerts

Email notifications configured via Resend (see `.env`):
- Job failures
- Drift detection
- Order execution errors

## Emergency Procedures

### Stop All Trading

```bash
# Unload live trader job
launchctl unload ~/Library/LaunchAgents/com.hrp.live-trader.plist

# Or kill all HRP jobs
scripts/manage_launchd.sh uninstall
```

### Rollback Strategy

```python
from hrp.api.platform import PlatformAPI

api = PlatformAPI()

# Manual rollback
api.rollback_deployment(
    model_name="momentum_v1",
    to_version=None,  # Previous version
    actor="user",
    reason="Manual emergency rollback"
)
```

### Close All Positions

Manually close positions via TWS/Gateway, then sync:

```python
from hrp.execution.broker import IBKRBroker, BrokerConfig
from hrp.execution.positions import PositionTracker
from hrp.api.platform import PlatformAPI

api = PlatformAPI()
config = BrokerConfig(...)  # From env

with IBKRBroker(config) as broker:
    tracker = PositionTracker(broker, api)
    positions = tracker.sync_positions()
    tracker.persist_positions()
```

## Troubleshooting

### No Predictions Generated

**Check:**
1. Deployed strategies exist: `api.get_deployed_strategies()`
2. Model has production version in MLflow
3. Universe has symbols: `api.get_universe()`
4. Feature data up to date

### Orders Not Submitted

**Check:**
1. Broker connection: Test connection script
2. TWS/Gateway running
3. API enabled in TWS settings
4. Portfolio value > 0
5. Signals generated (check logs)

### Drift False Positives

**Solutions:**
1. Increase drift threshold in job config
2. Collect more baseline data
3. Disable auto-rollback
4. Manual drift investigation

## Next Steps

1. Run in paper trading for 1-2 weeks
2. Compare live execution to backtest performance
3. Monitor slippage and commissions
4. Adjust position sizing if needed
5. Enable auto-rollback once drift monitoring validated
6. Document any edge cases or issues

## Support

- **IBKR Issues:** See `docs/operations/ibkr-setup-guide.md`
- **Job Failures:** Check `~/hrp-data/logs/` for error details
- **API Errors:** Enable debug logging in `.env`: `LOG_LEVEL=DEBUG`
```

**Step 3: Update Project Status**

Modify `docs/plans/Project-Status.md`:

```markdown
## Tier 4: Trading (Complete)

### Live Execution

- IBKR broker integration (ib_insync)
- Order management system (market and limit orders)
- Position tracking and synchronization
- Signal-to-order conversion with risk limits
- Daily prediction job (scheduled)
- Live trading agent (disabled by default)
- Model drift monitoring with auto-rollback
- Trading dashboard page

### Database

- `executed_trades` table for trade history
- `live_positions` table for broker position sync
- Broker order ID tracking and reconciliation

### Jobs

| Job | Schedule | Purpose |
|-----|----------|---------|
| Predictions | Daily 6:15 PM | Generate predictions for deployed models |
| Drift Monitor | Daily 7:00 PM | Check for model drift, optional rollback |
| Live Trader | Daily 6:30 PM | Execute trades (DISABLED by default) |

### API Methods

- `get_live_positions()` - Query current positions
- `get_executed_trades()` - Trade history
- `record_trade()` - Persist trade execution
- `get_portfolio_value()` - Portfolio metrics

### Safety Features

- Dry-run mode for live trader
- Position limits (max 20 positions, 10% each)
- Minimum order value ($100)
- Drift monitoring before execution
- Manual enable required for live trading
- Paper trading default configuration
```

**Step 4: Update CLAUDE.md**

Modify `CLAUDE.md`:

```markdown
## Services

| Service | Command | Port |
|---------|---------|------|
| Dashboard | `streamlit run hrp/dashboard/app.py` | 8501 |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///~/hrp-data/mlflow/mlflow.db` | 5000 |
| Ops Server | `python -m hrp.ops` | 8080 |
| IBKR TWS | Launch manually | - |
| Single job | `python -m hrp.agents.run_job --job prices` | - |
| Scheduler | `python -m hrp.agents.run_scheduler` | - |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| ... existing vars ...
| `IBKR_HOST` | IBKR host (default: `127.0.0.1`) | For trading |
| `IBKR_PORT` | IBKR port (7497 paper, 7496 live) | For trading |
| `IBKR_CLIENT_ID` | IBKR client ID (default: `1`) | For trading |
| `IBKR_ACCOUNT` | IBKR account (DU for paper, U for live) | For trading |
| `IBKR_PAPER_TRADING` | Enable paper trading (default: `true`) | For trading |

## Key Modules

| Module | Purpose |
|--------|---------|
| ... existing modules ...
| `hrp/execution/broker.py` | IBKR broker connection manager |
| `hrp/execution/orders.py` | Order management and submission |
| `hrp/execution/positions.py` | Position tracking and sync |
| `hrp/execution/signal_converter.py` | Signal-to-order conversion with risk limits |
| `hrp/agents/prediction_job.py` | Daily prediction generation |
| `hrp/agents/live_trader.py` | Live trading execution (disabled by default) |
| `hrp/agents/drift_monitor_job.py` | Model drift monitoring and rollback |
```

**Step 5: Commit**

```bash
git add docs/operations/tier4-trading-setup.md docs/operations/ibkr-setup-guide.md docs/plans/Project-Status.md CLAUDE.md
git commit -m "docs: add Tier 4 trading setup and configuration guides

- IBKR setup guide with troubleshooting
- Tier 4 setup guide with verification checklist
- Emergency procedures documentation
- Updated Project Status to mark Tier 4 complete
- Added trading environment variables to CLAUDE.md"
```

---

## Verification and Testing

### End-to-End Test Procedure

**Step 1: Verify all tests pass**

```bash
pytest tests/test_execution/ -v
pytest tests/test_agents/test_prediction_job.py -v
pytest tests/test_agents/test_live_trader.py -v
pytest tests/test_agents/test_drift_monitor_job.py -v
pytest tests/test_api/test_trading_api.py -v
```

**Step 2: Database schema**

```bash
python -c "
from hrp.data.db import get_db
conn = get_db(read_only=True)
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name IN ('executed_trades', 'live_positions')\").fetchall()
print('Trading tables:', [t[0] for t in tables])
"
```

Expected output: `Trading tables: ['executed_trades', 'live_positions']`

**Step 3: Broker connection test**

```bash
# Ensure TWS/Gateway running first
python -c "
from hrp.execution.broker import IBKRBroker, BrokerConfig

config = BrokerConfig(
    host='127.0.0.1',
    port=7497,
    client_id=1,
    account='DU123456',
    paper_trading=True,
)

with IBKRBroker(config) as broker:
    print(f'Connected: {broker.is_connected()}')
"
```

**Step 4: Run prediction job (dry run)**

```bash
python -m hrp.agents.run_job --job predictions --dry-run
```

**Step 5: Run drift monitor (dry run)**

```bash
python -m hrp.agents.run_job --job drift-monitor --dry-run
```

**Step 6: Run live trader (trading dry run)**

```bash
python -m hrp.agents.run_job --job live-trader --trading-dry-run
```

**Step 7: Test dashboard**

```bash
streamlit run hrp/dashboard/app.py
```

Navigate to Trading page and verify all sections render.

**Step 8: Install scheduled jobs**

```bash
scripts/manage_launchd.sh reload
launchctl list | grep com.hrp
```

Verify jobs installed:
- `com.hrp.predictions`
- `com.hrp.drift-monitor`

---

## Success Criteria

- [ ] All tests pass (execution, agents, API)
- [ ] Database tables created
- [ ] Broker connection works
- [ ] Prediction job generates predictions
- [ ] Drift monitor detects drift
- [ ] Live trader converts signals to orders (dry run)
- [ ] Trading dashboard displays correctly
- [ ] Jobs scheduled via launchd
- [ ] Documentation complete

---

## Notes

- Live trading disabled by default (safety)
- Paper trading recommended for initial testing
- Auto-rollback disabled initially (manual review recommended)
- All operations logged to lineage for audit trail
- Emergency stop procedures documented

