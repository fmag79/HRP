# Design: TASK-010 — Robinhood Order Execution Engine

## Overview

Replace the Interactive Brokers (IBKR) execution layer with a Robinhood broker integration using the `robin_stocks` library. Build a complete order execution engine supporting market, limit, and stop-loss orders with status tracking, rate limiting, error handling, and integration with the existing VaR/CVaR risk management framework (TASK-006) and position sizing system.

## Why Robinhood?

The existing Tier 4 execution layer was designed for IBKR (`ib_insync`). This task adds Robinhood as an alternative broker, keeping the broker-agnostic architecture so either can be used. Robinhood's API (via `robin_stocks`) provides:
- Zero-commission stock trading
- Fractional share support
- Simple REST-based API (no persistent socket connection needed)
- All required order types (market, limit, stop-loss, stop-limit, trailing stop)

## Current State

### Existing Execution Layer (`hrp/execution/`)
- `broker.py` — `IBKRBroker`, `BrokerConfig` (IBKR-specific)
- `orders.py` — `Order`, `OrderManager`, `OrderType` (market/limit only), `OrderSide`, `OrderStatus`
- `signal_converter.py` — `SignalConverter`, `ConversionConfig` (broker-agnostic)
- `positions.py` — `Position`, `PositionTracker` (IBKR-specific)
- `__init__.py` — Exports all above

### Existing Risk Layer
- `hrp/risk/limits.py` — `RiskLimits`, `PreTradeValidator`, `LimitViolation` (pre-trade validation)
- `hrp/data/risk/var_calculator.py` — `VaRCalculator` with parametric, historical, Monte Carlo methods
- `hrp/data/risk/risk_config.py` — `VaRConfig`, `VaRMethod`, `Distribution`
- `hrp/data/features/risk_features.py` — 5 registered VaR/CVaR features
- `hrp/agents/risk_manager.py` — `RiskManager` agent (drawdown, concentration, correlation checks)

### Existing Trading Agent
- `hrp/agents/live_trader.py` — `LiveTradingAgent`, `TradingConfig` (uses IBKR broker)

### Dashboard
- `hrp/dashboard/pages/10_Trading.py` — Trading page (queries `live_positions`, `executed_trades`)

## Architecture Decision: Broker Abstraction

**Decision: Introduce a `BaseBroker` protocol/ABC** so both IBKR and Robinhood can be used interchangeably.

### Rationale
- The existing `IBKRBroker` is tightly coupled to `ib_insync`
- `OrderManager` currently imports IBKR-specific classes
- `PositionTracker` calls IBKR-specific methods
- A broker abstraction allows switching brokers via config, not code changes
- Follows the existing project principle: "Consistency — Follow existing patterns"

### Rejected Alternative
- Direct replacement of IBKR with Robinhood — rejected because IBKR code already works and the user may want both

## Proposed Changes

### Layer 1: Broker Abstraction

#### Modified Files
- `hrp/execution/broker.py` — Add `BaseBroker` protocol, keep `IBKRBroker` unchanged

#### New Files
- `hrp/execution/robinhood_broker.py` — `RobinhoodBroker` implementing `BaseBroker`
- `hrp/execution/robinhood_auth.py` — Auth, session management, MFA
- `hrp/execution/rate_limiter.py` — Token-bucket rate limiter for API calls

### Layer 2: Order Types Extension

#### Modified Files
- `hrp/execution/orders.py` — Add `STOP_LOSS`, `STOP_LIMIT`, `TRAILING_STOP` to `OrderType` enum; add `stop_price`, `trail_amount`, `trail_type` fields to `Order` dataclass

### Layer 3: Risk-Integrated Position Sizing

#### New Files
- `hrp/execution/position_sizer.py` — VaR/CVaR-aware position sizing

#### Modified Files
- `hrp/execution/signal_converter.py` — Integrate position sizer into order generation

### Layer 4: Trading Agent Update

#### Modified Files
- `hrp/agents/live_trader.py` — Support broker selection via config

### Layer 5: Tests

#### New Files
- `tests/test_execution/test_robinhood_broker.py`
- `tests/test_execution/test_robinhood_auth.py`
- `tests/test_execution/test_rate_limiter.py`
- `tests/test_execution/test_position_sizer.py`

#### Modified Files
- `tests/test_execution/test_orders.py` — Tests for new order types

---

## Detailed Component Design

### 1. Robinhood Authentication (`hrp/execution/robinhood_auth.py`)

```python
@dataclass
class RobinhoodAuthConfig:
    """Robinhood authentication configuration."""
    username: str                    # Email
    password: str                    # Password
    totp_secret: str | None = None   # Base32 TOTP secret for automated MFA
    session_expiry: int = 86400      # 24 hours default
    pickle_name: str = ""            # Session file suffix
    device_token: str | None = None  # Persistent device token

class RobinhoodSession:
    """Manages Robinhood authentication lifecycle."""

    def __init__(self, config: RobinhoodAuthConfig) -> None: ...

    def login(self) -> bool:
        """Authenticate with Robinhood. Uses pyotp for MFA if totp_secret provided."""

    def logout(self) -> None:
        """End session and clean up."""

    def is_authenticated(self) -> bool:
        """Check if session is valid."""

    def ensure_authenticated(self) -> None:
        """Re-authenticate if session expired. Called before each API operation."""
```

**Key Decisions:**
- MFA handled via `pyotp` TOTP (no interactive prompts in production)
- Session cached via `robin_stocks` pickle mechanism
- `ensure_authenticated()` called by broker before every operation
- TOTP secret stored as env var `ROBINHOOD_TOTP_SECRET` (never in code/config files)

### 2. Rate Limiter (`hrp/execution/rate_limiter.py`)

```python
@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    max_requests_per_window: int = 5    # Max requests per window
    window_seconds: float = 15.0         # Window duration
    order_cooldown: float = 2.0          # Min seconds between orders
    backoff_base: float = 2.0            # Exponential backoff base
    max_backoff: float = 60.0            # Max backoff seconds
    max_retries: int = 3                 # Max retry attempts

class RateLimiter:
    """Token-bucket rate limiter for Robinhood API."""

    def __init__(self, config: RateLimitConfig | None = None) -> None: ...

    def acquire(self) -> None:
        """Block until a request slot is available."""

    def handle_throttle(self, response_text: str) -> float:
        """Parse throttle response, return wait seconds."""

    def reset(self) -> None:
        """Reset rate limiter state."""
```

**Key Decisions:**
- Conservative default: 5 requests per 15 seconds (within observed ~8/15s limit)
- Order-specific cooldown of 2 seconds between consecutive orders
- Parse "Expected available in N seconds" from throttle responses
- Exponential backoff on repeated throttling

### 3. Robinhood Broker (`hrp/execution/robinhood_broker.py`)

```python
@dataclass
class RobinhoodConfig:
    """Robinhood broker configuration."""
    username: str
    password: str
    totp_secret: str | None = None
    account_number: str | None = None  # Multi-account support
    paper_trading: bool = True          # Safety flag (logs but blocks real orders)
    rate_limit: RateLimitConfig | None = None

class RobinhoodBroker(BaseBroker):
    """Robinhood broker implementation."""

    def __init__(self, config: RobinhoodConfig) -> None: ...

    # BaseBroker interface
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

    # Order operations
    def place_order(self, order: Order) -> Order: ...
    def cancel_order(self, broker_order_id: str) -> bool: ...
    def get_order_status(self, broker_order_id: str) -> OrderStatus: ...
    def get_open_orders(self) -> list[dict]: ...

    # Position operations
    def get_positions(self) -> list[dict]: ...
    def get_account_info(self) -> dict: ...
    def get_portfolio_value(self) -> Decimal: ...

    # Market data
    def get_quote(self, symbol: str) -> Decimal: ...
    def get_quotes(self, symbols: list[str]) -> dict[str, Decimal]: ...

    # Context manager
    def __enter__(self) -> "RobinhoodBroker": ...
    def __exit__(self, *args) -> None: ...
```

**Order Type Mapping:**

| HRP OrderType | robin_stocks Function |
|---|---|
| `MARKET` | `order_buy_market()` / `order_sell_market()` |
| `LIMIT` | `order_buy_limit()` / `order_sell_limit()` |
| `STOP_LOSS` | `order_buy_stop_loss()` / `order_sell_stop_loss()` |
| `STOP_LIMIT` | `order_buy_stop_limit()` / `order_sell_stop_limit()` |
| `TRAILING_STOP` | `order_buy_trailing_stop()` / `order_sell_trailing_stop()` |

**Order Status Mapping:**

| Robinhood State | HRP OrderStatus |
|---|---|
| `queued`, `unconfirmed`, `confirmed` | `SUBMITTED` |
| `partially_filled` | `PARTIALLY_FILLED` |
| `filled` | `FILLED` |
| `cancelled` | `CANCELLED` |
| `rejected`, `failed` | `REJECTED` |

**Paper Trading Safety:**
- When `paper_trading=True`, `place_order()` logs the order but does NOT call `robin_stocks`
- Returns a synthetic order with `SUBMITTED` status and a generated order ID
- This mirrors IBKR's paper trading but implemented at our layer (Robinhood has no paper trading)

### 4. Broker Abstraction (`hrp/execution/broker.py` — Modified)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class BaseBroker(Protocol):
    """Protocol for broker implementations."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def place_order(self, order: "Order") -> "Order": ...
    def cancel_order(self, broker_order_id: str) -> bool: ...
    def get_order_status(self, broker_order_id: str) -> "OrderStatus": ...
    def get_positions(self) -> list[dict]: ...
    def get_portfolio_value(self) -> Decimal: ...
    def get_quote(self, symbol: str) -> Decimal: ...
    def get_quotes(self, symbols: list[str]) -> dict[str, Decimal]: ...

    def __enter__(self) -> "BaseBroker": ...
    def __exit__(self, *args) -> None: ...
```

`IBKRBroker` will be updated to implement `BaseBroker` (backward-compatible addition of missing methods).

### 5. Extended Order Types (`hrp/execution/orders.py` — Modified)

```python
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"           # NEW
    STOP_LIMIT = "stop_limit"         # NEW
    TRAILING_STOP = "trailing_stop"   # NEW

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None          # NEW: for stop-loss/stop-limit
    trail_amount: float | None = None          # NEW: for trailing stop
    trail_type: str = "percentage"             # NEW: "percentage" or "amount"
    # ... existing fields unchanged
```

**Validation Rules:**
- `STOP_LOSS` requires `stop_price`
- `STOP_LIMIT` requires both `stop_price` and `limit_price`
- `TRAILING_STOP` requires `trail_amount`
- `trail_type` must be `"percentage"` or `"amount"`

### 6. VaR/CVaR Position Sizer (`hrp/execution/position_sizer.py`)

```python
@dataclass
class PositionSizingConfig:
    """VaR-aware position sizing configuration."""
    portfolio_value: Decimal
    max_portfolio_var_pct: float = 0.02     # 2% daily portfolio VaR limit
    max_position_var_pct: float = 0.005     # 0.5% daily VaR per position
    confidence_level: float = 0.95
    use_cvar: bool = False                   # Use CVaR for more conservative sizing
    min_position_value: Decimal = Decimal("100.00")
    max_position_pct: float = 0.10           # Hard cap: 10% per position
    lookback_days: int = 252                 # VaR calculation lookback

class PositionSizer:
    """VaR/CVaR-aware position sizing engine."""

    def __init__(self, config: PositionSizingConfig, api: PlatformAPI) -> None: ...

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_portfolio_var: float | None = None,
    ) -> int:
        """Calculate position size constrained by VaR limits.

        Algorithm:
        1. Get symbol's historical VaR from feature store
        2. Calculate max $ allocation: max_position_var / symbol_var
        3. Apply signal-strength scaling (stronger signal → larger position)
        4. Apply hard caps (max_position_pct, portfolio-level VaR budget)
        5. Convert to shares using current price
        """

    def calculate_portfolio_var_budget(
        self,
        current_positions: dict[str, int],
    ) -> float:
        """Calculate remaining VaR budget for new positions.

        Returns the VaR capacity available (portfolio_limit - current_var).
        """

    def size_all_positions(
        self,
        signals: pd.DataFrame,
        current_positions: dict[str, int],
    ) -> dict[str, int]:
        """Size all positions from signals, respecting portfolio VaR limit.

        Allocates VaR budget across signals, strongest first.
        """
```

**Integration with Existing VaR:**
- Reads `var_95_1d` and `cvar_95_1d` features from the feature store via `PlatformAPI`
- Falls back to parametric VaR calculation if features aren't available for a symbol
- Portfolio-level VaR aggregation uses simple sum (conservative — ignores diversification benefit)

### 7. Updated Signal Converter (`hrp/execution/signal_converter.py` — Modified)

Add optional `PositionSizer` integration:

```python
class SignalConverter:
    def __init__(
        self,
        config: ConversionConfig,
        position_sizer: PositionSizer | None = None,  # NEW
    ) -> None: ...

    def signals_to_orders(self, signals, method, current_prices, use_var_sizing=False):
        """If use_var_sizing and position_sizer available, use VaR-based sizing."""
```

When `position_sizer` is provided and `use_var_sizing=True`, position sizes come from the VaR engine rather than equal-weight allocation. Falls back to existing logic if sizer is not configured.

### 8. Updated Trading Agent (`hrp/agents/live_trader.py` — Modified)

```python
@dataclass
class TradingConfig:
    # ... existing fields ...
    broker_type: str = "robinhood"       # NEW: "robinhood" or "ibkr"
    use_var_sizing: bool = True          # NEW: enable VaR-based position sizing
    stop_loss_pct: float | None = None   # NEW: auto-attach stop-loss (e.g., 0.05 = 5%)
```

The agent creates the appropriate broker based on `broker_type` and optionally attaches stop-loss orders to every buy.

---

## Data Flow

```
1. LiveTradingAgent.execute()
   ├── Get deployed strategies & predictions
   ├── Combine predictions into signals DataFrame
   │
2. PositionSizer.size_all_positions(signals, current_positions)
   ├── Read VaR features from feature store (var_95_1d per symbol)
   ├── Calculate portfolio VaR budget (limit - current exposure)
   ├── Allocate budget across signals (strongest first)
   └── Return {symbol: quantity} dict
   │
3. SignalConverter.rebalance_to_orders(current, target, prices)
   ├── Generate BUY orders for new/increased positions
   ├── Generate SELL orders for exits/decreases
   └── Optionally attach STOP_LOSS orders to buys
   │
4. PreTradeValidator.validate(orders, portfolio)  [existing]
   ├── Position limit checks
   ├── Sector concentration checks
   └── Turnover limit checks
   │
5. RobinhoodBroker.place_order(order)  [for each validated order]
   ├── RateLimiter.acquire()  → wait if needed
   ├── RobinhoodSession.ensure_authenticated()  → re-login if expired
   ├── robin_stocks.order_*()  → submit to Robinhood
   ├── Map response to Order with status + broker_order_id
   └── Log to lineage
   │
6. OrderStatusTracker.poll_until_settled(orders)
   ├── Poll open orders every 10s
   ├── Update Order status from Robinhood state
   ├── Record fills to executed_trades table
   └── Alert on rejections/failures
```

## Environment Variables

```bash
# Robinhood Configuration
ROBINHOOD_USERNAME=user@email.com
ROBINHOOD_PASSWORD=<password>            # Store in secrets manager
ROBINHOOD_TOTP_SECRET=<base32_secret>    # For automated MFA
ROBINHOOD_ACCOUNT_NUMBER=               # Optional for multi-account
ROBINHOOD_PAPER_TRADING=true            # Safety: true by default

# Position Sizing
HRP_MAX_PORTFOLIO_VAR_PCT=0.02          # 2% daily portfolio VaR limit
HRP_MAX_POSITION_VAR_PCT=0.005          # 0.5% per-position VaR limit
HRP_USE_VAR_SIZING=true                 # Enable VaR-based sizing
HRP_AUTO_STOP_LOSS_PCT=0.05             # Auto stop-loss at 5%

# Rate Limiting
ROBINHOOD_RATE_LIMIT_REQUESTS=5         # Max requests per window
ROBINHOOD_RATE_LIMIT_WINDOW=15          # Window in seconds
ROBINHOOD_ORDER_COOLDOWN=2.0            # Seconds between orders
```

## Implementation Tasks

### Phase 1: Core Robinhood Broker (Complexity: Medium)

- [ ] **Task 1.1:** Create `hrp/execution/rate_limiter.py` — Token-bucket rate limiter with throttle parsing (low)
- [ ] **Task 1.2:** Create `hrp/execution/robinhood_auth.py` — Auth session with pyotp MFA, auto-refresh (medium)
- [ ] **Task 1.3:** Add `BaseBroker` protocol to `hrp/execution/broker.py` (low)
- [ ] **Task 1.4:** Create `hrp/execution/robinhood_broker.py` — Full broker with order placement, status, positions (high)
- [ ] **Task 1.5:** Write tests for rate limiter, auth, and broker (medium)

### Phase 2: Extended Order Types (Complexity: Low)

- [ ] **Task 2.1:** Add `STOP_LOSS`, `STOP_LIMIT`, `TRAILING_STOP` to `OrderType` enum and `Order` dataclass with validation (low)
- [ ] **Task 2.2:** Update `OrderManager.submit_order()` to handle new types (low)
- [ ] **Task 2.3:** Write tests for new order types (low)

### Phase 3: VaR-Integrated Position Sizing (Complexity: Medium)

- [ ] **Task 3.1:** Create `hrp/execution/position_sizer.py` — VaR-aware sizing with portfolio budget (medium)
- [ ] **Task 3.2:** Integrate `PositionSizer` into `SignalConverter` (low)
- [ ] **Task 3.3:** Write tests for position sizer with mocked VaR features (medium)

### Phase 4: Trading Agent Integration (Complexity: Medium)

- [ ] **Task 4.1:** Update `TradingConfig` with `broker_type`, `use_var_sizing`, `stop_loss_pct` (low)
- [ ] **Task 4.2:** Update `LiveTradingAgent.execute()` for broker selection and VaR sizing (medium)
- [ ] **Task 4.3:** Add auto-stop-loss order generation after buy orders (low)
- [ ] **Task 4.4:** Add order status polling loop (medium)
- [ ] **Task 4.5:** Write integration tests with mocked Robinhood API (medium)

### Phase 5: Polish & Safety (Complexity: Low)

- [ ] **Task 5.1:** Update `hrp/execution/__init__.py` with new exports (low)
- [ ] **Task 5.2:** Update `.env.example` with Robinhood vars (low)
- [ ] **Task 5.3:** Add `robin-stocks` and `pyotp` to `pyproject.toml` dependencies (low)
- [ ] **Task 5.4:** Run full test suite, ensure no regressions (low)

## Testing Strategy

### Unit Tests
- `test_rate_limiter.py` — Token bucket logic, throttle parsing, backoff
- `test_robinhood_auth.py` — Login flow, MFA with pyotp, session refresh (all mocked)
- `test_robinhood_broker.py` — Order placement, status mapping, position retrieval (mocked robin_stocks)
- `test_position_sizer.py` — VaR budget allocation, position sizing, edge cases
- `test_orders.py` — New order type validation (stop-loss, stop-limit, trailing stop)

### Integration Tests
- End-to-end: signals → position sizing → order generation → broker submission (all mocked)
- Paper trading mode: verify no real API calls when `paper_trading=True`
- VaR integration: verify position sizes respect VaR budget

### Key Invariants to Test
- `paper_trading=True` NEVER calls `robin_stocks` order functions
- Rate limiter never exceeds configured limits
- Position sizes never exceed `max_position_pct` regardless of VaR sizing
- Portfolio VaR never exceeds `max_portfolio_var_pct` after new orders
- Stop-loss orders always paired with buy orders when configured
- All orders logged to lineage

## Edge Cases & Error Handling

### Authentication Failures
- **Expired session:** `ensure_authenticated()` re-logins before each operation
- **Invalid credentials:** Raise `AuthenticationError`, log, alert via email
- **MFA failure:** Retry once, then fail with clear error message
- **Device challenge:** Use `respond_to_challenge()` if SMS/email code needed

### Order Failures
- **Throttled:** Parse wait time, sleep, retry (max 3 retries)
- **Rejected by Robinhood:** Log rejection reason, mark order `REJECTED`, continue with remaining orders
- **None response (silent failure):** Treat as failure, log warning, retry once
- **Partial fill:** Track `cumulative_quantity`, allow completion or cancellation
- **Insufficient buying power:** Log error, skip order, continue with others

### Position Sizing Edge Cases
- **No VaR data for symbol:** Fall back to equal-weight sizing
- **VaR is zero/NaN:** Use conservative fallback (1% of portfolio per position)
- **Entire VaR budget consumed:** Log warning, skip remaining orders
- **Signal for symbol already in portfolio:** Adjust for existing exposure

### Rate Limiting Edge Cases
- **Burst of status checks:** Rate limiter covers all API calls, not just orders
- **Multiple concurrent sessions:** Each `RobinhoodBroker` instance has its own rate limiter
- **Clock skew:** Use monotonic time for rate limiting

## Security Considerations

1. **Credentials:** `ROBINHOOD_PASSWORD` and `ROBINHOOD_TOTP_SECRET` MUST be in environment variables or a secrets manager — never in code, config files, or git
2. **Pickle files:** `robin_stocks` stores auth tokens in `~/.tokens/` — ensure this directory has restricted permissions (700)
3. **Paper trading default:** `ROBINHOOD_PAPER_TRADING=true` by default — requires explicit opt-in for live trading
4. **Logging:** NEVER log passwords, TOTP secrets, or auth tokens. Use `hrp.utils.log_filter.filter_secrets()` for all Robinhood-related logs
5. **Order confirmation:** Live orders are logged to lineage BEFORE and AFTER submission for complete audit trail

## Open Questions

1. **Fractional shares:** Should we support fractional share orders? The `robin_stocks` library supports it, but it adds complexity (market orders only for fractional). **Recommendation:** Skip for v1, add in v2.
2. **Crypto trading:** `robin_stocks` supports crypto. Out of scope for this task but architecture supports future extension.
3. **Extended hours:** Should we support pre/post-market trading? **Recommendation:** No for v1 — stick to regular hours for simplicity.
4. **Multi-account:** `robin_stocks` supports multiple accounts. **Recommendation:** Support via `ROBINHOOD_ACCOUNT_NUMBER` env var but default to single account.
