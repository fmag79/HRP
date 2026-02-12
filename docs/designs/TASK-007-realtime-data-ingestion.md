# Design: TASK-007 — Real-time Data Ingestion via Polygon.io WebSocket

## Overview

Add real-time / intraday data ingestion to HRP via Polygon.io's WebSocket streaming API. This enables intraday price monitoring, real-time feature computation, and live signal generation — bridging the gap between the current daily-batch pipeline and the Tier 4 live trading system.

**Why:** The existing pipeline runs post-market (6 PM ET) and operates on daily bars only. Real-time data enables: (1) intraday risk monitoring for deployed strategies, (2) live signal generation for the trading execution layer, (3) real-time feature computation for latency-sensitive models, and (4) market-hours dashboard monitoring.

## Current State

### What Exists
- **Daily ingestion** via `hrp/data/ingestion/prices.py` — batch fetch after market close
- **Polygon REST client** at `hrp/data/sources/polygon_source.py` — rate-limited, retry-aware
- **DuckDB storage** with `prices` table keyed on `(symbol, date)` — daily granularity only
- **APScheduler** at `hrp/agents/scheduler.py` — cron-based jobs, no continuous services
- **Feature computation** at `hrp/data/features/computation.py` — 45 features, daily bars only
- **No WebSocket infrastructure** — no streaming client, no intraday tables, no real-time jobs

### What Polygon.io Provides (WebSocket)
- **Channels:** `T.*` (trades), `Q.*` (quotes), `A.*` (per-second aggs), `AM.*` (per-minute aggs)
- **Python SDK:** `polygon.WebSocketClient` — blocking `ws.run(handler)`, subscribe/unsubscribe API
- **Market:** `Market.Stocks` for US equities
- **Existing dependency:** `polygon-api-client >= 1.12.0` already in `requirements.txt`

## Proposed Changes

### Architecture Decision: Layered Approach

We use a **3-layer architecture** mirroring HRP's existing patterns:

```
┌─────────────────────────────────────────────────┐
│ Layer 1: WebSocket Client (connection manager)  │
│   polygon_websocket.py — connect, reconnect,    │
│   subscribe, heartbeat, message dispatch        │
├─────────────────────────────────────────────────┤
│ Layer 2: Intraday Ingestion Service             │
│   intraday_ingestion.py — buffer, batch-write,  │
│   aggregate minute bars, upsert to DuckDB       │
├─────────────────────────────────────────────────┤
│ Layer 3: Intraday Feature Engine (Phase 2)      │
│   intraday_features.py — real-time feature      │
│   computation from streaming bars               │
└─────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Subscribe to `AM.*` (per-minute aggregates) primarily** — not raw trades.
   - *Reason:* HRP is a daily-timeframe systematic platform. Minute bars are the right granularity for intraday monitoring without the complexity of tick-level processing. Raw trades at ~500K msgs/day for S&P 500 would overwhelm DuckDB's write path.
   - *Supplementary:* Subscribe to `T.*` for a watchlist subset (≤20 symbols) for real-time last-price.

2. **New `intraday_bars` table** — separate from `prices`.
   - *Reason:* Different granularity (timestamp vs date), different retention policy (7-day hot, 30-day warm), different query patterns. Mixing into `prices` would break the `(symbol, date)` primary key and confuse the daily pipeline.

3. **Buffered writes** — accumulate bars in memory, flush to DuckDB every N seconds.
   - *Reason:* DuckDB is OLAP-optimized, not designed for row-by-row inserts. Batch inserts of 50-100 rows are 10-50x faster than individual inserts.

4. **Background thread, not separate process** — run WebSocket in a daemon thread alongside the scheduler.
   - *Reason:* Simpler deployment, shared connection pool, no IPC needed. The `IngestionScheduler` already uses `BackgroundScheduler` (threading). The WebSocket client blocks on its own thread anyway.

5. **Graceful degradation** — if WebSocket disconnects, fall back to REST polling at 1-minute intervals until reconnected.
   - *Reason:* Market hours data cannot have gaps. REST fallback (using existing `PolygonSource.get_daily_bars` with `timespan="minute"`) ensures continuity.

### Components Affected

| File | Change |
|------|--------|
| `hrp/data/schema.py` | Add `intraday_bars` + `intraday_features` tables, indexes |
| `hrp/data/sources/polygon_source.py` | Add `get_minute_bars()` REST method (fallback) |
| `hrp/agents/scheduler.py` | Add `setup_realtime_ingestion()` method |
| `hrp/agents/jobs.py` | Add `IntradayIngestionJob` class |
| `hrp/api/platform.py` | Add `get_intraday_prices()`, `get_intraday_features()` |
| `hrp/utils/config.py` | Add `realtime` config section |
| `requirements.txt` | No change needed (`polygon-api-client` already included) |

### New Components

| File | Purpose |
|------|---------|
| `hrp/data/sources/polygon_websocket.py` | WebSocket client wrapper with reconnection, heartbeat, subscription management |
| `hrp/data/ingestion/intraday.py` | Intraday bar buffering, batch upsert, minute-bar aggregation |
| `hrp/data/features/intraday_features.py` | Real-time feature computation from intraday bars (Phase 2) |
| `hrp/dashboard/pages/realtime.py` | Real-time monitoring dashboard page (Phase 3) |

### Data Flow

```
                    Market Hours (9:30 AM - 4:00 PM ET)
                    ===================================

 Polygon.io WS ──AM.*──> PolygonWebSocketClient
                              │
                              ▼
                    IntradayBarBuffer (in-memory)
                              │
                         every 10s
                              │
                              ▼
                    _batch_upsert_intraday()
                              │
                              ▼
                    DuckDB: intraday_bars table
                              │
                              ▼ (Phase 2)
                    IntradayFeatureEngine
                              │
                              ▼
                    DuckDB: intraday_features table
                              │
                              ▼ (Phase 3)
                    Dashboard: realtime.py page
```

```
                    After Hours / Disconnect
                    ========================

 APScheduler cron ──> close_realtime_session()
                              │
                              ▼
                    Flush remaining buffer
                    Disconnect WebSocket
                    Log session stats to ingestion_log
                    Compute EOD intraday summary → lineage
```

## Schema Design

### `intraday_bars` Table

```sql
CREATE TABLE IF NOT EXISTS intraday_bars (
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,    -- Minute-bar start time (UTC)
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT,
    vwap DECIMAL(12,4),              -- Polygon provides VWAP per bar
    trade_count INTEGER,             -- Number of trades in bar
    source VARCHAR DEFAULT 'polygon_ws',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp),
    CHECK (close > 0),
    CHECK (volume IS NULL OR volume >= 0),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
)
```

### `intraday_features` Table (Phase 2)

```sql
CREATE TABLE IF NOT EXISTS intraday_features (
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,    -- Aligned to minute boundary
    feature_name VARCHAR NOT NULL,
    value DECIMAL(24,6),
    version VARCHAR DEFAULT 'v1',
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp, feature_name, version),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
)
```

### Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_intraday_bars_symbol_ts
    ON intraday_bars(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_intraday_bars_ts
    ON intraday_bars(timestamp);
CREATE INDEX IF NOT EXISTS idx_intraday_features_symbol_ts
    ON intraday_features(symbol, timestamp, feature_name);
```

### Retention Policy

| Tier | Age | Action |
|------|-----|--------|
| HOT | 0-7 days | Full minute-bar resolution in `intraday_bars` |
| WARM | 7-30 days | Compress to 5-minute bars (optional Phase 3) |
| COLD | 30+ days | Delete. Daily bars in `prices` table provide historical coverage |

## Implementation Tasks

### Phase 1: WebSocket Client & Storage (12 tasks)

1. [ ] **Add `intraday_bars` table to `hrp/data/schema.py`** (low)
   - Add CREATE TABLE statement to `TABLES` dict
   - Add indexes to `INDEXES` list
   - Add migration function for existing DBs

2. [ ] **Create `hrp/data/sources/polygon_websocket.py`** (high)
   - `PolygonWebSocketClient` class
   - Constructor: API key from config, market=Stocks
   - `start(symbols, channels)` — connect + subscribe in daemon thread
   - `stop()` — graceful disconnect, flush buffer
   - `_handle_message(msgs)` — dispatch to registered callbacks
   - Automatic reconnection with exponential backoff (5s, 10s, 30s, 60s max)
   - Heartbeat monitoring (log warning if no messages for 30s during market hours)
   - Thread-safe message queue for cross-thread communication

3. [ ] **Create `IntradayBarBuffer` in `hrp/data/ingestion/intraday.py`** (medium)
   - Thread-safe buffer using `collections.deque` with `threading.Lock`
   - `add_bar(symbol, timestamp, o, h, l, c, v, vwap, trade_count)`
   - `flush() -> list[dict]` — drain buffer and return records
   - Max buffer size: 10,000 bars (safety valve — force flush if exceeded)

4. [ ] **Create `_batch_upsert_intraday()` in `hrp/data/ingestion/intraday.py`** (medium)
   - Follow existing `_upsert_prices()` pattern from `prices.py`
   - Temp table → INSERT OR REPLACE → drop temp
   - Accept list of dicts from buffer flush
   - Return row count inserted

5. [ ] **Create `IntradayIngestionService` in `hrp/data/ingestion/intraday.py`** (high)
   - Owns `PolygonWebSocketClient` + `IntradayBarBuffer`
   - `start(symbols)` — start WS, register flush timer (10s interval)
   - `stop()` — stop WS, final flush, log stats
   - `_on_minute_agg(msg)` — parse AM message, add to buffer
   - `_flush_timer()` — periodic buffer flush + DB write
   - `is_market_hours() -> bool` — check if currently 9:30 AM - 4:00 PM ET
   - Session stats: bars_received, bars_written, reconnect_count, last_message_at

6. [ ] **Add `get_minute_bars()` to `PolygonSource`** (low)
   - REST fallback for intraday data
   - Uses `client.get_aggs(timespan="minute")`
   - Same pattern as `get_daily_bars()` but returns minute-level data
   - Used during WS disconnection as gap-fill

7. [ ] **Add `realtime` config section to `hrp/utils/config.py`** (low)
   - `enabled: bool = False`
   - `symbols: list[str] | None = None` (None = top 50 by volume from universe)
   - `channels: list[str] = ["AM"]` (minute aggs by default)
   - `buffer_flush_interval_seconds: int = 10`
   - `max_buffer_size: int = 10000`
   - `reconnect_max_delay_seconds: int = 60`
   - `watchlist_symbols: list[str] = []` (for raw trade feed, ≤20)

8. [ ] **Add `IntradayIngestionJob` to `hrp/agents/jobs.py`** (medium)
   - Extends `IngestionJob` base class
   - `execute()` starts `IntradayIngestionService`, blocks until market close
   - Data requirements: `symbols` table must have rows
   - Logs session to `ingestion_log` with source_id = "intraday_ingestion"
   - Override `run()` to handle the long-running nature (no retry on market-hours service)

9. [ ] **Add `setup_realtime_ingestion()` to scheduler** (medium)
   - Schedule `IntradayIngestionJob` start at 9:25 AM ET (5 min before open)
   - Schedule stop at 4:05 PM ET (5 min after close)
   - Weekdays only (mon-fri)
   - Skip market holidays via `exchange_calendars` or hardcoded holiday list

10. [ ] **Add `get_intraday_prices()` to `PlatformAPI`** (low)
    - `get_intraday_prices(symbols, start_time, end_time) -> pd.DataFrame`
    - Query `intraday_bars` table
    - Follow existing `get_prices()` pattern

11. [ ] **Add lineage events for intraday ingestion** (low)
    - New event types: `intraday_session_start`, `intraday_session_end`
    - Log to lineage table with session stats (bars received, symbols tracked, etc.)

12. [ ] **Write unit tests for Phase 1** (medium)
    - `tests/data/sources/test_polygon_websocket.py`
      - Test connection lifecycle (start/stop)
      - Test message parsing for AM channel
      - Test reconnection logic (mock disconnect)
      - Test subscription management
    - `tests/data/ingestion/test_intraday.py`
      - Test `IntradayBarBuffer` thread safety
      - Test `_batch_upsert_intraday()` with DuckDB
      - Test `IntradayIngestionService` start/stop lifecycle
      - Test buffer flush mechanics
    - `tests/data/test_schema_intraday.py`
      - Test table creation and indexes

### Phase 2: Intraday Features (6 tasks)

13. [ ] **Add `intraday_features` table to schema** (low)
    - CREATE TABLE + indexes
    - Migration function

14. [ ] **Create `hrp/data/features/intraday_features.py`** (high)
    - `IntradayFeatureEngine` class
    - Compute subset of existing features at minute granularity:
      - `intraday_vwap` — running VWAP
      - `intraday_rsi_14` — 14-bar RSI on minute data
      - `intraday_momentum_20` — 20-minute momentum
      - `intraday_volatility_20` — 20-minute realized vol
      - `intraday_volume_ratio` — current bar volume vs 20-bar avg
      - `intraday_price_to_open` — current price / day's open
      - `intraday_range_position` — (close - day_low) / (day_high - day_low)
    - Windowed computation: keep rolling window of last 60 bars in memory
    - Batch insert to `intraday_features` table

15. [ ] **Register intraday features in feature_definitions table** (low)
    - Add 7 intraday feature definitions with `version='v1'`
    - Mark as `is_active=True`

16. [ ] **Wire feature engine into `IntradayIngestionService`** (medium)
    - After each buffer flush, trigger feature computation on new bars
    - Optional: configurable feature computation interval (every flush vs every N flushes)

17. [ ] **Add `get_intraday_features()` to `PlatformAPI`** (low)
    - Query `intraday_features` table
    - Return DataFrame with (symbol, timestamp, feature_name, value)

18. [ ] **Write tests for Phase 2** (medium)
    - Test each intraday feature computation against known values
    - Test rolling window behavior
    - Test feature registration and retrieval

### Phase 3: Dashboard & Polish (5 tasks)

19. [ ] **Create `hrp/dashboard/pages/realtime.py`** (medium)
    - Real-time monitoring page with:
      - Connection status indicator (connected/disconnected/reconnecting)
      - Last message timestamp + bars/minute throughput
      - Live price table for watchlist symbols
      - Intraday candlestick chart (Plotly) for selected symbol
      - Session stats (total bars, symbols tracked, uptime)
    - Auto-refresh via `st.rerun()` with 10s interval during market hours

20. [ ] **Add intraday data to daily retention engine** (low)
    - Register `intraday_bars` in `hrp/data/retention.py`
    - HOT: 7 days, WARM: 30 days, COLD: delete
    - Register `intraday_features` with same policy

21. [ ] **Add REST gap-fill logic** (medium)
    - On WebSocket reconnection, detect gap in `intraday_bars`
    - Use `PolygonSource.get_minute_bars()` to backfill missing bars
    - Deduplicate via upsert

22. [ ] **Add market calendar awareness** (low)
    - Check for market holidays before starting WebSocket session
    - Use `exchange_calendars` library or hardcoded US holiday list
    - Skip half-days (1 PM close) with adjusted schedule

23. [ ] **Write integration tests** (medium)
    - End-to-end test: mock WebSocket → buffer → DuckDB → feature → API
    - Dashboard page smoke test (Streamlit testing)

## Testing Strategy

### Unit Tests
- `PolygonWebSocketClient`: connection lifecycle, message parsing, reconnection (all mocked)
- `IntradayBarBuffer`: thread safety under concurrent add/flush, max buffer enforcement
- `_batch_upsert_intraday`: DuckDB upsert correctness, duplicate handling
- `IntradayFeatureEngine`: feature math correctness, rolling window edge cases
- Config loading: realtime section defaults, validation

### Integration Tests
- Full pipeline: simulated WS messages → buffer → DuckDB → query via PlatformAPI
- Scheduler integration: job start/stop lifecycle
- Feature computation triggered by new bar arrival

### Manual Testing Checklist
- [ ] Start WebSocket during market hours, verify bars flowing into DB
- [ ] Kill network connection, verify reconnection and gap-fill
- [ ] Verify daily retention job cleans up old intraday data
- [ ] Dashboard page renders correctly with live data
- [ ] Verify no interference with daily batch pipeline (6 PM job)

## Edge Cases & Error Handling

| Case | Handling |
|------|----------|
| **WS disconnect during market hours** | Exponential backoff reconnection (5s → 60s). REST gap-fill on reconnect. Log `intraday_session_reconnect` to lineage. |
| **Polygon API key invalid/expired** | Fail fast on startup with clear error message. Do not retry auth failures. |
| **DuckDB write failure** | Keep bars in buffer (up to max_buffer_size). Retry on next flush cycle. Log error. If buffer full, drop oldest bars and log warning. |
| **Market holiday** | Check calendar before starting session. Log skip to lineage. |
| **Half trading day** | Adjust stop time from 4:05 PM to 1:05 PM on half days. |
| **Stale data (no messages for 60s during market hours)** | Log warning. After 120s, attempt reconnect. Could indicate API issue or market halt. |
| **Symbol delisted mid-session** | Polygon sends status messages. Unsubscribe from symbol. Log to lineage. |
| **Buffer overflow (>10K bars)** | Force flush immediately. If flush fails, drop oldest 50% and log critical warning. |
| **Concurrent access to DuckDB** | Use existing `ConnectionPool` with `threading.Lock`. Intraday writes use separate connection from daily pipeline. |

## Open Questions

1. **Polygon API tier** — What Polygon subscription tier is active? Basic (5 req/min) has WebSocket access but limited to delayed data. Stocks Starter or higher needed for real-time. *Impacts: whether data is real-time or 15-min delayed.*

2. **Symbol count** — Should we stream all ~450 S&P 500 symbols or a focused watchlist? Streaming all generates ~180K bars/day (450 symbols × 390 minutes). *Recommendation: Start with top 50 by daily volume, configurable.*

3. **Intraday features scope** — The 7 proposed intraday features are a minimal set. Should we port more of the 45 daily features to intraday? *Recommendation: Start minimal (Phase 2), expand based on research needs.*

4. **Retention aggressive vs conservative** — 7-day hot retention means only 1 week of minute data for backtesting. Should we keep 30 days at full resolution? *Storage impact: ~5.4M rows/month at 50 symbols. DuckDB handles this fine.*

5. **exchange_calendars dependency** — Should we add `exchange_calendars` package for holiday detection, or use a simpler hardcoded approach? *Recommendation: Hardcoded for 2026, add package later if needed.*

## Dependencies

- `polygon-api-client >= 1.12.0` — already installed (provides `WebSocketClient`)
- `exchange_calendars` — optional, for market holiday detection (defer to Phase 3)
- No other new dependencies needed

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Polygon WS rate limiting | Low | Medium | Already using their SDK which handles auth. AM channel is lightweight. |
| DuckDB contention with daily pipeline | Low | Low | Different tables. Use connection pool. Daily pipeline runs 6 PM, WS runs 9:30 AM-4 PM. |
| Increased DuckDB file size | Medium | Low | Retention policy limits growth. 50 symbols × 390 bars × 30 days ≈ 585K rows. Negligible. |
| WebSocket stability | Medium | Medium | Robust reconnection logic. REST fallback. Lineage logging for monitoring. |
| Feature compute latency | Low | Low | 7 features on 50 symbols = 350 computations per flush. Trivial for vectorized pandas. |
