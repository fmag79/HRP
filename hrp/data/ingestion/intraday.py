"""
Intraday data ingestion service.

Handles real-time minute bar ingestion from Polygon.io WebSocket,
buffering, and batch writes to DuckDB.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dt_time
from decimal import Decimal

import pandas as pd
import pytz
from loguru import logger

from hrp.data.connection_pool import ConnectionPool
from hrp.data.features.intraday_features import IntradayBar as FeatureBar
from hrp.data.features.intraday_features import IntradayFeatureEngine
from hrp.data.sources.polygon_websocket import PolygonWebSocketClient, WebSocketConfig


@dataclass
class IntradayBar:
    """Represents a single intraday minute bar."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal | None = None
    trade_count: int | None = None
    source: str = "polygon_ws"


class IntradayBarBuffer:
    """
    Thread-safe buffer for intraday bars.

    Accumulates bars in memory before batch write to DuckDB.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize buffer.

        Args:
            max_size: Maximum bars to buffer before forcing flush
        """
        self._buffer: deque[IntradayBar] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._max_size = max_size
        self._bars_added = 0
        self._bars_flushed = 0

    def add_bar(
        self,
        symbol: str,
        timestamp: datetime,
        open_: float | Decimal,
        high: float | Decimal,
        low: float | Decimal,
        close: float | Decimal,
        volume: int,
        vwap: float | Decimal | None = None,
        trade_count: int | None = None,
    ) -> None:
        """
        Add a bar to the buffer.

        Args:
            symbol: Stock ticker
            timestamp: Bar timestamp (UTC)
            open_: Opening price
            high: High price
            low: Low price
            close: Closing price
            volume: Trading volume
            vwap: Volume-weighted average price
            trade_count: Number of trades in bar
        """
        bar = IntradayBar(
            symbol=symbol,
            timestamp=timestamp,
            open=Decimal(str(open_)),
            high=Decimal(str(high)),
            low=Decimal(str(low)),
            close=Decimal(str(close)),
            volume=volume,
            vwap=Decimal(str(vwap)) if vwap is not None else None,
            trade_count=trade_count,
        )

        with self._lock:
            self._buffer.append(bar)
            self._bars_added += 1

            # Log warning if approaching max size
            if len(self._buffer) > self._max_size * 0.9:
                logger.warning(
                    f"Buffer at {len(self._buffer)}/{self._max_size} "
                    f"(90% full) - consider flushing"
                )

    def flush(self) -> list[dict]:
        """
        Drain buffer and return all bars as dicts.

        Returns:
            List of bar dicts ready for DataFrame conversion
        """
        with self._lock:
            if not self._buffer:
                return []

            bars = [
                {
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                    "trade_count": bar.trade_count,
                    "source": bar.source,
                }
                for bar in self._buffer
            ]

            count = len(bars)
            self._buffer.clear()
            self._bars_flushed += count

            return bars

    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self._max_size,
                "bars_added": self._bars_added,
                "bars_flushed": self._bars_flushed,
                "utilization_pct": len(self._buffer) / self._max_size * 100,
            }


def _batch_upsert_intraday(bars: list[dict], conn_pool: ConnectionPool) -> int:
    """
    Batch upsert intraday bars to DuckDB.

    Uses temp table pattern for efficient upsert:
    1. Create temp table
    2. Insert new data to temp
    3. ON CONFLICT upsert from temp to main table
    4. Drop temp table

    Args:
        bars: List of bar dicts (from IntradayBarBuffer.flush())
        conn_pool: Database connection pool

    Returns:
        Number of rows upserted
    """
    if not bars:
        return 0

    # Convert to DataFrame for efficient bulk insert
    df = pd.DataFrame(bars)

    # Ensure timestamp is datetime type
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure all required columns exist with defaults
    default_columns = {
        "trade_count": None,
        "vwap": None,
        "source": "polygon_ws",
    }
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val

    with conn_pool.connection() as conn:
        try:
            # Create temporary table
            conn.execute(
                """
                CREATE TEMPORARY TABLE temp_intraday_bars (
                    symbol VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL(12,4),
                    high DECIMAL(12,4),
                    low DECIMAL(12,4),
                    close DECIMAL(12,4) NOT NULL,
                    volume BIGINT,
                    vwap DECIMAL(12,4),
                    trade_count INTEGER,
                    source VARCHAR DEFAULT 'polygon_ws',
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Insert data to temp table (excluding ingested_at - will use DEFAULT)
            conn.execute("""
                INSERT INTO temp_intraday_bars (
                    symbol, timestamp, open, high, low, close,
                    volume, vwap, trade_count, source
                )
                SELECT
                    symbol, timestamp, open, high, low, close,
                    volume, vwap, trade_count, source
                FROM df
            """)

            # Upsert: ON CONFLICT updates existing rows, preserves ingested_at
            result = conn.execute(
                """
                INSERT INTO intraday_bars (
                    symbol, timestamp, open, high, low, close,
                    volume, vwap, trade_count, source
                )
                SELECT symbol, timestamp, open, high, low, close,
                       volume, vwap, trade_count, source
                FROM temp_intraday_bars
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    vwap = EXCLUDED.vwap,
                    trade_count = EXCLUDED.trade_count,
                    source = EXCLUDED.source
                """
            )

            row_count = result.fetchone()[0] if result else len(df)

            # Drop temp table
            conn.execute("DROP TABLE temp_intraday_bars")

            logger.debug(f"Upserted {row_count} intraday bars")
            return row_count

        except Exception as e:
            logger.error(f"Error upserting intraday bars: {e}", exc_info=True)
            # Try to clean up temp table if it exists
            try:
                conn.execute("DROP TABLE IF EXISTS temp_intraday_bars")
            except Exception:
                pass
            raise


class IntradayIngestionService:
    """
    Service for managing real-time intraday data ingestion.

    Owns the WebSocket client and bar buffer, coordinates periodic flushes,
    and provides lifecycle management for the ingestion session.
    """

    def __init__(
        self,
        conn_pool: ConnectionPool,
        ws_config: WebSocketConfig | None = None,
        flush_interval: int = 10,
        max_buffer_size: int = 10000,
        compute_features: bool = True,
    ):
        """
        Initialize ingestion service.

        Args:
            conn_pool: Database connection pool
            ws_config: WebSocket configuration (None = use defaults)
            flush_interval: Seconds between buffer flushes
            max_buffer_size: Maximum bars to buffer
            compute_features: Whether to compute intraday features after each flush
        """
        self.conn_pool = conn_pool
        self.ws_client = PolygonWebSocketClient(ws_config)
        self.buffer = IntradayBarBuffer(max_size=max_buffer_size)
        self.flush_interval = flush_interval
        self.compute_features = compute_features

        # Initialize feature engine if enabled
        self.feature_engine = IntradayFeatureEngine() if compute_features else None

        self._flush_thread: threading.Thread | None = None
        self._stop_flush = threading.Event()
        self._session_start: datetime | None = None
        self._bars_received = 0
        self._bars_written = 0
        self._features_written = 0

        # Register WebSocket callback
        self.ws_client.register_callback(self._on_message)

        logger.info(
            f"IntradayIngestionService initialized (features: {compute_features})"
        )

    def start(self, symbols: list[str], channels: list[str] | None = None) -> None:
        """
        Start real-time ingestion.

        Args:
            symbols: List of symbols to stream
            channels: WebSocket channels (defaults to ['AM'] for minute bars)
        """
        if channels is None:
            channels = ["AM"]

        self._session_start = datetime.now(UTC)
        self._bars_received = 0
        self._bars_written = 0
        self._stop_flush.clear()

        # Start WebSocket
        self.ws_client.start(symbols, channels)

        # Start flush timer thread
        self._flush_thread = threading.Thread(
            target=self._flush_timer_loop, daemon=True, name="IntradayFlushTimer"
        )
        self._flush_thread.start()

        logger.info(
            f"Intraday ingestion started: {len(symbols)} symbols, "
            f"flush interval: {self.flush_interval}s"
        )

    def stop(self) -> dict:
        """
        Stop ingestion and return session statistics.

        Returns:
            Dict with session stats (bars_received, bars_written, etc.)
        """
        logger.info("Stopping intraday ingestion...")

        # Stop flush timer
        self._stop_flush.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)

        # Final flush
        self._flush_buffer()

        # Stop WebSocket
        self.ws_client.stop()

        session_duration = (
            (datetime.now(UTC) - self._session_start).total_seconds()
            if self._session_start
            else 0
        )

        stats = {
            "bars_received": self._bars_received,
            "bars_written": self._bars_written,
            "features_written": self._features_written,
            "session_duration_seconds": session_duration,
            "ws_reconnects": self.ws_client.get_stats()["reconnect_count"],
            "buffer_stats": self.buffer.get_stats(),
        }

        logger.info(
            f"Intraday ingestion stopped. Stats: {self._bars_received} received, "
            f"{self._bars_written} written, {session_duration:.0f}s duration"
        )

        return stats

    def is_market_hours(self) -> bool:
        """
        Check if currently in market hours (9:30 AM - 4:00 PM ET).

        Returns:
            True if in market hours, False otherwise
        """
        et_tz = pytz.timezone("America/New_York")
        now_et = datetime.now(et_tz).time()

        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)

        return market_open <= now_et <= market_close

    def gap_fill_after_reconnect(self, symbols: list[str]) -> int:
        """
        Fill data gaps after WebSocket reconnection using REST API.

        Detects gaps in intraday_bars by finding missing timestamps
        for each symbol, then backfills using PolygonSource.get_minute_bars().

        Args:
            symbols: List of symbols being tracked

        Returns:
            Number of bars backfilled
        """
        from hrp.data.sources.polygon_source import PolygonSource

        logger.info("Starting gap-fill after reconnection...")

        try:
            polygon_source = PolygonSource()
        except Exception as e:
            logger.error(f"Could not initialize PolygonSource for gap-fill: {e}")
            return 0

        total_backfilled = 0

        for symbol in symbols:
            try:
                # Find last bar timestamp in DB for this symbol
                with self.conn_pool.connection() as conn:
                    result = conn.execute(
                        """
                        SELECT MAX(timestamp) as last_timestamp
                        FROM intraday_bars
                        WHERE symbol = ?
                        """,
                        (symbol,)
                    ).fetchone()

                    last_timestamp = result[0] if result and result[0] else None

                if not last_timestamp:
                    logger.debug(f"No existing data for {symbol}, skipping gap-fill")
                    continue

                # Convert to datetime if needed
                if isinstance(last_timestamp, str):
                    last_timestamp = datetime.fromisoformat(last_timestamp)

                # Check if gap exists (more than 2 minutes since last bar during market hours)
                now = datetime.now(UTC)
                gap_minutes = (now - last_timestamp).total_seconds() / 60

                if gap_minutes < 2:
                    logger.debug(f"No gap for {symbol} (last bar {gap_minutes:.1f}m ago)")
                    continue

                if gap_minutes > 120:  # More than 2 hours - likely not a reconnect gap
                    logger.warning(
                        f"Large gap detected for {symbol} ({gap_minutes:.0f} minutes). "
                        f"Skipping gap-fill to avoid large REST fetch."
                    )
                    continue

                logger.info(f"Gap detected for {symbol}: {gap_minutes:.0f} minutes")

                # Backfill using REST API
                start_time = last_timestamp
                end_time = now

                bars_df = polygon_source.get_minute_bars(
                    symbol=symbol,
                    start=start_time,
                    end=end_time,
                )

                if bars_df.empty:
                    logger.debug(f"No gap-fill data returned for {symbol}")
                    continue

                # Convert to IntradayBar format
                bars_to_insert = []
                for _, row in bars_df.iterrows():
                    bars_to_insert.append({
                        "symbol": symbol,
                        "timestamp": row["timestamp"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "vwap": row.get("vwap"),
                        "trade_count": row.get("trade_count"),
                        "source": "polygon_rest_gapfill",
                    })

                # Upsert backfilled bars
                if bars_to_insert:
                    rows_written = _batch_upsert_intraday(bars_to_insert, self.conn_pool)
                    total_backfilled += rows_written
                    logger.info(f"Gap-filled {rows_written} bars for {symbol}")

            except Exception as e:
                logger.error(f"Error gap-filling {symbol}: {e}", exc_info=True)
                continue

        logger.info(f"Gap-fill complete: {total_backfilled} bars backfilled")
        return total_backfilled

    def _on_message(self, msgs: list[dict]) -> None:
        """
        Handle incoming WebSocket messages.

        Args:
            msgs: List of message dicts from Polygon
        """
        for msg in msgs:
            try:
                # Parse minute aggregate (AM) message
                # Polygon AM message format:
                # {
                #   "ev": "AM",  # event type
                #   "sym": "AAPL",  # symbol
                #   "o": 150.25,  # open
                #   "h": 150.50,  # high
                #   "l": 150.20,  # low
                #   "c": 150.45,  # close
                #   "v": 125000,  # volume
                #   "vw": 150.35,  # VWAP
                #   "s": 1609459200000,  # start timestamp (ms)
                #   "e": 1609459260000,  # end timestamp (ms)
                #   "n": 1250  # number of trades
                # }

                if msg.get("ev") != "AM":
                    continue  # Skip non-minute-aggregate messages

                symbol = msg.get("sym")
                if not symbol:
                    continue

                # Convert timestamp from milliseconds to datetime
                timestamp_ms = msg.get("s")
                if not timestamp_ms:
                    continue

                timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

                # Add to buffer
                self.buffer.add_bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_=msg.get("o"),
                    high=msg.get("h"),
                    low=msg.get("l"),
                    close=msg.get("c"),
                    volume=msg.get("v", 0),
                    vwap=msg.get("vw"),
                    trade_count=msg.get("n"),
                )

                self._bars_received += 1

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

    def _flush_timer_loop(self) -> None:
        """Periodic buffer flush loop."""
        while not self._stop_flush.is_set():
            time.sleep(self.flush_interval)
            if not self._stop_flush.is_set():
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffer to database and optionally compute features."""
        try:
            bars = self.buffer.flush()
            if not bars:
                return

            # Write bars to database
            rows_written = _batch_upsert_intraday(bars, self.conn_pool)
            self._bars_written += rows_written
            logger.debug(
                f"Flushed {len(bars)} bars, {rows_written} rows written to DB"
            )

            # Compute and persist features if enabled
            if self.feature_engine:
                try:
                    # Convert IntradayBar dataclass to FeatureBar
                    feature_bars = [
                        FeatureBar(
                            symbol=b["symbol"],
                            timestamp=b["timestamp"],
                            open=float(b["open"]),
                            high=float(b["high"]),
                            low=float(b["low"]),
                            close=float(b["close"]),
                            volume=b["volume"],
                            vwap=float(b["vwap"]) if b["vwap"] is not None else None,
                        )
                        for b in bars
                    ]

                    # Compute features
                    features_df = self.feature_engine.compute_features(feature_bars)

                    # Persist features
                    if not features_df.empty:
                        feature_count = self.feature_engine.persist_features(
                            features_df, conn_pool=self.conn_pool
                        )
                        self._features_written += feature_count
                        logger.debug(f"Computed and persisted {feature_count} features")

                except Exception as e:
                    logger.error(f"Error computing features: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """Get current session statistics."""
        return {
            "bars_received": self._bars_received,
            "bars_written": self._bars_written,
            "features_written": self._features_written,
            "buffer_stats": self.buffer.get_stats(),
            "ws_stats": self.ws_client.get_stats(),
            "is_connected": self.ws_client.is_connected(),
            "session_start": self._session_start,
        }

    def __enter__(self) -> IntradayIngestionService:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()
