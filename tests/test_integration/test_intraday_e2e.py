"""
End-to-end integration tests for intraday data ingestion.

Tests the full pipeline: WebSocket → buffer → DuckDB → features → API.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from hrp.data.db import ConnectionPool
from hrp.data.ingestion.intraday import IntradayIngestionService
from hrp.data.sources.polygon_websocket import WebSocketConfig


@pytest.fixture
def conn_pool(tmp_path):
    """Create a temporary connection pool for testing."""
    db_path = tmp_path / "test_intraday_e2e.db"
    pool = ConnectionPool(str(db_path))

    # Initialize schema
    from hrp.data.schema import TABLES, INDEXES

    with pool.connection() as conn:
        # Create required tables
        for table_name, create_sql in TABLES.items():
            if table_name in ["symbols", "intraday_bars", "intraday_features"]:
                conn.execute(create_sql)

        # Create indexes
        for index_sql in INDEXES:
            if "intraday" in index_sql.lower():
                conn.execute(index_sql)

        # Insert test symbols
        conn.execute(
            """
            INSERT INTO symbols (symbol, name, sector, industry)
            VALUES ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics'),
                   ('MSFT', 'Microsoft Corp.', 'Technology', 'Software')
            """
        )

    yield pool


def test_e2e_pipeline_mock_websocket(conn_pool):
    """
    Test full pipeline with mocked WebSocket messages.

    Flow: Mock WS → IntradayIngestionService → DuckDB → query data
    """
    # Create mock WebSocket config
    mock_config = WebSocketConfig(
        api_key="test_key",
        market="stocks",
    )

    # Create service
    service = IntradayIngestionService(
        conn_pool=conn_pool,
        ws_config=mock_config,
        flush_interval=1,  # Fast flush for testing
        max_buffer_size=100,
        compute_features=False,  # Disable features for simpler test
    )

    # Mock the WebSocket client to avoid actual connection
    with patch.object(service.ws_client, "start") as mock_start:
        with patch.object(service.ws_client, "stop") as mock_stop:
            with patch.object(service.ws_client, "is_connected", return_value=True):
                with patch.object(service.ws_client, "get_stats", return_value={
                    "reconnect_count": 0,
                    "messages_received": 10,
                }):
                    # Start service (mocked, doesn't actually connect)
                    service.start(symbols=["AAPL", "MSFT"], channels=["AM"])

                    # Simulate incoming WebSocket messages
                    mock_messages = [
                        {
                            "ev": "AM",
                            "sym": "AAPL",
                            "o": 150.25,
                            "h": 150.50,
                            "l": 150.20,
                            "c": 150.45,
                            "v": 125000,
                            "vw": 150.35,
                            "s": int(datetime.now(UTC).timestamp() * 1000),
                            "n": 1250,
                        },
                        {
                            "ev": "AM",
                            "sym": "MSFT",
                            "o": 320.10,
                            "h": 320.30,
                            "l": 320.05,
                            "c": 320.25,
                            "v": 80000,
                            "vw": 320.18,
                            "s": int(datetime.now(UTC).timestamp() * 1000),
                            "n": 800,
                        },
                    ]

                    # Process messages through service
                    service._on_message(mock_messages)

                    # Force flush to write to DB
                    service._flush_buffer()

                    # Stop service
                    service.stop()

    # Verify data was written to intraday_bars
    with conn_pool.connection() as conn:
        result = conn.execute(
            "SELECT COUNT(*) FROM intraday_bars"
        ).fetchone()
        assert result[0] == 2, "Expected 2 bars in DB"

        # Verify AAPL bar
        aapl_bar = conn.execute(
            "SELECT symbol, open, high, low, close, volume FROM intraday_bars WHERE symbol = 'AAPL'"
        ).fetchone()
        assert aapl_bar is not None
        assert aapl_bar[0] == "AAPL"
        assert float(aapl_bar[1]) == pytest.approx(150.25, abs=0.01)
        assert float(aapl_bar[2]) == pytest.approx(150.50, abs=0.01)
        assert float(aapl_bar[3]) == pytest.approx(150.20, abs=0.01)
        assert float(aapl_bar[4]) == pytest.approx(150.45, abs=0.01)
        assert aapl_bar[5] == 125000


def test_e2e_with_features(conn_pool):
    """
    Test full pipeline including feature computation.

    Flow: Mock WS → buffer → DB → features → query features
    """
    mock_config = WebSocketConfig(
        api_key="test_key",
        market="stocks",
    )

    # Create service with features enabled
    service = IntradayIngestionService(
        conn_pool=conn_pool,
        ws_config=mock_config,
        flush_interval=1,
        max_buffer_size=100,
        compute_features=True,  # Enable feature computation
    )

    with patch.object(service.ws_client, "start"):
        with patch.object(service.ws_client, "stop"):
            with patch.object(service.ws_client, "is_connected", return_value=True):
                with patch.object(service.ws_client, "get_stats", return_value={
                    "reconnect_count": 0,
                    "messages_received": 30,
                }):
                    # Start service
                    service.start(symbols=["AAPL"], channels=["AM"])

                    # Send 30 bars to build rolling windows
                    base_time = datetime.now(UTC)
                    for i in range(30):
                        msg = {
                            "ev": "AM",
                            "sym": "AAPL",
                            "o": 150.0 + i * 0.1,
                            "h": 150.5 + i * 0.1,
                            "l": 149.8 + i * 0.1,
                            "c": 150.2 + i * 0.1,
                            "v": 100000 + i * 1000,
                            "vw": 150.1 + i * 0.1,
                            "s": int((base_time + timedelta(minutes=i)).timestamp() * 1000),
                            "n": 1000,
                        }
                        service._on_message([msg])

                    # Flush to trigger feature computation
                    service._flush_buffer()

                    # Stop service
                    service.stop()

    # Verify features were computed
    with conn_pool.connection() as conn:
        feature_count = conn.execute(
            "SELECT COUNT(*) FROM intraday_features WHERE symbol = 'AAPL'"
        ).fetchone()[0]

        # Should have features for at least some bars
        assert feature_count > 0, "Expected features to be computed"

        # Check for specific features
        features = conn.execute(
            """
            SELECT DISTINCT feature_name
            FROM intraday_features
            WHERE symbol = 'AAPL'
            ORDER BY feature_name
            """
        ).fetchall()

        feature_names = [f[0] for f in features]
        assert "intraday_vwap" in feature_names
        # Other features may not compute if not enough data


def test_gap_fill_integration(conn_pool):
    """
    Test gap-fill logic after simulated reconnection.

    Flow: Write initial bars → detect gap → call gap_fill → verify backfill
    """
    mock_config = WebSocketConfig(api_key="test_key", market="stocks")

    service = IntradayIngestionService(
        conn_pool=conn_pool,
        ws_config=mock_config,
        compute_features=False,
    )

    # Insert initial bars with a gap
    base_time = datetime.now(UTC)
    initial_bars = [
        {
            "symbol": "AAPL",
            "timestamp": base_time - timedelta(minutes=10),
            "open": Decimal("150.00"),
            "high": Decimal("150.50"),
            "low": Decimal("149.80"),
            "close": Decimal("150.20"),
            "volume": 100000,
            "source": "test",
        }
    ]

    from hrp.data.ingestion.intraday import _batch_upsert_intraday

    _batch_upsert_intraday(initial_bars, conn_pool)

    # Verify initial bar exists
    with conn_pool.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM intraday_bars").fetchone()[0]
        assert count == 1

    # Mock PolygonSource.get_minute_bars to return gap-fill data
    mock_gap_bars = [
        {
            "timestamp": base_time - timedelta(minutes=i),
            "open": 150.0 + i * 0.1,
            "high": 150.5 + i * 0.1,
            "low": 149.8 + i * 0.1,
            "close": 150.2 + i * 0.1,
            "volume": 100000 + i * 1000,
        }
        for i in range(1, 6)  # 5 bars to fill the gap
    ]

    import pandas as pd
    mock_df = pd.DataFrame(mock_gap_bars)

    with patch("hrp.data.ingestion.intraday.PolygonSource") as mock_polygon:
        mock_instance = MagicMock()
        mock_instance.get_minute_bars.return_value = mock_df
        mock_polygon.return_value = mock_instance

        # Run gap-fill
        backfilled = service.gap_fill_after_reconnect(symbols=["AAPL"])

    # Verify gap was filled
    with conn_pool.connection() as conn:
        total_bars = conn.execute("SELECT COUNT(*) FROM intraday_bars").fetchone()[0]
        # Should have original bar + gap-fill bars
        assert total_bars > 1, f"Expected more than 1 bar after gap-fill, got {total_bars}"


def test_api_query_integration(conn_pool, tmp_path):
    """
    Test querying intraday data via PlatformAPI.

    Flow: Insert bars → query via API → verify results
    """
    # Insert test bars
    base_time = datetime.now(UTC)
    test_bars = [
        {
            "symbol": "AAPL",
            "timestamp": base_time - timedelta(minutes=i),
            "open": Decimal("150.00"),
            "high": Decimal("150.50"),
            "low": Decimal("149.80"),
            "close": Decimal("150.20"),
            "volume": 100000,
            "source": "test",
        }
        for i in range(10)
    ]

    from hrp.data.ingestion.intraday import _batch_upsert_intraday

    _batch_upsert_intraday(test_bars, conn_pool)

    # Query via PlatformAPI
    from hrp.api.platform import PlatformAPI

    # Mock get_db to return our test connection pool
    with patch("hrp.api.platform.get_db", return_value=conn_pool._db):
        api = PlatformAPI(read_only=True)

        # Query intraday prices
        result = api.get_intraday_prices(
            symbols=["AAPL"],
            start_time=base_time - timedelta(minutes=20),
            end_time=base_time,
        )

        assert not result.empty, "Expected query to return data"
        assert len(result) == 10, f"Expected 10 bars, got {len(result)}"
        assert result["symbol"].iloc[0] == "AAPL"
