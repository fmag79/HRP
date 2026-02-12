"""
Tests for Polygon WebSocket client.

Following TDD: comprehensive tests for connection lifecycle, message handling,
reconnection logic, and subscription management.
"""

import pytest
import threading
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from hrp.data.sources.polygon_websocket import (
    PolygonWebSocketClient,
    WebSocketConfig,
)


@pytest.fixture
def ws_config():
    """Create test WebSocket configuration."""
    return WebSocketConfig(
        api_key="test_api_key_12345",
        market="stocks",
        max_reconnect_delay=5,  # Shorter for tests
        heartbeat_timeout=5,
        queue_max_size=100,
    )


def test_websocket_config_initialization():
    """Test WebSocketConfig dataclass initialization."""
    config = WebSocketConfig(api_key="test_key")

    assert config.api_key == "test_key"
    assert config.market == "stocks"  # Default
    assert config.max_reconnect_delay == 60  # Default
    assert config.heartbeat_timeout == 30  # Default
    assert config.queue_max_size == 10000  # Default


def test_websocket_client_initialization(ws_config):
    """Test PolygonWebSocketClient initialization."""
    client = PolygonWebSocketClient(ws_config)

    assert client.config == ws_config
    assert client._is_connected is False
    assert client._reconnect_count == 0
    assert len(client._callbacks) == 0
    assert len(client._subscriptions) == 0


def test_websocket_client_requires_api_key():
    """Test that WebSocket client requires API key."""
    config = WebSocketConfig(api_key="")

    with pytest.raises(ValueError) as exc_info:
        PolygonWebSocketClient(config)

    assert "POLYGON_API_KEY not found" in str(exc_info.value)


def test_register_callback(ws_config):
    """Test callback registration."""
    client = PolygonWebSocketClient(ws_config)

    def test_callback(msgs):
        pass

    client.register_callback(test_callback)

    assert len(client._callbacks) == 1
    assert client._callbacks[0] == test_callback


def test_register_multiple_callbacks(ws_config):
    """Test multiple callback registration."""
    client = PolygonWebSocketClient(ws_config)

    def callback1(msgs):
        pass

    def callback2(msgs):
        pass

    client.register_callback(callback1)
    client.register_callback(callback2)

    assert len(client._callbacks) == 2


def test_start_creates_daemon_thread(ws_config):
    """Test that start() creates a daemon thread."""
    client = PolygonWebSocketClient(ws_config)

    # Mock the WebSocket to prevent actual connection
    with patch("hrp.data.sources.polygon_websocket.WebSocketClient"):
        client.start(symbols=["AAPL"], channels=["AM"])

        # Give thread time to start
        time.sleep(0.1)

        assert client._thread is not None
        assert client._thread.is_alive()
        assert client._thread.daemon is True

        # Clean up
        client.stop()


def test_start_default_channels(ws_config):
    """Test that start() defaults to ['AM'] channel."""
    client = PolygonWebSocketClient(ws_config)

    with patch("hrp.data.sources.polygon_websocket.WebSocketClient"):
        client.start(symbols=["AAPL"])  # No channels specified

        assert "AM" in client._subscriptions
        assert "AAPL" in client._subscriptions["AM"]

        client.stop()


def test_stop_gracefully(ws_config):
    """Test graceful stop."""
    client = PolygonWebSocketClient(ws_config)

    with patch("hrp.data.sources.polygon_websocket.WebSocketClient"):
        client.start(symbols=["AAPL"], channels=["AM"])
        time.sleep(0.1)

        client.stop()

        # Thread should terminate
        time.sleep(0.5)
        assert client._is_connected is False
        assert not client._thread.is_alive()


def test_handle_message_dispatches_to_callbacks(ws_config):
    """Test that messages are dispatched to all registered callbacks."""
    client = PolygonWebSocketClient(ws_config)

    received_messages = []

    def callback(msgs):
        received_messages.extend(msgs)

    client.register_callback(callback)

    # Simulate incoming messages
    test_msgs = [
        {"ev": "AM", "sym": "AAPL", "c": 150.0},
        {"ev": "AM", "sym": "MSFT", "c": 300.0},
    ]

    client._handle_message(test_msgs)

    assert len(received_messages) == 2
    assert received_messages[0]["sym"] == "AAPL"
    assert received_messages[1]["sym"] == "MSFT"


def test_handle_message_updates_last_message_time(ws_config):
    """Test that message handling updates last message time."""
    client = PolygonWebSocketClient(ws_config)

    initial_time = client._last_message_time
    assert initial_time == 0

    client._handle_message([{"ev": "AM", "sym": "AAPL"}])

    assert client._last_message_time > initial_time
    assert client._last_message_time > 0


def test_handle_message_adds_to_queue(ws_config):
    """Test that messages are added to queue."""
    client = PolygonWebSocketClient(ws_config)

    test_msgs = [{"ev": "AM", "sym": "AAPL"}]
    client._handle_message(test_msgs)

    assert len(client._message_queue) == 1
    assert client._message_queue[0]["sym"] == "AAPL"


def test_get_pending_messages_drains_queue(ws_config):
    """Test that get_pending_messages() drains queue."""
    client = PolygonWebSocketClient(ws_config)

    # Add messages to queue
    client._handle_message([{"ev": "AM", "sym": "AAPL"}])
    client._handle_message([{"ev": "AM", "sym": "MSFT"}])

    assert len(client._message_queue) == 2

    # Get messages
    messages = client.get_pending_messages()

    assert len(messages) == 2
    assert len(client._message_queue) == 0  # Queue cleared


def test_monitor_heartbeat_detects_stale_data(ws_config):
    """Test heartbeat monitoring detects stale data."""
    client = PolygonWebSocketClient(ws_config)

    # No messages received yet
    assert not client.monitor_heartbeat()

    # Simulate old message
    client._last_message_time = time.time() - 100  # 100 seconds ago

    # Should detect stale data (timeout is 5s in test config)
    assert not client.monitor_heartbeat()


def test_monitor_heartbeat_passes_with_recent_data(ws_config):
    """Test heartbeat monitoring passes with recent messages."""
    client = PolygonWebSocketClient(ws_config)

    # Simulate recent message
    client._last_message_time = time.time() - 1  # 1 second ago

    # Should pass (timeout is 5s in test config)
    assert client.monitor_heartbeat()


def test_get_stats(ws_config):
    """Test get_stats() returns correct statistics."""
    client = PolygonWebSocketClient(ws_config)

    # Simulate some activity
    client._reconnect_count = 2
    client._last_message_time = time.time()
    client._handle_message([{"ev": "AM", "sym": "AAPL"}])

    stats = client.get_stats()

    assert stats["reconnect_count"] == 2
    assert stats["queue_size"] == 1
    assert stats["last_message_time"] is not None
    assert stats["is_connected"] is False  # Not actually connected in test


def test_is_connected_reflects_state(ws_config):
    """Test is_connected() method."""
    client = PolygonWebSocketClient(ws_config)

    assert not client.is_connected()

    client._is_connected = True
    assert client.is_connected()

    client._is_connected = False
    assert not client.is_connected()


def test_context_manager(ws_config):
    """Test WebSocket client can be used as context manager."""
    with patch("hrp.data.sources.polygon_websocket.WebSocketClient"):
        with PolygonWebSocketClient(ws_config) as client:
            assert client is not None
            client.start(symbols=["AAPL"], channels=["AM"])

        # Should be stopped after exiting context
        assert not client._is_connected


def test_callback_exception_does_not_crash(ws_config):
    """Test that exception in callback doesn't crash message handling."""
    client = PolygonWebSocketClient(ws_config)

    def bad_callback(msgs):
        raise ValueError("Intentional test error")

    def good_callback(msgs):
        pass  # Should still run despite bad_callback failing

    client.register_callback(bad_callback)
    client.register_callback(good_callback)

    # Should not raise exception even though callback fails
    client._handle_message([{"ev": "AM", "sym": "AAPL"}])

    # Message should still be in queue
    assert len(client._message_queue) == 1
