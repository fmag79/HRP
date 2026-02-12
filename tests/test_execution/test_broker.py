"""Tests for IBKR broker connection."""
from unittest.mock import patch

import pytest

from hrp.execution.broker import BrokerConfig, IBKRBroker


def test_broker_config_validation():
    """Test broker config requires all fields."""
    with pytest.raises(ValueError, match="host is required"):
        BrokerConfig(host="", port=7497, client_id=1, account="")


def test_broker_config_account_required():
    """Test broker config requires account."""
    with pytest.raises(ValueError, match="account is required"):
        BrokerConfig(host="127.0.0.1", port=7497, client_id=1, account="")


def test_broker_connection_paper_trading():
    """Test broker connects to paper trading account."""
    config = BrokerConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=1,
        account="DU123456",
        paper_trading=True,
    )

    with patch("ib_insync.IB") as mock_ib:
        mock_ib.return_value.connect.return_value = None
        mock_ib.return_value.isConnected.return_value = True

        broker = IBKRBroker(config)
        broker.connect()

        assert broker.is_connected()
        mock_ib.return_value.connect.assert_called_once_with(
            "127.0.0.1", 7497, clientId=1, readonly=False, timeout=10
        )


def test_broker_disconnect():
    """Test broker disconnects cleanly."""
    config = BrokerConfig(
        host="127.0.0.1", port=7497, client_id=1,
        account="DU123456", paper_trading=True
    )

    with patch("ib_insync.IB") as mock_ib:
        mock_ib.return_value.isConnected.return_value = True
        broker = IBKRBroker(config)
        broker.connect()
        broker.disconnect()

        mock_ib.return_value.disconnect.assert_called_once()


def test_broker_context_manager():
    """Test broker works as context manager."""
    config = BrokerConfig(
        host="127.0.0.1", port=7497, client_id=1,
        account="DU123456", paper_trading=True
    )

    with patch("ib_insync.IB") as mock_ib:
        mock_ib.return_value.isConnected.return_value = True

        with IBKRBroker(config) as broker:
            assert broker.is_connected()

        mock_ib.return_value.disconnect.assert_called_once()


def test_broker_paper_trading_port_warning(caplog):
    """Test warning when paper trading uses non-standard port."""
    import logging
    caplog.set_level(logging.WARNING)

    _ = BrokerConfig(
        host="127.0.0.1",
        port=7496,  # Live port, not paper
        client_id=1,
        account="DU123456",
        paper_trading=True,
    )

    assert "Paper trading typically uses port 7497" in caplog.text
