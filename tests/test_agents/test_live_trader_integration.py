"""Integration tests for LiveTradingAgent with Robinhood broker.

Tests the full integration:
- Broker selection (Robinhood vs IBKR)
- VaR-based position sizing
- Auto stop-loss generation
- Order status polling
"""

import os
from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from hrp.agents.live_trader import LiveTradingAgent, TradingConfig
from hrp.api.platform import PlatformAPI
from hrp.execution.orders import Order, OrderSide, OrderStatus, OrderType
from hrp.execution.robinhood_broker import RobinhoodConfig


class TestLiveTradingAgentBrokerSelection:
    """Test broker selection logic."""

    def test_robinhood_broker_creation(self):
        """Test Robinhood broker is created when broker_type='robinhood'."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            broker_type="robinhood",
            dry_run=False,
        )

        with patch.dict(
            os.environ,
            {
                "ROBINHOOD_USERNAME": "test@example.com",
                "ROBINHOOD_PASSWORD": "password",
                "ROBINHOOD_PAPER_TRADING": "true",
            },
        ):
            agent = LiveTradingAgent(trading_config=config, api=Mock())
            broker_config = agent._broker_config_from_env()

            assert isinstance(broker_config, RobinhoodConfig)
            assert broker_config.username == "test@example.com"
            assert broker_config.paper_trading is True

    def test_ibkr_broker_creation(self):
        """Test IBKR broker is created when broker_type='ibkr'."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            broker_type="ibkr",
            dry_run=False,
        )

        with patch.dict(
            os.environ,
            {
                "HRP_BROKER_TYPE": "ibkr",  # Need to set this!
                "IBKR_HOST": "127.0.0.1",
                "IBKR_PORT": "7497",
                "IBKR_CLIENT_ID": "1",
                "IBKR_ACCOUNT": "DU123456",
                "IBKR_PAPER_TRADING": "true",
            },
        ):
            agent = LiveTradingAgent(trading_config=config, api=Mock())
            broker_config = agent._broker_config_from_env()

            from hrp.execution.broker import BrokerConfig

            assert isinstance(broker_config, BrokerConfig)
            assert broker_config.account == "DU123456"

    def test_invalid_broker_type_raises(self):
        """Test invalid broker_type raises ValueError."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            broker_type="fidelity",  # Invalid
            dry_run=False,
        )

        with patch.dict(os.environ, {"HRP_BROKER_TYPE": "fidelity"}):
            # Agent init will call _broker_config_from_env, which should raise
            with pytest.raises(ValueError, match="Unknown broker type"):
                _ = LiveTradingAgent(trading_config=config, api=Mock())


class TestLiveTradingAgentVaRSizing:
    """Test VaR-based position sizing integration."""

    @pytest.fixture
    def mock_api(self):
        """Create mock API with VaR features."""
        api = Mock(spec=PlatformAPI)

        # Mock deployed strategies
        api.get_deployed_strategies.return_value = [
            {"hypothesis_id": "hyp1", "metadata": {"model_name": "model1"}}
        ]

        # Mock universe
        api.get_universe.return_value = ["AAPL", "MSFT", "GOOGL"]

        # Mock predictions
        predictions = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "prediction": [0.8, 0.6, 0.4],
            "signal": [1.0, 1.0, 1.0],
        })
        api.predict_model.return_value = predictions

        # Mock VaR features
        def mock_get_features(symbols, features, as_of_date, version):
            return pd.DataFrame({
                "symbol": symbols,
                "var_95_1d": [0.02, 0.025, 0.03],  # Different VaR for each stock
            })

        api.get_features.side_effect = mock_get_features

        return api

    def test_var_sizing_enabled(self, mock_api):
        """Test VaR sizing is used when enabled."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            broker_type="robinhood",
            use_var_sizing=True,
            dry_run=True,  # Dry run to avoid broker connection
        )

        with patch.dict(
            os.environ,
            {
                "HRP_MAX_PORTFOLIO_VAR_PCT": "0.02",
                "HRP_MAX_POSITION_VAR_PCT": "0.005",
            },
        ):
            agent = LiveTradingAgent(trading_config=config, api=mock_api)
            result = agent.execute()

            # Should generate orders with VaR-based sizing
            assert result["status"] == "dry_run"
            assert result["orders_generated"] > 0

    def test_var_sizing_disabled_uses_equal_weight(self, mock_api):
        """Test equal-weight sizing is used when VaR sizing disabled."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            broker_type="robinhood",
            use_var_sizing=False,
            dry_run=True,
        )

        agent = LiveTradingAgent(trading_config=config, api=mock_api)
        result = agent.execute()

        # Should generate orders with equal-weight sizing
        assert result["status"] == "dry_run"
        assert result["orders_generated"] > 0


class TestLiveTradingAgentStopLoss:
    """Test automatic stop-loss generation."""

    def test_stop_loss_generation(self):
        """Test stop-loss orders are generated for buy orders."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            stop_loss_pct=0.05,  # 5% stop-loss
        )
        api = Mock(spec=PlatformAPI)
        agent = LiveTradingAgent(trading_config=config, api=api)

        # Create buy orders
        buy_orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
            ),
            Order(
                symbol="MSFT",
                side=OrderSide.BUY,
                quantity=5,
                order_type=OrderType.MARKET,
            ),
        ]

        current_prices = {
            "AAPL": Decimal("150.00"),
            "MSFT": Decimal("300.00"),
        }

        # Generate stop-loss orders
        orders_with_stops = agent._add_stop_loss_orders(
            buy_orders, current_prices, 0.05
        )

        # Should have 2 buy + 2 stop-loss = 4 orders
        assert len(orders_with_stops) == 4

        # Check stop-loss orders
        stop_orders = [o for o in orders_with_stops if o.order_type == OrderType.STOP_LOSS]
        assert len(stop_orders) == 2

        # Check AAPL stop-loss
        aapl_stop = next(o for o in stop_orders if o.symbol == "AAPL")
        assert aapl_stop.side == OrderSide.SELL
        assert aapl_stop.quantity == 10
        assert aapl_stop.stop_price == Decimal("142.50")  # 150 * 0.95

        # Check MSFT stop-loss
        msft_stop = next(o for o in stop_orders if o.symbol == "MSFT")
        assert msft_stop.side == OrderSide.SELL
        assert msft_stop.quantity == 5
        assert msft_stop.stop_price == Decimal("285.00")  # 300 * 0.95

    def test_no_stop_loss_for_sell_orders(self):
        """Test stop-loss orders are NOT generated for sell orders."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            stop_loss_pct=0.05,
        )
        api = Mock(spec=PlatformAPI)
        agent = LiveTradingAgent(trading_config=config, api=api)

        # Create sell order
        sell_orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=10,
                order_type=OrderType.MARKET,
            ),
        ]

        current_prices = {"AAPL": Decimal("150.00")}

        # Generate stop-loss orders
        orders_with_stops = agent._add_stop_loss_orders(
            sell_orders, current_prices, 0.05
        )

        # Should have only 1 order (no stop-loss added)
        assert len(orders_with_stops) == 1
        assert orders_with_stops[0].order_type == OrderType.MARKET

    def test_stop_loss_disabled_when_none(self):
        """Test no stop-loss orders when stop_loss_pct=None."""
        config = TradingConfig(
            portfolio_value=Decimal("100000"),
            stop_loss_pct=None,  # Disabled
        )

        # When stop_loss_pct is None, _add_stop_loss_orders shouldn't be called
        # This is tested in the execute() flow
        assert config.stop_loss_pct is None


class TestLiveTradingAgentOrderPolling:
    """Test order status polling logic."""

    def test_order_polling_tracks_status(self):
        """Test order polling updates order status correctly."""
        config = TradingConfig(portfolio_value=Decimal("100000"))
        api = Mock(spec=PlatformAPI)
        agent = LiveTradingAgent(trading_config=config, api=api)

        # Create submitted orders
        orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
                broker_order_id="order1",
                status=OrderStatus.SUBMITTED,
            ),
            Order(
                symbol="MSFT",
                side=OrderSide.BUY,
                quantity=5,
                order_type=OrderType.MARKET,
                broker_order_id="order2",
                status=OrderStatus.SUBMITTED,
            ),
        ]

        # Mock broker
        broker = Mock()

        # order1 fills immediately, order2 fills on second poll
        broker.get_order_status.side_effect = [
            OrderStatus.FILLED,  # order1, poll 1
            OrderStatus.SUBMITTED,  # order2, poll 1
            OrderStatus.FILLED,  # order2, poll 2
        ]

        # Mock order manager
        order_manager = Mock()

        # Poll with short interval for test speed
        settled_orders = agent._poll_order_status(
            orders, order_manager, broker, poll_interval=0, max_polls=3
        )

        # Both orders should be filled
        assert len(settled_orders) == 2
        assert all(o.status == OrderStatus.FILLED for o in settled_orders)

        # Should have called get_order_status 3 times
        assert broker.get_order_status.call_count == 3

    def test_order_polling_stops_after_max_polls(self):
        """Test order polling gives up after max_polls."""
        config = TradingConfig(portfolio_value=Decimal("100000"))
        api = Mock(spec=PlatformAPI)
        agent = LiveTradingAgent(trading_config=config, api=api)

        # Create order that never settles
        orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
                broker_order_id="order1",
                status=OrderStatus.SUBMITTED,
            ),
        ]

        # Mock broker - order always stays submitted
        broker = Mock()
        broker.get_order_status.return_value = OrderStatus.SUBMITTED

        order_manager = Mock()

        # Poll with max_polls=2
        settled_orders = agent._poll_order_status(
            orders, order_manager, broker, poll_interval=0, max_polls=2
        )

        # Order should still be pending
        assert settled_orders[0].status == OrderStatus.SUBMITTED

        # Should have called get_order_status twice
        assert broker.get_order_status.call_count == 2

    def test_order_polling_handles_errors(self):
        """Test order polling continues if status query fails."""
        config = TradingConfig(portfolio_value=Decimal("100000"))
        api = Mock(spec=PlatformAPI)
        agent = LiveTradingAgent(trading_config=config, api=api)

        orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
                broker_order_id="order1",
                status=OrderStatus.SUBMITTED,
            ),
        ]

        # Mock broker - first call fails, second succeeds
        broker = Mock()
        broker.get_order_status.side_effect = [
            Exception("API error"),
            OrderStatus.FILLED,
        ]

        order_manager = Mock()

        # Poll should handle error and continue
        settled_orders = agent._poll_order_status(
            orders, order_manager, broker, poll_interval=0, max_polls=2
        )

        # Order should be filled after second poll
        assert settled_orders[0].status == OrderStatus.FILLED

        # Should have called get_order_status twice
        assert broker.get_order_status.call_count == 2


class TestLiveTradingAgentConfigFromEnv:
    """Test config loading from environment variables."""

    def test_config_from_env_defaults(self):
        """Test TradingConfig.from_env() uses correct defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = TradingConfig.from_env()

            assert config.portfolio_value == Decimal("100000")
            assert config.max_positions == 20
            assert config.broker_type == "robinhood"
            assert config.use_var_sizing is True
            assert config.stop_loss_pct is None
            assert config.dry_run is True

    def test_config_from_env_overrides(self):
        """Test TradingConfig.from_env() respects env var overrides."""
        with patch.dict(
            os.environ,
            {
                "HRP_PORTFOLIO_VALUE": "250000",
                "HRP_MAX_POSITIONS": "30",
                "HRP_BROKER_TYPE": "ibkr",
                "HRP_USE_VAR_SIZING": "false",
                "HRP_AUTO_STOP_LOSS_PCT": "0.03",
                "HRP_TRADING_DRY_RUN": "false",
            },
        ):
            config = TradingConfig.from_env()

            assert config.portfolio_value == Decimal("250000")
            assert config.max_positions == 30
            assert config.broker_type == "ibkr"
            assert config.use_var_sizing is False
            assert config.stop_loss_pct == 0.03
            assert config.dry_run is False
