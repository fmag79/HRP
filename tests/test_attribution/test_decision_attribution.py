"""Tests for decision attribution."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from hrp.data.attribution.decision_attribution import (
    DecisionAttributor,
    RebalanceAnalyzer,
    TradeDecision,
)


class TestTradeDecision:
    """Tests for TradeDecision dataclass."""

    def test_valid_creation(self):
        """Test creating valid TradeDecision."""
        trade = TradeDecision(
            trade_id="TRADE-001",
            asset="AAPL",
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 10),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100.0,
            pnl=500.0,
        )
        assert trade.trade_id == "TRADE-001"
        assert trade.asset == "AAPL"
        assert trade.quantity == 100.0
        assert trade.pnl == 500.0

    def test_zero_quantity_raises(self):
        """Test that zero quantity raises ValueError."""
        with pytest.raises(ValueError, match="quantity cannot be zero"):
            TradeDecision(
                trade_id="TRADE-001",
                asset="AAPL",
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=150.0,
                exit_price=155.0,
                quantity=0.0,
                pnl=0.0,
            )

    def test_negative_entry_price_raises(self):
        """Test that negative entry price raises ValueError."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            TradeDecision(
                trade_id="TRADE-001",
                asset="AAPL",
                entry_date=datetime(2024, 1, 1),
                exit_date=None,
                entry_price=-150.0,
                exit_price=None,
                quantity=100.0,
                pnl=None,
            )

    def test_open_trade(self):
        """Test creating open trade (no exit)."""
        trade = TradeDecision(
            trade_id="TRADE-001",
            asset="AAPL",
            entry_date=datetime(2024, 1, 1),
            exit_date=None,
            entry_price=150.0,
            exit_price=None,
            quantity=100.0,
            pnl=None,
        )
        assert trade.exit_date is None
        assert trade.exit_price is None
        assert trade.pnl is None


class TestDecisionAttributor:
    """Tests for DecisionAttributor."""

    def test_attribute_trade_basic(self):
        """Test basic trade attribution."""
        trade = TradeDecision(
            trade_id="TRADE-001",
            asset="AAPL",
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 10),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100.0,
            pnl=None,  # Will be computed
        )

        attributor = DecisionAttributor()

        # Attribute with benchmark prices
        # Entry: bought at 150, benchmark was 151 → bad entry timing (-100)
        # Exit: sold at 155, benchmark was 154 → good exit timing (+100)
        attributed = attributor.attribute_trade(
            trade,
            benchmark_entry_price=151.0,
            benchmark_exit_price=154.0,
        )

        # Check P&L computation
        assert attributed.pnl == (155.0 - 150.0) * 100.0  # 500.0

        # Check timing attribution
        entry_timing = (151.0 - 150.0) * 100.0  # +100 (good entry)
        exit_timing = (155.0 - 154.0) * 100.0  # +100 (good exit)
        assert abs(attributed.timing_contribution - (entry_timing + exit_timing)) < 1e-6  # type: ignore

        # Check residual (should be close to 0 with no sizing attribution)
        assert abs(attributed.residual) < 1e-6  # type: ignore

    def test_attribute_trade_with_sizing(self):
        """Test trade attribution with position sizing component."""
        trade = TradeDecision(
            trade_id="TRADE-001",
            asset="AAPL",
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 10),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100.0,
            pnl=500.0,
        )

        attributor = DecisionAttributor()

        # Optimal size was 120 (undersized)
        attributed = attributor.attribute_trade(
            trade,
            benchmark_entry_price=150.0,  # Same as actual
            benchmark_exit_price=155.0,  # Same as actual
            optimal_quantity=120.0,
        )

        # Sizing attribution: (100 - 120) × (155 - 150) = -20 × 5 = -100
        assert abs(attributed.sizing_contribution - (-100.0)) < 1e-6  # type: ignore

        # Timing should be 0 (benchmark = actual)
        assert abs(attributed.timing_contribution) < 1e-6  # type: ignore

        # Residual should account for difference
        expected_residual = 500.0 - 0.0 - (-100.0)  # pnl - timing - sizing
        assert abs(attributed.residual - expected_residual) < 1e-6  # type: ignore

    def test_attribute_trade_open_raises(self):
        """Test that attributing open trade raises ValueError."""
        trade = TradeDecision(
            trade_id="TRADE-001",
            asset="AAPL",
            entry_date=datetime(2024, 1, 1),
            exit_date=None,
            entry_price=150.0,
            exit_price=None,
            quantity=100.0,
            pnl=None,
        )

        attributor = DecisionAttributor()

        with pytest.raises(ValueError, match="is not closed"):
            attributor.attribute_trade(trade)

    def test_attribute_portfolio(self):
        """Test attributing multiple trades."""
        trades = [
            TradeDecision(
                trade_id="TRADE-001",
                asset="AAPL",
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=150.0,
                exit_price=155.0,
                quantity=100.0,
                pnl=500.0,
            ),
            TradeDecision(
                trade_id="TRADE-002",
                asset="MSFT",
                entry_date=datetime(2024, 1, 2),
                exit_date=datetime(2024, 1, 11),
                entry_price=300.0,
                exit_price=295.0,
                quantity=50.0,
                pnl=-250.0,
            ),
            TradeDecision(
                trade_id="TRADE-003",
                asset="GOOGL",
                entry_date=datetime(2024, 1, 3),
                exit_date=None,  # Open trade, should be skipped
                entry_price=140.0,
                exit_price=None,
                quantity=75.0,
                pnl=None,
            ),
        ]

        attributor = DecisionAttributor()
        df = attributor.attribute_portfolio(trades)

        # Should have 2 rows (open trade excluded)
        assert len(df) == 2

        # Check columns
        expected_cols = [
            "trade_id",
            "asset",
            "entry_date",
            "exit_date",
            "pnl",
            "timing_contribution",
            "sizing_contribution",
            "residual",
        ]
        assert all(col in df.columns for col in expected_cols)

        # Check total P&L
        assert abs(df["pnl"].sum() - 250.0) < 1e-6  # 500 - 250

    def test_attribute_portfolio_with_benchmarks(self):
        """Test portfolio attribution with benchmark prices."""
        trades = [
            TradeDecision(
                trade_id="TRADE-001",
                asset="AAPL",
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=150.0,
                exit_price=155.0,
                quantity=100.0,
                pnl=500.0,
            ),
        ]

        # Benchmark prices
        benchmark_prices = pd.DataFrame(
            [
                {
                    "asset": "AAPL",
                    "date": datetime(2024, 1, 1).date(),
                    "price": 151.0,
                    "price_type": "close",
                },
                {
                    "asset": "AAPL",
                    "date": datetime(2024, 1, 10).date(),
                    "price": 154.0,
                    "price_type": "close",
                },
            ]
        )

        attributor = DecisionAttributor()
        df = attributor.attribute_portfolio(trades, benchmark_prices=benchmark_prices)

        # Timing should be non-zero (benchmarks differ from actuals)
        assert df["timing_contribution"].iloc[0] != 0

    def test_aggregate_by_component(self):
        """Test aggregating attribution by component."""
        trades = [
            TradeDecision(
                trade_id="TRADE-001",
                asset="AAPL",
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=150.0,
                exit_price=155.0,
                quantity=100.0,
                pnl=500.0,
            ),
            TradeDecision(
                trade_id="TRADE-002",
                asset="MSFT",
                entry_date=datetime(2024, 1, 2),
                exit_date=datetime(2024, 1, 11),
                entry_price=300.0,
                exit_price=295.0,
                quantity=50.0,
                pnl=-250.0,
            ),
        ]

        attributor = DecisionAttributor()
        df = attributor.attribute_portfolio(trades)

        # Aggregate
        aggregated = attributor.aggregate_by_component(df)

        # Should have timing, sizing, residual, total_pnl
        assert "timing" in aggregated
        assert "sizing" in aggregated
        assert "residual" in aggregated
        assert "total_pnl" in aggregated

        # Total P&L should match sum
        assert abs(aggregated["total_pnl"] - 250.0) < 1e-6  # 500 - 250


class TestRebalanceAnalyzer:
    """Tests for RebalanceAnalyzer."""

    def test_analyze_rebalance_event_basic(self):
        """Test basic rebalance analysis."""
        analyzer = RebalanceAnalyzer()

        pre_weights = {"AAPL": 0.5, "MSFT": 0.5}
        post_weights = {"AAPL": 0.7, "MSFT": 0.3}  # Increased AAPL
        returns = {"AAPL": 0.10, "MSFT": 0.05}  # AAPL outperformed

        result = analyzer.analyze_rebalance_event(
            date=datetime(2024, 1, 1),
            pre_rebalance_weights=pre_weights,
            post_rebalance_weights=post_weights,
            asset_returns_since_rebalance=returns,
        )

        # Check turnover (0.5 → 0.7 for AAPL = 0.2, 0.5 → 0.3 for MSFT = 0.2, total = 0.4/2 = 0.2)
        assert abs(result["turnover"] - 0.2) < 1e-6

        # Actual return: 0.7*0.10 + 0.3*0.05 = 0.085
        actual = 0.7 * 0.10 + 0.3 * 0.05
        assert abs(actual - 0.085) < 1e-6

        # Counterfactual return: 0.5*0.10 + 0.5*0.05 = 0.075
        counterfactual = 0.5 * 0.10 + 0.5 * 0.05
        assert abs(counterfactual - 0.075) < 1e-6

        # Value-add: 0.085 - 0.075 = 0.01
        assert abs(result["value_add"] - 0.01) < 1e-6

    def test_analyze_rebalance_event_negative_value_add(self):
        """Test rebalance that destroyed value."""
        analyzer = RebalanceAnalyzer()

        pre_weights = {"AAPL": 0.5, "MSFT": 0.5}
        post_weights = {"AAPL": 0.3, "MSFT": 0.7}  # Decreased AAPL
        returns = {"AAPL": 0.10, "MSFT": 0.05}  # AAPL outperformed

        result = analyzer.analyze_rebalance_event(
            date=datetime(2024, 1, 1),
            pre_rebalance_weights=pre_weights,
            post_rebalance_weights=post_weights,
            asset_returns_since_rebalance=returns,
        )

        # Value-add should be negative (wrong direction)
        assert result["value_add"] < 0

    def test_summarize_rebalancing(self):
        """Test summarizing multiple rebalance events."""
        analyzer = RebalanceAnalyzer()

        # Event 1
        analyzer.analyze_rebalance_event(
            date=datetime(2024, 1, 1),
            pre_rebalance_weights={"AAPL": 0.5, "MSFT": 0.5},
            post_rebalance_weights={"AAPL": 0.7, "MSFT": 0.3},
            asset_returns_since_rebalance={"AAPL": 0.10, "MSFT": 0.05},
        )

        # Event 2
        analyzer.analyze_rebalance_event(
            date=datetime(2024, 2, 1),
            pre_rebalance_weights={"AAPL": 0.7, "MSFT": 0.3},
            post_rebalance_weights={"AAPL": 0.5, "MSFT": 0.5},
            asset_returns_since_rebalance={"AAPL": 0.05, "MSFT": 0.08},
        )

        df = analyzer.summarize_rebalancing()

        # Should have 2 rows
        assert len(df) == 2

        # Check columns
        expected_cols = ["date", "turnover", "value_add", "optimal_value_add", "efficiency"]
        assert all(col in df.columns for col in expected_cols)

    def test_compute_aggregate_metrics(self):
        """Test computing aggregate rebalancing metrics."""
        analyzer = RebalanceAnalyzer()

        # Add multiple events
        for i in range(5):
            analyzer.analyze_rebalance_event(
                date=datetime(2024, 1, 1) + timedelta(days=30 * i),
                pre_rebalance_weights={"AAPL": 0.5, "MSFT": 0.5},
                post_rebalance_weights={"AAPL": 0.6, "MSFT": 0.4},
                asset_returns_since_rebalance={"AAPL": 0.10, "MSFT": 0.05},
            )

        metrics = analyzer.compute_aggregate_metrics()

        # Check metrics
        assert "total_value_add" in metrics
        assert "avg_turnover" in metrics
        assert "avg_efficiency" in metrics
        assert "n_rebalances" in metrics

        assert metrics["n_rebalances"] == 5
        assert metrics["avg_turnover"] > 0

    def test_empty_analyzer(self):
        """Test aggregate metrics with no events."""
        analyzer = RebalanceAnalyzer()

        # Empty summary
        df = analyzer.summarize_rebalancing()
        assert len(df) == 0

        # Empty metrics
        metrics = analyzer.compute_aggregate_metrics()
        assert metrics["n_rebalances"] == 0
        assert metrics["total_value_add"] == 0.0

    def test_perfect_foresight_optimal(self):
        """Test optimal value-add computation."""
        analyzer = RebalanceAnalyzer()

        pre_weights = {"AAPL": 0.5, "MSFT": 0.5}
        post_weights = {"AAPL": 0.6, "MSFT": 0.4}
        returns = {"AAPL": 0.15, "MSFT": 0.05}  # AAPL best

        result = analyzer.analyze_rebalance_event(
            date=datetime(2024, 1, 1),
            pre_rebalance_weights=pre_weights,
            post_rebalance_weights=post_weights,
            asset_returns_since_rebalance=returns,
        )

        # Optimal would be 100% AAPL → 0.15 return
        # Counterfactual: 0.5*0.15 + 0.5*0.05 = 0.10
        # Optimal value-add: 0.15 - 0.10 = 0.05
        assert abs(result["optimal_value_add"] - 0.05) < 1e-6

        # Efficiency: how close to optimal?
        # Actual: 0.6*0.15 + 0.4*0.05 = 0.11
        # Value-add: 0.11 - 0.10 = 0.01
        # Efficiency: 0.01 / 0.05 = 0.2 (20% of optimal)
        assert 0.0 <= result["efficiency"] <= 1.0
