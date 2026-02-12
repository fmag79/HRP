"""Decision-level attribution for trading strategies.

Attributes P&L to individual trading decisions:
- Entry timing
- Exit timing
- Position sizing
- Rebalancing decisions
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class TradeDecision:
    """Represents a single trade decision with P&L attribution.

    Attributes:
        trade_id: Unique identifier for the trade
        asset: Asset symbol
        entry_date: Entry timestamp
        exit_date: Exit timestamp (None if still open)
        entry_price: Entry price
        exit_price: Exit price (None if still open)
        quantity: Position size
        pnl: Realized P&L (None if still open)
        timing_contribution: P&L contribution from entry/exit timing
        sizing_contribution: P&L contribution from position sizing
        residual: Unexplained P&L component
    """

    trade_id: str
    asset: str
    entry_date: datetime
    exit_date: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: float
    pnl: float | None
    timing_contribution: float | None = None
    sizing_contribution: float | None = None
    residual: float | None = None

    def __post_init__(self):
        """Validate fields."""
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.exit_price is not None and self.exit_price <= 0:
            raise ValueError("exit_price must be positive if provided")


class DecisionAttributor:
    """Attributes P&L to individual trading decisions.

    Decomposes trade-level P&L into:
    - Timing: Entry and exit timing quality
    - Sizing: Position size optimality
    - Residual: Unexplained component
    """

    def __init__(
        self,
        benchmark_entry_method: Literal["open", "close", "vwap"] = "close",
        benchmark_exit_method: Literal["open", "close", "vwap"] = "close",
    ):
        """Initialize decision attributor.

        Args:
            benchmark_entry_method: Price to use as benchmark for entry timing
            benchmark_exit_method: Price to use as benchmark for exit timing
        """
        self.benchmark_entry_method = benchmark_entry_method
        self.benchmark_exit_method = benchmark_exit_method
        self.attribution_results_: list[TradeDecision] = []

    def attribute_trade(
        self,
        trade: TradeDecision,
        benchmark_entry_price: float | None = None,
        benchmark_exit_price: float | None = None,
        optimal_quantity: float | None = None,
    ) -> TradeDecision:
        """Attribute P&L for a single trade.

        Args:
            trade: TradeDecision object
            benchmark_entry_price: Benchmark price for entry (e.g., day's VWAP)
            benchmark_exit_price: Benchmark price for exit
            optimal_quantity: Optimal position size for sizing attribution

        Returns:
            Updated TradeDecision with attribution components filled in

        Raises:
            ValueError: If trade is not closed (exit_price/exit_date missing)
        """
        if trade.exit_price is None or trade.exit_date is None:
            raise ValueError(f"Trade {trade.trade_id} is not closed, cannot attribute P&L")

        # Compute actual P&L if not provided
        if trade.pnl is None:
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity

        # Default benchmarks to actual prices if not provided
        if benchmark_entry_price is None:
            benchmark_entry_price = trade.entry_price
        if benchmark_exit_price is None:
            benchmark_exit_price = trade.exit_price

        # 1. Timing attribution
        # Entry timing: (benchmark_entry - actual_entry) * quantity
        entry_timing = (benchmark_entry_price - trade.entry_price) * trade.quantity

        # Exit timing: (actual_exit - benchmark_exit) * quantity
        exit_timing = (trade.exit_price - benchmark_exit_price) * trade.quantity

        total_timing = entry_timing + exit_timing

        # 2. Sizing attribution
        if optimal_quantity is not None:
            # Compare actual size vs optimal size
            # Sizing attribution = (actual_quantity - optimal_quantity) × (exit - entry)
            sizing_attr = (
                (trade.quantity - optimal_quantity)
                * (trade.exit_price - trade.entry_price)
            )
        else:
            sizing_attr = 0.0

        # 3. Residual (should be small if attribution is complete)
        residual = trade.pnl - total_timing - sizing_attr

        # Update trade with attribution
        trade.timing_contribution = total_timing
        trade.sizing_contribution = sizing_attr
        trade.residual = residual

        return trade

    def attribute_portfolio(
        self,
        trades: list[TradeDecision],
        benchmark_prices: pd.DataFrame | None = None,
        optimal_sizes: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Attribute P&L for a portfolio of trades.

        Args:
            trades: List of TradeDecision objects
            benchmark_prices: DataFrame with columns ['asset', 'date', 'price', 'price_type']
                            where price_type is 'open', 'close', or 'vwap'
            optimal_sizes: Dictionary mapping trade_id -> optimal quantity

        Returns:
            DataFrame with columns: trade_id, asset, pnl, timing_contribution,
                                   sizing_contribution, residual

        Raises:
            ValueError: If any trade is not closed
        """
        results = []

        for trade in trades:
            # Skip open trades
            if trade.exit_price is None or trade.exit_date is None:
                continue

            # Get benchmark prices if provided
            benchmark_entry = None
            benchmark_exit = None

            if benchmark_prices is not None:
                # Find benchmark entry price
                entry_mask = (
                    (benchmark_prices["asset"] == trade.asset)
                    & (benchmark_prices["date"] == trade.entry_date.date())
                    & (benchmark_prices["price_type"] == self.benchmark_entry_method)
                )
                if entry_mask.any():
                    benchmark_entry = benchmark_prices.loc[entry_mask, "price"].iloc[0]

                # Find benchmark exit price
                exit_mask = (
                    (benchmark_prices["asset"] == trade.asset)
                    & (benchmark_prices["date"] == trade.exit_date.date())
                    & (benchmark_prices["price_type"] == self.benchmark_exit_method)
                )
                if exit_mask.any():
                    benchmark_exit = benchmark_prices.loc[exit_mask, "price"].iloc[0]

            # Get optimal size if provided
            optimal_qty = optimal_sizes.get(trade.trade_id) if optimal_sizes else None

            # Attribute trade
            attributed_trade = self.attribute_trade(
                trade, benchmark_entry, benchmark_exit, optimal_qty
            )

            # Store result
            results.append(
                {
                    "trade_id": attributed_trade.trade_id,
                    "asset": attributed_trade.asset,
                    "entry_date": attributed_trade.entry_date,
                    "exit_date": attributed_trade.exit_date,
                    "pnl": attributed_trade.pnl,
                    "timing_contribution": attributed_trade.timing_contribution,
                    "sizing_contribution": attributed_trade.sizing_contribution,
                    "residual": attributed_trade.residual,
                }
            )

        # Store results
        self.attribution_results_ = [
            r for r in trades if r.timing_contribution is not None
        ]

        return pd.DataFrame(results)

    def aggregate_by_component(self, df: pd.DataFrame) -> dict[str, float]:
        """Aggregate attribution results by component type.

        Args:
            df: DataFrame returned by attribute_portfolio

        Returns:
            Dictionary mapping component -> total P&L contribution
        """
        return {
            "timing": df["timing_contribution"].sum(),
            "sizing": df["sizing_contribution"].sum(),
            "residual": df["residual"].sum(),
            "total_pnl": df["pnl"].sum(),
        }


class RebalanceAnalyzer:
    """Analyzes value-add from portfolio rebalancing events.

    Compares actual rebalancing decisions against:
    - Buy-and-hold baseline
    - Constant-weight rebalancing
    - Threshold-based rebalancing
    """

    def __init__(self):
        """Initialize rebalance analyzer."""
        self.rebalance_events_: list[dict] = []

    def analyze_rebalance_event(
        self,
        date: datetime,
        pre_rebalance_weights: dict[str, float],
        post_rebalance_weights: dict[str, float],
        asset_returns_since_rebalance: dict[str, float],
    ) -> dict[str, float]:
        """Analyze a single rebalance event.

        Args:
            date: Rebalance date
            pre_rebalance_weights: Asset weights before rebalancing
            post_rebalance_weights: Asset weights after rebalancing
            asset_returns_since_rebalance: Returns from rebalance to evaluation date

        Returns:
            Dictionary with:
                - turnover: Portfolio turnover from rebalancing
                - value_add: P&L added by rebalancing vs holding pre-rebalance weights
                - optimal_value_add: P&L from perfect foresight rebalancing
        """
        # Compute turnover
        turnover = sum(
            abs(post_rebalance_weights.get(asset, 0) - pre_rebalance_weights.get(asset, 0))
            for asset in set(pre_rebalance_weights) | set(post_rebalance_weights)
        ) / 2

        # Actual return with rebalancing
        actual_return = sum(
            post_rebalance_weights.get(asset, 0) * asset_returns_since_rebalance.get(asset, 0)
            for asset in post_rebalance_weights
        )

        # Counterfactual return (hold pre-rebalance weights)
        counterfactual_return = sum(
            pre_rebalance_weights.get(asset, 0) * asset_returns_since_rebalance.get(asset, 0)
            for asset in pre_rebalance_weights
        )

        # Value-add from rebalancing
        value_add = actual_return - counterfactual_return

        # Optimal rebalancing (perfect foresight: max weight on best asset)
        best_asset = max(asset_returns_since_rebalance, key=asset_returns_since_rebalance.get)
        optimal_return = asset_returns_since_rebalance[best_asset]
        optimal_value_add = optimal_return - counterfactual_return

        # Store event
        event = {
            "date": date,
            "turnover": turnover,
            "value_add": value_add,
            "optimal_value_add": optimal_value_add,
            "efficiency": value_add / optimal_value_add if optimal_value_add != 0 else 0.0,
        }
        self.rebalance_events_.append(event)

        return event

    def summarize_rebalancing(self) -> pd.DataFrame:
        """Summarize all rebalancing events.

        Returns:
            DataFrame with columns: date, turnover, value_add, optimal_value_add, efficiency
        """
        if not self.rebalance_events_:
            return pd.DataFrame()

        return pd.DataFrame(self.rebalance_events_)

    def compute_aggregate_metrics(self) -> dict[str, float]:
        """Compute aggregate rebalancing metrics.

        Returns:
            Dictionary with:
                - total_value_add: Sum of value-add across all rebalances
                - avg_turnover: Average turnover per rebalance
                - avg_efficiency: Average rebalancing efficiency
                - n_rebalances: Number of rebalancing events
        """
        if not self.rebalance_events_:
            return {
                "total_value_add": 0.0,
                "avg_turnover": 0.0,
                "avg_efficiency": 0.0,
                "n_rebalances": 0,
            }

        df = self.summarize_rebalancing()

        return {
            "total_value_add": df["value_add"].sum(),
            "avg_turnover": df["turnover"].mean(),
            "avg_efficiency": df["efficiency"].mean(),
            "n_rebalances": len(df),
        }
