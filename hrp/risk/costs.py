"""
Market impact cost model for realistic transaction cost estimation.

Implements square-root market impact model (industry standard) with
IBKR-style commission structure.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CostBreakdown:
    """Breakdown of transaction costs."""

    commission: float
    spread: float
    market_impact: float
    total: float
    total_pct: float


@dataclass
class MarketImpactModel:
    """
    Square-root market impact cost model.

    Cost Components:
        - Commission: IBKR tiered (per-share with min/max)
        - Spread: Half bid-ask spread in basis points
        - Market Impact: k * sigma * sqrt(shares / ADV)

    Attributes:
        eta: Market impact coefficient (default 0.1 for US large-cap)
        spread_bps: Half bid-ask spread in basis points
        commission_per_share: Per-share commission
        commission_min: Minimum commission per trade
        commission_max_pct: Maximum commission as % of trade value
    """

    eta: float = 0.1
    spread_bps: float = 5.0
    commission_per_share: float = 0.005
    commission_min: float = 1.00
    commission_max_pct: float = 0.005

    def estimate_cost(
        self,
        shares: int,
        price: float,
        adv: float,
        volatility: float,
    ) -> CostBreakdown:
        """
        Estimate transaction cost for a trade.

        Args:
            shares: Number of shares to trade
            price: Current share price
            adv: Average daily volume (shares)
            volatility: Daily volatility (decimal, e.g., 0.02 for 2%)

        Returns:
            CostBreakdown with commission, spread, market impact, and totals
        """
        trade_value = shares * price

        # Commission (IBKR tiered)
        per_share_cost = shares * self.commission_per_share
        max_commission = trade_value * self.commission_max_pct
        commission = max(self.commission_min, min(per_share_cost, max_commission))

        # Spread cost
        spread_cost = (self.spread_bps / 10_000) * trade_value

        # Market impact (square-root law)
        if adv > 0:
            participation_rate = shares / adv
            impact_cost = self.eta * volatility * np.sqrt(participation_rate) * trade_value
        else:
            # No volume data: assume 100% participation (very high impact)
            # Cap at 5% of trade value to avoid infinity
            impact_cost = min(0.05 * trade_value, self.eta * volatility * trade_value)

        total = commission + spread_cost + impact_cost
        total_pct = total / trade_value if trade_value > 0 else 0.0

        return CostBreakdown(
            commission=commission,
            spread=spread_cost,
            market_impact=impact_cost,
            total=total,
            total_pct=total_pct,
        )
