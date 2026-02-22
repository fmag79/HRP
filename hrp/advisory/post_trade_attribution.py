"""
Post-trade attribution for the advisory service.

Decomposes closed recommendation outcomes into signal quality,
timing, sizing, and cost components to feed back into model improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hrp.api.platform import PlatformAPI


@dataclass
class Attribution:
    """Decomposition of a single recommendation outcome."""

    recommendation_id: str
    symbol: str
    total_return: float
    signal_correct: bool  # Was direction right?
    signal_contribution: float  # Return from directional call
    timing_contribution: float  # Return from entry/exit timing vs. period average
    sizing_contribution: float  # Impact of position size vs. equal-weight
    benchmark_return: float  # What SPY did over same period


@dataclass
class AggregateAttribution:
    """Aggregated attribution across many recommendations."""

    total_recommendations: int
    signal_accuracy: float  # % of correct directional calls
    avg_signal_contribution: float
    avg_timing_contribution: float
    avg_sizing_contribution: float
    avg_benchmark_return: float
    information_ratio: float | None  # Excess return / tracking error


class PostTradeAttributor:
    """Decomposes recommendation outcomes into signal, timing, and sizing."""

    def __init__(self, api: PlatformAPI):
        self.api = api

    def attribute(self, recommendation_id: str) -> Attribution | None:
        """
        Attribute a single closed recommendation's return.

        Decomposition:
        - signal_contribution: Did we get the direction right?
        - timing_contribution: Did we enter/exit at good prices vs period average?
        - sizing_contribution: Did position size help or hurt vs equal-weight?
        """
        rec = self.api.fetchone_readonly(
            "SELECT recommendation_id, symbol, action, entry_price, close_price, "
            "realized_return, position_pct, created_at, closed_at "
            "FROM recommendations WHERE recommendation_id = ? AND status != 'active'",
            [recommendation_id],
        )
        if not rec:
            return None

        rec_id, symbol, action, entry_price, close_price = rec[0], rec[1], rec[2], float(rec[3]), float(rec[4])
        realized_return = float(rec[5])
        position_pct = float(rec[6])
        created_at = rec[7]
        closed_at = rec[8]

        # Get benchmark return over same period
        benchmark_return = self._get_benchmark_return(created_at, closed_at)

        # Signal contribution: was direction correct?
        if action == "BUY":
            signal_correct = close_price > entry_price
        else:
            signal_correct = close_price < entry_price

        # Timing: compare actual entry/exit to period average price
        avg_price = self._get_average_price(symbol, created_at, closed_at)
        if avg_price and avg_price > 0 and entry_price > 0:
            timing_contribution = (avg_price - entry_price) / entry_price
        else:
            timing_contribution = 0.0

        # Sizing: compare weighted return to equal-weight return
        equal_weight = 1.0 / 20  # Assume 20-position portfolio
        sizing_contribution = realized_return * (position_pct - equal_weight)

        # Signal is the residual
        signal_contribution = realized_return - timing_contribution - sizing_contribution

        return Attribution(
            recommendation_id=rec_id,
            symbol=symbol,
            total_return=realized_return,
            signal_correct=signal_correct,
            signal_contribution=signal_contribution,
            timing_contribution=timing_contribution,
            sizing_contribution=sizing_contribution,
            benchmark_return=benchmark_return,
        )

    def aggregate_attribution(
        self, start_date: date, end_date: date
    ) -> AggregateAttribution | None:
        """Aggregate attribution across all closed recommendations in period."""
        closed = self.api.query_readonly(
            "SELECT recommendation_id FROM recommendations "
            "WHERE closed_at IS NOT NULL AND closed_at >= ? AND closed_at <= ?",
            [start_date, end_date],
        )
        if closed.empty:
            return None

        attributions = []
        for _, row in closed.iterrows():
            attr = self.attribute(row["recommendation_id"])
            if attr:
                attributions.append(attr)

        if not attributions:
            return None

        n = len(attributions)
        signal_accuracy = sum(1 for a in attributions if a.signal_correct) / n
        avg_signal = np.mean([a.signal_contribution for a in attributions])
        avg_timing = np.mean([a.timing_contribution for a in attributions])
        avg_sizing = np.mean([a.sizing_contribution for a in attributions])
        avg_bench = np.mean([a.benchmark_return for a in attributions])

        # Information ratio
        excess_returns = [a.total_return - a.benchmark_return for a in attributions]
        tracking_error = np.std(excess_returns) if len(excess_returns) > 1 else None
        ir = None
        if tracking_error and tracking_error > 0:
            ir = float(np.mean(excess_returns) / tracking_error)

        return AggregateAttribution(
            total_recommendations=n,
            signal_accuracy=signal_accuracy,
            avg_signal_contribution=float(avg_signal),
            avg_timing_contribution=float(avg_timing),
            avg_sizing_contribution=float(avg_sizing),
            avg_benchmark_return=float(avg_bench),
            information_ratio=ir,
        )

    def _get_benchmark_return(self, start_date, end_date) -> float:
        """Get SPY return between two dates."""
        try:
            if isinstance(start_date, str):
                start_date = date.fromisoformat(start_date[:10])
            if isinstance(end_date, str):
                end_date = date.fromisoformat(end_date[:10])

            spy = self.api.get_prices(["SPY"], start_date, end_date)
            if spy.empty or len(spy) < 2:
                return 0.0
            first = float(spy.iloc[0]["close"])
            last = float(spy.iloc[-1]["close"])
            return (last - first) / first if first > 0 else 0.0
        except Exception:
            return 0.0

    def _get_average_price(self, symbol: str, start_date, end_date) -> float | None:
        """Get average close price over a period."""
        try:
            if isinstance(start_date, str):
                start_date = date.fromisoformat(start_date[:10])
            if isinstance(end_date, str):
                end_date = date.fromisoformat(end_date[:10])

            prices = self.api.get_prices([symbol], start_date, end_date)
            if prices.empty:
                return None
            return float(prices["close"].astype(float).mean())
        except Exception:
            return None
