"""
Track record tracker for HRP advisory service.

Tracks recommendation outcomes with full transparency, computes
performance metrics, and generates weekly reports with honest
comparison to benchmarks.
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
class TrackRecordSummary:
    """Aggregated track record metrics."""

    period_start: date
    period_end: date
    total_recommendations: int
    closed_recommendations: int
    profitable: int
    unprofitable: int
    win_rate: float
    avg_return: float
    avg_win: float
    avg_loss: float
    best_pick: str | None
    best_return: float
    worst_pick: str | None
    worst_return: float
    total_return: float
    benchmark_return: float
    excess_return: float
    sharpe_ratio: float | None


@dataclass
class WeeklyReport:
    """Weekly recommendation report."""

    report_date: date
    new_recommendations: list[dict]
    open_positions: list[dict]
    closed_this_week: list[dict]
    track_record: TrackRecordSummary


class TrackRecordTracker:
    """Tracks and reports recommendation outcomes with full transparency."""

    def __init__(self, api: PlatformAPI):
        self.api = api

    def compute_track_record(
        self, start_date: date, end_date: date
    ) -> TrackRecordSummary:
        """
        Compute track record for a date range.

        Queries all closed recommendations and computes win rate,
        average returns, and comparison to SPY benchmark.
        """
        closed = self.api.query_readonly(
            "SELECT symbol, action, entry_price, close_price, "
            "realized_return, status, created_at, closed_at "
            "FROM recommendations "
            "WHERE closed_at IS NOT NULL "
            "AND created_at >= ? AND created_at <= ? "
            "ORDER BY closed_at",
            [start_date, end_date],
        )

        total_recs = self._count_total_recommendations(start_date, end_date)

        if closed.empty:
            return TrackRecordSummary(
                period_start=start_date,
                period_end=end_date,
                total_recommendations=total_recs,
                closed_recommendations=0,
                profitable=0,
                unprofitable=0,
                win_rate=0.0,
                avg_return=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                best_pick=None,
                best_return=0.0,
                worst_pick=None,
                worst_return=0.0,
                total_return=0.0,
                benchmark_return=self._get_benchmark_return(start_date, end_date),
                excess_return=0.0,
                sharpe_ratio=None,
            )

        returns = closed["realized_return"].astype(float)
        profitable = returns[returns > 0]
        unprofitable = returns[returns <= 0]

        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()

        avg_return = float(returns.mean())
        benchmark_return = self._get_benchmark_return(start_date, end_date)

        sharpe = None
        if len(returns) >= 5 and returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(52))

        return TrackRecordSummary(
            period_start=start_date,
            period_end=end_date,
            total_recommendations=total_recs,
            closed_recommendations=len(closed),
            profitable=len(profitable),
            unprofitable=len(unprofitable),
            win_rate=float(len(profitable) / len(closed)) if len(closed) > 0 else 0.0,
            avg_return=avg_return,
            avg_win=float(profitable.mean()) if len(profitable) > 0 else 0.0,
            avg_loss=float(unprofitable.mean()) if len(unprofitable) > 0 else 0.0,
            best_pick=closed.loc[best_idx, "symbol"],
            best_return=float(returns.max()),
            worst_pick=closed.loc[worst_idx, "symbol"],
            worst_return=float(returns.min()),
            total_return=float(returns.sum()),
            benchmark_return=benchmark_return,
            excess_return=avg_return - benchmark_return if benchmark_return else avg_return,
            sharpe_ratio=sharpe,
        )

    def generate_weekly_report(self, as_of_date: date) -> WeeklyReport:
        """
        Generate a weekly report with new recommendations, open positions,
        recent closings, and cumulative track record.
        """
        week_start = as_of_date - pd.Timedelta(days=7)

        # New recommendations this week
        new_recs = self.api.query_readonly(
            "SELECT recommendation_id, symbol, action, confidence, "
            "signal_strength, entry_price, target_price, stop_price, "
            "thesis_plain, risk_plain "
            "FROM recommendations "
            "WHERE created_at >= ? AND created_at <= ? "
            "ORDER BY signal_strength DESC",
            [week_start, as_of_date],
        )

        # Open positions
        open_recs = self.api.query_readonly(
            "SELECT recommendation_id, symbol, action, entry_price, "
            "signal_strength, created_at "
            "FROM recommendations WHERE status = 'active'"
        )

        # Closed this week
        closed_week = self.api.query_readonly(
            "SELECT recommendation_id, symbol, action, entry_price, "
            "close_price, realized_return, status "
            "FROM recommendations "
            "WHERE closed_at >= ? AND closed_at <= ?",
            [week_start, as_of_date],
        )

        # Cumulative track record (all time)
        track_record = self.compute_track_record(
            start_date=date(2020, 1, 1), end_date=as_of_date
        )

        return WeeklyReport(
            report_date=as_of_date,
            new_recommendations=new_recs.to_dict("records") if not new_recs.empty else [],
            open_positions=open_recs.to_dict("records") if not open_recs.empty else [],
            closed_this_week=closed_week.to_dict("records") if not closed_week.empty else [],
            track_record=track_record,
        )

    def compute_rolling_metrics(self, window_days: int = 90) -> pd.DataFrame:
        """Compute rolling win rate and excess return for trend analysis."""
        closed = self.api.query_readonly(
            "SELECT closed_at, realized_return FROM recommendations "
            "WHERE closed_at IS NOT NULL ORDER BY closed_at"
        )
        if closed.empty or len(closed) < 5:
            return pd.DataFrame()

        closed["closed_at"] = pd.to_datetime(closed["closed_at"])
        closed = closed.set_index("closed_at").sort_index()
        closed["realized_return"] = closed["realized_return"].astype(float)

        rolling = closed["realized_return"].rolling(f"{window_days}D", min_periods=5)

        result = pd.DataFrame(
            {
                "rolling_avg_return": rolling.mean(),
                "rolling_win_rate": rolling.apply(
                    lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
                ),
                "rolling_count": rolling.count(),
            }
        )
        return result.dropna()

    def persist_track_record(self, summary: TrackRecordSummary) -> None:
        """Write track record summary to database."""
        self.api.execute_write(
            "INSERT INTO track_record ("
            "period_start, period_end, total_recommendations, profitable, "
            "unprofitable, avg_return, avg_win, avg_loss, "
            "best_pick, worst_pick, benchmark_return, excess_return"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [
                summary.period_start, summary.period_end,
                summary.total_recommendations, summary.profitable,
                summary.unprofitable, summary.avg_return,
                summary.avg_win, summary.avg_loss,
                summary.best_pick, summary.worst_pick,
                summary.benchmark_return, summary.excess_return,
            ],
        )

    def _count_total_recommendations(self, start_date: date, end_date: date) -> int:
        """Count total recommendations (open + closed) in period."""
        result = self.api.fetchone_readonly(
            "SELECT COUNT(*) FROM recommendations "
            "WHERE created_at >= ? AND created_at <= ?",
            [start_date, end_date],
        )
        return result[0] if result else 0

    def _get_benchmark_return(self, start_date: date, end_date: date) -> float:
        """Get SPY return for the period as benchmark."""
        try:
            spy = self.api.get_prices(["SPY"], start_date, end_date)
            if spy.empty or len(spy) < 2:
                return 0.0
            first_close = float(spy.iloc[0]["close"])
            last_close = float(spy.iloc[-1]["close"])
            if first_close > 0:
                return (last_close - first_close) / first_close
        except Exception:
            pass
        return 0.0
