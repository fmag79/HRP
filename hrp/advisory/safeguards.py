"""
Pre-trade safety checks and circuit breakers for the advisory service.

Prevents recommendations from being generated or executed under
dangerous conditions: stale data, extreme market regimes, excessive
concentration, or portfolio-level drawdowns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hrp.api.platform import PlatformAPI


@dataclass
class CheckResult:
    """Result of a pre-trade safety check."""

    passed: bool
    check_name: str
    message: str
    severity: str = "info"  # info, warning, error


class PreTradeChecks:
    """Sanity checks before recommendation generation or execution."""

    def __init__(self, api: PlatformAPI):
        self.api = api

    def check_data_freshness(self, as_of_date: date) -> CheckResult:
        """
        Verify that price data is current (not stale by more than 1 trading day).

        Running models on stale data produces stale (potentially harmful) signals.
        """
        result = self.api.fetchone_readonly(
            "SELECT MAX(date) FROM prices"
        )
        if not result or result[0] is None:
            return CheckResult(
                passed=False,
                check_name="data_freshness",
                message="No price data found in database",
                severity="error",
            )

        latest_date = result[0]
        if isinstance(latest_date, str):
            latest_date = date.fromisoformat(latest_date)

        # Allow 3 calendar days gap (weekends + holidays)
        gap = (as_of_date - latest_date).days
        if gap > 3:
            return CheckResult(
                passed=False,
                check_name="data_freshness",
                message=(
                    f"Price data is {gap} days old (latest: {latest_date}). "
                    f"Recommendations require data no older than 3 calendar days."
                ),
                severity="error",
            )

        return CheckResult(
            passed=True,
            check_name="data_freshness",
            message=f"Price data is current (latest: {latest_date})",
        )

    def check_market_regime(self, as_of_date: date) -> CheckResult:
        """
        Detect extreme market conditions that may invalidate model predictions.

        Checks: recent S&P 500 drawdown exceeding -5% in 5 days.
        """
        try:
            start = as_of_date - timedelta(days=10)
            spy = self.api.get_prices(["SPY"], start, as_of_date)
            if spy.empty or len(spy) < 2:
                return CheckResult(
                    passed=True,
                    check_name="market_regime",
                    message="Insufficient SPY data for regime check",
                    severity="warning",
                )

            closes = spy["close"].astype(float)
            five_day_return = (closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5] if len(closes) >= 5 else 0

            if five_day_return < -0.05:
                return CheckResult(
                    passed=False,
                    check_name="market_regime",
                    message=(
                        f"Extreme market stress detected: SPY 5-day return = {five_day_return:.1%}. "
                        f"Recommendations suspended during extreme conditions."
                    ),
                    severity="error",
                )

            return CheckResult(
                passed=True,
                check_name="market_regime",
                message=f"Market conditions normal (SPY 5d return: {five_day_return:+.1%})",
            )
        except Exception as e:
            return CheckResult(
                passed=True,
                check_name="market_regime",
                message=f"Could not check market regime: {e}",
                severity="warning",
            )

    def check_portfolio_concentration(
        self, weights: dict[str, float], sector_map: dict[str, str]
    ) -> CheckResult:
        """Verify no single sector dominates the portfolio."""
        max_sector_pct = float(os.getenv("HRP_PORTFOLIO_MAX_SECTOR_PCT", "0.30"))

        sector_exposure: dict[str, float] = {}
        for symbol, weight in weights.items():
            sector = sector_map.get(symbol, "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

        for sector, exposure in sector_exposure.items():
            if exposure > max_sector_pct:
                return CheckResult(
                    passed=False,
                    check_name="portfolio_concentration",
                    message=(
                        f"Sector {sector} has {exposure:.1%} weight, "
                        f"exceeding {max_sector_pct:.0%} limit."
                    ),
                    severity="error",
                )

        return CheckResult(
            passed=True,
            check_name="portfolio_concentration",
            message="Portfolio sector concentration within limits",
        )

    def run_all_checks(self, as_of_date: date) -> list[CheckResult]:
        """Run all pre-trade checks and return results."""
        results = [
            self.check_data_freshness(as_of_date),
            self.check_market_regime(as_of_date),
        ]
        return results


class CircuitBreaker:
    """Emergency halt mechanism for the advisory service."""

    def __init__(self, api: PlatformAPI):
        self.api = api
        self.daily_loss_limit = float(
            os.getenv("HRP_CIRCUIT_BREAKER_DAILY_LOSS", "-0.03")
        )
        self.weekly_loss_limit = float(
            os.getenv("HRP_CIRCUIT_BREAKER_WEEKLY_LOSS", "-0.05")
        )

    def should_halt(self, as_of_date: date) -> tuple[bool, str]:
        """
        Check if trading should be halted.

        Returns (should_halt, reason).
        """
        # Check portfolio daily P&L
        daily_return = self._get_portfolio_daily_return(as_of_date)
        if daily_return is not None and daily_return < self.daily_loss_limit:
            return True, (
                f"Daily loss circuit breaker: portfolio return {daily_return:.1%} "
                f"exceeds limit {self.daily_loss_limit:.1%}"
            )

        # Check 5-day rolling
        weekly_return = self._get_portfolio_weekly_return(as_of_date)
        if weekly_return is not None and weekly_return < self.weekly_loss_limit:
            return True, (
                f"Weekly loss circuit breaker: 5-day return {weekly_return:.1%} "
                f"exceeds limit {self.weekly_loss_limit:.1%}"
            )

        return False, "All clear"

    def should_reduce_size(self, as_of_date: date) -> tuple[bool, float]:
        """
        Check if position sizes should be reduced.

        Returns (should_reduce, size_multiplier).
        """
        weekly_return = self._get_portfolio_weekly_return(as_of_date)
        if weekly_return is not None:
            # Linearly reduce between -3% and -5%
            if self.weekly_loss_limit < weekly_return < self.daily_loss_limit:
                # Interpolate: at -3% → 0.75x, at -5% → 0.5x
                pct = (weekly_return - self.daily_loss_limit) / (
                    self.weekly_loss_limit - self.daily_loss_limit
                )
                multiplier = 1.0 - (0.5 * pct)
                return True, max(0.5, multiplier)

        return False, 1.0

    def _get_portfolio_daily_return(self, as_of_date: date) -> float | None:
        """Compute portfolio daily return from active recommendations."""
        active = self.api.query_readonly(
            "SELECT symbol, entry_price, position_pct "
            "FROM recommendations WHERE status = 'active'"
        )
        if active.empty:
            return None

        total_return = 0.0
        for _, row in active.iterrows():
            symbol = row["symbol"]
            entry = float(row["entry_price"])
            weight = float(row["position_pct"])

            yesterday = as_of_date - timedelta(days=1)
            prices = self.api.get_prices([symbol], yesterday, as_of_date)
            if len(prices) >= 2:
                prev_close = float(prices.iloc[-2]["close"])
                curr_close = float(prices.iloc[-1]["close"])
                if prev_close > 0:
                    total_return += weight * (curr_close - prev_close) / prev_close

        return total_return

    def _get_portfolio_weekly_return(self, as_of_date: date) -> float | None:
        """Compute 5-day portfolio return."""
        active = self.api.query_readonly(
            "SELECT symbol, entry_price, position_pct "
            "FROM recommendations WHERE status = 'active'"
        )
        if active.empty:
            return None

        total_return = 0.0
        start = as_of_date - timedelta(days=7)
        for _, row in active.iterrows():
            symbol = row["symbol"]
            weight = float(row["position_pct"])

            prices = self.api.get_prices([symbol], start, as_of_date)
            if len(prices) >= 2:
                first_close = float(prices.iloc[0]["close"])
                last_close = float(prices.iloc[-1]["close"])
                if first_close > 0:
                    total_return += weight * (last_close - first_close) / first_close

        return total_return
