"""
Data quality checks for HRP.

Provides individual quality check classes for:
- Price anomaly detection (>50% moves without corporate actions)
- Completeness checks (missing prices for active symbols)
- Gap detection (missing dates in price history)
- Stale data detection (symbols not updated recently)

Each check returns a standardized QualityIssue result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.utils.calendar import get_trading_days


class IssueSeverity(Enum):
    """Severity levels for quality issues."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class QualityIssue:
    """
    Represents a single data quality issue.

    Immutable after creation to ensure issues cannot be modified
    after being recorded.
    """

    check_name: str
    severity: IssueSeverity
    symbol: str | None
    date: date | None
    description: str
    details: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self):
        """Validate required fields."""
        if not self.check_name:
            raise ValueError("check_name cannot be empty")
        if not self.description:
            raise ValueError("description cannot be empty")

    @classmethod
    def create(
        cls,
        check_name: str,
        severity: IssueSeverity,
        description: str,
        symbol: str | None = None,
        date: date | None = None,
        details: dict[str, Any] | None = None,
    ) -> "QualityIssue":
        """
        Factory method for creating QualityIssue with dict-based details.

        Args:
            check_name: Name of the check that found this issue
            severity: Issue severity level
            description: Human-readable description of the issue
            symbol: Optional ticker symbol related to the issue
            date: Optional date when issue was detected
            details: Optional dict of additional details (converted to tuple)

        Returns:
            QualityIssue instance
        """
        details_tuple = tuple(details.items()) if details else ()
        return cls(check_name, severity, symbol, date, description, details_tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "check_name": self.check_name,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "date": str(self.date) if self.date else None,
            "description": self.description,
            "details": dict(self.details) if self.details else {},
        }


@dataclass
class CheckResult:
    """Result of running a quality check."""

    check_name: str
    issues: list[QualityIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    run_time_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """Check passes if no critical issues."""
        return self.critical_count == 0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count of warning issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)


class QualityCheck(ABC):
    """Base class for data quality checks."""

    name: str = "base_check"
    description: str = "Base quality check"

    def __init__(self, db_path: str | None = None, read_only: bool = True):
        # Default to read-only connection since checks only read data
        # This allows checks to run even when MCP server holds write lock
        # Pass read_only=False in tests that already have read-write connections
        self._db = get_db(db_path, read_only=read_only)
        self._read_only = read_only

    @abstractmethod
    def run(self, as_of_date: date) -> CheckResult:
        """
        Run the quality check.

        Args:
            as_of_date: Date to run the check for.

        Returns:
            CheckResult with pass/fail status and any issues found.
        """
        pass


class PriceAnomalyCheck(QualityCheck):
    """
    Detects abnormal price movements without corporate actions.

    Flags price changes >50% day-over-day that don't have a corresponding
    stock split or dividend in the corporate_actions table.
    """

    name = "price_anomaly"
    description = "Detects price changes >50% without corporate actions"

    def __init__(
        self,
        db_path: str | None = None,
        threshold: float = 0.5,
        lookback_days: int = 1,
        read_only: bool = True,
    ):
        super().__init__(db_path, read_only=read_only)
        self.threshold = threshold  # 50% by default
        self.lookback_days = lookback_days

    def run(self, as_of_date: date) -> CheckResult:
        """Check for anomalous price movements."""
        import time

        start_time = time.time()
        issues = []

        # Get price changes with corporate action info
        query = """
            WITH price_changes AS (
                SELECT
                    p.symbol,
                    p.date,
                    p.close,
                    LAG(p.close) OVER (PARTITION BY p.symbol ORDER BY p.date) as prev_close,
                    ca.action_type,
                    ca.factor
                FROM prices p
                LEFT JOIN corporate_actions ca
                    ON p.symbol = ca.symbol AND p.date = ca.date
                WHERE p.date BETWEEN ? AND ?
            )
            SELECT
                symbol,
                date,
                close,
                prev_close,
                (close - prev_close) / prev_close as pct_change,
                action_type,
                factor
            FROM price_changes
            WHERE prev_close IS NOT NULL
              AND ABS((close - prev_close) / prev_close) > ?
              AND action_type IS NULL
            ORDER BY ABS((close - prev_close) / prev_close) DESC
        """

        start_date = as_of_date - timedelta(days=self.lookback_days + 30)
        results = self._db.fetchall(query, (start_date, as_of_date, self.threshold))

        for row in results:
            symbol, dt, close, prev_close, pct_change, _, _ = row
            severity = IssueSeverity.CRITICAL if abs(pct_change) > 0.9 else IssueSeverity.WARNING

            issues.append(
                QualityIssue.create(
                    check_name=self.name,
                    severity=severity,
                    symbol=symbol,
                    date=dt,
                    description=f"Price changed {pct_change:.1%} without corporate action",
                    details={
                        "previous_close": float(prev_close),
                        "current_close": float(close),
                        "pct_change": float(pct_change),
                    },
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000

        return CheckResult(
            check_name=self.name,
            issues=issues,
            stats={"anomalies_found": len(issues), "threshold": self.threshold},
            run_time_ms=elapsed_ms,
        )


class CompletenessCheck(QualityCheck):
    """
    Checks for missing prices for active universe symbols.

    Compares the active universe against available price data
    to find symbols that should have prices but don't.
    """

    name = "completeness"
    description = "Checks for missing prices for active symbols"

    def run(self, as_of_date: date) -> CheckResult:
        """Check for missing price data."""
        import time

        start_time = time.time()
        issues = []

        # Get active universe symbols
        universe_query = """
            WITH latest_universe AS (
                SELECT symbol, MAX(date) as latest_date
                FROM universe
                WHERE date <= ? AND in_universe = TRUE
                GROUP BY symbol
            )
            SELECT u.symbol
            FROM latest_universe lu
            JOIN universe u ON lu.symbol = u.symbol AND lu.latest_date = u.date
            WHERE u.in_universe = TRUE
        """
        universe_result = self._db.fetchall(universe_query, (as_of_date,))
        universe_symbols = {row[0] for row in universe_result}

        if not universe_symbols:
            # No universe defined, check all symbols with recent prices
            symbols_query = """
                SELECT DISTINCT symbol
                FROM prices
                WHERE date >= ?
            """
            recent_date = as_of_date - timedelta(days=30)
            universe_result = self._db.fetchall(symbols_query, (recent_date,))
            universe_symbols = {row[0] for row in universe_result}

        # Get symbols with prices on the given date
        prices_query = """
            SELECT DISTINCT symbol
            FROM prices
            WHERE date = ?
        """
        prices_result = self._db.fetchall(prices_query, (as_of_date,))
        symbols_with_prices = {row[0] for row in prices_result}

        # Find missing symbols
        missing_symbols = universe_symbols - symbols_with_prices

        for symbol in sorted(missing_symbols):
            issues.append(
                QualityIssue.create(
                    check_name=self.name,
                    severity=IssueSeverity.WARNING,
                    symbol=symbol,
                    date=as_of_date,
                    description="Missing price data for active symbol",
                    details={"expected_date": str(as_of_date)},
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000

        return CheckResult(
            check_name=self.name,
            issues=issues,
            stats={
                "universe_size": len(universe_symbols),
                "symbols_with_prices": len(symbols_with_prices),
                "missing_count": len(missing_symbols),
            },
            run_time_ms=elapsed_ms,
        )


class GapDetectionCheck(QualityCheck):
    """
    Detects gaps in price history (missing trading days).

    Identifies symbols with missing dates in their price history,
    accounting for weekends and market holidays.
    """

    name = "gap_detection"
    description = "Detects missing dates in price history"

    def __init__(self, db_path: str | None = None, lookback_days: int = 30, read_only: bool = True):
        super().__init__(db_path, read_only=read_only)
        self.lookback_days = lookback_days

    def run(self, as_of_date: date) -> CheckResult:
        """Check for gaps in price history."""
        import time

        start_time = time.time()
        issues = []

        start_date = as_of_date - timedelta(days=self.lookback_days)

        # Get NYSE trading days using the calendar (excludes weekends and holidays)
        trading_days_index = get_trading_days(start_date, as_of_date)

        # Exclude the last trading day from expected days
        # (data may not be available yet for the most recent trading day)
        if len(trading_days_index) > 1:
            trading_days_index = trading_days_index[:-1]

        trading_dates = {d.date() for d in trading_days_index}

        if len(trading_dates) < 2:
            elapsed_ms = (time.time() - start_time) * 1000
            return CheckResult(
                check_name=self.name,
                issues=[],
                stats={"trading_days": len(trading_dates), "symbols_checked": 0},
                run_time_ms=elapsed_ms,
            )

        # Get symbols that might have gaps
        # First get all symbols with prices in the date range
        symbols_query = """
            SELECT DISTINCT symbol
            FROM prices
            WHERE date BETWEEN ? AND ?
        """
        symbols_result = self._db.fetchall(symbols_query, (start_date, as_of_date))
        all_symbols = [row[0] for row in symbols_result]

        for symbol in all_symbols:
            # Get the actual dates this symbol has data for
            symbol_dates_query = """
                SELECT DISTINCT date FROM prices
                WHERE symbol = ? AND date BETWEEN ? AND ?
            """
            symbol_dates_result = self._db.fetchall(
                symbol_dates_query, (symbol, start_date, as_of_date)
            )
            symbol_dates = {row[0] for row in symbol_dates_result}

            # Count only NYSE trading days (exclude weekends and holidays)
            symbol_nyse_dates = symbol_dates & trading_dates
            price_days = len(symbol_nyse_dates)

            # Count missing NYSE trading days
            missing_nyse_days = trading_dates - symbol_dates
            missing_days = len(missing_nyse_days)

            # Expect at least 80% of trading days
            expected_days = int(len(trading_dates) * 0.8)

            # Expect at least 80% of trading days
            expected_days = int(len(trading_dates) * 0.8)

            if price_days < expected_days and missing_days > 0:
                # Get sample of missing dates
                missing_dates = sorted(list(missing_nyse_days))[:5]  # First 5 missing

                severity = IssueSeverity.CRITICAL if missing_days > 10 else IssueSeverity.WARNING

                issues.append(
                    QualityIssue.create(
                        check_name=self.name,
                        severity=severity,
                        symbol=symbol,
                        date=as_of_date,
                        description=f"Missing {missing_days} trading days in last {self.lookback_days} days",
                        details={
                            "price_days": price_days,
                            "expected_days": len(trading_dates),
                            "missing_days": missing_days,
                            "sample_missing_dates": [str(d) for d in missing_dates],
                        },
                    )
                )

        elapsed_ms = (time.time() - start_time) * 1000

        return CheckResult(
            check_name=self.name,
            issues=issues,
            stats={
                "trading_days": len(trading_dates),
                "symbols_with_gaps": len(issues),
                "lookback_days": self.lookback_days,
            },
            run_time_ms=elapsed_ms,
        )


class StaleDataCheck(QualityCheck):
    """
    Detects symbols with stale (not recently updated) data.

    Flags symbols in the active universe that haven't been
    updated within the expected timeframe.
    """

    name = "stale_data"
    description = "Detects symbols not updated recently"

    def __init__(self, db_path: str | None = None, stale_threshold_days: int = 3, read_only: bool = True):
        super().__init__(db_path, read_only=read_only)
        self.stale_threshold_days = stale_threshold_days

    def run(self, as_of_date: date) -> CheckResult:
        """Check for stale data."""
        import time

        start_time = time.time()
        issues = []

        stale_cutoff = as_of_date - timedelta(days=self.stale_threshold_days)

        # Find symbols with no recent prices
        query = """
            WITH latest_prices AS (
                SELECT symbol, MAX(date) as last_price_date
                FROM prices
                GROUP BY symbol
            ),
            active_universe AS (
                SELECT DISTINCT symbol
                FROM universe
                WHERE in_universe = TRUE
            )
            SELECT
                au.symbol,
                lp.last_price_date
            FROM active_universe au
            LEFT JOIN latest_prices lp ON au.symbol = lp.symbol
            WHERE lp.last_price_date IS NULL OR lp.last_price_date < ?
        """

        results = self._db.fetchall(query, (stale_cutoff,))

        for row in results:
            symbol, last_price_date = row
            days_stale = (
                (as_of_date - last_price_date).days if last_price_date else None
            )

            severity = IssueSeverity.CRITICAL if days_stale is None or days_stale > 7 else IssueSeverity.WARNING

            issues.append(
                QualityIssue.create(
                    check_name=self.name,
                    severity=severity,
                    symbol=symbol,
                    date=as_of_date,
                    description=(
                        "No price data" if last_price_date is None
                        else f"Data is {days_stale} days stale"
                    ),
                    details={
                        "last_price_date": str(last_price_date) if last_price_date else None,
                        "days_stale": days_stale,
                        "stale_threshold": self.stale_threshold_days,
                    },
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000

        return CheckResult(
            check_name=self.name,
            issues=issues,
            stats={
                "stale_symbols": len(issues),
                "stale_threshold_days": self.stale_threshold_days,
            },
            run_time_ms=elapsed_ms,
        )


class VolumeAnomalyCheck(QualityCheck):
    """
    Detects abnormal volume patterns.

    Flags days with zero volume or volume significantly different
    from the rolling average.
    """

    name = "volume_anomaly"
    description = "Detects abnormal trading volume"

    def __init__(
        self,
        db_path: str | None = None,
        volume_threshold: float = 10.0,  # 10x average
        lookback_days: int = 20,
        read_only: bool = True,
    ):
        super().__init__(db_path, read_only=read_only)
        self.volume_threshold = volume_threshold
        self.lookback_days = lookback_days

    def run(self, as_of_date: date) -> CheckResult:
        """Check for volume anomalies."""
        import time

        start_time = time.time()
        issues = []

        # Find zero volume days
        zero_volume_query = """
            SELECT symbol, date, volume
            FROM prices
            WHERE date = ? AND (volume = 0 OR volume IS NULL)
        """
        zero_results = self._db.fetchall(zero_volume_query, (as_of_date,))

        for row in zero_results:
            symbol, dt, volume = row
            issues.append(
                QualityIssue.create(
                    check_name=self.name,
                    severity=IssueSeverity.WARNING,
                    symbol=symbol,
                    date=dt,
                    description="Zero or null trading volume",
                    details={"volume": volume},
                )
            )

        # Find extreme volume days
        extreme_volume_query = """
            WITH volume_stats AS (
                SELECT
                    symbol,
                    date,
                    volume,
                    AVG(volume) OVER (
                        PARTITION BY symbol
                        ORDER BY date
                        ROWS BETWEEN ? PRECEDING AND 1 PRECEDING
                    ) as avg_volume
                FROM prices
                WHERE date BETWEEN ? AND ?
            )
            SELECT symbol, date, volume, avg_volume
            FROM volume_stats
            WHERE date = ?
              AND avg_volume > 0
              AND volume > avg_volume * ?
        """

        start_date = as_of_date - timedelta(days=self.lookback_days + 30)
        extreme_results = self._db.fetchall(
            extreme_volume_query,
            (self.lookback_days, start_date, as_of_date, as_of_date, self.volume_threshold),
        )

        for row in extreme_results:
            symbol, dt, volume, avg_volume = row
            multiplier = volume / avg_volume if avg_volume else 0

            issues.append(
                QualityIssue.create(
                    check_name=self.name,
                    severity=IssueSeverity.INFO,
                    symbol=symbol,
                    date=dt,
                    description=f"Volume {multiplier:.1f}x above average",
                    details={
                        "volume": int(volume),
                        "avg_volume": int(avg_volume),
                        "multiplier": float(multiplier),
                    },
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000

        return CheckResult(
            check_name=self.name,
            issues=issues,
            stats={
                "zero_volume_count": len(zero_results),
                "extreme_volume_count": len(extreme_results),
            },
            run_time_ms=elapsed_ms,
        )


# Default checks to run
DEFAULT_CHECKS = [
    PriceAnomalyCheck,
    CompletenessCheck,
    GapDetectionCheck,
    StaleDataCheck,
    VolumeAnomalyCheck,
]
