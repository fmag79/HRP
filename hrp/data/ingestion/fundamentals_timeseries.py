"""
Time-series fundamentals for HRP.

Extends quarterly fundamentals with daily fundamental values for
backtesting accuracy and research.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional
import logging

import pandas as pd

from hrp.data.db import get_db

logger = logging.getLogger(__name__)

# Valuation metrics to include in time-series (from snapshot fundamentals)
VALUATION_METRICS = [
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "dividend_yield",
    "ev_ebitda",
    "shares_outstanding",
]


def backfill_fundamentals_timeseries(
    symbols: list[str],
    start: date,
    end: date,
    metrics: list[str] | None = None,
    batch_size: int = 10,
    source: str = "yfinance",
    progress_file: Optional[Path] = None,
    db_path: Optional[str] = None,
    include_valuation_metrics: bool = False,
) -> dict[str, Any]:
    """
    Backfill daily fundamental time-series data.

    For each day in the range:
    1. Fetch the latest quarterly fundamental data available as of that day
    2. Forward-fill values until next quarter report
    3. Store in features table with ts_ prefix (time-series)

    This provides point-in-time correctness for backtesting.

    Args:
        symbols: List of tickers to backfill
        start: Start date for time-series
        end: End date for time-series
        metrics: Fundamental metrics to track (default: revenue, eps, book_value)
        batch_size: Number of symbols per batch
        source: Data source ('yfinance' or 'simfin')
        progress_file: Path to progress tracking file
        db_path: Optional database path
        include_valuation_metrics: If True, also backfill valuation metrics
            (ts_market_cap, ts_pe_ratio, ts_pb_ratio, ts_dividend_yield,
            ts_ev_ebitda, ts_shares_outstanding) from snapshot fundamentals

    Returns:
        Dictionary with backfill statistics
    """
    metrics = metrics or ["revenue", "eps", "book_value"]

    # Initialize database connection
    db = get_db(db_path)

    # Track statistics
    stats: dict[str, Any] = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "rows_inserted": 0,
        "failed_symbols": [],
        "valuation_rows_inserted": 0,
    }

    # Get trading days in range
    trading_days_result = db.execute(
        """
        SELECT DISTINCT date FROM prices
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        params=[start, end],
    )
    trading_days = [row[0] for row in trading_days_result.fetchall()]

    logger.info(f"Computing time-series for {len(trading_days)} trading days")

    for symbol in symbols:
        try:
            # Build dynamic IN clause for metrics
            metrics_placeholders = ",".join(["?" for _ in metrics])
            
            # Fetch quarterly fundamentals with point-in-time
            # Use report_date to determine when data became available
            quarterly_data_result = db.execute(
                f"""
                SELECT report_date, period_end, metric, value
                FROM fundamentals
                WHERE symbol = ? AND metric IN ({metrics_placeholders})
                ORDER BY report_date
                """,
                params=[symbol, *metrics],
            )
            quarterly_data = quarterly_data_result.fetchall()

            if not quarterly_data:
                logger.warning(f"No quarterly data for {symbol}")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append(symbol)
                continue

            # Convert to DataFrame
            df = pd.DataFrame(quarterly_data, columns=["report_date", "period_end", "metric", "value"])

            # Pivot to have metrics as columns, using report_date as index
            df_pivot = df.pivot(index="report_date", columns="metric", values="value")

            # Forward-fill to create daily time-series
            all_rows = []
            for trading_day in trading_days:
                # Find latest report as of this day (point-in-time correctness)
                latest_report = df_pivot[df_pivot.index <= trading_day]
                if latest_report.empty:
                    continue

                # Get last row (latest report)
                latest_values = latest_report.iloc[-1]

                # Store time-series value for each metric
                for metric in metrics:
                    if pd.notna(latest_values[metric]):
                        all_rows.append({
                            "symbol": symbol,
                            "date": trading_day,
                            "feature_name": f"ts_{metric}",
                            "value": float(latest_values[metric]),
                        })

            # Bulk insert time-series fundamentals
            if all_rows:
                # Use upsert pattern similar to features
                for row in all_rows:
                    db.execute(
                        """
                        INSERT OR REPLACE INTO features (symbol, date, feature_name, value, version, computed_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (row["symbol"], row["date"], row["feature_name"], row["value"], "v1"),
                    )

                stats["rows_inserted"] += len(all_rows)

            stats["symbols_success"] += 1
            logger.info(f"Computed {len(all_rows)} time-series values for {symbol}")

        except Exception as e:
            logger.error(f"Failed to compute time-series for {symbol}: {e}")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)

    # Backfill valuation metrics if requested
    if include_valuation_metrics:
        valuation_stats = _backfill_valuation_timeseries(
            db=db,
            symbols=symbols,
            trading_days=trading_days,
        )
        stats["valuation_rows_inserted"] = valuation_stats["rows_inserted"]
        stats["rows_inserted"] += valuation_stats["rows_inserted"]

    return stats


def _backfill_valuation_timeseries(
    db,
    symbols: list[str],
    trading_days: list,
) -> dict[str, Any]:
    """
    Backfill valuation metrics as time-series from snapshot fundamentals.

    For each symbol and trading day, fetches the most recent snapshot
    fundamental value and stores it as ts_<metric> in the features table.

    This provides daily coverage of valuation metrics for backtesting,
    using the snapshot approach (current values replicated historically).

    Args:
        db: Database connection manager
        symbols: List of stock symbols
        trading_days: List of trading days to backfill

    Returns:
        Dictionary with backfill statistics
    """
    stats: dict[str, Any] = {
        "rows_inserted": 0,
        "symbols_processed": 0,
    }

    if not trading_days:
        logger.warning("No trading days provided for valuation backfill")
        return stats

    logger.info(
        f"Backfilling valuation time-series for {len(symbols)} symbols "
        f"across {len(trading_days)} trading days"
    )

    # Get the latest snapshot fundamentals for all symbols
    # Query the features table for the most recent snapshot values
    symbols_str = ",".join(f"'{s}'" for s in symbols)
    metrics_str = ",".join(f"'{m}'" for m in VALUATION_METRICS)

    snapshot_query = f"""
        SELECT f1.symbol, f1.feature_name, f1.value
        FROM features f1
        INNER JOIN (
            SELECT symbol, feature_name, MAX(date) as max_date
            FROM features
            WHERE symbol IN ({symbols_str})
              AND feature_name IN ({metrics_str})
            GROUP BY symbol, feature_name
        ) f2 ON f1.symbol = f2.symbol
            AND f1.feature_name = f2.feature_name
            AND f1.date = f2.max_date
    """

    snapshot_df = db.fetchdf(snapshot_query)

    if snapshot_df.empty:
        logger.warning("No snapshot fundamentals found for valuation backfill")
        return stats

    # Pivot to get metrics as columns per symbol
    snapshot_pivot = snapshot_df.pivot(
        index="symbol",
        columns="feature_name",
        values="value",
    )

    # Build time-series records for each symbol and trading day
    all_rows = []
    for symbol in snapshot_pivot.index:
        for trading_day in trading_days:
            for metric in VALUATION_METRICS:
                if metric in snapshot_pivot.columns:
                    value = snapshot_pivot.loc[symbol, metric]
                    if pd.notna(value):
                        all_rows.append({
                            "symbol": symbol,
                            "date": trading_day,
                            "feature_name": f"ts_{metric}",
                            "value": float(value),
                        })

        stats["symbols_processed"] += 1

    if not all_rows:
        logger.warning("No valuation time-series records to insert")
        return stats

    # Bulk insert using upsert pattern
    for row in all_rows:
        db.execute(
            """
            INSERT OR REPLACE INTO features (symbol, date, feature_name, value, version, computed_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (row["symbol"], row["date"], row["feature_name"], row["value"], "v1"),
        )

    stats["rows_inserted"] = len(all_rows)
    logger.info(
        f"Inserted {stats['rows_inserted']} valuation time-series records "
        f"for {stats['symbols_processed']} symbols"
    )

    return stats
