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


def backfill_fundamentals_timeseries(
    symbols: list[str],
    start: date,
    end: date,
    metrics: list[str] | None = None,
    batch_size: int = 10,
    source: str = "yfinance",
    progress_file: Optional[Path] = None,
    db_path: Optional[str] = None,
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

    return stats
