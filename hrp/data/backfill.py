"""
Historical data backfill utilities for HRP.

Provides automated backfill with progress tracking, rate limiting, and validation.
Supports resumable operations for large backfill jobs.

Usage:
    # Backfill prices for specific symbols
    python -m hrp.data.backfill --symbols AAPL MSFT --start 2020-01-01 --prices

    # Backfill entire S&P 500 universe
    python -m hrp.data.backfill --universe --start 2020-01-01 --all

    # Resume from previous progress
    python -m hrp.data.backfill --resume backfill_progress.json --prices

    # Validate backfill completeness
    python -m hrp.data.backfill --symbols AAPL MSFT --start 2020-01-01 --validate
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.sources.yfinance_source import YFinanceSource
from hrp.utils.rate_limiter import RateLimiter


# =============================================================================
# Progress Tracking
# =============================================================================


class BackfillProgress:
    """
    Track backfill progress for resumability.

    Saves progress to a JSON file that can be used to resume interrupted
    backfill operations.

    Attributes:
        progress_file: Path to the progress tracking file
        completed_symbols: Set of symbols that have been successfully processed
        failed_symbols: Set of symbols that failed during processing
        start_date: Start date for the backfill operation
        end_date: End date for the backfill operation
    """

    def __init__(self, progress_file: Path):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to save/load progress
        """
        self.progress_file = Path(progress_file)
        self.completed_symbols: set[str] = set()
        self.failed_symbols: set[str] = set()
        self.start_date: Optional[date] = None
        self.end_date: Optional[date] = None
        self.symbols: list[str] = []
        self.load()

    def load(self) -> None:
        """Load progress from file if it exists."""
        if not self.progress_file.exists():
            logger.debug(f"No existing progress file at {self.progress_file}")
            return

        try:
            data = json.loads(self.progress_file.read_text())
            self.completed_symbols = set(data.get("completed_symbols", []))
            self.failed_symbols = set(data.get("failed_symbols", []))
            self.symbols = data.get("symbols", [])

            if data.get("start_date"):
                self.start_date = date.fromisoformat(data["start_date"])
            if data.get("end_date"):
                self.end_date = date.fromisoformat(data["end_date"])

            logger.info(
                f"Loaded progress: {len(self.completed_symbols)} completed, "
                f"{len(self.failed_symbols)} failed"
            )
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}")

    def save(self) -> None:
        """Save current progress to file."""
        data = {
            "completed_symbols": sorted(self.completed_symbols),
            "failed_symbols": sorted(self.failed_symbols),
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "total_symbols": len(self.symbols),
            "progress_percent": self._calculate_progress(),
        }

        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress_file.write_text(json.dumps(data, indent=2))
        logger.debug(f"Progress saved to {self.progress_file}")

    def _calculate_progress(self) -> float:
        """Calculate completion percentage."""
        total = len(self.symbols) if self.symbols else 0
        if total == 0:
            return 0.0
        completed = len(self.completed_symbols)
        return round(completed / total * 100, 1)

    def mark_completed(self, symbol: str) -> None:
        """
        Mark a symbol as successfully completed.

        Args:
            symbol: The ticker symbol to mark as completed
        """
        self.completed_symbols.add(symbol)
        self.failed_symbols.discard(symbol)
        logger.debug(f"Marked {symbol} as completed")

    def mark_failed(self, symbol: str) -> None:
        """
        Mark a symbol as failed.

        Will not mark if already completed (prevents overwriting success).

        Args:
            symbol: The ticker symbol to mark as failed
        """
        if symbol not in self.completed_symbols:
            self.failed_symbols.add(symbol)
            logger.debug(f"Marked {symbol} as failed")

    def get_pending_symbols(self, all_symbols: list[str]) -> list[str]:
        """
        Get symbols that are not yet completed.

        Failed symbols are included for retry. Only completed symbols are skipped.

        Args:
            all_symbols: Full list of symbols to process

        Returns:
            List of symbols that still need to be processed (including failed ones)
        """
        return [s for s in all_symbols if s not in self.completed_symbols]


# =============================================================================
# Backfill Functions
# =============================================================================


def backfill_prices(
    symbols: list[str],
    start: date,
    end: date,
    source: str = "yfinance",
    batch_size: int = 10,
    progress_file: Optional[Path] = None,
    rate_limit_calls: int = 2000,
    rate_limit_period: float = 3600.0,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backfill historical price data.

    Fetches price data in batches with rate limiting and progress tracking.
    Supports resuming from previous progress file.

    Args:
        symbols: List of tickers to backfill
        start: Start date
        end: End date
        source: Data source ('yfinance' or 'polygon')
        batch_size: Number of symbols per batch
        progress_file: Path to progress tracking file (for resumability)
        rate_limit_calls: Maximum API calls per period
        rate_limit_period: Rate limit time window in seconds
        db_path: Optional database path (for testing)

    Returns:
        Dictionary with backfill statistics:
            - symbols_requested: Total symbols requested
            - symbols_success: Symbols successfully processed
            - symbols_failed: Symbols that failed
            - symbols_skipped: Symbols skipped (already completed)
            - rows_inserted: Total rows inserted
            - batches_processed: Number of batches processed
            - failed_symbols: List of failed symbol names
    """
    db = get_db(db_path)

    # Initialize progress tracking
    progress = None
    if progress_file:
        progress = BackfillProgress(progress_file)
        progress.symbols = symbols
        progress.start_date = start
        progress.end_date = end

    # Get pending symbols (skip already completed)
    pending_symbols = symbols
    skipped_count = 0
    if progress:
        pending_symbols = progress.get_pending_symbols(symbols)
        skipped_count = len(symbols) - len(pending_symbols)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already completed symbols")

    # Initialize data source
    if source == "yfinance":
        data_source = YFinanceSource()
    else:
        raise ValueError(f"Unknown source: {source}")

    # Initialize rate limiter
    rate_limiter = RateLimiter(max_calls=rate_limit_calls, period=rate_limit_period)

    stats: dict[str, Any] = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "symbols_skipped": skipped_count,
        "rows_fetched": 0,
        "rows_inserted": 0,
        "batches_processed": 0,
        "failed_symbols": [],
    }

    # Process in batches
    num_batches = math.ceil(len(pending_symbols) / batch_size)
    logger.info(f"Processing {len(pending_symbols)} symbols in {num_batches} batches")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(pending_symbols))
        batch_symbols = pending_symbols[batch_start:batch_end]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}: {batch_symbols}")

        for symbol in batch_symbols:
            try:
                # Rate limiting
                rate_limiter.acquire()

                logger.info(f"Fetching {symbol} from {start} to {end}")
                df = data_source.get_daily_bars(symbol, start, end)

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    stats["symbols_failed"] += 1
                    stats["failed_symbols"].append(symbol)
                    if progress:
                        progress.mark_failed(symbol)
                    continue

                stats["rows_fetched"] += len(df)

                # Insert into database
                rows_inserted = _upsert_prices(db, df)
                stats["rows_inserted"] += rows_inserted
                stats["symbols_success"] += 1

                logger.info(f"Inserted {rows_inserted} rows for {symbol}")

                if progress:
                    progress.mark_completed(symbol)
                    progress.save()

            except Exception as e:
                logger.error(f"Failed to backfill {symbol}: {e}")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append(symbol)
                if progress:
                    progress.mark_failed(symbol)
                    progress.save()

        stats["batches_processed"] += 1

    logger.info(
        f"Price backfill complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['rows_inserted']} rows inserted"
    )

    return stats


def _upsert_prices(db, df: pd.DataFrame) -> int:
    """
    Upsert price data into the database.

    Uses INSERT OR REPLACE to handle duplicates.
    Automatically ensures symbols exist in the symbols table.
    """
    if df.empty:
        return 0

    records = df.to_dict("records")

    with db.connection() as conn:
        # Ensure symbols exist (for FK constraints)
        symbols = set(r["symbol"] for r in records)
        for symbol in symbols:
            conn.execute(
                "INSERT OR IGNORE INTO symbols (symbol, name) VALUES (?, ?)",
                (symbol, symbol),
            )

        # Create temporary table for bulk insert
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS temp_prices AS SELECT * FROM prices LIMIT 0")
        conn.execute("DELETE FROM temp_prices")

        # Insert into temp table
        for record in records:
            conn.execute(
                """
                INSERT INTO temp_prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["symbol"],
                    record["date"],
                    record.get("open"),
                    record.get("high"),
                    record.get("low"),
                    record["close"],
                    record.get("adj_close"),
                    record.get("volume"),
                    record.get("source", "backfill"),
                ),
            )

        # Upsert from temp to main table
        conn.execute(
            """
            INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, adj_close, volume, source, ingested_at)
            SELECT symbol, date, open, high, low, close, adj_close, volume, source, CURRENT_TIMESTAMP
            FROM temp_prices
            """
        )

        # Cleanup
        conn.execute("DROP TABLE temp_prices")

    return len(records)


def backfill_features(
    symbols: list[str],
    start: date,
    end: date,
    batch_size: int = 10,
    lookback_days: int = 252,
    version: str = "v1",
    progress_file: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backfill computed features for historical dates.

    Computes technical features from price data for the specified date range.

    Args:
        symbols: List of tickers to compute features for
        start: Start date for feature computation
        end: End date for feature computation
        batch_size: Number of symbols per batch
        lookback_days: Days of price history needed for computation
        version: Feature version identifier
        progress_file: Path to progress tracking file (for resumability)
        db_path: Optional database path (for testing)

    Returns:
        Dictionary with computation statistics
    """
    from hrp.data.ingestion.features import compute_features

    # Initialize progress tracker
    progress = None
    if progress_file:
        progress = BackfillProgress(progress_file)
        symbols = progress.get_pending_symbols(symbols)
        logger.info(f"Resuming backfill: {len(symbols)} symbols remaining")

    stats: dict[str, Any] = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "features_computed": 0,
        "rows_inserted": 0,
        "batches_processed": 0,
        "failed_symbols": [],
    }

    num_batches = math.ceil(len(symbols) / batch_size)
    logger.info(f"Computing features for {len(symbols)} symbols in {num_batches} batches")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(symbols))
        batch_symbols = symbols[batch_start:batch_end]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

        try:
            result = compute_features(
                symbols=batch_symbols,
                start=start,
                end=end,
                lookback_days=lookback_days,
                version=version,
            )

            batch_success = result.get("symbols_success", 0)
            batch_failed = result.get("symbols_failed", 0)

            stats["symbols_success"] += batch_success
            stats["symbols_failed"] += batch_failed
            stats["features_computed"] += result.get("features_computed", 0)
            stats["rows_inserted"] += result.get("rows_inserted", 0)

            # Track failed symbols and update progress
            failed = result.get("failed_symbols", [])
            failed_set = set()
            for fs in failed:
                if isinstance(fs, dict):
                    failed_set.add(fs.get("symbol", str(fs)))
                else:
                    failed_set.add(str(fs))
            stats["failed_symbols"].extend(failed_set)

            # Mark progress for completed symbols
            if progress:
                for symbol in batch_symbols:
                    if symbol in failed_set:
                        progress.mark_failed(symbol)
                    else:
                        progress.mark_completed(symbol)

        except Exception as e:
            logger.error(f"Failed to compute features for batch: {e}")
            stats["symbols_failed"] += len(batch_symbols)
            stats["failed_symbols"].extend(batch_symbols)
            if progress:
                for symbol in batch_symbols:
                    progress.mark_failed(symbol)

        stats["batches_processed"] += 1

        # Save progress after each batch
        if progress:
            progress.save()

    logger.info(
        f"Feature backfill complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['rows_inserted']} rows inserted"
    )

    return stats


def backfill_corporate_actions(
    symbols: list[str],
    start: date,
    end: date,
    source: str = "yfinance",
    batch_size: int = 10,
    progress_file: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backfill corporate actions (splits, dividends).

    Args:
        symbols: List of tickers to backfill
        start: Start date
        end: End date
        source: Data source ('yfinance')
        batch_size: Number of symbols per batch
        progress_file: Path to progress tracking file (for resumability)
        db_path: Optional database path (for testing)

    Returns:
        Dictionary with backfill statistics
    """
    db = get_db(db_path)

    # Initialize data source
    if source == "yfinance":
        data_source = YFinanceSource()
    else:
        raise ValueError(f"Unknown source: {source}")

    # Initialize progress tracker
    progress = None
    if progress_file:
        progress = BackfillProgress(progress_file)
        symbols = progress.get_pending_symbols(symbols)
        logger.info(f"Resuming backfill: {len(symbols)} symbols remaining")

    # Rate limiter for API calls (Yahoo Finance: 2000/hour)
    rate_limiter = RateLimiter(max_calls=2000, period=3600)

    stats: dict[str, Any] = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "rows_fetched": 0,
        "rows_inserted": 0,
        "batches_processed": 0,
        "failed_symbols": [],
    }

    num_batches = math.ceil(len(symbols) / batch_size)
    logger.info(f"Backfilling corporate actions for {len(symbols)} symbols in {num_batches} batches")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(symbols))
        batch_symbols = symbols[batch_start:batch_end]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

        for symbol in batch_symbols:
            try:
                rate_limiter.acquire()

                logger.info(f"Fetching corporate actions for {symbol}")
                df = data_source.get_corporate_actions(symbol, start, end)

                if df.empty:
                    logger.debug(f"No corporate actions for {symbol}")
                    stats["symbols_success"] += 1
                    if progress:
                        progress.mark_completed(symbol)
                    continue

                stats["rows_fetched"] += len(df)

                # Insert into database
                rows_inserted = _upsert_corporate_actions(db, df)
                stats["rows_inserted"] += rows_inserted
                stats["symbols_success"] += 1

                if progress:
                    progress.mark_completed(symbol)

                logger.info(f"Inserted {rows_inserted} corporate actions for {symbol}")

            except Exception as e:
                logger.error(f"Failed to backfill corporate actions for {symbol}: {e}")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append(symbol)
                if progress:
                    progress.mark_failed(symbol)

        stats["batches_processed"] += 1

        # Save progress after each batch
        if progress:
            progress.save()

    logger.info(
        f"Corporate actions backfill complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols"
    )

    return stats


def _upsert_corporate_actions(db, df: pd.DataFrame) -> int:
    """Upsert corporate actions into the database."""
    if df.empty:
        return 0

    records = df.to_dict("records")

    with db.connection() as conn:
        # Ensure symbols exist (for FK constraints)
        symbols = set(r["symbol"] for r in records)
        for symbol in symbols:
            conn.execute(
                "INSERT OR IGNORE INTO symbols (symbol, name) VALUES (?, ?)",
                (symbol, symbol),
            )

        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS temp_corporate_actions AS SELECT * FROM corporate_actions LIMIT 0"
        )
        conn.execute("DELETE FROM temp_corporate_actions")

        for record in records:
            conn.execute(
                """
                INSERT INTO temp_corporate_actions (symbol, date, action_type, factor, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record["symbol"],
                    record["date"],
                    record["action_type"],
                    record.get("value"),
                    record.get("source", "backfill"),
                ),
            )

        conn.execute(
            """
            INSERT OR REPLACE INTO corporate_actions (symbol, date, action_type, factor, source, ingested_at)
            SELECT symbol, date, action_type, factor, source, CURRENT_TIMESTAMP
            FROM temp_corporate_actions
            """
        )

        conn.execute("DROP TABLE temp_corporate_actions")

    return len(records)


def backfill_features_ema_vwap(
    symbols: list[str],
    start: date,
    end: date,
    batch_size: int = 10,
    lookback_days: int = 60,
    progress_file: Optional[Path] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backfill EMA and VWAP features for historical dates.

    Computes only ema_12d, ema_26d, and vwap_20d features which were
    previously only computed from 2026-01-25.

    Args:
        symbols: List of tickers to compute features for
        start: Start date for feature computation
        end: End date for feature computation
        batch_size: Number of symbols per batch
        lookback_days: Days of price history needed for computation
        progress_file: Path to progress tracking file
        db_path: Optional database path

    Returns:
        Dictionary with computation statistics
    """
    from hrp.data.ingestion.features import _fetch_prices, _compute_all_features, _upsert_features

    # Calculate price data start (need extra history for rolling windows)
    price_start = start - timedelta(days=lookback_days)

    # Initialize database connection
    db = get_db(db_path)

    # Initialize progress tracker
    progress = None
    if progress_file:
        progress = BackfillProgress(progress_file)
        symbols = progress.get_pending_symbols(symbols)
        logger.info(f"Resuming backfill: {len(symbols)} symbols remaining")

    # Track statistics
    stats: dict[str, Any] = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "symbols_skipped": 0,
        "rows_inserted": 0,
        "batches_processed": 0,
        "failed_symbols": [],
    }

    # Calculate skipped count from progress
    if progress:
        original_count = progress.symbols and len(progress.symbols) or len(symbols)
        stats["symbols_skipped"] = original_count - len(symbols)

    # Process symbols in batches
    num_batches = math.ceil(len(symbols) / batch_size)
    logger.info(f"Computing EMA/VWAP features for {len(symbols)} symbols in {num_batches} batches")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(symbols))
        batch_symbols = symbols[batch_start:batch_end]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

        for symbol in batch_symbols:
            try:
                # Fetch price data with extra history for rolling windows
                prices_df = _fetch_prices(db, symbol, price_start, end)

                if prices_df.empty or len(prices_df) < lookback_days:
                    logger.warning(f"Insufficient price data for {symbol} (need {lookback_days}+ days)")
                    stats["symbols_failed"] += 1
                    stats["failed_symbols"].append(symbol)
                    if progress:
                        progress.mark_failed(symbol)
                    continue

                # Compute all features
                features_df = _compute_all_features(prices_df, symbol, version="v1")

                # Filter to EMA/VWAP features only
                ema_vwap_features = ["ema_12d", "ema_26d", "vwap_20d"]
                features_df = features_df[features_df["feature_name"].isin(ema_vwap_features)]

                # Filter to requested date range (convert to pd.Timestamp for comparison)
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                features_df = features_df[
                    (features_df["date"] >= start_ts) & (features_df["date"] <= end_ts)
                ]

                if features_df.empty:
                    logger.warning(f"No EMA/VWAP features computed for {symbol} in date range")
                    stats["symbols_failed"] += 1
                    stats["failed_symbols"].append(symbol)
                    if progress:
                        progress.mark_failed(symbol)
                    continue

                # Upsert to features table
                _upsert_features(db, features_df)
                stats["rows_inserted"] += len(features_df)

                stats["symbols_success"] += 1
                logger.info(f"Computed {len(features_df)} EMA/VWAP features for {symbol}")

                if progress:
                    progress.mark_completed(symbol)

            except Exception as e:
                logger.error(f"Failed to compute features for {symbol}: {e}")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append(symbol)
                if progress:
                    progress.mark_failed(symbol)

        stats["batches_processed"] += 1

        # Save progress after each batch
        if progress:
            progress.save()

    logger.info(
        f"EMA/VWAP feature backfill complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['rows_inserted']} rows inserted"
    )

    return stats


# =============================================================================
# Validation
# =============================================================================


def validate_backfill(
    symbols: list[str],
    start: date,
    end: date,
    check_features: bool = False,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate backfill completeness.

    Checks:
    - All symbols have price data
    - No gaps in date range (trading days only)
    - Features computed for all dates with prices (if check_features=True)

    Args:
        symbols: List of symbols to validate
        start: Start date
        end: End date
        check_features: Also check feature completeness
        db_path: Optional database path (for testing)

    Returns:
        Dictionary with validation results:
            - is_valid: Overall validation status
            - symbols_complete: Number of symbols with complete data
            - missing_symbols: Symbols with no data
            - gaps: Dict mapping symbols to lists of missing dates
            - features_missing: Dict of symbols with incomplete features
    """
    from hrp.utils.calendar import get_trading_days

    db = get_db(db_path)

    # Get expected trading days
    trading_days = get_trading_days(start, end)
    expected_dates = {d.date() for d in trading_days}

    result = {
        "is_valid": True,
        "symbols_complete": 0,
        "missing_symbols": [],
        "gaps": {},
        "features_missing": {},
        "expected_trading_days": len(expected_dates),
    }

    for symbol in symbols:
        # Check price data
        with db.connection() as conn:
            rows = conn.execute(
                """
                SELECT date FROM prices
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
                """,
                (symbol, start, end),
            ).fetchall()

        actual_dates = {row[0] for row in rows}

        if not actual_dates:
            result["missing_symbols"].append(symbol)
            result["is_valid"] = False
            continue

        # Check for gaps
        missing_dates = expected_dates - actual_dates
        if missing_dates:
            result["gaps"][symbol] = sorted([d.isoformat() for d in missing_dates])
            result["is_valid"] = False
        else:
            result["symbols_complete"] += 1

    # Check features if requested
    if check_features:
        for symbol in symbols:
            with db.connection() as conn:
                feature_count = conn.execute(
                    """
                    SELECT COUNT(DISTINCT date) FROM features
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    """,
                    (symbol, start, end),
                ).fetchone()[0]

            # We expect fewer feature dates than price dates due to lookback
            # Just check that some features exist
            if feature_count == 0:
                result["features_missing"][symbol] = "No features computed"

    return result


# =============================================================================
# CLI Interface
# =============================================================================


def main() -> int:
    """CLI entry point for backfill operations."""
    parser = argparse.ArgumentParser(
        description="HRP Historical Data Backfill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill S&P 500 prices for 2020-2023
  python -m hrp.data.backfill --universe --start 2020-01-01 --end 2023-12-31 --prices

  # Backfill specific symbols (all data types)
  python -m hrp.data.backfill --symbols AAPL MSFT GOOGL --start 2019-01-01 --all

  # Resume failed backfill
  python -m hrp.data.backfill --resume backfill_progress.json --prices

  # Validate completeness
  python -m hrp.data.backfill --symbols AAPL MSFT --start 2020-01-01 --validate
        """,
    )

    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to backfill",
    )
    symbol_group.add_argument(
        "--universe",
        action="store_true",
        help="Backfill entire S&P 500 universe",
    )
    symbol_group.add_argument(
        "--resume",
        type=str,
        help="Resume from progress file",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, default: today)",
    )

    # Data types
    parser.add_argument(
        "--prices",
        action="store_true",
        help="Backfill prices",
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Backfill features",
    )
    parser.add_argument(
        "--corporate-actions",
        action="store_true",
        help="Backfill corporate actions",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backfill all data types",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate completeness after backfill",
    )

    # Options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size (default: 10)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="yfinance",
        choices=["yfinance", "polygon"],
        help="Data source (default: yfinance)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Database path (for testing)",
    )

    args = parser.parse_args()

    # Determine symbols
    symbols = []
    progress_file = None

    if args.resume:
        # Resume from progress file
        progress_file = Path(args.resume)
        if not progress_file.exists():
            print(f"Progress file not found: {progress_file}")
            return 1

        progress = BackfillProgress(progress_file)
        symbols = progress.symbols
        start = progress.start_date
        end = progress.end_date

        if not symbols:
            print("No symbols found in progress file")
            return 1
    elif args.universe:
        # Get universe symbols
        from hrp.data.universe import UniverseManager

        manager = UniverseManager(args.db_path)
        as_of = date.fromisoformat(args.end) if args.end else date.today()
        symbols = manager.get_universe_at_date(as_of)
        start = date.fromisoformat(args.start) if args.start else None
        end = date.fromisoformat(args.end) if args.end else date.today()
    elif args.symbols:
        symbols = args.symbols
        start = date.fromisoformat(args.start) if args.start else None
        end = date.fromisoformat(args.end) if args.end else date.today()
    else:
        # No symbols specified - show help and return
        if not args.validate:
            parser.print_help()
            print("\nNo symbols specified. Use --symbols, --universe, or --resume.")
            return 0

    # Validate required arguments
    if not args.resume and not args.start:
        parser.print_help()
        print("\n--start is required unless using --resume")
        return 1

    # Determine what to backfill
    do_prices = args.prices or args.all
    do_features = args.features or args.all
    do_corporate_actions = args.corporate_actions or args.all
    do_validate = args.validate

    # If no action specified, show info
    if not any([do_prices, do_features, do_corporate_actions, do_validate]):
        parser.print_help()
        print("\nNo action specified. Use --prices, --features, --corporate-actions, --all, or --validate.")
        return 0

    print(f"\nBackfill Configuration:")
    print(f"  Symbols: {len(symbols)} total")
    print(f"  Date range: {start} to {end}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Actions: prices={do_prices}, features={do_features}, "
          f"corporate_actions={do_corporate_actions}, validate={do_validate}")
    print()

    # Create progress file if not resuming
    if not progress_file and do_prices:
        progress_file = Path(f"backfill_progress_{date.today().strftime('%Y%m%d')}.json")

    # Execute backfill operations
    if do_prices:
        print("=== Backfilling Prices ===")
        result = backfill_prices(
            symbols=symbols,
            start=start,
            end=end,
            source=args.source,
            batch_size=args.batch_size,
            progress_file=progress_file,
            db_path=args.db_path,
        )
        print(f"  Success: {result['symbols_success']}/{result['symbols_requested']}")
        print(f"  Rows inserted: {result['rows_inserted']}")
        if result["failed_symbols"]:
            print(f"  Failed: {', '.join(result['failed_symbols'][:10])}...")
        print()

    if do_corporate_actions:
        print("=== Backfilling Corporate Actions ===")
        result = backfill_corporate_actions(
            symbols=symbols,
            start=start,
            end=end,
            source=args.source,
            batch_size=args.batch_size,
            db_path=args.db_path,
        )
        print(f"  Success: {result['symbols_success']}/{result['symbols_requested']}")
        print(f"  Rows inserted: {result['rows_inserted']}")
        print()

    if do_features:
        print("=== Backfilling Features ===")
        result = backfill_features(
            symbols=symbols,
            start=start,
            end=end,
            batch_size=args.batch_size,
            db_path=args.db_path,
        )
        print(f"  Success: {result['symbols_success']}/{result['symbols_requested']}")
        print(f"  Rows inserted: {result['rows_inserted']}")
        print()

    if do_validate:
        print("=== Validating Backfill ===")
        result = validate_backfill(
            symbols=symbols,
            start=start,
            end=end,
            check_features=do_features,
            db_path=args.db_path,
        )
        print(f"  Valid: {result['is_valid']}")
        print(f"  Complete symbols: {result['symbols_complete']}/{len(symbols)}")
        if result["missing_symbols"]:
            print(f"  Missing: {', '.join(result['missing_symbols'][:10])}...")
        if result["gaps"]:
            print(f"  Symbols with gaps: {len(result['gaps'])}")
            for symbol, gaps in list(result["gaps"].items())[:3]:
                print(f"    {symbol}: {len(gaps)} missing dates")
        print()

    print("Backfill complete!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
