"""
Corporate actions ingestion for HRP.

Fetches and stores corporate actions data (splits, dividends) from configured sources.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.sources.polygon_source import PolygonSource


# Default test symbols (companies with frequent dividends)
TEST_SYMBOLS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "JNJ",    # Johnson & Johnson
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
]


def ingest_corporate_actions(
    symbols: list[str],
    start: date,
    end: date,
    action_types: list[str] | None = None,
    source: str = "polygon",
) -> dict[str, Any]:
    """
    Ingest corporate actions data for given symbols.

    Currently supports Polygon.io only (YFinance corporate actions not yet implemented).

    Args:
        symbols: List of stock tickers
        start: Start date
        end: End date
        action_types: List of action types to fetch ('split', 'dividend').
                     If None, fetches all types.
        source: Data source to use ('polygon', default: 'polygon')

    Returns:
        Dictionary with ingestion stats
    """
    db = get_db()

    # Initialize data source
    primary_source = None

    if source == "polygon":
        try:
            primary_source = PolygonSource()
            logger.info("Using Polygon.io for corporate actions")
        except ValueError as e:
            logger.error(f"Polygon.io unavailable ({e}), no fallback available for corporate actions")
            raise
    else:
        raise ValueError(f"Unknown source: {source}. Currently only 'polygon' is supported for corporate actions")

    stats = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "actions_fetched": 0,
        "actions_inserted": 0,
        "failed_symbols": [],
        "action_types": action_types or ['split', 'dividend'],
    }

    for symbol in symbols:
        df = pd.DataFrame()

        try:
            logger.info(f"Fetching corporate actions for {symbol} from {start} to {end}")

            # Fetch corporate actions
            df = primary_source.get_corporate_actions(symbol, start, end, action_types)

            if df.empty:
                logger.info(f"No corporate actions found for {symbol}")
                stats["symbols_success"] += 1
                continue

        except Exception as e:
            logger.error(f"Failed to fetch corporate actions for {symbol}: {e}")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)
            continue

        # Process results
        try:
            stats["actions_fetched"] += len(df)

            # Insert into database (upsert)
            rows_inserted = _upsert_corporate_actions(db, df)
            stats["actions_inserted"] += rows_inserted
            stats["symbols_success"] += 1

            logger.info(f"Inserted {rows_inserted} corporate actions for {symbol}")

        except Exception as e:
            logger.error(f"Failed to insert corporate actions for {symbol} into database: {e}")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)

    logger.info(
        f"Ingestion complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['actions_inserted']} actions inserted"
    )

    return stats


def _upsert_corporate_actions(db, df: pd.DataFrame) -> int:
    """
    Upsert corporate actions data into the database.

    Uses INSERT OR REPLACE to handle duplicates.
    """
    if df.empty:
        return 0

    # Prepare data for insertion
    records = df.to_dict('records')

    with db.connection() as conn:
        # Create temporary table for bulk insert
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS temp_corporate_actions AS SELECT * FROM corporate_actions LIMIT 0")
        conn.execute("DELETE FROM temp_corporate_actions")

        # Insert into temp table
        for record in records:
            conn.execute("""
                INSERT INTO temp_corporate_actions (symbol, date, action_type, factor, source)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record['symbol'],
                record['date'],
                record['action_type'],
                record['factor'],
                record.get('source', 'unknown'),
            ))

        # Upsert from temp to main table
        conn.execute("""
            INSERT OR REPLACE INTO corporate_actions (symbol, date, action_type, factor, source, ingested_at)
            SELECT symbol, date, action_type, factor, source, CURRENT_TIMESTAMP
            FROM temp_corporate_actions
        """)

        # Cleanup
        conn.execute("DROP TABLE temp_corporate_actions")

    return len(records)


def main() -> None:
    """CLI entry point for corporate actions ingestion."""
    parser = argparse.ArgumentParser(description="Ingest corporate actions data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=TEST_SYMBOLS,
        help="Stock symbols to fetch (default: test symbols)"
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=(date.today() - timedelta(days=365)).isoformat(),
        help="Start date (YYYY-MM-DD, default: 1 year ago)"
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--action-types",
        nargs="+",
        choices=["split", "dividend"],
        default=None,
        help="Action types to fetch (default: all)"
    )
    parser.add_argument(
        "--source",
        choices=["polygon"],
        default="polygon",
        help="Data source (default: polygon)"
    )

    args = parser.parse_args()

    logger.info(f"Starting corporate actions ingestion for {len(args.symbols)} symbols")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Source: {args.source}")
    if args.action_types:
        logger.info(f"Action types: {args.action_types}")

    stats = ingest_corporate_actions(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        action_types=args.action_types,
        source=args.source,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CORPORATE ACTIONS INGESTION SUMMARY")
    print("=" * 60)
    print(f"Symbols requested:   {stats['symbols_requested']}")
    print(f"Symbols succeeded:   {stats['symbols_success']}")
    print(f"Symbols failed:      {stats['symbols_failed']}")
    print(f"Actions fetched:     {stats['actions_fetched']}")
    print(f"Actions inserted:    {stats['actions_inserted']}")

    if stats['failed_symbols']:
        print(f"\nFailed symbols: {', '.join(stats['failed_symbols'])}")

    print("=" * 60)

    if stats['symbols_failed'] > 0:
        exit(1)


if __name__ == "__main__":
    main()
