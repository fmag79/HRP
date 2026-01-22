"""
Price data ingestion for HRP.

Fetches and stores daily price data from configured sources.
"""

import argparse
from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.sources.polygon_source import PolygonSource
from hrp.data.sources.yfinance_source import YFinanceSource


# Default test symbols (large-cap, liquid)
TEST_SYMBOLS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta
    "TSLA",   # Tesla
    "V",      # Visa
    "UNH",    # UnitedHealth
    "JNJ",    # Johnson & Johnson
]


def ingest_prices(
    symbols: list[str],
    start: date,
    end: date,
    source: str = "polygon",
) -> dict[str, Any]:
    """
    Ingest price data for given symbols.

    Supports multiple data sources with automatic fallback:
    - polygon: Official Polygon.io data (requires API key)
    - yfinance: Free Yahoo Finance data (fallback)

    If Polygon fails to initialize (missing API key), falls back to YFinance.
    If Polygon fails for a specific symbol, tries YFinance before giving up.

    Args:
        symbols: List of stock tickers
        start: Start date
        end: End date
        source: Data source to use ('polygon' or 'yfinance', default: 'polygon')

    Returns:
        Dictionary with ingestion stats
    """
    db = get_db()

    # Initialize primary data source with fallback
    primary_source = None
    fallback_source = None

    if source == "polygon":
        try:
            primary_source = PolygonSource()
            fallback_source = YFinanceSource()
            logger.info("Using Polygon.io as primary source with YFinance fallback")
        except ValueError as e:
            # Polygon initialization failed (likely missing API key)
            logger.warning(f"Polygon.io unavailable ({e}), falling back to YFinance")
            primary_source = YFinanceSource()
            fallback_source = None
    elif source == "yfinance":
        primary_source = YFinanceSource()
        fallback_source = None
        logger.info("Using YFinance as primary source")
    else:
        raise ValueError(f"Unknown source: {source}. Use 'polygon' or 'yfinance'")

    stats = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "rows_fetched": 0,
        "rows_inserted": 0,
        "failed_symbols": [],
        "fallback_used": 0,
    }

    for symbol in symbols:
        df = pd.DataFrame()
        used_fallback = False

        try:
            logger.info(f"Fetching {symbol} from {start} to {end} using {primary_source.source_name}")

            # Try primary source
            df = primary_source.get_daily_bars(symbol, start, end)

            if df.empty:
                logger.warning(f"No data for {symbol} from {primary_source.source_name}")

        except Exception as e:
            logger.warning(f"Primary source failed for {symbol}: {e}")

        # Try fallback if primary failed and fallback is available
        if df.empty and fallback_source is not None:
            try:
                logger.info(f"Trying fallback source {fallback_source.source_name} for {symbol}")
                df = fallback_source.get_daily_bars(symbol, start, end)
                used_fallback = True

                if df.empty:
                    logger.warning(f"No data for {symbol} from fallback source")

            except Exception as e:
                logger.error(f"Fallback source also failed for {symbol}: {e}")

        # Process results
        if df.empty:
            logger.error(f"Failed to fetch data for {symbol} from any source")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)
            continue

        try:
            stats["rows_fetched"] += len(df)
            if used_fallback:
                stats["fallback_used"] += 1

            # Insert into database (upsert)
            rows_inserted = _upsert_prices(db, df)
            stats["rows_inserted"] += rows_inserted
            stats["symbols_success"] += 1

            source_used = fallback_source.source_name if used_fallback else primary_source.source_name
            logger.info(f"Inserted {rows_inserted} rows for {symbol} from {source_used}")

        except Exception as e:
            logger.error(f"Failed to insert {symbol} into database: {e}")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)

    logger.info(
        f"Ingestion complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['rows_inserted']} rows inserted"
    )
    if stats["fallback_used"] > 0:
        logger.info(f"Fallback source used for {stats['fallback_used']} symbols")

    return stats


def _upsert_prices(db, df: pd.DataFrame) -> int:
    """
    Upsert price data into the database.

    Uses INSERT OR REPLACE to handle duplicates.
    """
    if df.empty:
        return 0

    # Prepare data for insertion
    records = df.to_dict('records')

    with db.connection() as conn:
        # Create temporary table for bulk insert
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS temp_prices AS SELECT * FROM prices LIMIT 0")
        conn.execute("DELETE FROM temp_prices")

        # Insert into temp table
        for record in records:
            conn.execute("""
                INSERT INTO temp_prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record['symbol'],
                record['date'],
                record.get('open'),
                record.get('high'),
                record.get('low'),
                record['close'],
                record.get('adj_close'),
                record.get('volume'),
                record.get('source', 'unknown'),
            ))

        # Upsert from temp to main table
        conn.execute("""
            INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, adj_close, volume, source, ingested_at)
            SELECT symbol, date, open, high, low, close, adj_close, volume, source, CURRENT_TIMESTAMP
            FROM temp_prices
        """)

        # Cleanup
        conn.execute("DROP TABLE temp_prices")

    return len(records)


def get_price_stats() -> dict[str, Any]:
    """Get statistics about stored price data."""
    db = get_db()

    with db.connection() as conn:
        # Total rows
        total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

        # Unique symbols
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]

        # Date range
        date_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM prices"
        ).fetchone()

        # Rows per symbol
        per_symbol = conn.execute("""
            SELECT symbol, COUNT(*) as rows, MIN(date) as start, MAX(date) as end
            FROM prices
            GROUP BY symbol
            ORDER BY symbol
        """).fetchall()

    return {
        "total_rows": total,
        "unique_symbols": symbols,
        "date_range": {
            "start": date_range[0],
            "end": date_range[1],
        },
        "per_symbol": [
            {"symbol": r[0], "rows": r[1], "start": r[2], "end": r[3]}
            for r in per_symbol
        ],
    }


def main():
    """CLI entry point for price ingestion."""
    parser = argparse.ArgumentParser(description="HRP Price Data Ingestion")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=TEST_SYMBOLS,
        help="Symbols to ingest (default: test symbols)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2019-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="polygon",
        choices=["polygon", "yfinance"],
        help="Data source (default: polygon with yfinance fallback)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show price data statistics",
    )

    args = parser.parse_args()

    if args.stats:
        stats = get_price_stats()
        print(f"\nPrice Data Statistics:")
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Unique symbols: {stats['unique_symbols']}")
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"\nPer Symbol:")
        for s in stats['per_symbol']:
            print(f"  {s['symbol']:6} {s['rows']:6,} rows  ({s['start']} to {s['end']})")
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    stats = ingest_prices(
        symbols=args.symbols,
        start=start,
        end=end,
        source=args.source,
    )

    print(f"\nIngestion Complete:")
    print(f"  Symbols: {stats['symbols_success']}/{stats['symbols_requested']} success")
    print(f"  Rows: {stats['rows_fetched']} fetched, {stats['rows_inserted']} inserted")
    if stats['failed_symbols']:
        print(f"  Failed: {', '.join(stats['failed_symbols'])}")


if __name__ == "__main__":
    main()
