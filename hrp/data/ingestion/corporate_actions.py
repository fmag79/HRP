"""
Corporate actions data ingestion for HRP.

Fetches and stores corporate actions (splits, dividends) from configured sources.
"""

import argparse
from datetime import date
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
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


def ingest_corporate_actions(
    symbols: list[str],
    start: date,
    end: date,
    source: str = "yfinance",
) -> dict[str, Any]:
    """
    Ingest corporate actions for given symbols.

    Args:
        symbols: List of stock tickers
        start: Start date
        end: End date
        source: Data source to use ('yfinance')

    Returns:
        Dictionary with ingestion stats
    """
    db = get_db()

    # Initialize data source
    if source == "yfinance":
        data_source = YFinanceSource()
    else:
        raise ValueError(f"Unknown source: {source}")

    stats = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "rows_fetched": 0,
        "rows_inserted": 0,
        "failed_symbols": [],
    }

    for symbol in symbols:
        try:
            logger.info(f"Fetching corporate actions for {symbol} from {start} to {end}")

            # Fetch data
            df = data_source.get_corporate_actions(symbol, start, end)

            if df.empty:
                logger.warning(f"No corporate actions for {symbol}")
                stats["symbols_success"] += 1  # Not an error, just no actions
                continue

            stats["rows_fetched"] += len(df)

            # Insert into database (upsert)
            rows_inserted = _upsert_corporate_actions(db, df)
            stats["rows_inserted"] += rows_inserted
            stats["symbols_success"] += 1

            logger.info(f"Inserted {rows_inserted} corporate actions for {symbol}")

        except Exception as e:
            logger.error(f"Failed to ingest corporate actions for {symbol}: {e}")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)

    logger.info(
        f"Ingestion complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['rows_inserted']} rows inserted"
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
                record.get('value'),  # YFinanceSource uses 'value', DB uses 'factor'
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


def get_corporate_action_stats() -> dict[str, Any]:
    """Get statistics about stored corporate actions data."""
    db = get_db()

    with db.connection() as conn:
        # Total rows
        total = conn.execute("SELECT COUNT(*) FROM corporate_actions").fetchone()[0]

        # Unique symbols
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM corporate_actions").fetchone()[0]

        # Date range
        date_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM corporate_actions"
        ).fetchone()

        # Actions by type
        by_type = conn.execute("""
            SELECT action_type, COUNT(*) as count
            FROM corporate_actions
            GROUP BY action_type
            ORDER BY action_type
        """).fetchall()

        # Rows per symbol
        per_symbol = conn.execute("""
            SELECT symbol, COUNT(*) as rows, MIN(date) as start, MAX(date) as end
            FROM corporate_actions
            GROUP BY symbol
            ORDER BY symbol
        """).fetchall()

    return {
        "total_rows": total,
        "unique_symbols": symbols,
        "date_range": {
            "start": date_range[0] if date_range[0] else None,
            "end": date_range[1] if date_range[1] else None,
        },
        "by_type": [
            {"action_type": r[0], "count": r[1]}
            for r in by_type
        ],
        "per_symbol": [
            {"symbol": r[0], "rows": r[1], "start": r[2], "end": r[3]}
            for r in per_symbol
        ],
    }


def main():
    """CLI entry point for corporate actions ingestion."""
    parser = argparse.ArgumentParser(description="HRP Corporate Actions Data Ingestion")
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
        default="yfinance",
        choices=["yfinance"],
        help="Data source",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show corporate actions data statistics",
    )

    args = parser.parse_args()

    if args.stats:
        stats = get_corporate_action_stats()
        print(f"\nCorporate Actions Data Statistics:")
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Unique symbols: {stats['unique_symbols']}")
        if stats['date_range']['start']:
            print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"\nBy Action Type:")
        for t in stats['by_type']:
            print(f"  {t['action_type']:10} {t['count']:6,} actions")
        print(f"\nPer Symbol:")
        for s in stats['per_symbol']:
            print(f"  {s['symbol']:6} {s['rows']:6,} actions  ({s['start']} to {s['end']})")
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    stats = ingest_corporate_actions(
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
