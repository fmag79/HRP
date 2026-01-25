"""
Load S&P 500 universe using Polygon.io with strict rate limiting.

Polygon Basic tier: 5 API calls per minute
- We use 4 calls/min to have safety margin
- That's 15 seconds between each symbol
- ~400 symbols = ~100 minutes total

Run with: nohup python scripts/load_universe_polygon.py > polygon_load.log 2>&1 &

Requires POLYGON_API_KEY in .env file or environment.
"""

import os
import time

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()
from datetime import date
from io import StringIO
import urllib.request

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.universe import EXCLUDED_SECTORS, REIT_SUBINDUSTRIES
from hrp.data.sources.polygon_source import PolygonSource
from hrp.data.ingestion.prices import _upsert_prices


# Rate limit config: 4 calls per minute (15 sec between calls)
# Polygon Basic tier is 5/min, we use 4 for safety margin
CALLS_PER_MINUTE = 4
SECONDS_BETWEEN_CALLS = 60 / CALLS_PER_MINUTE  # 15 seconds


def fetch_sp500_symbols() -> list[str]:
    """Fetch S&P 500 symbols from Wikipedia with exclusions applied."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) HRP/1.0'}
    )

    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')

    tables = pd.read_html(StringIO(html))
    df = tables[0]

    symbols = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).replace(".", "-")
        sector = str(row.get("GICS Sector", ""))
        sub_industry = str(row.get("GICS Sub-Industry", ""))

        if sector in EXCLUDED_SECTORS:
            continue
        if sub_industry in REIT_SUBINDUSTRIES:
            continue

        symbols.append(symbol)

    return symbols


def get_symbols_needing_data() -> list[str]:
    """Get symbols that need full historical data (< 5000 rows)."""
    db = get_db()

    # Get all S&P 500 symbols
    all_symbols = set(fetch_sp500_symbols())

    # Get symbols already loaded with enough data
    loaded = db.fetchall("""
        SELECT symbol, COUNT(*) as rows
        FROM prices
        GROUP BY symbol
        HAVING COUNT(*) >= 5000
    """)
    loaded_symbols = {r[0] for r in loaded}

    # Return symbols needing data
    return [s for s in all_symbols if s not in loaded_symbols]


def load_symbol_with_rate_limit(
    source: PolygonSource,
    symbol: str,
    start: date,
    end: date,
    db,
) -> dict:
    """Load a single symbol with rate limiting."""
    try:
        logger.info(f"Fetching {symbol}...")
        df = source.get_daily_bars(symbol, start, end)

        if df.empty:
            logger.warning(f"No data for {symbol}")
            return {"symbol": symbol, "status": "no_data", "rows": 0}

        # Upsert to database
        rows_inserted = _upsert_prices(db, df)
        logger.info(f"Inserted {rows_inserted} rows for {symbol}")

        return {"symbol": symbol, "status": "success", "rows": rows_inserted}

    except Exception as e:
        logger.error(f"Failed to load {symbol}: {e}")
        return {"symbol": symbol, "status": "failed", "error": str(e), "rows": 0}


def main():
    print("="*70)
    print("S&P 500 DATA LOAD - POLYGON.IO (Rate Limited)")
    print("="*70)
    print(f"Rate limit: {CALLS_PER_MINUTE} calls/minute ({SECONDS_BETWEEN_CALLS:.0f}s between calls)")

    # Get symbols needing data
    symbols = get_symbols_needing_data()
    print(f"\nSymbols to load: {len(symbols)}")

    if not symbols:
        print("All symbols already have full data!")
        return

    # Estimate time
    estimated_minutes = len(symbols) * SECONDS_BETWEEN_CALLS / 60
    print(f"Estimated time: {estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")
    print(f"\nDate range: 2001-01-01 to {date.today()}")
    print("-"*70)

    # Initialize Polygon source
    try:
        source = PolygonSource()
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nTo fix, add POLYGON_API_KEY to your .env file:")
        print("  echo 'POLYGON_API_KEY=your_key_here' >> .env")
        print("\nOr set it as environment variable:")
        print("  export POLYGON_API_KEY=your_key_here")
        return

    db = get_db()
    start_date = date(2001, 1, 1)
    end_date = date.today()

    # Stats
    stats = {"success": 0, "failed": 0, "no_data": 0, "total_rows": 0}
    failed_symbols = []

    start_time = time.time()

    for i, symbol in enumerate(symbols, 1):
        # Progress
        elapsed = time.time() - start_time
        rate = i / (elapsed / 60) if elapsed > 0 else 0
        remaining = (len(symbols) - i) / rate if rate > 0 else 0

        print(f"\n[{i}/{len(symbols)}] {symbol} (ETA: {remaining:.0f} min)")

        # Load symbol
        result = load_symbol_with_rate_limit(source, symbol, start_date, end_date, db)

        # Update stats
        stats[result["status"]] = stats.get(result["status"], 0) + 1
        stats["total_rows"] += result["rows"]

        if result["status"] == "failed":
            failed_symbols.append(symbol)

        # Rate limit: wait before next call
        if i < len(symbols):
            logger.debug(f"Rate limiting: waiting {SECONDS_BETWEEN_CALLS:.0f}s...")
            time.sleep(SECONDS_BETWEEN_CALLS)

    # Final report
    total_time = (time.time() - start_time) / 60

    print("\n" + "="*70)
    print("LOAD COMPLETE")
    print("="*70)
    print(f"Time: {total_time:.1f} minutes")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"No data: {stats.get('no_data', 0)}")
    print(f"Total rows inserted: {stats['total_rows']:,}")

    if failed_symbols:
        print(f"\nFailed symbols: {', '.join(failed_symbols)}")

    # Database stats
    final = db.fetchone('SELECT COUNT(*), COUNT(DISTINCT symbol) FROM prices')
    print(f"\nDatabase total: {final[0]:,} rows, {final[1]} symbols")


if __name__ == "__main__":
    main()
