"""Load full S&P 500 universe with historical price data."""

from datetime import date
import pandas as pd
from io import StringIO
import urllib.request

from hrp.data.db import get_db
from hrp.data.universe import EXCLUDED_SECTORS, REIT_SUBINDUSTRIES
from hrp.agents.jobs import PriceIngestionJob


def fetch_sp500_with_user_agent() -> list[str]:
    """Fetch S&P 500 symbols from Wikipedia with proper user agent."""
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
    excluded = {"sectors": 0, "reits": 0}

    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).replace(".", "-")
        sector = str(row.get("GICS Sector", ""))
        sub_industry = str(row.get("GICS Sub-Industry", ""))

        if sector in EXCLUDED_SECTORS:
            excluded["sectors"] += 1
            continue
        if sub_industry in REIT_SUBINDUSTRIES:
            excluded["reits"] += 1
            continue

        symbols.append(symbol)

    print(f"Fetched {len(df)} S&P 500 constituents")
    print(f"Excluded: {excluded['sectors']} financials/RE sectors, {excluded['reits']} REITs")
    print(f"Remaining: {len(symbols)} symbols")

    return symbols


def get_already_loaded_symbols() -> set[str]:
    """Get symbols that already have full data (6000+ rows)."""
    db = get_db()
    result = db.fetchall("""
        SELECT symbol, COUNT(*) as rows
        FROM prices
        GROUP BY symbol
        HAVING COUNT(*) >= 5000
    """)
    return {r[0] for r in result}


def main():
    print("="*60)
    print("FULL UNIVERSE DATA LOAD (yfinance only)")
    print("="*60)

    # 1. Get S&P 500 symbols
    print("\nFetching S&P 500 constituents...")
    all_symbols = fetch_sp500_with_user_agent()

    # 2. Skip symbols that already have full data
    already_loaded = get_already_loaded_symbols()
    symbols = [s for s in all_symbols if s not in already_loaded]

    print(f"\nAlready loaded (>=5000 rows): {len(already_loaded)} symbols")
    print(f"Remaining to load: {len(symbols)} symbols")

    if not symbols:
        print("All symbols already loaded!")
        return

    print(f"\nLoading {len(symbols)} symbols from 2001-01-01 to {date.today()}")
    print(f"Using: yfinance (full history, no rate limits)")

    # 3. Run price ingestion with yfinance only
    job = PriceIngestionJob(
        symbols=symbols,
        start=date(2001, 1, 1),
        end=date.today(),
        source="yfinance",  # Direct yfinance, no Polygon
        max_retries=2,
    )
    result = job.run()

    # 4. Print results
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"Symbols requested: {len(symbols)}")
    print(f"Symbols success: {result.get('symbols_success', 0)}")
    print(f"Symbols failed: {result.get('symbols_failed', 0)}")
    print(f"Records fetched: {result.get('records_fetched', 0):,}")
    print(f"Records inserted: {result.get('records_inserted', 0):,}")

    if result.get('failed_symbols'):
        print(f"\nFailed symbols ({len(result['failed_symbols'])}):")
        for i in range(0, len(result['failed_symbols']), 10):
            print(f"  {', '.join(result['failed_symbols'][i:i+10])}")

    # 5. Final stats
    db = get_db()
    final = db.fetchone('SELECT COUNT(*), COUNT(DISTINCT symbol) FROM prices')
    print(f"\nFinal database: {final[0]:,} rows, {final[1]} symbols")


if __name__ == "__main__":
    main()
