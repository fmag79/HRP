"""
Feature computation for HRP.

Computes and stores technical indicators and features from price data.
"""

import argparse
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from hrp.data.db import get_db


# Default lookback periods for feature computation
DEFAULT_LOOKBACK_DAYS = 252  # 1 year of trading days


def compute_features(
    symbols: Optional[list[str]] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    version: str = "v1",
) -> dict[str, Any]:
    """
    Compute technical features for given symbols.

    Computes:
    - returns_1d: 1-day return
    - returns_5d: 5-day return
    - returns_20d: 20-day return (monthly)
    - momentum_20d: 20-day momentum
    - momentum_60d: 60-day momentum (quarterly)
    - volatility_20d: 20-day rolling volatility
    - volatility_60d: 60-day rolling volatility

    Args:
        symbols: List of stock tickers (None = all symbols in price data)
        start: Start date for feature computation (None = lookback_days ago)
        end: End date (None = today)
        lookback_days: Days of price history needed for computation
        version: Feature version identifier

    Returns:
        Dictionary with computation stats
    """
    db = get_db()

    # Set date defaults
    if end is None:
        end = date.today()
    if start is None:
        start = end - timedelta(days=30)  # Default to last 30 days

    # Calculate price data start (need extra history for rolling windows)
    price_start = start - timedelta(days=lookback_days)

    logger.info(f"Computing features from {start} to {end}")

    # Get symbols from database if not specified
    if symbols is None:
        with db.connection() as conn:
            result = conn.execute("""
                SELECT DISTINCT symbol
                FROM prices
                WHERE date >= ? AND date <= ?
                ORDER BY symbol
            """, (price_start, end)).fetchall()
            symbols = [row[0] for row in result]
        logger.info(f"Found {len(symbols)} symbols in database")

    if not symbols:
        logger.warning("No symbols to process")
        return {
            "symbols_requested": 0,
            "symbols_success": 0,
            "symbols_failed": 0,
            "features_computed": 0,
            "rows_inserted": 0,
            "failed_symbols": [],
        }

    stats = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "features_computed": 0,
        "rows_inserted": 0,
        "failed_symbols": [],
    }

    for symbol in symbols:
        try:
            logger.info(f"Computing features for {symbol}")

            # Fetch price data with extra history for rolling windows
            prices_df = _fetch_prices(db, symbol, price_start, end)

            if prices_df.empty or len(prices_df) < 60:
                logger.warning(f"Insufficient price data for {symbol} (need 60+ days)")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append(symbol)
                continue

            # Compute all features
            features_df = _compute_all_features(prices_df, symbol, version)

            # Filter to requested date range (convert to pd.Timestamp for comparison)
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            features_df = features_df[
                (features_df['date'] >= start_ts) & (features_df['date'] <= end_ts)
            ]

            if features_df.empty:
                logger.warning(f"No features computed for {symbol} in date range")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append(symbol)
                continue

            # Count unique features
            feature_count = len(features_df['feature_name'].unique())
            stats["features_computed"] += feature_count

            # Insert into database
            rows_inserted = _upsert_features(db, features_df)
            stats["rows_inserted"] += rows_inserted
            stats["symbols_success"] += 1

            logger.info(
                f"Inserted {rows_inserted} feature rows for {symbol} "
                f"({feature_count} features)"
            )

        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Failed to compute features for {symbol}: {error_type}: {e}",
                extra={"symbol": symbol, "error_type": error_type},
            )
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append({
                "symbol": symbol,
                "error": str(e),
                "error_type": error_type,
            })

    logger.info(
        f"Feature computation complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['rows_inserted']} rows inserted"
    )

    return stats


def _fetch_prices(db, symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch price data for a symbol.

    Automatically filters to NYSE trading days only (excludes weekends and holidays).
    """
    from hrp.utils.calendar import get_trading_days

    # Filter date range to trading days only
    trading_days = get_trading_days(start, end)
    if len(trading_days) == 0:
        logger.warning(f"No trading days found between {start} and {end}")
        return pd.DataFrame()

    # Update start/end to first and last trading days
    filtered_start = trading_days[0].date()
    filtered_end = trading_days[-1].date()

    with db.connection() as conn:
        df = conn.execute("""
            SELECT date, close, adj_close, volume
            FROM prices
            WHERE symbol = ?
            AND date >= ?
            AND date <= ?
            ORDER BY date
        """, (symbol, filtered_start, filtered_end)).df()

    return df


def _compute_all_features(df: pd.DataFrame, symbol: str, version: str) -> pd.DataFrame:
    """
    Compute all technical features from price data.

    Returns long-form DataFrame with columns: symbol, date, feature_name, value, version
    """
    if df.empty:
        return pd.DataFrame()

    # Use adjusted close for returns
    df = df.copy()
    df['price'] = df['adj_close'].fillna(df['close'])

    features = []

    # === RETURNS ===
    # 1-day return
    df['returns_1d'] = df['price'].pct_change(1)

    # 5-day return
    df['returns_5d'] = df['price'].pct_change(5)

    # 20-day return (monthly)
    df['returns_20d'] = df['price'].pct_change(20)

    # === MOMENTUM ===
    # 20-day momentum (20-day return)
    df['momentum_20d'] = df['price'].pct_change(20)

    # 60-day momentum (quarterly)
    df['momentum_60d'] = df['price'].pct_change(60)

    # === VOLATILITY ===
    # Calculate log returns for volatility
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))

    # 20-day rolling volatility (annualized)
    df['volatility_20d'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)

    # 60-day rolling volatility (annualized)
    df['volatility_60d'] = df['log_return'].rolling(window=60).std() * np.sqrt(252)

    # === VOLUME ===
    # 20-day average volume
    df['volume_20d'] = df['volume'].rolling(window=20).mean()

    # Convert wide format to long format
    feature_columns = [
        'returns_1d', 'returns_5d', 'returns_20d',
        'momentum_20d', 'momentum_60d',
        'volatility_20d', 'volatility_60d',
        'volume_20d',
    ]

    for feature_name in feature_columns:
        feature_data = df[['date', feature_name]].copy()
        feature_data['symbol'] = symbol
        feature_data['feature_name'] = feature_name
        feature_data['value'] = feature_data[feature_name]
        feature_data['version'] = version
        feature_data = feature_data[['symbol', 'date', 'feature_name', 'value', 'version']]

        # Drop NaN values
        feature_data = feature_data.dropna(subset=['value'])

        features.append(feature_data)

    if not features:
        return pd.DataFrame()

    return pd.concat(features, ignore_index=True)


def _upsert_features(db, df: pd.DataFrame) -> int:
    """
    Upsert features into the database.

    Uses INSERT OR REPLACE to handle duplicates.
    """
    if df.empty:
        return 0

    records = df.to_dict('records')

    with db.connection() as conn:
        # Create temporary table for bulk insert
        conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_features
            AS SELECT * FROM features LIMIT 0
        """)
        conn.execute("DELETE FROM temp_features")

        # Insert into temp table
        for record in records:
            conn.execute("""
                INSERT INTO temp_features (symbol, date, feature_name, value, version)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record['symbol'],
                record['date'],
                record['feature_name'],
                record['value'],
                record['version'],
            ))

        # Upsert from temp to main table
        conn.execute("""
            INSERT OR REPLACE INTO features
            (symbol, date, feature_name, value, version, computed_at)
            SELECT symbol, date, feature_name, value, version, CURRENT_TIMESTAMP
            FROM temp_features
        """)

        # Cleanup
        conn.execute("DROP TABLE temp_features")

    return len(records)


def get_feature_stats() -> dict[str, Any]:
    """Get statistics about computed features."""
    db = get_db()

    with db.connection() as conn:
        # Total rows
        total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]

        # Unique symbols
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM features").fetchone()[0]

        # Unique features
        feature_names = conn.execute(
            "SELECT COUNT(DISTINCT feature_name) FROM features"
        ).fetchone()[0]

        # Date range
        date_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM features"
        ).fetchone()

        # Features per symbol
        per_symbol = conn.execute("""
            SELECT symbol, COUNT(*) as rows,
                   COUNT(DISTINCT feature_name) as features,
                   MIN(date) as start, MAX(date) as end
            FROM features
            GROUP BY symbol
            ORDER BY symbol
        """).fetchall()

        # Feature coverage
        feature_coverage = conn.execute("""
            SELECT feature_name, COUNT(DISTINCT symbol) as symbols, COUNT(*) as rows
            FROM features
            GROUP BY feature_name
            ORDER BY feature_name
        """).fetchall()

    return {
        "total_rows": total,
        "unique_symbols": symbols,
        "unique_features": feature_names,
        "date_range": {
            "start": date_range[0],
            "end": date_range[1],
        },
        "per_symbol": [
            {
                "symbol": r[0],
                "rows": r[1],
                "features": r[2],
                "start": r[3],
                "end": r[4],
            }
            for r in per_symbol
        ],
        "feature_coverage": [
            {"feature_name": r[0], "symbols": r[1], "rows": r[2]}
            for r in feature_coverage
        ],
    }


def main():
    """CLI entry point for feature computation."""
    parser = argparse.ArgumentParser(description="HRP Feature Computation")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to compute features for (default: all symbols in database)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD, default: 30 days ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Lookback days for price data (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Feature version (default: v1)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show feature statistics",
    )

    args = parser.parse_args()

    if args.stats:
        stats = get_feature_stats()
        print("\n=== Feature Statistics ===")
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"Unique symbols: {stats['unique_symbols']}")
        print(f"Unique features: {stats['unique_features']}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print("\n=== Feature Coverage ===")
        for fc in stats['feature_coverage']:
            print(f"  {fc['feature_name']:20} {fc['symbols']:>4} symbols, {fc['rows']:>6} rows")
        return

    # Parse dates
    start_date = date.fromisoformat(args.start) if args.start else None
    end_date = date.fromisoformat(args.end) if args.end else None

    # Run feature computation
    result = compute_features(
        symbols=args.symbols,
        start=start_date,
        end=end_date,
        lookback_days=args.lookback,
        version=args.version,
    )

    print("\n=== Feature Computation Results ===")
    print(f"Symbols processed: {result['symbols_success']}/{result['symbols_requested']}")
    print(f"Features computed: {result['features_computed']}")
    print(f"Rows inserted: {result['rows_inserted']}")

    if result['failed_symbols']:
        print(f"\nFailed symbols ({len(result['failed_symbols'])}):")
        for symbol in result['failed_symbols']:
            print(f"  - {symbol}")


if __name__ == "__main__":
    main()
