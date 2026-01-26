"""
Fundamentals data ingestion for HRP.

Fetches and stores quarterly fundamental data (revenue, EPS, book value, etc.)
with point-in-time correctness for backtesting.
"""

import argparse
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from hrp.data.db import get_db


# Default metrics to fetch
DEFAULT_METRICS = [
    "revenue",
    "eps",
    "book_value",
    "net_income",
    "total_assets",
    "total_liabilities",
]

# Point-in-time buffer for YFinance (lacks publish_date)
# Assume data is available 45 days after period end
YFINANCE_PIT_BUFFER_DAYS = 45


class YFinanceFundamentalsAdapter:
    """
    YFinance adapter for fundamental data.

    Fallback source when SimFin is unavailable. Lacks true point-in-time
    correctness (no publish_date), so we apply a conservative buffer.
    """

    source_name = "yfinance"

    def __init__(self):
        """Initialize the YFinance fundamentals adapter."""
        logger.info("YFinance fundamentals adapter initialized")

    def get_fundamentals(
        self,
        symbol: str,
        metrics: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for a single symbol.

        Since YFinance doesn't provide publish_date, we estimate it as
        period_end + 45 days (conservative buffer for 10-Q filing).

        Args:
            symbol: Stock ticker
            metrics: List of metrics to fetch (default: all available)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            DataFrame with columns: symbol, report_date, period_end, metric, value, source
        """
        metrics = metrics or DEFAULT_METRICS

        try:
            ticker = yf.Ticker(symbol)

            # Get quarterly financials
            income_stmt = ticker.quarterly_income_stmt
            balance_sheet = ticker.quarterly_balance_sheet

            if income_stmt.empty and balance_sheet.empty:
                logger.warning(f"No fundamental data for {symbol}")
                return pd.DataFrame()

            rows = []

            # Map our metric names to yfinance column names
            metric_mapping = {
                "revenue": ["Total Revenue", "Revenue"],
                "net_income": ["Net Income", "Net Income Common Stockholders"],
                "eps": ["Diluted EPS", "Basic EPS"],
                "book_value": ["Stockholders Equity", "Total Stockholder Equity"],
                "total_assets": ["Total Assets"],
                "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
            }

            for metric in metrics:
                yf_columns = metric_mapping.get(metric, [])
                if not yf_columns:
                    continue

                # Choose source based on metric
                if metric in ["revenue", "net_income", "eps"]:
                    source_df = income_stmt
                else:
                    source_df = balance_sheet

                if source_df.empty:
                    continue

                # Find the first matching column
                value_col = None
                for col_name in yf_columns:
                    if col_name in source_df.index:
                        value_col = col_name
                        break

                if not value_col:
                    continue

                # Extract values for each period
                for period_end in source_df.columns:
                    value = source_df.loc[value_col, period_end]

                    if pd.isna(value):
                        continue

                    # Convert period_end to date
                    if isinstance(period_end, pd.Timestamp):
                        period_end_date = period_end.date()
                    else:
                        period_end_date = pd.Timestamp(period_end).date()

                    # Estimate report_date as period_end + buffer
                    # This is conservative; actual filing could be earlier
                    report_date = period_end_date + timedelta(days=YFINANCE_PIT_BUFFER_DAYS)

                    # Apply date filters
                    if start_date and report_date < start_date:
                        continue
                    if end_date and report_date > end_date:
                        continue

                    # For book value per share, we'd need shares outstanding
                    # For now, use total equity as proxy
                    rows.append({
                        "symbol": symbol,
                        "report_date": report_date,
                        "period_end": period_end_date,
                        "metric": metric,
                        "value": float(value),
                        "source": self.source_name,
                    })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)

            # Sort by report_date descending
            df = df.sort_values("report_date", ascending=False)

            logger.debug(f"Fetched {len(df)} fundamental records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            raise

    def get_fundamentals_batch(
        self,
        symbols: list[str],
        metrics: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for multiple symbols.

        Args:
            symbols: List of stock tickers
            metrics: List of metrics to fetch (default: all available)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            Combined DataFrame for all symbols
        """
        all_data = []
        failed_symbols = []

        for symbol in symbols:
            try:
                df = self.get_fundamentals(
                    symbol=symbol,
                    metrics=metrics,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
                failed_symbols.append(symbol)
                continue

        if failed_symbols:
            logger.warning(f"Failed symbols: {failed_symbols}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)


def _validate_point_in_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and filter records for point-in-time correctness.

    Enforces that period_end <= report_date to prevent look-ahead bias.

    Args:
        df: DataFrame with report_date and period_end columns

    Returns:
        Filtered DataFrame with only valid records
    """
    if df.empty:
        return df

    # Convert to dates if needed
    if df["report_date"].dtype == "object":
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
    if df["period_end"].dtype == "object":
        df["period_end"] = pd.to_datetime(df["period_end"]).dt.date

    # Filter out records where period_end > report_date (look-ahead bias)
    initial_count = len(df)
    valid_mask = df["period_end"] <= df["report_date"]
    df = df[valid_mask].copy()

    invalid_count = initial_count - len(df)
    if invalid_count > 0:
        logger.warning(
            f"Filtered {invalid_count} records with point-in-time violations "
            f"(period_end > report_date)"
        )

    return df


def _upsert_fundamentals(db, df: pd.DataFrame) -> int:
    """
    Upsert fundamental data into the database.

    Uses temp table pattern for atomic upsert.

    Args:
        db: Database connection manager
        df: DataFrame with fundamental data

    Returns:
        Number of records inserted/updated
    """
    if df.empty:
        return 0

    records = df.to_dict("records")

    with db.connection() as conn:
        # Create temporary table for bulk insert
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS temp_fundamentals AS "
            "SELECT * FROM fundamentals LIMIT 0"
        )
        conn.execute("DELETE FROM temp_fundamentals")

        # Insert into temp table
        for record in records:
            conn.execute(
                """
                INSERT INTO temp_fundamentals
                (symbol, report_date, period_end, metric, value, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["symbol"],
                    record["report_date"],
                    record["period_end"],
                    record["metric"],
                    record["value"],
                    record.get("source", "unknown"),
                ),
            )

        # Upsert from temp to main table
        conn.execute(
            """
            INSERT OR REPLACE INTO fundamentals
            (symbol, report_date, period_end, metric, value, source, ingested_at)
            SELECT symbol, report_date, period_end, metric, value, source, CURRENT_TIMESTAMP
            FROM temp_fundamentals
            """
        )

        # Cleanup
        conn.execute("DROP TABLE temp_fundamentals")

    return len(records)


def ingest_fundamentals(
    symbols: list[str],
    metrics: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    source: str = "simfin",
) -> dict[str, Any]:
    """
    Ingest fundamental data for given symbols.

    Supports multiple data sources with automatic fallback:
    - simfin: SimFin data with point-in-time via publish_date (requires API key)
    - yfinance: Free YFinance data (estimates publish_date with 45-day buffer)

    Args:
        symbols: List of stock tickers
        metrics: List of metrics to fetch (default: all available)
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
        source: Data source to use ('simfin' or 'yfinance', default: 'simfin')

    Returns:
        Dictionary with ingestion stats
    """
    db = get_db()
    metrics = metrics or DEFAULT_METRICS

    # Initialize data source with fallback
    primary_source = None
    fallback_source = None

    if source == "simfin":
        try:
            from hrp.data.sources.simfin_source import SimFinSource
            primary_source = SimFinSource()
            fallback_source = YFinanceFundamentalsAdapter()
            logger.info("Using SimFin as primary source with YFinance fallback")
        except (ImportError, ValueError) as e:
            logger.warning(f"SimFin unavailable ({e}), falling back to YFinance")
            primary_source = YFinanceFundamentalsAdapter()
            fallback_source = None
    elif source == "yfinance":
        primary_source = YFinanceFundamentalsAdapter()
        fallback_source = None
        logger.info("Using YFinance as primary source")
    else:
        raise ValueError(f"Unknown source: {source}. Use 'simfin' or 'yfinance'")

    stats = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "records_fetched": 0,
        "records_inserted": 0,
        "failed_symbols": [],
        "fallback_used": 0,
        "pit_violations_filtered": 0,
    }

    for symbol in symbols:
        df = pd.DataFrame()
        used_fallback = False

        try:
            logger.info(f"Fetching fundamentals for {symbol} using {primary_source.source_name}")
            df = primary_source.get_fundamentals(
                symbol=symbol,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
            )

            if df.empty:
                logger.warning(f"No data for {symbol} from {primary_source.source_name}")

        except Exception as e:
            logger.warning(f"Primary source failed for {symbol}: {e}")

        # Try fallback if primary failed
        if df.empty and fallback_source is not None:
            try:
                logger.info(f"Trying fallback source {fallback_source.source_name} for {symbol}")
                df = fallback_source.get_fundamentals(
                    symbol=symbol,
                    metrics=metrics,
                    start_date=start_date,
                    end_date=end_date,
                )
                used_fallback = True

                if df.empty:
                    logger.warning(f"No data for {symbol} from fallback source")

            except Exception as e:
                logger.error(f"Fallback source also failed for {symbol}: {e}")

        # Process results
        if df.empty:
            logger.error(f"Failed to fetch fundamentals for {symbol} from any source")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)
            continue

        try:
            # Validate point-in-time correctness
            initial_count = len(df)
            df = _validate_point_in_time(df)
            stats["pit_violations_filtered"] += initial_count - len(df)

            stats["records_fetched"] += len(df)
            if used_fallback:
                stats["fallback_used"] += 1

            # Insert into database
            rows_inserted = _upsert_fundamentals(db, df)
            stats["records_inserted"] += rows_inserted
            stats["symbols_success"] += 1

            source_used = fallback_source.source_name if used_fallback else primary_source.source_name
            logger.info(f"Inserted {rows_inserted} records for {symbol} from {source_used}")

        except Exception as e:
            logger.error(f"Failed to insert {symbol} into database: {e}")
            stats["symbols_failed"] += 1
            stats["failed_symbols"].append(symbol)

    logger.info(
        f"Fundamentals ingestion complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['records_inserted']} records inserted"
    )
    if stats["fallback_used"] > 0:
        logger.info(f"Fallback source used for {stats['fallback_used']} symbols")
    if stats["pit_violations_filtered"] > 0:
        logger.info(f"Filtered {stats['pit_violations_filtered']} records with point-in-time violations")

    return stats


def get_fundamentals_stats() -> dict[str, Any]:
    """Get statistics about stored fundamental data."""
    db = get_db()

    with db.connection() as conn:
        # Total rows
        total = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]

        # Unique symbols
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM fundamentals").fetchone()[0]

        # Date range
        date_range = conn.execute(
            "SELECT MIN(report_date), MAX(report_date) FROM fundamentals"
        ).fetchone()

        # Metrics breakdown
        metrics = conn.execute(
            """
            SELECT metric, COUNT(*) as count
            FROM fundamentals
            GROUP BY metric
            ORDER BY count DESC
            """
        ).fetchall()

        # Per symbol summary
        per_symbol = conn.execute(
            """
            SELECT symbol, COUNT(*) as records,
                   MIN(report_date) as start, MAX(report_date) as end
            FROM fundamentals
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()

    return {
        "total_records": total,
        "unique_symbols": symbols,
        "date_range": {
            "start": date_range[0],
            "end": date_range[1],
        },
        "metrics": [{"metric": m[0], "count": m[1]} for m in metrics],
        "per_symbol": [
            {"symbol": r[0], "records": r[1], "start": r[2], "end": r[3]}
            for r in per_symbol
        ],
    }


def main():
    """CLI entry point for fundamentals ingestion."""
    parser = argparse.ArgumentParser(description="HRP Fundamentals Data Ingestion")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to ingest (default: all universe symbols)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Metrics to fetch (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
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
        default="simfin",
        choices=["simfin", "yfinance"],
        help="Data source (default: simfin with yfinance fallback)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show fundamentals data statistics",
    )

    args = parser.parse_args()

    if args.stats:
        stats = get_fundamentals_stats()
        print("\nFundamentals Data Statistics:")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Unique symbols: {stats['unique_symbols']}")
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print("\nMetrics:")
        for m in stats["metrics"]:
            print(f"  {m['metric']:20} {m['count']:6,} records")
        print("\nPer Symbol:")
        for s in stats["per_symbol"][:20]:  # Limit to 20 symbols
            print(f"  {s['symbol']:6} {s['records']:6,} records  ({s['start']} to {s['end']})")
        if len(stats["per_symbol"]) > 20:
            print(f"  ... and {len(stats['per_symbol']) - 20} more symbols")
        return

    # Get symbols from universe if not specified
    symbols = args.symbols
    if not symbols:
        from hrp.data.universe import UniverseManager
        manager = UniverseManager()
        symbols = manager.get_universe_at_date(date.today())
        if not symbols:
            # Fallback to some test symbols
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        logger.info(f"Using {len(symbols)} symbols from universe")

    start_date = date.fromisoformat(args.start) if args.start else None
    end_date = date.fromisoformat(args.end) if args.end else None

    stats = ingest_fundamentals(
        symbols=symbols,
        metrics=args.metrics,
        start_date=start_date,
        end_date=end_date,
        source=args.source,
    )

    print("\nIngestion Complete:")
    print(f"  Symbols: {stats['symbols_success']}/{stats['symbols_requested']} success")
    print(f"  Records: {stats['records_fetched']} fetched, {stats['records_inserted']} inserted")
    if stats["failed_symbols"]:
        print(f"  Failed: {', '.join(stats['failed_symbols'][:10])}")
        if len(stats["failed_symbols"]) > 10:
            print(f"  ... and {len(stats['failed_symbols']) - 10} more")
    if stats["pit_violations_filtered"] > 0:
        print(f"  Point-in-time violations filtered: {stats['pit_violations_filtered']}")


# =============================================================================
# Snapshot Fundamentals (market_cap, pe_ratio, pb_ratio, dividend_yield, ev_ebitda)
# =============================================================================

# Snapshot metrics are current values (not historical quarterly data)
SNAPSHOT_METRICS = [
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "dividend_yield",
    "ev_ebitda",
]


def ingest_snapshot_fundamentals(
    symbols: list[str],
    as_of_date: date | None = None,
) -> dict[str, Any]:
    """
    Ingest snapshot fundamental metrics for given symbols.

    These are current values like P/E ratio, market cap, etc. that represent
    point-in-time snapshots (unlike quarterly financials which have report dates).

    Args:
        symbols: List of stock tickers
        as_of_date: Date to record for these values (default: today)

    Returns:
        Dictionary with ingestion stats
    """
    from hrp.data.sources.fundamental_source import FundamentalSource

    if as_of_date is None:
        as_of_date = date.today()

    db = get_db()
    source = FundamentalSource()

    stats = {
        "symbols_requested": len(symbols),
        "symbols_success": 0,
        "symbols_failed": 0,
        "records_fetched": 0,
        "records_inserted": 0,
        "failed_symbols": [],
    }

    logger.info(f"Ingesting snapshot fundamentals for {len(symbols)} symbols as of {as_of_date}")

    # Fetch data for all symbols
    df = source.get_fundamentals_batch(symbols, as_of_date)

    if df.empty:
        logger.warning("No snapshot fundamentals data fetched")
        stats["symbols_failed"] = len(symbols)
        stats["failed_symbols"] = symbols.copy()
        return stats

    # Convert wide format to long format for storage
    records = []
    successful_symbols = set()

    for _, row in df.iterrows():
        symbol = row["symbol"]

        for metric in SNAPSHOT_METRICS:
            value = row.get(metric)
            if value is not None and not pd.isna(value):
                records.append({
                    "symbol": symbol,
                    "date": as_of_date,
                    "feature_name": metric,
                    "value": float(value),
                    "version": "v1",
                })
                successful_symbols.add(symbol)

    if not records:
        logger.warning("No valid snapshot fundamental values to store")
        return stats

    stats["records_fetched"] = len(records)
    stats["symbols_success"] = len(successful_symbols)
    stats["symbols_failed"] = len(symbols) - len(successful_symbols)
    stats["failed_symbols"] = [s for s in symbols if s not in successful_symbols]

    # Store in features table (same as other computed features)
    records_df = pd.DataFrame(records)
    rows_inserted = _upsert_snapshot_fundamentals(db, records_df)
    stats["records_inserted"] = rows_inserted

    logger.info(
        f"Snapshot fundamentals ingestion complete: {stats['symbols_success']}/{stats['symbols_requested']} symbols, "
        f"{stats['records_inserted']} records inserted"
    )

    return stats


def _upsert_snapshot_fundamentals(db, df: pd.DataFrame) -> int:
    """
    Upsert snapshot fundamental data into the features table.

    Args:
        db: Database connection manager
        df: DataFrame with columns: symbol, date, feature_name, value, version

    Returns:
        Number of records inserted/updated
    """
    if df.empty:
        return 0

    with db.connection() as conn:
        # Register DataFrame for bulk insert
        conn.register("fundamentals_to_insert", df)

        # Upsert into features table
        conn.execute("""
            INSERT OR REPLACE INTO features (symbol, date, feature_name, value, version, computed_at)
            SELECT symbol, date, feature_name, value, version, CURRENT_TIMESTAMP
            FROM fundamentals_to_insert
        """)

        conn.unregister("fundamentals_to_insert")

    return len(df)


def get_latest_fundamentals(
    symbols: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Get the latest snapshot fundamental values for given symbols.

    Args:
        symbols: List of stock tickers
        metrics: List of metrics to retrieve (default: all snapshot metrics)

    Returns:
        DataFrame with latest fundamental values
    """
    db = get_db()
    metrics = metrics or SNAPSHOT_METRICS

    symbols_str = ",".join(f"'{s}'" for s in symbols)
    metrics_str = ",".join(f"'{m}'" for m in metrics)

    query = f"""
        SELECT f1.symbol, f1.date, f1.feature_name, f1.value
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
        ORDER BY f1.symbol, f1.feature_name
    """

    df = db.fetchdf(query)

    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format
    result = df.pivot_table(
        index=["symbol", "date"],
        columns="feature_name",
        values="value",
        aggfunc="first",
    ).reset_index()

    return result


if __name__ == "__main__":
    main()
