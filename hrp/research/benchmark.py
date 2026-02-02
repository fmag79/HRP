"""
Benchmark comparison utilities.
"""

from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.sources.yfinance_source import YFinanceSource


# Standard benchmarks
BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
}


def get_benchmark_prices(
    benchmark: str = "SPY",
    start: date | None = None,
    end: date | None = None,
    db=None,
) -> pd.DataFrame:
    """
    Get benchmark price data.

    First tries database, falls back to Yahoo Finance.

    Args:
        benchmark: Benchmark ticker (default SPY)
        start: Start date
        end: End date
        db: Optional database connection (uses default if not provided)

    Returns:
        DataFrame with date, close, adj_close columns
    """
    db = db or get_db()

    # Try database first
    query = """
        SELECT date, close, adj_close
        FROM prices
        WHERE symbol = ?
    """
    params: list[str | date] = [benchmark]

    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    query += " ORDER BY date"

    df = db.fetchdf(query, tuple(params))

    if not df.empty:
        logger.debug(f"Loaded {len(df)} rows for {benchmark} from database")
        return df

    # Fallback to Yahoo Finance
    logger.info(f"Fetching {benchmark} from Yahoo Finance")
    source = YFinanceSource()

    if start is None:
        start = date(2010, 1, 1)
    if end is None:
        end = date.today()

    df = source.get_daily_bars(benchmark, start, end)

    if df.empty:
        raise ValueError(f"No data found for benchmark {benchmark}")

    return df[["date", "close", "adj_close"]]


def get_benchmark_returns(
    benchmark: str = "SPY",
    start: date | None = None,
    end: date | None = None,
    use_adjusted: bool = True,
) -> pd.Series:
    """
    Get benchmark daily returns.

    Args:
        benchmark: Benchmark ticker
        start: Start date
        end: End date
        use_adjusted: Use adjusted close (default True)

    Returns:
        Series of daily returns indexed by date
    """
    df = get_benchmark_prices(benchmark, start, end)

    price_col = "adj_close" if use_adjusted and "adj_close" in df.columns else "close"

    df = df.sort_values("date")
    df["return"] = df[price_col].pct_change()

    returns = df.set_index("date")["return"].dropna()
    returns.name = benchmark

    return returns


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark: str = "SPY",
) -> dict:
    """
    Compare strategy returns to a benchmark.

    Args:
        strategy_returns: Daily strategy returns
        benchmark: Benchmark ticker

    Returns:
        Dictionary with comparison metrics
    """
    from hrp.research.metrics import calculate_metrics

    # Get benchmark returns for the same period
    start = strategy_returns.index.min()
    end = strategy_returns.index.max()

    # Handle date types
    if hasattr(start, 'date'):
        start = start.date()
    if hasattr(end, 'date'):
        end = end.date()

    benchmark_returns = get_benchmark_returns(benchmark, start, end)

    # Align dates
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["strategy", "benchmark"]

    # Calculate metrics for both
    strategy_metrics = calculate_metrics(
        aligned["strategy"],
        benchmark_returns=aligned["benchmark"]
    )
    benchmark_metrics = calculate_metrics(aligned["benchmark"])

    return {
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "excess_return": strategy_metrics.get("total_return", 0) - benchmark_metrics.get("total_return", 0),
        "excess_sharpe": strategy_metrics.get("sharpe_ratio", 0) - benchmark_metrics.get("sharpe_ratio", 0),
    }


def ensure_benchmark_data(benchmark: str = "SPY", years: int = 10, db=None) -> None:
    """
    Ensure benchmark data is loaded in the database.

    Args:
        benchmark: Benchmark ticker
        years: Years of history to load
    """
    from hrp.data.ingestion.prices import ingest_prices
    from datetime import timedelta

    end = date.today()
    start = date(end.year - years, 1, 1)

    db = db or get_db()

    # Check if we have data
    result = db.fetchone(
        "SELECT COUNT(*) FROM prices WHERE symbol = ?",
        (benchmark,)
    )

    if result[0] == 0:
        logger.info(f"Loading {benchmark} data for {years} years")
        ingest_prices([benchmark], start, end)
    else:
        logger.debug(f"{benchmark} already has {result[0]} rows")
