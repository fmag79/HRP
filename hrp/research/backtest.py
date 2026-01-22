"""
VectorBT backtesting wrapper for HRP.
"""

import argparse
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from hrp.data.db import get_db
from hrp.research.config import BacktestConfig, BacktestResult, CostModel
from hrp.research.metrics import calculate_metrics
from hrp.research.benchmark import get_benchmark_returns


def get_price_data(
    symbols: list[str],
    start: date,
    end: date,
    adjust_splits: bool = True,
) -> pd.DataFrame:
    """
    Load price data from database.

    Args:
        symbols: List of ticker symbols
        start: Start date (inclusive)
        end: End date (inclusive)
        adjust_splits: If True, apply split adjustments to close prices (default: True)

    Returns:
        DataFrame with MultiIndex columns (symbol, field)
    """
    db = get_db()

    symbols_str = ",".join(f"'{s}'" for s in symbols)
    query = f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM prices
        WHERE symbol IN ({symbols_str})
          AND date >= ?
          AND date <= ?
        ORDER BY date, symbol
    """

    df = db.fetchdf(query, (start, end))

    if df.empty:
        raise ValueError(f"No price data found for {symbols} from {start} to {end}")

    # Apply split adjustments if requested
    if adjust_splits:
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()
        df = api.adjust_prices_for_splits(df)
        logger.debug(f"Applied split adjustments to price data for {symbols}")

    # Pivot to get symbol columns
    df['date'] = pd.to_datetime(df['date'])
    pivot = df.pivot(index='date', columns='symbol')

    return pivot


def run_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig,
    prices: pd.DataFrame = None,
) -> BacktestResult:
    """
    Run a backtest using VectorBT.

    Args:
        signals: DataFrame of signals (1 = long, 0 = no position)
                 Index = dates, columns = symbols
        config: Backtest configuration
        prices: Optional price data (loaded if not provided)

    Returns:
        BacktestResult with metrics, equity curve, and trades
    """
    # Load prices if not provided
    if prices is None:
        prices = get_price_data(config.symbols, config.start_date, config.end_date)

    # Get close prices
    close = prices['close'] if 'close' in prices.columns.get_level_values(0) else prices

    # Align signals with prices
    signals = signals.reindex(close.index).fillna(0)

    # Calculate fees as percentage
    fees = config.costs.total_cost_pct()

    # Run portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=signals > 0,
        exits=signals <= 0,
        fees=fees,
        freq='D',
        init_cash=100000,
        size_type='percent',
        size=1.0 / config.max_positions,  # Equal weight
    )

    # Get returns
    returns = portfolio.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.sum(axis=1)  # Aggregate if multiple symbols

    # Get equity curve
    equity = portfolio.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.sum(axis=1)

    # Calculate metrics
    benchmark_returns = None
    try:
        benchmark_returns = get_benchmark_returns(
            "SPY",
            config.start_date,
            config.end_date
        )
        # Align with strategy returns
        benchmark_returns = benchmark_returns.reindex(returns.index)
    except Exception as e:
        logger.warning(f"Could not load benchmark: {e}")

    metrics = calculate_metrics(returns, benchmark_returns)

    # Get trades
    trades = portfolio.trades.records_readable
    if trades is None or (hasattr(trades, 'empty') and trades.empty):
        trades = pd.DataFrame()

    # Benchmark metrics
    benchmark_metrics = None
    if benchmark_returns is not None:
        benchmark_metrics = calculate_metrics(benchmark_returns.dropna())

    return BacktestResult(
        config=config,
        metrics=metrics,
        equity_curve=equity,
        trades=trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(),
        benchmark_metrics=benchmark_metrics,
    )


def generate_momentum_signals(
    prices: pd.DataFrame,
    lookback: int = 252,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate simple momentum signals.

    Long top N stocks by trailing return.

    Args:
        prices: Price data with 'close' level
        lookback: Lookback period in days
        top_n: Number of stocks to hold

    Returns:
        Signal DataFrame (1 = long, 0 = no position)
    """
    close = prices['close'] if 'close' in prices.columns.get_level_values(0) else prices

    # Calculate momentum (trailing return)
    momentum = close.pct_change(lookback)

    # Rank stocks each day
    ranks = momentum.rank(axis=1, ascending=False)

    # Signal = 1 for top N stocks
    signals = (ranks <= top_n).astype(float)

    # Don't trade first `lookback` days
    signals.iloc[:lookback] = 0

    return signals


def main():
    """CLI for running backtests."""
    parser = argparse.ArgumentParser(description="HRP Backtesting")
    parser.add_argument("--strategy", type=str, default="momentum", help="Strategy to run")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="Symbols to trade")

    args = parser.parse_args()

    # Default symbols from database
    if args.symbols is None:
        db = get_db()
        result = db.fetchall("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
        symbols = [r[0] for r in result]
    else:
        symbols = args.symbols

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    config = BacktestConfig(
        symbols=symbols,
        start_date=start,
        end_date=end,
        name=f"{args.strategy}_backtest",
    )

    logger.info(f"Running {args.strategy} backtest on {len(symbols)} symbols")

    # Load prices
    prices = get_price_data(symbols, start, end)

    # Generate signals
    if args.strategy == "momentum":
        signals = generate_momentum_signals(prices, lookback=252, top_n=min(5, len(symbols)))
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Run backtest
    result = run_backtest(signals, config, prices)

    # Print results
    print(f"\n{'='*50}")
    print(f"Backtest Results: {args.strategy}")
    print(f"{'='*50}")
    print(f"Period: {start} to {end}")
    print(f"Symbols: {len(symbols)}")
    print(f"\nStrategy Metrics:")
    from hrp.research.metrics import format_metrics
    print(format_metrics(result.metrics))

    if result.benchmark_metrics:
        print(f"\nBenchmark (SPY) Metrics:")
        print(format_metrics(result.benchmark_metrics))


if __name__ == "__main__":
    main()
