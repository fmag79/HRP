"""
VectorBT backtesting wrapper for HRP.
"""

import argparse
from datetime import date
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from hrp.data.db import get_db
from hrp.research.config import BacktestConfig, BacktestResult, CostModel, StopLossConfig
from hrp.research.metrics import calculate_metrics
from hrp.research.benchmark import get_benchmark_returns
from hrp.research.stops import apply_trailing_stops
from hrp.risk.limits import PreTradeValidator, ValidationReport


def _load_sector_mapping(symbols: list[str], db=None) -> pd.Series:
    """Load sector mapping from database.

    Args:
        symbols: List of ticker symbols
        db: Optional database connection (uses default if not provided)

    Returns:
        Series mapping symbol -> sector
    """
    db = db or get_db()
    symbols_str = ",".join(f"'{s}'" for s in symbols)

    try:
        result = db.fetchdf(
            f"SELECT symbol, sector FROM symbols WHERE symbol IN ({symbols_str})"
        )
        if result.empty:
            return pd.Series({s: "Unknown" for s in symbols})
        return result.set_index("symbol")["sector"]
    except Exception:
        return pd.Series({s: "Unknown" for s in symbols})


def _compute_adv(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute average daily volume from price data.

    Args:
        prices: Price DataFrame with volume level
        window: Rolling window period

    Returns:
        DataFrame with ADV per symbol
    """
    if "volume" in prices.columns.get_level_values(0):
        return prices["volume"].rolling(window).mean()
    return pd.DataFrame()


def _load_volatility(symbols: list[str], start: date, end: date, db=None) -> pd.Series:
    """Load volatility data for cost estimation.

    Args:
        symbols: List of ticker symbols
        start: Start date
        end: End date
        db: Optional database connection (uses default if not provided)

    Returns:
        Series with average volatility per symbol
    """
    db = db or get_db()
    symbols_str = ",".join(f"'{s}'" for s in symbols)

    try:
        result = db.fetchdf(f"""
            SELECT symbol, AVG(volatility_20d) as avg_vol
            FROM features
            WHERE symbol IN ({symbols_str})
              AND feature_name = 'volatility_20d'
              AND date >= ?
              AND date <= ?
            GROUP BY symbol
        """, (start, end))

        if result.empty:
            # Return default volatility if no data
            return pd.Series({s: 0.02 for s in symbols})
        return result.set_index("symbol")["avg_vol"]
    except Exception:
        return pd.Series({s: 0.02 for s in symbols})


def get_price_data(
    symbols: list[str],
    start: date,
    end: date,
    db=None,
) -> pd.DataFrame:
    """
    Load price data from database.

    Automatically filters to NYSE trading days only (excludes weekends and holidays).
    Price adjustments (splits, dividends) are pre-computed in the adj_close column.

    Args:
        symbols: List of ticker symbols
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        DataFrame with MultiIndex columns (symbol, field)
    """
    from hrp.utils.calendar import get_trading_days

    # Filter date range to trading days only
    trading_days = get_trading_days(start, end)
    if len(trading_days) == 0:
        raise ValueError(f"No trading days found between {start} and {end}")

    # Update start/end to first and last trading days
    filtered_start = trading_days[0].date()
    filtered_end = trading_days[-1].date()

    logger.debug(
        f"Filtered date range to {len(trading_days)} trading days: "
        f"{filtered_start} to {filtered_end}"
    )

    db = db or get_db()

    symbols_str = ",".join(f"'{s}'" for s in symbols)
    query = f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM prices
        WHERE symbol IN ({symbols_str})
          AND date >= ?
          AND date <= ?
        ORDER BY date, symbol
    """

    df = db.fetchdf(query, (filtered_start, filtered_end))

    if df.empty:
        raise ValueError(f"No price data found for {symbols} from {start} to {end}")

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
        BacktestResult with metrics, equity curve, trades, and validation report
    """
    # Load prices if not provided
    if prices is None:
        prices = get_price_data(config.symbols, config.start_date, config.end_date)

    # Get close prices
    close = prices['close'] if 'close' in prices.columns.get_level_values(0) else prices

    # Align signals with prices
    signals = signals.reindex(close.index).fillna(0)

    # PRE-TRADE VALIDATION
    validation_report = None
    if config.risk_limits is not None:
        logger.info(f"Applying risk limits (mode: {config.validation_mode})")

        # Load sector data
        sectors = _load_sector_mapping(config.symbols)

        # Compute ADV
        adv = _compute_adv(prices)

        # Load volatility for cost estimation
        volatility = _load_volatility(config.symbols, config.start_date, config.end_date)

        # Run validation
        validator = PreTradeValidator(
            limits=config.risk_limits,
            cost_model=config.cost_model,
            mode=config.validation_mode,
        )
        signals, validation_report = validator.validate(
            signals=signals,
            prices=close,
            sectors=sectors,
            adv=adv if not adv.empty else None,
        )

        logger.info(f"Validation complete: {len(validation_report.clips)} clips, "
                   f"{len(validation_report.warnings)} warnings, "
                   f"{len(validation_report.violations)} violations")

    # Apply trailing stops if configured
    stop_events = None
    if config.stop_loss is not None and config.stop_loss.enabled:
        logger.info(
            f"Applying {config.stop_loss.type} trailing stops "
            f"(ATR multiplier: {config.stop_loss.atr_multiplier})"
        )
        signals, stop_events = apply_trailing_stops(
            signals=signals,
            prices=prices,
            stop_config=config.stop_loss,
        )
        n_stops = (stop_events > 0).sum().sum() if stop_events is not None else 0
        logger.debug(f"Trailing stops triggered {n_stops} exit signals")

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
        validation_report=validation_report,
    )


def get_fundamentals_for_backtest(
    symbols: list[str],
    metrics: list[str],
    dates: pd.DatetimeIndex,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load point-in-time fundamentals for backtest.

    For each date in the backtest, gets fundamentals as they would
    have been known on that date. This ensures no look-ahead bias
    in backtests using fundamental data.

    Args:
        symbols: List of ticker symbols
        metrics: List of fundamental metric names (e.g., 'revenue', 'eps')
        dates: DatetimeIndex of backtest dates
        db_path: Optional path to database (uses default if not provided)

    Returns:
        DataFrame with MultiIndex (date, symbol) and metric columns.
        Each row contains the most recent fundamental values available
        as of that date.

    Example:
        # Load fundamentals for a backtest
        prices = get_price_data(['AAPL', 'MSFT'], start, end)
        fundamentals = get_fundamentals_for_backtest(
            symbols=['AAPL', 'MSFT'],
            metrics=['revenue', 'eps', 'book_value'],
            dates=prices.index
        )

        # Access fundamentals for a specific date
        date_fundamentals = fundamentals.loc['2023-01-15']
    """
    db = get_db(db_path) if db_path else get_db()
    all_data = []

    # Build query parts
    symbols_str = ",".join(f"'{s}'" for s in symbols)
    metrics_str = ",".join(f"'{m}'" for m in metrics)

    for trade_date in dates:
        # Convert to date if it's a Timestamp
        as_of = trade_date.date() if hasattr(trade_date, 'date') else trade_date

        # Use window function to get the most recent report for each symbol/metric
        # where report_date <= as_of_date
        query = f"""
            WITH ranked_fundamentals AS (
                SELECT
                    symbol,
                    metric,
                    value,
                    report_date,
                    period_end,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, metric
                        ORDER BY report_date DESC
                    ) as rn
                FROM fundamentals
                WHERE symbol IN ({symbols_str})
                  AND metric IN ({metrics_str})
                  AND report_date <= ?
            )
            SELECT symbol, metric, value, report_date, period_end
            FROM ranked_fundamentals
            WHERE rn = 1
            ORDER BY symbol, metric
        """

        df = db.fetchdf(query, (as_of,))

        if not df.empty:
            # Pivot to get metrics as columns
            pivoted = df.pivot(index='symbol', columns='metric', values='value')
            pivoted['trade_date'] = trade_date
            all_data.append(pivoted.reset_index())

    if not all_data:
        # Return empty DataFrame with expected structure
        logger.warning(f"No fundamentals found for {symbols} with metrics {metrics}")
        return pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=['date', 'symbol']),
            columns=metrics
        )

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    # Set MultiIndex (date, symbol)
    combined = combined.set_index(['trade_date', 'symbol'])
    combined.index = combined.index.set_names(['date', 'symbol'])

    # Sort by date and symbol
    combined = combined.sort_index()

    logger.debug(
        f"Loaded fundamentals for {len(dates)} dates, "
        f"{len(combined.index.get_level_values('symbol').unique())} symbols"
    )

    return combined


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
        from hrp.api.platform import PlatformAPI
        symbols = PlatformAPI().get_available_symbols()
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
