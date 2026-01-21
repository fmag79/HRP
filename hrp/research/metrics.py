"""
Standard metrics calculation for backtests.
"""

import numpy as np
import pandas as pd
from typing import Any


def calculate_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Calculate comprehensive backtest metrics.

    Args:
        returns: Daily returns series
        benchmark_returns: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading days per year (default 252)

    Returns:
        Dictionary of metric name -> value
    """
    if returns.empty:
        return {}

    # Clean returns
    returns = returns.dropna()

    metrics = {}

    # Basic returns
    metrics["total_return"] = (1 + returns).prod() - 1
    metrics["cagr"] = _calculate_cagr(returns, periods_per_year)

    # Risk metrics
    metrics["volatility"] = returns.std() * np.sqrt(periods_per_year)
    metrics["downside_volatility"] = _downside_volatility(returns, periods_per_year)

    # Risk-adjusted returns
    excess_returns = returns - risk_free_rate / periods_per_year
    metrics["sharpe_ratio"] = _sharpe_ratio(excess_returns, periods_per_year)
    metrics["sortino_ratio"] = _sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown
    metrics["max_drawdown"] = _max_drawdown(returns)
    metrics["calmar_ratio"] = metrics["cagr"] / abs(metrics["max_drawdown"]) if metrics["max_drawdown"] != 0 else 0

    # Trade statistics (from returns)
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    metrics["win_rate"] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    metrics["avg_win"] = positive_returns.mean() if len(positive_returns) > 0 else 0
    metrics["avg_loss"] = negative_returns.mean() if len(negative_returns) > 0 else 0
    metrics["profit_factor"] = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')

    # Benchmark comparison
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 10:
            strat_ret = aligned.iloc[:, 0]
            bench_ret = aligned.iloc[:, 1]

            metrics["alpha"], metrics["beta"] = _calculate_alpha_beta(strat_ret, bench_ret, periods_per_year)

            excess = strat_ret - bench_ret
            tracking_error = excess.std() * np.sqrt(periods_per_year)
            metrics["tracking_error"] = tracking_error
            metrics["information_ratio"] = excess.mean() * periods_per_year / tracking_error if tracking_error > 0 else 0

    return metrics


def _calculate_cagr(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate Compound Annual Growth Rate."""
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1 / n_years) - 1


def _sharpe_ratio(excess_returns: pd.Series, periods_per_year: int) -> float:
    """Calculate Sharpe ratio."""
    std = excess_returns.std()
    # Handle near-zero volatility (floating point precision)
    if std < 1e-10:
        return 0.0
    return excess_returns.mean() / std * np.sqrt(periods_per_year)


def _sortino_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    """Calculate Sortino ratio (downside risk only)."""
    excess = returns - risk_free_rate / periods_per_year
    downside = _downside_volatility(returns, periods_per_year)
    if downside == 0:
        return 0.0
    return excess.mean() * periods_per_year / downside


def _downside_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate downside volatility."""
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0
    std = negative_returns.std()
    # Handle near-zero volatility (floating point precision)
    if std < 1e-10:
        return 0.0
    return std * np.sqrt(periods_per_year)


def _max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    # Start with initial equity of 1.0
    cumulative = (1 + returns).cumprod()
    # Prepend 1.0 for initial equity value
    cumulative = pd.concat([pd.Series([1.0]), cumulative]).reset_index(drop=True)
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def _calculate_alpha_beta(returns: pd.Series, benchmark: pd.Series, periods_per_year: int) -> tuple[float, float]:
    """Calculate alpha and beta vs benchmark."""
    cov_matrix = np.cov(returns, benchmark)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0

    alpha = (returns.mean() - beta * benchmark.mean()) * periods_per_year

    return alpha, beta


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics for display."""
    lines = []

    formatters = {
        "total_return": lambda v: f"{v*100:.2f}%",
        "cagr": lambda v: f"{v*100:.2f}%",
        "volatility": lambda v: f"{v*100:.2f}%",
        "sharpe_ratio": lambda v: f"{v:.2f}",
        "sortino_ratio": lambda v: f"{v:.2f}",
        "max_drawdown": lambda v: f"{v*100:.2f}%",
        "calmar_ratio": lambda v: f"{v:.2f}",
        "win_rate": lambda v: f"{v*100:.1f}%",
        "profit_factor": lambda v: f"{v:.2f}",
        "alpha": lambda v: f"{v*100:.2f}%",
        "beta": lambda v: f"{v:.2f}",
        "information_ratio": lambda v: f"{v:.2f}",
    }

    for key, value in metrics.items():
        formatter = formatters.get(key, lambda v: f"{v:.4f}")
        lines.append(f"{key:20} {formatter(value)}")

    return "\n".join(lines)
