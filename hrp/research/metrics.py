"""
Standard metrics calculation for backtests.

Uses empyrical-reloaded for battle-tested portfolio performance metrics.

Stability Score metrics:
    - stability_score_v1: Walk-forward validation stability (2026-01-29)

Components:
    1. Sharpe coefficient of variation (CV): Measures performance consistency
    2. Drawdown dispersion: Measures risk consistency
    3. Sign flip penalty: Penalizes direction changes

Threshold:
    - ≤ 1.0: Stable
    - > 1.0: Unstable
"""

import empyrical as ep
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
    Calculate comprehensive backtest metrics using Empyrical.

    Args:
        returns: Daily returns series
        benchmark_returns: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading days per year (default 252)

    Returns:
        Dictionary of metric name -> value

    Metrics included:
        Core metrics (existing):
            - total_return: Cumulative return over period
            - cagr: Compound Annual Growth Rate
            - volatility: Annualized standard deviation
            - downside_volatility: Annualized downside deviation
            - sharpe_ratio: Risk-adjusted return (excess return / volatility)
            - sortino_ratio: Risk-adjusted return using downside volatility
            - max_drawdown: Maximum peak-to-trough decline
            - calmar_ratio: CAGR / |max_drawdown|
            - win_rate: Percentage of positive return days
            - avg_win: Average positive return
            - avg_loss: Average negative return
            - profit_factor: Sum of wins / |sum of losses|

        Benchmark metrics (if benchmark provided):
            - alpha: Excess return vs benchmark (annualized)
            - beta: Sensitivity to benchmark
            - tracking_error: Volatility of excess returns
            - information_ratio: Alpha / tracking_error

        New Empyrical metrics:
            - omega_ratio: Probability-weighted ratio of gains vs losses
            - value_at_risk: 5th percentile of returns (95% VaR)
            - conditional_value_at_risk: Expected shortfall below VaR
            - tail_ratio: Ratio of 95th to 5th percentile returns
            - stability: R-squared of cumulative log returns regression
    """
    if returns.empty:
        return {}

    # Clean returns (remove NaN values)
    returns = returns.dropna()
    if returns.empty:
        return {}

    # Convert risk_free_rate to per-period rate for Empyrical
    rf_per_period = risk_free_rate / periods_per_year

    metrics = {}

    # Basic returns (Empyrical)
    metrics["total_return"] = float(ep.cum_returns_final(returns))
    metrics["cagr"] = _calculate_cagr(returns, periods_per_year)

    # Risk metrics (Empyrical)
    metrics["volatility"] = float(
        ep.annual_volatility(returns, period="daily", annualization=periods_per_year)
    )
    metrics["downside_volatility"] = _downside_volatility(returns, periods_per_year)

    # Risk-adjusted returns (Empyrical)
    metrics["sharpe_ratio"] = _sharpe_ratio(
        returns - rf_per_period, periods_per_year
    )
    metrics["sortino_ratio"] = _sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown (Empyrical)
    metrics["max_drawdown"] = _max_drawdown(returns)
    metrics["calmar_ratio"] = (
        metrics["cagr"] / abs(metrics["max_drawdown"])
        if metrics["max_drawdown"] != 0
        else 0.0
    )

    # Trade statistics (custom - not in Empyrical)
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    metrics["win_rate"] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    metrics["avg_win"] = float(positive_returns.mean()) if len(positive_returns) > 0 else 0
    metrics["avg_loss"] = float(negative_returns.mean()) if len(negative_returns) > 0 else 0
    metrics["profit_factor"] = (
        abs(positive_returns.sum() / negative_returns.sum())
        if negative_returns.sum() != 0
        else float("inf")
    )

    # NEW: Additional Empyrical metrics
    try:
        omega = ep.omega_ratio(returns, risk_free=rf_per_period, required_return=0.0)
        # Handle NaN/inf (can occur with all positive or all negative returns)
        if np.isnan(omega) or np.isinf(omega):
            metrics["omega_ratio"] = float("inf") if len(negative_returns) == 0 else 0.0
        else:
            metrics["omega_ratio"] = float(omega)
    except Exception:
        metrics["omega_ratio"] = 0.0

    try:
        metrics["value_at_risk"] = float(ep.value_at_risk(returns, cutoff=0.05))
    except Exception:
        metrics["value_at_risk"] = 0.0

    try:
        metrics["conditional_value_at_risk"] = float(
            ep.conditional_value_at_risk(returns, cutoff=0.05)
        )
    except Exception:
        metrics["conditional_value_at_risk"] = 0.0

    try:
        metrics["tail_ratio"] = float(ep.tail_ratio(returns))
    except Exception:
        metrics["tail_ratio"] = 0.0

    try:
        metrics["stability"] = float(ep.stability_of_timeseries(returns))
    except Exception:
        metrics["stability"] = 0.0

    # Benchmark comparison (Empyrical)
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 10:
            strat_ret = aligned.iloc[:, 0]
            bench_ret = aligned.iloc[:, 1]

            alpha, beta = _calculate_alpha_beta(strat_ret, bench_ret, periods_per_year)
            metrics["alpha"] = alpha
            metrics["beta"] = beta

            excess = strat_ret - bench_ret
            tracking_error = float(excess.std() * np.sqrt(periods_per_year))
            metrics["tracking_error"] = tracking_error
            metrics["information_ratio"] = (
                float(excess.mean() * periods_per_year / tracking_error)
                if tracking_error > 0
                else 0.0
            )

    return metrics


def _calculate_cagr(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate Compound Annual Growth Rate (Empyrical with fallback)."""
    try:
        cagr = ep.cagr(returns, period="daily", annualization=periods_per_year)
        if np.isnan(cagr) or np.isinf(cagr):
            return 0.0
        return float(cagr)
    except Exception:
        # Fallback for edge cases
        total_return = (1 + returns).prod()
        n_years = len(returns) / periods_per_year
        if n_years <= 0 or total_return <= 0:
            return 0.0
        return float(total_return ** (1 / n_years) - 1)


def _sharpe_ratio(excess_returns: pd.Series, periods_per_year: int) -> float:
    """Calculate Sharpe ratio from excess returns."""
    std = excess_returns.std()
    # Handle near-zero volatility
    if std < 1e-10:
        return 0.0
    return float(excess_returns.mean() / std * np.sqrt(periods_per_year))


def _sortino_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    """Calculate Sortino ratio (Empyrical with fallback)."""
    try:
        rf_per_period = risk_free_rate / periods_per_year
        sortino = ep.sortino_ratio(
            returns,
            required_return=rf_per_period,
            period="daily",
            annualization=periods_per_year,
        )
        if np.isnan(sortino) or np.isinf(sortino):
            return 0.0
        return float(sortino)
    except Exception:
        # Fallback for edge cases
        excess = returns - risk_free_rate / periods_per_year
        downside = _downside_volatility(returns, periods_per_year)
        if downside == 0:
            return 0.0
        return float(excess.mean() * periods_per_year / downside)


def _downside_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate downside volatility (annualized)."""
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0
    std = negative_returns.std()
    # Handle near-zero volatility
    if std < 1e-10:
        return 0.0
    return float(std * np.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown (Empyrical with fallback)."""
    try:
        mdd = ep.max_drawdown(returns)
        if np.isnan(mdd):
            return 0.0
        return float(mdd)
    except Exception:
        # Fallback for edge cases
        cumulative = (1 + returns).cumprod()
        cumulative = pd.concat([pd.Series([1.0]), cumulative]).reset_index(drop=True)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())


def _calculate_alpha_beta(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int
) -> tuple[float, float]:
    """Calculate alpha and beta vs benchmark (numpy covariance method)."""
    try:
        # Use numpy covariance for beta (more stable than Empyrical for this)
        cov_matrix = np.cov(returns, benchmark)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.0

        # Alpha = strategy excess return over beta-adjusted benchmark
        alpha = (returns.mean() - beta * benchmark.mean()) * periods_per_year

        return float(alpha), float(beta)
    except Exception:
        return 0.0, 0.0


def calculate_stability_score_v1(
    fold_sharpes: list[float],
    fold_drawdowns: list[float],
    mean_fold_ic: float,
) -> float:
    """
    Stability Score v1 - Lower is better.

    Components:
    1. Sharpe coefficient of variation (CV)
    2. Drawdown dispersion penalty
    3. Sign flip penalty

    Args:
        fold_sharpes: List of Sharpe ratios from each fold
        fold_drawdowns: List of max drawdowns from each fold
        mean_fold_ic: Mean Information Coefficient across folds

    Returns:
        float: Stability score (≤ 1.0 is stable)

    Examples:
        >>> fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]
        >>> fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]
        >>> score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, 0.05)
        >>> assert score <= 1.0
    """
    # Component 1: Sharpe CV
    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes)
    sharpe_cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float('inf')

    # Component 2: Drawdown dispersion
    mean_dd = np.mean(fold_drawdowns)
    std_dd = np.std(fold_drawdowns)
    dd_dispersion = (std_dd / mean_dd) if mean_dd > 0 else 0

    # Component 3: Sign flip penalty (simplified - uses mean IC sign)
    # For multi-fold sign flips, count positive vs negative fold ICs
    # This is a simplified version using mean IC
    sign_flip_penalty = 0.5 if mean_fold_ic < 0 else 0

    stability_score = sharpe_cv + dd_dispersion + sign_flip_penalty

    return stability_score


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics for display."""
    lines = []

    formatters = {
        # Percentages
        "total_return": lambda v: f"{v*100:.2f}%",
        "cagr": lambda v: f"{v*100:.2f}%",
        "volatility": lambda v: f"{v*100:.2f}%",
        "downside_volatility": lambda v: f"{v*100:.2f}%",
        "max_drawdown": lambda v: f"{v*100:.2f}%",
        "win_rate": lambda v: f"{v*100:.1f}%",
        "alpha": lambda v: f"{v*100:.2f}%",
        "tracking_error": lambda v: f"{v*100:.2f}%",
        "value_at_risk": lambda v: f"{v*100:.2f}%",
        "conditional_value_at_risk": lambda v: f"{v*100:.2f}%",
        # Ratios
        "sharpe_ratio": lambda v: f"{v:.2f}",
        "sortino_ratio": lambda v: f"{v:.2f}",
        "calmar_ratio": lambda v: f"{v:.2f}",
        "profit_factor": lambda v: f"{v:.2f}",
        "beta": lambda v: f"{v:.2f}",
        "information_ratio": lambda v: f"{v:.2f}",
        "omega_ratio": lambda v: f"{v:.2f}",
        "tail_ratio": lambda v: f"{v:.2f}",
        "stability": lambda v: f"{v:.4f}",
    }

    for key, value in metrics.items():
        formatter = formatters.get(key, lambda v: f"{v:.4f}")
        lines.append(f"{key:20} {formatter(value)}")

    return "\n".join(lines)
