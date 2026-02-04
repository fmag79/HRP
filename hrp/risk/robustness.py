"""
Robustness checks for trading strategies.

Tests parameter sensitivity, time stability, and regime robustness.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RobustnessResult:
    """Result of robustness checks."""
    
    passed: bool
    checks: dict[str, Any]
    failures: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"Robustness {status}: {len(self.failures)} checks failed"


def check_parameter_sensitivity(
    experiments: dict[str, dict[str, Any]],
    baseline_key: str,
    threshold: float = 0.5,
) -> RobustnessResult:
    """
    Check parameter sensitivity.
    
    Tests if strategy performance degrades gracefully when parameters vary.
    
    Args:
        experiments: Dict mapping experiment name to metrics
        baseline_key: Key for baseline experiment
        threshold: Minimum ratio of variation/baseline Sharpe (default 0.5)
        
    Returns:
        RobustnessResult indicating if parameters are stable
    """
    if baseline_key not in experiments:
        raise ValueError(f"Baseline experiment '{baseline_key}' not found")
    
    baseline_sharpe = experiments[baseline_key]["sharpe"]
    failures = []
    variations = {}
    
    for name, exp in experiments.items():
        if name == baseline_key:
            continue
        
        var_sharpe = exp["sharpe"]
        ratio = var_sharpe / baseline_sharpe if baseline_sharpe > 0 else 0
        variations[name] = {
            "sharpe": var_sharpe,
            "ratio": ratio,
            "params": exp.get("params", {}),
        }
        
        if ratio < threshold:
            failures.append(
                f"{name}: Sharpe {var_sharpe:.2f} is {ratio:.1%} of baseline "
                f"(threshold: {threshold:.1%})"
            )
    
    passed = len(failures) == 0
    
    logger.info(
        f"Parameter sensitivity: {len(failures)} failures, "
        f"{len(variations)} variations tested"
    )
    
    return RobustnessResult(
        passed=passed,
        checks={
            "parameter_sensitivity": {
                "baseline_sharpe": baseline_sharpe,
                "variations": variations,
                "threshold": threshold,
            }
        },
        failures=failures,
    )


def check_time_stability(
    period_metrics: list[dict[str, Any]],
    min_profitable_ratio: float = 0.67,
) -> RobustnessResult:
    """
    Check time period stability.
    
    Tests if strategy is profitable across multiple time periods.
    
    Args:
        period_metrics: List of dicts with period metrics
            Each must have: 'period', 'sharpe', 'profitable'
        min_profitable_ratio: Minimum ratio of profitable periods (default 2/3)
        
    Returns:
        RobustnessResult indicating if strategy is time-stable
    """
    if not period_metrics:
        raise ValueError("No period metrics provided")

    n_periods = len(period_metrics)
    # Derive 'profitable' from total_return if not explicitly set
    n_profitable = sum(
        1 for p in period_metrics
        if p.get("profitable", p.get("total_return", 0) > 0)
    )
    profitable_ratio = n_profitable / n_periods
    
    failures = []
    
    if profitable_ratio < min_profitable_ratio:
        failures.append(
            f"Only {n_profitable}/{n_periods} periods profitable "
            f"({profitable_ratio:.1%} < {min_profitable_ratio:.1%})"
        )
    
    # Calculate Sharpe stability (std of Sharpes)
    sharpes = [p["sharpe"] for p in period_metrics]
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float("inf")
    
    # High coefficient of variation indicates instability
    if cv > 1.0:
        failures.append(
            f"High Sharpe variability: CV={cv:.2f} (mean={mean_sharpe:.2f}, "
            f"std={std_sharpe:.2f})"
        )
    
    passed = len(failures) == 0
    
    logger.info(
        f"Time stability: {n_profitable}/{n_periods} profitable, "
        f"mean Sharpe={mean_sharpe:.2f}, CV={cv:.2f}"
    )
    
    return RobustnessResult(
        passed=passed,
        checks={
            "time_stability": {
                "n_periods": n_periods,
                "n_profitable": n_profitable,
                "profitable_ratio": profitable_ratio,
                "mean_sharpe": float(mean_sharpe),
                "sharpe_cv": float(cv),
                "periods": period_metrics,
            }
        },
        failures=failures,
    )


def check_regime_stability(
    regime_metrics: dict[str, dict[str, Any]],
    min_regimes_profitable: int = 2,
) -> RobustnessResult:
    """
    Check market regime stability.
    
    Tests if strategy works in different market regimes (bull, bear, sideways).
    
    Args:
        regime_metrics: Dict mapping regime name to metrics
            Each must have: 'sharpe', 'profitable'
        min_regimes_profitable: Minimum number of regimes that must be profitable
        
    Returns:
        RobustnessResult indicating if strategy is regime-robust
    """
    if not regime_metrics:
        raise ValueError("No regime metrics provided")

    n_regimes = len(regime_metrics)
    # Derive 'profitable' from total_return or sharpe if not explicitly set
    n_profitable = sum(
        1 for m in regime_metrics.values()
        if m.get("profitable", m.get("total_return", m.get("sharpe", 0)) > 0)
    )
    
    failures = []
    
    if n_profitable < min_regimes_profitable:
        failures.append(
            f"Only {n_profitable}/{n_regimes} regimes profitable "
            f"(required: {min_regimes_profitable})"
        )
    
    # List unprofitable regimes
    unprofitable = [
        name for name, m in regime_metrics.items() if not m["profitable"]
    ]
    
    if unprofitable:
        failures.append(
            f"Unprofitable in regimes: {', '.join(unprofitable)}"
        )
    
    passed = len(failures) == 0
    
    logger.info(
        f"Regime stability: {n_profitable}/{n_regimes} profitable regimes"
    )
    
    return RobustnessResult(
        passed=passed,
        checks={
            "regime_stability": {
                "n_regimes": n_regimes,
                "n_profitable": n_profitable,
                "min_required": min_regimes_profitable,
                "regimes": regime_metrics,
                "unprofitable_regimes": unprofitable,
            }
        },
        failures=failures,
    )


def check_regime_stability_hmm(
    returns: pd.Series,
    prices: pd.DataFrame,
    strategy_metrics_by_date: pd.DataFrame,
    n_regimes: int = 3,
    min_regimes_profitable: int = 2,
) -> RobustnessResult:
    """
    Check strategy performance across HMM-detected regimes.

    Uses Hidden Markov Model to automatically detect market regimes
    and evaluates strategy performance in each regime.

    Args:
        returns: Strategy returns series, indexed by date
        prices: Price DataFrame for regime detection (OHLCV)
        strategy_metrics_by_date: DataFrame with strategy metrics per date
            Must have 'return' or 'pnl' column
        n_regimes: Number of HMM regimes to detect (default 3)
        min_regimes_profitable: Minimum regimes that must be profitable

    Returns:
        RobustnessResult indicating if strategy is regime-robust
    """
    try:
        from hrp.ml.regime import HMMConfig, RegimeDetector, MarketRegime
    except ImportError as e:
        logger.warning(f"Could not import regime module: {e}")
        return RobustnessResult(
            passed=False,
            checks={"regime_stability_hmm": {"error": "regime module not available"}},
            failures=["HMM regime detection not available"],
        )

    # Fit HMM to detect regimes
    try:
        config = HMMConfig(n_regimes=n_regimes)
        detector = RegimeDetector(config)
        detector.fit(prices)
        regime_stats = detector.get_regime_statistics(prices)
    except Exception as e:
        logger.warning(f"HMM fitting failed: {e}")
        return RobustnessResult(
            passed=False,
            checks={"regime_stability_hmm": {"error": str(e)}},
            failures=[f"HMM fitting failed: {e}"],
        )

    # Get regime labels for each date
    regimes = regime_stats.regimes

    # Align returns with regimes
    aligned_returns = returns.reindex(regimes.index)

    # Calculate metrics per regime
    regime_metrics = {}
    for regime_idx in range(n_regimes):
        mask = regimes == regime_idx
        regime_returns = aligned_returns[mask].dropna()

        if len(regime_returns) < 10:
            # Not enough data for this regime
            regime_label = regime_stats.regime_labels.get(regime_idx, MarketRegime.SIDEWAYS)
            regime_metrics[regime_label.value] = {
                "sharpe": 0.0,
                "profitable": False,
                "n_periods": len(regime_returns),
                "total_return": 0.0,
                "insufficient_data": True,
            }
            continue

        # Calculate Sharpe ratio
        mean_return = regime_returns.mean() * 252  # Annualized
        std_return = regime_returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0.0

        # Calculate total return
        total_return = (1 + regime_returns).prod() - 1

        # Is regime profitable?
        profitable = total_return > 0

        # Get regime label
        regime_label = regime_stats.regime_labels.get(regime_idx, MarketRegime.SIDEWAYS)

        regime_metrics[regime_label.value] = {
            "sharpe": float(sharpe),
            "profitable": profitable,
            "n_periods": len(regime_returns),
            "total_return": float(total_return),
            "mean_return": float(mean_return),
            "volatility": float(std_return),
        }

    # Count profitable regimes
    n_profitable = sum(1 for m in regime_metrics.values() if m.get("profitable", False))

    failures = []

    if n_profitable < min_regimes_profitable:
        failures.append(
            f"Only {n_profitable}/{n_regimes} HMM regimes profitable "
            f"(required: {min_regimes_profitable})"
        )

    # List unprofitable regimes
    unprofitable = [
        name for name, m in regime_metrics.items()
        if not m.get("profitable", False) and not m.get("insufficient_data", False)
    ]

    if unprofitable:
        failures.append(
            f"Unprofitable in HMM regimes: {', '.join(unprofitable)}"
        )

    # Check for severe underperformance in any regime
    for name, m in regime_metrics.items():
        if m.get("sharpe", 0) < -0.5 and not m.get("insufficient_data", False):
            failures.append(
                f"Severely negative Sharpe in {name} regime: {m['sharpe']:.2f}"
            )

    passed = len(failures) == 0

    logger.info(
        f"HMM regime stability: {n_profitable}/{n_regimes} profitable regimes, "
        f"labels={list(regime_metrics.keys())}"
    )

    return RobustnessResult(
        passed=passed,
        checks={
            "regime_stability_hmm": {
                "n_regimes": n_regimes,
                "n_profitable": n_profitable,
                "min_required": min_regimes_profitable,
                "regimes": regime_metrics,
                "unprofitable_regimes": unprofitable,
                "transition_matrix": regime_stats.transition_matrix.tolist(),
                "regime_durations": regime_stats.regime_durations,
            }
        },
        failures=failures,
    )
