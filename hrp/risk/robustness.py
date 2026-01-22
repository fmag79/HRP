"""
Robustness checks for trading strategies.

Tests parameter sensitivity, time stability, and regime robustness.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
    n_profitable = sum(1 for p in period_metrics if p["profitable"])
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
    n_profitable = sum(1 for m in regime_metrics.values() if m["profitable"])
    
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
