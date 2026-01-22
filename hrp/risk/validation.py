"""
Statistical validation for trading strategies.

Implements validation criteria, significance testing, and robustness checks.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class ValidationCriteria:
    """
    Criteria for strategy validation.
    
    A strategy must meet ALL criteria to be validated.
    """
    
    # Performance thresholds
    min_sharpe: float = 0.5
    min_trades: int = 100
    max_drawdown: float = 0.25
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.2
    
    # Time requirements
    min_oos_period_days: int = 730  # 2 years minimum
    
    # Statistical requirements
    significance_level: float = 0.05


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    
    passed: bool
    metrics: dict[str, float]
    failed_criteria: list[str]
    warnings: list[str] = field(default_factory=list)
    confidence_score: float | None = None
    
    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"Validation {status}: {len(self.failed_criteria)} criteria failed"


def significance_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Test if strategy significantly outperforms benchmark.
    
    Args:
        strategy_returns: Daily strategy returns
        benchmark_returns: Daily benchmark returns
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary with test results:
        - t_statistic: T-statistic value
        - p_value: One-sided p-value
        - significant: Boolean indicating significance
        - excess_return_annualized: Annualized excess return
    """
    excess_returns = strategy_returns - benchmark_returns
    
    # One-sample t-test: Is mean excess return > 0?
    t_stat, p_value_two_sided = stats.ttest_1samp(excess_returns.dropna(), 0)
    
    # Convert to one-sided p-value
    p_value = p_value_two_sided / 2 if t_stat > 0 else 1 - p_value_two_sided / 2
    
    result = {
        "excess_return_annualized": float(excess_returns.mean() * 252),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }
    
    logger.info(
        f"Significance test: t={t_stat:.2f}, p={p_value:.4f}, "
        f"significant={result['significant']}"
    )
    
    return result


def calculate_bootstrap_ci(
    returns: pd.Series,
    metric: str = "sharpe",
    confidence: float = 0.95,
    n_bootstraps: int = 10000,
) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        returns: Daily returns series
        metric: Metric to calculate ('sharpe', 'mean', 'std')
        confidence: Confidence level (default 0.95)
        n_bootstraps: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    np.random.seed(42)
    
    def calculate_metric(r: pd.Series) -> float:
        if metric == "sharpe":
            return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
        elif metric == "mean":
            return r.mean() * 252
        elif metric == "std":
            return r.std() * np.sqrt(252)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Bootstrap
    bootstrap_values = []
    n = len(returns)
    
    for _ in range(n_bootstraps):
        sample = returns.sample(n=n, replace=True)
        bootstrap_values.append(calculate_metric(sample))
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_values, alpha / 2 * 100))
    upper = float(np.percentile(bootstrap_values, (1 - alpha / 2) * 100))
    
    logger.debug(
        f"Bootstrap CI ({confidence*100:.0f}%): [{lower:.4f}, {upper:.4f}]"
    )
    
    return lower, upper


def validate_strategy(
    metrics: dict[str, float],
    criteria: ValidationCriteria | None = None,
) -> ValidationResult:
    """
    Validate strategy against criteria.
    
    Args:
        metrics: Dictionary of strategy metrics
        criteria: Validation criteria (uses defaults if None)
        
    Returns:
        ValidationResult with pass/fail and details
    """
    if criteria is None:
        criteria = ValidationCriteria()
    
    failed = []
    warnings = []
    
    # Check each criterion
    if metrics.get("sharpe", 0) < criteria.min_sharpe:
        failed.append(
            f"Sharpe {metrics.get('sharpe'):.2f} < {criteria.min_sharpe:.2f}"
        )
    
    if metrics.get("num_trades", 0) < criteria.min_trades:
        failed.append(
            f"Trades {metrics.get('num_trades')} < {criteria.min_trades}"
        )
    
    if metrics.get("max_drawdown", 1.0) > criteria.max_drawdown:
        failed.append(
            f"Max DD {metrics.get('max_drawdown'):.2%} > {criteria.max_drawdown:.2%}"
        )
    
    if metrics.get("win_rate", 0) < criteria.min_win_rate:
        failed.append(
            f"Win rate {metrics.get('win_rate'):.2%} < {criteria.min_win_rate:.2%}"
        )
    
    if metrics.get("profit_factor", 0) < criteria.min_profit_factor:
        failed.append(
            f"Profit factor {metrics.get('profit_factor'):.2f} < "
            f"{criteria.min_profit_factor:.2f}"
        )
    
    if metrics.get("oos_period_days", 0) < criteria.min_oos_period_days:
        failed.append(
            f"OOS period {metrics.get('oos_period_days')} days < "
            f"{criteria.min_oos_period_days} days"
        )
    
    # Calculate confidence score (0-1)
    # Higher = more confident in validation
    confidence_factors = []
    
    if metrics.get("sharpe"):
        confidence_factors.append(
            min(1.0, metrics["sharpe"] / (criteria.min_sharpe * 2))
        )
    
    if metrics.get("num_trades"):
        confidence_factors.append(
            min(1.0, metrics["num_trades"] / (criteria.min_trades * 2))
        )
    
    confidence_score = float(np.mean(confidence_factors)) if confidence_factors else 0.0
    
    passed = len(failed) == 0
    
    logger.info(
        f"Validation {'PASSED' if passed else 'FAILED'}: "
        f"{len(failed)} criteria failed, confidence={confidence_score:.2f}"
    )
    
    return ValidationResult(
        passed=passed,
        metrics=metrics,
        failed_criteria=failed,
        warnings=warnings,
        confidence_score=confidence_score,
    )


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """
    Apply Bonferroni correction for multiple hypothesis testing.
    
    Conservative method: divides significance threshold by number of tests.
    
    Args:
        p_values: List of p-values
        alpha: Desired family-wise error rate
        
    Returns:
        List of booleans indicating which hypotheses to reject (significant)
    """
    n = len(p_values)
    adjusted_alpha = alpha / n
    
    rejected = [p <= adjusted_alpha for p in p_values]
    
    logger.info(
        f"Bonferroni correction: {sum(rejected)}/{n} significant "
        f"(adjusted α={adjusted_alpha:.4f})"
    )
    
    return rejected


def benjamini_hochberg(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Less conservative than Bonferroni, controls false discovery rate.
    
    Args:
        p_values: List of p-values
        alpha: Desired false discovery rate
        
    Returns:
        List of booleans indicating which hypotheses to reject (significant)
    """
    n = len(p_values)
    
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]
    
    # Find largest k where p(k) <= k/n * alpha
    thresholds = [(i + 1) / n * alpha for i in range(n)]
    passes_threshold = sorted_pvals <= thresholds
    
    # Determine rejection cutoff
    rejected_sorted = np.zeros(n, dtype=bool)
    if passes_threshold.any():
        max_rejected_idx = np.max(np.where(passes_threshold)[0])
        rejected_sorted[:max_rejected_idx + 1] = True
    
    # Map back to original order
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices] = rejected_sorted
    
    logger.info(
        f"Benjamini-Hochberg: {sum(rejected)}/{n} significant (FDR={alpha})"
    )
    
    return rejected.tolist()
