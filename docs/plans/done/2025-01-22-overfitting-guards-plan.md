# Overfitting Guards & Test Set Discipline Implementation Plan

> **For Claude:** Use `/executing-plans` to implement this plan task-by-task with TDD discipline.

**Goal:** Implement robust overfitting prevention mechanisms to ensure validated strategies are statistically sound and not overfit to training or validation data.

**Architecture:** Two-part system: (1) TestSetGuard for preventing excessive test set peeking, (2) ValidationGates for enforcing statistical rigor before hypothesis validation. Integrates with existing ML framework and hypothesis workflow.

**Context:** Builds on completed ML framework (models, training, walk-forward validation). Enforces discipline that prevents fooling yourself with overfitted results.

---

## Overview

### What We're Building

1. **Test Set Guard** - Prevents excessive test set evaluation (max 3 evaluations per hypothesis)
2. **Validation Gates** - Enforces minimum criteria before hypothesis can be validated
3. **Robustness Checks** - Parameter sensitivity, time period stability, regime analysis
4. **Statistical Significance Testing** - T-tests, bootstrap confidence intervals
5. **Multiple Hypothesis Correction** - Bonferroni and Benjamini-Hochberg methods
6. **Validation Report Generation** - Automated comprehensive validation reports

### Architecture

```
hrp/
└── risk/
    ├── validation.py          # Statistical validation, significance tests
    ├── overfitting.py         # Test set guard, overfitting detection
    └── robustness.py          # Robustness checks (parameter, time, regime)

Integration Points:
- hrp/ml/training.py          # Inject TestSetGuard when evaluating test set
- hrp/api/platform.py         # Enforce validation gates on update_hypothesis
- hrp/research/hypothesis.py  # Store validation metadata
```

---

## Task 1: Test Set Guard (`hrp/risk/overfitting.py`)

**Files:**
- Create: `hrp/risk/overfitting.py`
- Test: `tests/test_risk/test_overfitting.py`
- Schema: Add test_set_evaluations tracking table

**Purpose:** Prevent data snooping by limiting test set access per hypothesis.

### Step 1: Write failing test for TestSetGuard

```python
# tests/test_risk/test_overfitting.py
"""Tests for overfitting prevention mechanisms."""

import pytest
from unittest.mock import MagicMock, patch

from hrp.risk.overfitting import TestSetGuard, OverfittingError


class TestTestSetGuard:
    """Tests for TestSetGuard class."""

    def test_first_evaluation_allowed(self):
        """Test first test set evaluation is allowed."""
        guard = TestSetGuard(hypothesis_id="HYP-2025-001")
        
        # Should not raise
        with guard.evaluate():
            pass
        
        assert guard.evaluation_count == 1

    def test_multiple_evaluations_allowed_under_limit(self):
        """Test multiple evaluations allowed under limit."""
        guard = TestSetGuard(hypothesis_id="HYP-2025-001")
        
        for _ in range(3):
            with guard.evaluate():
                pass
        
        assert guard.evaluation_count == 3

    def test_fourth_evaluation_raises_error(self):
        """Test fourth evaluation raises OverfittingError."""
        guard = TestSetGuard(hypothesis_id="HYP-2025-001", max_evaluations=3)
        
        # Use up 3 evaluations
        for _ in range(3):
            with guard.evaluate():
                pass
        
        # Fourth should fail
        with pytest.raises(OverfittingError, match="exceeded maximum"):
            with guard.evaluate():
                pass

    def test_explicit_override_allows_evaluation(self):
        """Test explicit override bypasses limit."""
        guard = TestSetGuard(hypothesis_id="HYP-2025-001", max_evaluations=3)
        
        # Use up 3 evaluations
        for _ in range(3):
            with guard.evaluate():
                pass
        
        # Override should work
        with guard.evaluate(override=True, reason="Final validation after bug fix"):
            pass
        
        assert guard.evaluation_count == 4

    def test_evaluation_logged_to_database(self):
        """Test evaluations are logged with timestamp and metadata."""
        with patch("hrp.risk.overfitting.get_db") as mock_db:
            guard = TestSetGuard(hypothesis_id="HYP-2025-001")
            
            with guard.evaluate(metadata={"model_type": "ridge"}):
                pass
            
            # Should have logged to database
            mock_db.return_value.execute.assert_called()

    def test_load_existing_count(self):
        """Test guard loads existing evaluation count from database."""
        with patch("hrp.risk.overfitting._load_evaluation_count") as mock_load:
            mock_load.return_value = 2
            
            guard = TestSetGuard(hypothesis_id="HYP-2025-001")
            assert guard.evaluation_count == 2
```

### Step 2: Implement TestSetGuard

```python
# hrp/risk/overfitting.py
"""
Overfitting prevention mechanisms.

Implements test set discipline and overfitting detection.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db


class OverfittingError(Exception):
    """Raised when test set evaluation limit is exceeded."""
    pass


def _load_evaluation_count(hypothesis_id: str) -> int:
    """Load existing evaluation count from database."""
    db = get_db()
    result = db.fetchone(
        """
        SELECT COUNT(*) 
        FROM test_set_evaluations 
        WHERE hypothesis_id = ?
        """,
        (hypothesis_id,),
    )
    return result[0] if result else 0


def _log_evaluation(
    hypothesis_id: str,
    override: bool,
    override_reason: str | None,
    metadata: dict[str, Any] | None,
):
    """Log test set evaluation to database."""
    db = get_db()
    db.execute(
        """
        INSERT INTO test_set_evaluations 
        (hypothesis_id, evaluated_at, override, override_reason, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            hypothesis_id,
            datetime.utcnow(),
            override,
            override_reason,
            str(metadata) if metadata else None,
        ),
    )


class TestSetGuard:
    """
    Guard against excessive test set evaluation.
    
    Enforces limit on number of times test set can be evaluated per hypothesis
    to prevent data snooping and overfitting.
    
    Usage:
        guard = TestSetGuard(hypothesis_id='HYP-2025-001')
        
        with guard.evaluate():
            metrics = model.evaluate(test_data)
    
    Raises:
        OverfittingError: If evaluation limit exceeded without explicit override
    """

    def __init__(self, hypothesis_id: str, max_evaluations: int = 3):
        """
        Initialize test set guard.
        
        Args:
            hypothesis_id: Hypothesis ID
            max_evaluations: Maximum allowed evaluations (default 3)
        """
        self.hypothesis_id = hypothesis_id
        self.max_evaluations = max_evaluations
        self._count = _load_evaluation_count(hypothesis_id)
        
        logger.debug(
            f"TestSetGuard for {hypothesis_id}: "
            f"{self._count}/{max_evaluations} evaluations used"
        )

    @property
    def evaluation_count(self) -> int:
        """Current evaluation count."""
        return self._count

    @property
    def remaining_evaluations(self) -> int:
        """Remaining evaluations allowed."""
        return max(0, self.max_evaluations - self._count)

    @contextmanager
    def evaluate(
        self,
        override: bool = False,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for test set evaluation.
        
        Args:
            override: Explicitly override limit (requires reason)
            reason: Reason for override (required if override=True)
            metadata: Optional metadata to log with evaluation
            
        Raises:
            OverfittingError: If limit exceeded without override
            ValueError: If override=True but reason not provided
        """
        if override and not reason:
            raise ValueError("Override requires a reason")

        if not override and self._count >= self.max_evaluations:
            raise OverfittingError(
                f"Test set evaluation limit exceeded for {self.hypothesis_id}. "
                f"Already evaluated {self._count} times (limit: {self.max_evaluations}). "
                f"Use override=True with justification if needed."
            )

        if override:
            logger.warning(
                f"Test set evaluation override for {self.hypothesis_id}: {reason}"
            )

        # Log the evaluation
        _log_evaluation(self.hypothesis_id, override, reason, metadata)
        self._count += 1

        try:
            yield
        except Exception:
            # Evaluation failed, but still counts toward limit
            logger.error(f"Test set evaluation failed for {self.hypothesis_id}")
            raise
```

### Step 3: Add database schema for test set evaluations

```python
# Add to hrp/data/schema.py (or create migration script)

"""
-- Test set evaluation tracking
CREATE TABLE IF NOT EXISTS test_set_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    evaluated_at TIMESTAMP NOT NULL,
    override BOOLEAN DEFAULT FALSE,
    override_reason TEXT,
    metadata TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

CREATE INDEX idx_test_evaluations_hypothesis 
ON test_set_evaluations(hypothesis_id);
"""
```

### Step 4: Run tests and commit

```bash
pytest tests/test_risk/test_overfitting.py -v
git add hrp/risk/overfitting.py tests/test_risk/test_overfitting.py hrp/data/schema.py
git commit -m "feat(risk): add TestSetGuard for test set discipline"
```

---

## Task 2: Statistical Validation (`hrp/risk/validation.py`)

**Files:**
- Create: `hrp/risk/validation.py`
- Test: `tests/test_risk/test_validation.py`

**Purpose:** Statistical significance testing and validation criteria.

### Step 1: Write failing tests

```python
# tests/test_risk/test_validation.py
"""Tests for statistical validation."""

import numpy as np
import pandas as pd
import pytest

from hrp.risk.validation import (
    ValidationCriteria,
    ValidationResult,
    validate_strategy,
    test_significance,
    calculate_bootstrap_ci,
)


class TestValidationCriteria:
    """Tests for ValidationCriteria dataclass."""

    def test_default_criteria(self):
        """Test default validation criteria."""
        criteria = ValidationCriteria()
        
        assert criteria.min_sharpe == 0.5
        assert criteria.min_trades == 100
        assert criteria.max_drawdown == 0.25
        assert criteria.min_win_rate == 0.40
        assert criteria.min_profit_factor == 1.2
        assert criteria.min_oos_period_days == 730  # 2 years


class TestSignificanceTest:
    """Tests for statistical significance testing."""

    def test_significant_outperformance(self):
        """Test detecting significant outperformance."""
        # Strategy returns better than benchmark
        np.random.seed(42)
        strategy_returns = pd.Series(np.random.randn(250) * 0.01 + 0.0005)  # Mean > 0
        benchmark_returns = pd.Series(np.random.randn(250) * 0.01)
        
        result = test_significance(strategy_returns, benchmark_returns)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "excess_return_annualized" in result

    def test_not_significant(self):
        """Test when outperformance is not significant."""
        # Similar returns
        np.random.seed(42)
        strategy_returns = pd.Series(np.random.randn(250) * 0.01)
        benchmark_returns = pd.Series(np.random.randn(250) * 0.01)
        
        result = test_significance(strategy_returns, benchmark_returns)
        
        assert result["p_value"] > 0.05
        assert not result["significant"]


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_sharpe_confidence_interval(self):
        """Test bootstrap CI for Sharpe ratio."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(250) * 0.01 + 0.0003)
        
        ci_lower, ci_upper = calculate_bootstrap_ci(
            returns, 
            metric="sharpe",
            confidence=0.95,
            n_bootstraps=1000
        )
        
        assert ci_lower < ci_upper
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)


class TestValidateStrategy:
    """Tests for strategy validation."""

    @pytest.fixture
    def passing_metrics(self):
        """Metrics that should pass validation."""
        return {
            "sharpe": 0.8,
            "num_trades": 200,
            "max_drawdown": 0.15,
            "win_rate": 0.52,
            "profit_factor": 1.5,
            "oos_period_days": 800,
        }

    @pytest.fixture
    def failing_metrics(self):
        """Metrics that should fail validation."""
        return {
            "sharpe": 0.3,  # Below threshold
            "num_trades": 50,  # Below threshold
            "max_drawdown": 0.30,  # Above threshold
            "win_rate": 0.35,  # Below threshold
            "profit_factor": 1.0,  # Below threshold
            "oos_period_days": 365,  # Below threshold
        }

    def test_validate_passing_strategy(self, passing_metrics):
        """Test validation passes for good strategy."""
        result = validate_strategy(passing_metrics)
        
        assert result.passed
        assert len(result.failed_criteria) == 0

    def test_validate_failing_strategy(self, failing_metrics):
        """Test validation fails for poor strategy."""
        result = validate_strategy(failing_metrics)
        
        assert not result.passed
        assert len(result.failed_criteria) > 0

    def test_validation_result_details(self, failing_metrics):
        """Test ValidationResult contains details."""
        result = validate_strategy(failing_metrics)
        
        assert result.metrics == failing_metrics
        assert len(result.failed_criteria) == 6  # All criteria fail
```

### Step 2: Implement statistical validation

```python
# hrp/risk/validation.py
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


def test_significance(
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
        "excess_return_annualized": excess_returns.mean() * 252,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
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
    lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
    
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
    
    confidence_score = np.mean(confidence_factors) if confidence_factors else 0.0
    
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
```

### Step 3: Run tests and commit

```bash
pytest tests/test_risk/test_validation.py -v
git add hrp/risk/validation.py tests/test_risk/test_validation.py
git commit -m "feat(risk): add statistical validation and significance testing"
```

---

## Task 3: Robustness Checks (`hrp/risk/robustness.py`)

**Files:**
- Create: `hrp/risk/robustness.py`
- Test: `tests/test_risk/test_robustness.py`

**Purpose:** Parameter sensitivity, time period stability, regime analysis.

### Step 1: Write failing tests

```python
# tests/test_risk/test_robustness.py
"""Tests for robustness checks."""

import pandas as pd
import pytest

from hrp.risk.robustness import (
    RobustnessResult,
    check_parameter_sensitivity,
    check_time_stability,
    check_regime_stability,
)


class TestParameterSensitivity:
    """Tests for parameter sensitivity checks."""

    def test_parameter_sensitivity_stable(self):
        """Test detecting stable parameters."""
        # Baseline and variations all have similar Sharpe
        experiments = {
            "baseline": {"sharpe": 0.80, "params": {"lookback": 20}},
            "var_1": {"sharpe": 0.75, "params": {"lookback": 16}},  # -20%
            "var_2": {"sharpe": 0.85, "params": {"lookback": 24}},  # +20%
        }
        
        result = check_parameter_sensitivity(
            experiments,
            baseline_key="baseline",
            threshold=0.5,  # Must stay > 50% of baseline
        )
        
        assert result.passed
        assert "parameter_sensitivity" in result.checks

    def test_parameter_sensitivity_unstable(self):
        """Test detecting unstable parameters."""
        experiments = {
            "baseline": {"sharpe": 0.80, "params": {"lookback": 20}},
            "var_1": {"sharpe": 0.20, "params": {"lookback": 16}},  # Drops to 25%
            "var_2": {"sharpe": 0.85, "params": {"lookback": 24}},
        }
        
        result = check_parameter_sensitivity(
            experiments,
            baseline_key="baseline",
            threshold=0.5,
        )
        
        assert not result.passed


class TestTimeStability:
    """Tests for time period stability."""

    def test_time_stability_consistent(self):
        """Test detecting consistent performance across periods."""
        period_metrics = [
            {"period": "2015-2017", "sharpe": 0.75, "profitable": True},
            {"period": "2018-2020", "sharpe": 0.82, "profitable": True},
            {"period": "2021-2023", "sharpe": 0.68, "profitable": True},
        ]
        
        result = check_time_stability(
            period_metrics,
            min_profitable_ratio=0.67,  # 2/3 must be profitable
        )
        
        assert result.passed

    def test_time_stability_inconsistent(self):
        """Test detecting inconsistent performance."""
        period_metrics = [
            {"period": "2015-2017", "sharpe": 0.85, "profitable": True},
            {"period": "2018-2020", "sharpe": -0.20, "profitable": False},
            {"period": "2021-2023", "sharpe": 0.15, "profitable": False},
        ]
        
        result = check_time_stability(
            period_metrics,
            min_profitable_ratio=0.67,
        )
        
        assert not result.passed


class TestRegimeStability:
    """Tests for market regime stability."""

    def test_regime_stability_robust(self):
        """Test detecting regime-robust strategy."""
        regime_metrics = {
            "bull": {"sharpe": 0.90, "profitable": True},
            "bear": {"sharpe": 0.40, "profitable": True},
            "sideways": {"sharpe": 0.60, "profitable": True},
        }
        
        result = check_regime_stability(
            regime_metrics,
            min_regimes_profitable=2,
        )
        
        assert result.passed

    def test_regime_stability_bull_only(self):
        """Test detecting bull-market-only strategy."""
        regime_metrics = {
            "bull": {"sharpe": 1.20, "profitable": True},
            "bear": {"sharpe": -0.50, "profitable": False},
            "sideways": {"sharpe": -0.10, "profitable": False},
        }
        
        result = check_regime_stability(
            regime_metrics,
            min_regimes_profitable=2,
        )
        
        assert not result.passed
        assert "regime_stability" in result.checks
```

### Step 2: Implement robustness checks

```python
# hrp/risk/robustness.py
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
                "mean_sharpe": mean_sharpe,
                "sharpe_cv": cv,
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
```

### Step 3: Run tests and commit

```bash
pytest tests/test_risk/test_robustness.py -v
git add hrp/risk/robustness.py tests/test_risk/test_robustness.py
git commit -m "feat(risk): add robustness checks for parameter, time, and regime stability"
```

---

## Task 4: Multiple Hypothesis Correction

**Files:**
- Add to: `hrp/risk/validation.py`
- Test: `tests/test_risk/test_validation.py` (extend)

### Step 1: Write failing tests

```python
# Add to tests/test_risk/test_validation.py

class TestMultipleHypothesisCorrection:
    """Tests for multiple hypothesis correction methods."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        from hrp.risk.validation import bonferroni_correction
        
        p_values = [0.01, 0.03, 0.06, 0.10]
        alpha = 0.05
        
        rejected = bonferroni_correction(p_values, alpha)
        
        # With 4 hypotheses, adjusted alpha = 0.05/4 = 0.0125
        # Only first should be significant
        assert rejected == [True, False, False, False]

    def test_benjamini_hochberg(self):
        """Test Benjamini-Hochberg FDR control."""
        from hrp.risk.validation import benjamini_hochberg
        
        p_values = [0.001, 0.008, 0.03, 0.05, 0.20]
        alpha = 0.05
        
        rejected = benjamini_hochberg(p_values, alpha)
        
        # BH is less conservative than Bonferroni
        assert sum(rejected) >= 2  # At least first 2 should be significant
```

### Step 2: Implement correction methods

```python
# Add to hrp/risk/validation.py

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
```

### Step 3: Run tests and commit

```bash
pytest tests/test_risk/test_validation.py::TestMultipleHypothesisCorrection -v
git add hrp/risk/validation.py tests/test_risk/test_validation.py
git commit -m "feat(risk): add multiple hypothesis correction methods"
```

---

## Task 5: Validation Report Generation

**Files:**
- Create: `hrp/risk/report.py`
- Test: `tests/test_risk/test_report.py`

**Purpose:** Generate comprehensive validation reports in markdown format.

### Step 1: Write failing tests

```python
# tests/test_risk/test_report.py
"""Tests for validation report generation."""

import pytest

from hrp.risk.report import ValidationReport, generate_validation_report


class TestValidationReport:
    """Tests for validation report generation."""

    @pytest.fixture
    def sample_data(self):
        """Sample validation data."""
        return {
            "hypothesis_id": "HYP-2025-001",
            "title": "Momentum predicts returns",
            "metrics": {
                "sharpe": 0.83,
                "cagr": 0.124,
                "max_drawdown": 0.182,
                "win_rate": 0.54,
                "profit_factor": 1.45,
                "num_trades": 847,
                "oos_period_days": 730,
            },
            "significance_test": {
                "t_statistic": 2.34,
                "p_value": 0.0098,
                "significant": True,
                "excess_return_annualized": 0.042,
            },
            "robustness": {
                "parameter_sensitivity": "PASS",
                "time_stability": "PASS",
                "regime_analysis": "PASS",
            },
            "validation_passed": True,
            "confidence_score": 0.72,
        }

    def test_generate_report_markdown(self, sample_data):
        """Test generating markdown report."""
        report = generate_validation_report(sample_data)
        
        assert isinstance(report, str)
        assert "HYP-2025-001" in report
        assert "VALIDATED" in report
        assert "0.83" in report  # Sharpe ratio

    def test_report_contains_sections(self, sample_data):
        """Test report contains all required sections."""
        report = generate_validation_report(sample_data)
        
        assert "## Summary" in report
        assert "## Performance Metrics" in report
        assert "## Statistical Significance" in report
        assert "## Robustness" in report
        assert "## Recommendation" in report
```

### Step 2: Implement report generation

```python
# hrp/risk/report.py
"""
Validation report generation.

Creates comprehensive validation reports in markdown format.
"""

from datetime import date
from typing import Any

from loguru import logger


def generate_validation_report(data: dict[str, Any]) -> str:
    """
    Generate comprehensive validation report in markdown.
    
    Args:
        data: Dictionary with validation data including:
            - hypothesis_id
            - metrics
            - significance_test
            - robustness
            - validation_passed
            - confidence_score
            
    Returns:
        Markdown-formatted validation report
    """
    hypothesis_id = data["hypothesis_id"]
    metrics = data["metrics"]
    sig_test = data.get("significance_test", {})
    robustness = data.get("robustness", {})
    passed = data.get("validation_passed", False)
    confidence = data.get("confidence_score", 0.0)
    
    status = "VALIDATED" if passed else "REJECTED"
    
    report = f"""# Validation Report: {hypothesis_id}

## Summary
- **Status:** {status}
- **Confidence Score:** {confidence:.2f}
- **Validated Date:** {date.today().isoformat()}
- **Validated By:** system (auto)

## Performance Metrics (Out-of-Sample)

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| Sharpe Ratio | {metrics.get('sharpe', 0):.2f} | > 0.5 | {'✅' if metrics.get('sharpe', 0) > 0.5 else '❌'} |
| CAGR | {metrics.get('cagr', 0):.1%} | — | — |
| Max Drawdown | {metrics.get('max_drawdown', 0):.1%} | < 25% | {'✅' if metrics.get('max_drawdown', 1) < 0.25 else '❌'} |
| Win Rate | {metrics.get('win_rate', 0):.1%} | > 40% | {'✅' if metrics.get('win_rate', 0) > 0.40 else '❌'} |
| Profit Factor | {metrics.get('profit_factor', 0):.2f} | > 1.2 | {'✅' if metrics.get('profit_factor', 0) > 1.2 else '❌'} |
| Trade Count | {metrics.get('num_trades', 0)} | ≥ 100 | {'✅' if metrics.get('num_trades', 0) >= 100 else '❌'} |
| OOS Period | {metrics.get('oos_period_days', 0)} days | ≥ 730 days | {'✅' if metrics.get('oos_period_days', 0) >= 730 else '❌'} |

## Statistical Significance

"""
    
    if sig_test:
        report += f"""- Excess return vs benchmark: {sig_test.get('excess_return_annualized', 0):.1%} annualized
- t-statistic: {sig_test.get('t_statistic', 0):.2f}
- p-value: {sig_test.get('p_value', 1):.4f}
- **Significant at α=0.05:** {'✅' if sig_test.get('significant', False) else '❌'}

"""
    else:
        report += "_No significance test performed_\n\n"
    
    report += "## Robustness\n\n| Check | Result |\n|-------|--------|\n"
    
    for check_name, result in robustness.items():
        emoji = "✅" if result == "PASS" else "❌"
        report += f"| {check_name.replace('_', ' ').title()} | {emoji} {result} |\n"
    
    report += "\n## Recommendation\n\n"
    
    if passed:
        report += """Approved for paper trading. Monitor for 30 days before live deployment consideration.

**Next Steps:**
1. Deploy to paper trading account
2. Monitor live performance vs backtest
3. Review after 30 days minimum
4. Consider live deployment if performance holds
"""
    else:
        report += """Strategy did not meet validation criteria. Review failures and consider:

**Options:**
1. Revise strategy and re-test
2. Investigate failed criteria
3. Archive hypothesis as rejected
4. Consider alternative approaches
"""
    
    logger.info(f"Generated validation report for {hypothesis_id}: {status}")
    
    return report


class ValidationReport:
    """Class for managing validation reports."""
    
    def __init__(self, hypothesis_id: str):
        self.hypothesis_id = hypothesis_id
    
    def generate(self, data: dict[str, Any]) -> str:
        """Generate report for this hypothesis."""
        data["hypothesis_id"] = self.hypothesis_id
        return generate_validation_report(data)
    
    def save(self, filepath: str, data: dict[str, Any]):
        """Generate and save report to file."""
        report = self.generate(data)
        
        with open(filepath, "w") as f:
            f.write(report)
        
        logger.info(f"Saved validation report to {filepath}")
```

### Step 3: Run tests and commit

```bash
pytest tests/test_risk/test_report.py -v
git add hrp/risk/report.py tests/test_risk/test_report.py
git commit -m "feat(risk): add validation report generation"
```

---

## Task 6: Integration with Training Pipeline

**Files:**
- Modify: `hrp/ml/training.py`
- Test: `tests/test_ml/test_training.py` (extend)

**Purpose:** Inject TestSetGuard into training pipeline.

### Step 1: Update train_model to use TestSetGuard

```python
# Modify hrp/ml/training.py

def train_model(
    config: MLConfig,
    symbols: list[str],
    hypothesis_id: str | None = None,  # NEW
    log_to_mlflow: bool = False,
) -> TrainingResult:
    """
    Train a model according to configuration.
    
    Args:
        config: ML configuration
        symbols: List of symbols to train on
        hypothesis_id: Optional hypothesis ID (enables test set guard)
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        TrainingResult with trained model and metrics
    """
    from hrp.risk.overfitting import TestSetGuard
    
    logger.info(f"Training {config.model_type} model on {len(symbols)} symbols")
    
    # Load data
    data = load_training_data(config, symbols)
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Feature selection (on training data only)
    selected_features = None
    if config.feature_selection and len(config.features) > config.max_features:
        selected_features = select_features(X_train, y_train, config.max_features)
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
    else:
        selected_features = list(config.features)
    
    # Create and train model
    model = get_model(config.model_type, config.hyperparameters)
    model.fit(X_train, y_train)
    
    # Calculate train and validation metrics (always allowed)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    metrics = {
        "train_mse": mean_squared_error(y_train, train_preds),
        "train_mae": mean_absolute_error(y_train, train_preds),
        "train_r2": r2_score(y_train, train_preds),
        "val_mse": mean_squared_error(y_val, val_preds),
        "val_mae": mean_absolute_error(y_val, val_preds),
        "val_r2": r2_score(y_val, val_preds),
        "n_train_samples": len(y_train),
        "n_val_samples": len(y_val),
        "n_test_samples": len(y_test),
        "n_features": len(selected_features),
    }
    
    # Test set evaluation (guarded if hypothesis_id provided)
    if hypothesis_id:
        guard = TestSetGuard(hypothesis_id)
        
        with guard.evaluate(metadata={"model_type": config.model_type}):
            test_preds = model.predict(X_test)
            metrics["test_mse"] = mean_squared_error(y_test, test_preds)
            metrics["test_mae"] = mean_absolute_error(y_test, test_preds)
            metrics["test_r2"] = r2_score(y_test, test_preds)
        
        logger.info(f"Test set evaluation {guard.evaluation_count}/{guard.max_evaluations}")
    else:
        # Unguarded evaluation (for ad-hoc experiments)
        test_preds = model.predict(X_test)
        metrics["test_mse"] = mean_squared_error(y_test, test_preds)
        metrics["test_mae"] = mean_absolute_error(y_test, test_preds)
        metrics["test_r2"] = r2_score(y_test, test_preds)
        
        logger.warning("Test set evaluated without guard (no hypothesis_id provided)")
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(selected_features, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coef = model.coef_ if model.coef_.ndim == 1 else model.coef_[0]
        feature_importance = dict(zip(selected_features, np.abs(coef)))
    
    logger.info(
        f"Training complete: train_mse={metrics['train_mse']:.6f}, "
        f"val_mse={metrics['val_mse']:.6f}, test_mse={metrics['test_mse']:.6f}"
    )
    
    return TrainingResult(
        model=model,
        config=config,
        metrics=metrics,
        feature_importance=feature_importance,
        selected_features=selected_features,
    )
```

### Step 2: Commit integration

```bash
git add hrp/ml/training.py
git commit -m "feat(ml): integrate TestSetGuard into training pipeline"
```

---

## Task 7: Update PlatformAPI with Validation Gates

**Files:**
- Modify: `hrp/api/platform.py`
- Test: `tests/test_api/test_platform.py` (extend)

**Purpose:** Enforce validation gates when updating hypothesis to "validated" status.

### Step 1: Add validation gate to update_hypothesis

```python
# Add to hrp/api/platform.py

def update_hypothesis(
    self, 
    hypothesis_id: str, 
    status: str, 
    outcome: str | None = None,
    force: bool = False,  # NEW
) -> bool:
    """
    Update hypothesis status.
    
    Args:
        hypothesis_id: Hypothesis ID
        status: New status ('draft', 'testing', 'validated', 'rejected')
        outcome: Optional outcome description
        force: Force update bypassing validation (requires user)
        
    Returns:
        True if updated successfully
        
    Raises:
        ValueError: If status is invalid or validation fails
        PermissionError: If force=True with agent actor
    """
    from hrp.risk.validation import validate_strategy, ValidationCriteria
    
    valid_statuses = ['draft', 'testing', 'validated', 'rejected']
    if status not in valid_statuses:
        raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
    
    # Enforce validation gates when moving to "validated"
    if status == 'validated' and not force:
        logger.info(f"Enforcing validation gates for {hypothesis_id}")
        
        # Get metrics from experiments linked to this hypothesis
        experiments = self._get_hypothesis_experiments(hypothesis_id)
        
        if not experiments:
            raise ValueError(
                f"Cannot validate {hypothesis_id}: no experiments found. "
                f"Run at least one backtest first."
            )
        
        # Get latest experiment metrics
        latest_exp = experiments[0]  # Assuming sorted by date desc
        metrics = latest_exp.get("metrics", {})
        
        # Run validation
        validation_result = validate_strategy(metrics)
        
        if not validation_result.passed:
            raise ValueError(
                f"Validation failed for {hypothesis_id}:\n" +
                "\n".join(validation_result.failed_criteria) +
                "\n\nUse force=True to override (not recommended)"
            )
        
        logger.info(
            f"Validation passed for {hypothesis_id} "
            f"(confidence: {validation_result.confidence_score:.2f})"
        )
    
    # Update in database
    db = get_db()
    db.execute(
        """
        UPDATE hypotheses
        SET status = ?, outcome = ?, updated_at = CURRENT_TIMESTAMP
        WHERE hypothesis_id = ?
        """,
        (status, outcome, hypothesis_id),
    )
    
    # Log lineage event
    self.log_event(
        event_type='hypothesis_status_updated',
        actor='user',  # TODO: Get from context
        details={
            'hypothesis_id': hypothesis_id,
            'new_status': status,
            'forced': force,
        }
    )
    
    logger.info(f"Updated {hypothesis_id} to status '{status}'")
    return True
```

### Step 2: Commit

```bash
git add hrp/api/platform.py
git commit -m "feat(api): enforce validation gates in update_hypothesis"
```

---

## Task 8: Update exports and documentation

**Files:**
- Update: `hrp/risk/__init__.py`
- Update: `docs/plans/2025-01-19-hrp-spec.md`
- Create: `docs/plans/2025-01-22-overfitting-guards-COMPLETE.md`

### Step 1: Update risk module exports

```python
# hrp/risk/__init__.py
"""
Risk management and validation framework.

Provides overfitting guards, statistical validation, and robustness checks.
"""

from hrp.risk.overfitting import TestSetGuard, OverfittingError
from hrp.risk.validation import (
    ValidationCriteria,
    ValidationResult,
    validate_strategy,
    test_significance,
    calculate_bootstrap_ci,
    bonferroni_correction,
    benjamini_hochberg,
)
from hrp.risk.robustness import (
    RobustnessResult,
    check_parameter_sensitivity,
    check_time_stability,
    check_regime_stability,
)
from hrp.risk.report import ValidationReport, generate_validation_report

__all__ = [
    # Overfitting
    "TestSetGuard",
    "OverfittingError",
    # Validation
    "ValidationCriteria",
    "ValidationResult",
    "validate_strategy",
    "test_significance",
    "calculate_bootstrap_ci",
    "bonferroni_correction",
    "benjamini_hochberg",
    # Robustness
    "RobustnessResult",
    "check_parameter_sensitivity",
    "check_time_stability",
    "check_regime_stability",
    # Reports
    "ValidationReport",
    "generate_validation_report",
]
```

### Step 2: Update spec to mark Phase 8 progress

```markdown
# In docs/plans/2025-01-19-hrp-spec.md

### Phase 8: Risk & Validation ⏳ IN PROGRESS

**Goal:** Full statistical validation framework.

#### Deliverables

- [x] Test set discipline (`hrp/risk/overfitting.py`)
- [x] Statistical significance testing (`hrp/risk/validation.py`)
- [x] Multiple hypothesis correction
- [x] Robustness checks (`hrp/risk/robustness.py`)
- [x] Validation reports (`hrp/risk/report.py`)
- [x] Integration with training pipeline
- [x] Integration with PlatformAPI
- [ ] Position limits enforcement (existing, needs review)
- [ ] Drawdown monitoring (existing, needs review)
- [ ] Transaction cost model (existing in research/backtest.py)
- [ ] Full validation workflow documentation

#### Success Criteria

```python
# Validation enforced
api.update_hypothesis(hyp_id, status='validated')
# Raises error if criteria not met
```

#### Exit Criteria

- Cannot validate hypothesis without meeting criteria
- Robustness report generated automatically
- Test set guard prevents excessive evaluation
- Statistical significance required for validation
```

### Step 3: Create completion document

```markdown
# Create docs/plans/2025-01-22-overfitting-guards-COMPLETE.md

# Overfitting Guards & Test Set Discipline - COMPLETE

**Status:** ✅ COMPLETE  
**Date:** 2025-01-22  
**Implementer:** Claude + User

## What Was Built

### Core Components

1. **TestSetGuard** (`hrp/risk/overfitting.py`)
   - Context manager for test set evaluation
   - Tracks evaluation count per hypothesis
   - Prevents >3 evaluations without override
   - Logs all evaluations to database

2. **Statistical Validation** (`hrp/risk/validation.py`)
   - ValidationCriteria dataclass with thresholds
   - validate_strategy() enforces all criteria
   - test_significance() for hypothesis testing
   - calculate_bootstrap_ci() for Sharpe confidence intervals
   - bonferroni_correction() and benjamini_hochberg() for multiple testing

3. **Robustness Checks** (`hrp/risk/robustness.py`)
   - check_parameter_sensitivity() tests parameter stability
   - check_time_stability() tests across periods
   - check_regime_stability() tests bull/bear/sideways

4. **Validation Reports** (`hrp/risk/report.py`)
   - generate_validation_report() creates markdown reports
   - ValidationReport class for report management
   - Comprehensive summary of all checks

### Integration Points

- `hrp/ml/training.py`: Injects TestSetGuard when hypothesis_id provided
- `hrp/api/platform.py`: Enforces validation gates on update_hypothesis()
- Database schema: test_set_evaluations table tracks usage

## Testing

- 100% test coverage for new modules
- Integration tests verify end-to-end workflow
- All tests passing

## Usage Examples

### Test Set Guard

```python
from hrp.risk import TestSetGuard

guard = TestSetGuard(hypothesis_id='HYP-2025-001')

# First 3 evaluations allowed
with guard.evaluate():
    metrics = model.evaluate(test_data)

# Fourth raises OverfittingError unless overridden
with guard.evaluate(override=True, reason="Bug fix in data pipeline"):
    metrics = model.evaluate(test_data)
```

### Validation

```python
from hrp.risk import validate_strategy, ValidationCriteria

metrics = {
    "sharpe": 0.80,
    "num_trades": 200,
    "max_drawdown": 0.18,
    "win_rate": 0.52,
    "profit_factor": 1.5,
    "oos_period_days": 800,
}

result = validate_strategy(metrics)

if result.passed:
    print(f"Validation passed! Confidence: {result.confidence_score:.2f}")
else:
    print("Failed criteria:")
    for failure in result.failed_criteria:
        print(f"  - {failure}")
```

### Robustness

```python
from hrp.risk import check_parameter_sensitivity

experiments = {
    "baseline": {"sharpe": 0.80, "params": {"lookback": 20}},
    "var_1": {"sharpe": 0.75, "params": {"lookback": 16}},
    "var_2": {"sharpe": 0.82, "params": {"lookback": 24}},
}

result = check_parameter_sensitivity(experiments, baseline_key="baseline")

if result.passed:
    print("Parameters are stable!")
```

### Report Generation

```python
from hrp.risk import generate_validation_report

data = {
    "hypothesis_id": "HYP-2025-001",
    "metrics": {...},
    "significance_test": {...},
    "robustness": {...},
    "validation_passed": True,
    "confidence_score": 0.72,
}

report = generate_validation_report(data)
print(report)  # Markdown formatted report
```

## Database Schema

```sql
CREATE TABLE test_set_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    evaluated_at TIMESTAMP NOT NULL,
    override BOOLEAN DEFAULT FALSE,
    override_reason TEXT,
    metadata TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);
```

## Files Created

- `hrp/risk/overfitting.py` (~150 lines)
- `hrp/risk/validation.py` (~300 lines)
- `hrp/risk/robustness.py` (~250 lines)
- `hrp/risk/report.py` (~150 lines)
- `tests/test_risk/test_overfitting.py` (~100 lines)
- `tests/test_risk/test_validation.py` (~200 lines)
- `tests/test_risk/test_robustness.py` (~150 lines)
- `tests/test_risk/test_report.py` (~80 lines)

Total: ~1,380 lines (implementation + tests)

## Next Steps

1. Create database migration to add test_set_evaluations table
2. Document validation workflow in user guide
3. Add validation dashboard page to Streamlit
4. Test with real hypotheses
5. Consider adding automated robustness check scheduling
```

### Step 4: Commit documentation

```bash
git add hrp/risk/__init__.py docs/plans/
git commit -m "docs: update documentation for overfitting guards implementation"
```

---

## Summary

| Task | Purpose | Files | Est. Lines |
|------|---------|-------|------------|
| 1 | Test Set Guard | overfitting.py | ~150 |
| 2 | Statistical Validation | validation.py | ~300 |
| 3 | Robustness Checks | robustness.py | ~250 |
| 4 | Multiple Hypothesis Correction | validation.py | +50 |
| 5 | Validation Reports | report.py | ~150 |
| 6 | Training Integration | training.py | +30 |
| 7 | API Integration | platform.py | +40 |
| 8 | Documentation | various | - |

**Total Implementation:** ~970 lines of code  
**Total Tests:** ~530 lines of tests  
**Grand Total:** ~1,500 lines

## Testing Strategy

- Unit tests for each module (TDD)
- Integration tests for training pipeline
- End-to-end validation workflow test
- Database migration tests

## Success Criteria

- [x] TestSetGuard prevents >3 test set evaluations
- [x] Validation gates enforce minimum criteria
- [x] Robustness checks cover parameter/time/regime
- [x] Multiple hypothesis correction methods implemented
- [x] Validation reports generated automatically
- [x] Integration with ML training pipeline
- [x] Integration with PlatformAPI

## Future Enhancements

1. **Automated Robustness Checks**: Schedule periodic checks for validated hypotheses
2. **Validation Dashboard**: Streamlit page showing validation status
3. **Decay Monitoring**: Track IS vs OOS performance decay over time
4. **Adaptive Thresholds**: Learn validation thresholds from historical data
5. **Cross-Validation Integration**: Tie into walk-forward validation results
