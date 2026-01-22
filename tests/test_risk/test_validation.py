"""Tests for statistical validation."""

import numpy as np
import pandas as pd
import pytest

from hrp.risk.validation import (
    ValidationCriteria,
    ValidationResult,
    validate_strategy,
    significance_test,
    calculate_bootstrap_ci,
    bonferroni_correction,
    benjamini_hochberg,
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
        
        result = significance_test(strategy_returns, benchmark_returns)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "excess_return_annualized" in result

    def test_not_significant(self):
        """Test when outperformance is not significant."""
        # Similar returns
        np.random.seed(42)
        strategy_returns = pd.Series(np.random.randn(250) * 0.01)
        benchmark_returns = pd.Series(np.random.randn(250) * 0.01 + 0.00001)
        
        result = significance_test(strategy_returns, benchmark_returns)
        
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

    def test_mean_confidence_interval(self):
        """Test bootstrap CI for mean return."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(250) * 0.01 + 0.0003)
        
        ci_lower, ci_upper = calculate_bootstrap_ci(
            returns, 
            metric="mean",
            confidence=0.95,
            n_bootstraps=1000
        )
        
        assert ci_lower < ci_upper

    def test_invalid_metric_raises_error(self):
        """Test invalid metric raises ValueError."""
        returns = pd.Series(np.random.randn(250) * 0.01)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_bootstrap_ci(returns, metric="invalid")


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

    def test_custom_criteria(self):
        """Test validation with custom criteria."""
        metrics = {
            "sharpe": 0.6,
            "num_trades": 150,
            "max_drawdown": 0.20,
            "win_rate": 0.45,
            "profit_factor": 1.3,
            "oos_period_days": 800,
        }
        
        criteria = ValidationCriteria(
            min_sharpe=0.7,  # Higher threshold
            min_trades=100,
        )
        
        result = validate_strategy(metrics, criteria)
        
        # Should fail on Sharpe
        assert not result.passed
        assert any("Sharpe" in f for f in result.failed_criteria)

    def test_confidence_score_calculated(self, passing_metrics):
        """Test confidence score is calculated."""
        result = validate_strategy(passing_metrics)
        
        assert result.confidence_score is not None
        assert 0 <= result.confidence_score <= 1.0


class TestMultipleHypothesisCorrection:
    """Tests for multiple hypothesis correction methods."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.03, 0.06, 0.10]
        alpha = 0.05
        
        rejected = bonferroni_correction(p_values, alpha)
        
        # With 4 hypotheses, adjusted alpha = 0.05/4 = 0.0125
        # Only first should be significant
        assert rejected == [True, False, False, False]

    def test_bonferroni_all_significant(self):
        """Test Bonferroni with all p-values significant."""
        p_values = [0.001, 0.002, 0.003]
        alpha = 0.05
        
        rejected = bonferroni_correction(p_values, alpha)
        
        # Adjusted alpha = 0.05/3 = 0.0167, all pass
        assert all(rejected)

    def test_benjamini_hochberg(self):
        """Test Benjamini-Hochberg FDR control."""
        p_values = [0.001, 0.008, 0.03, 0.05, 0.20]
        alpha = 0.05
        
        rejected = benjamini_hochberg(p_values, alpha)
        
        # BH is less conservative than Bonferroni
        assert sum(rejected) >= 2  # At least first 2 should be significant

    def test_benjamini_hochberg_none_significant(self):
        """Test BH when no values are significant."""
        p_values = [0.5, 0.6, 0.7, 0.8]
        alpha = 0.05
        
        rejected = benjamini_hochberg(p_values, alpha)
        
        assert not any(rejected)

    def test_benjamini_hochberg_all_significant(self):
        """Test BH when all values are significant."""
        p_values = [0.001, 0.002, 0.003]
        alpha = 0.05
        
        rejected = benjamini_hochberg(p_values, alpha)
        
        assert all(rejected)
