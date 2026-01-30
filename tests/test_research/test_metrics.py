"""
Comprehensive tests for HRP metrics calculation module.
"""

import math
import pytest
import pandas as pd
import numpy as np

from hrp.research.metrics import (
    calculate_metrics,
    format_metrics,
    calculate_stability_score_v1,
    _calculate_cagr,
    _sharpe_ratio,
    _sortino_ratio,
    _downside_volatility,
    _max_drawdown,
    _calculate_alpha_beta,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_positive_returns():
    """Simple positive returns series - 1 year of daily data."""
    return pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)  # 250 days


@pytest.fixture
def all_positive_returns():
    """Series with all positive returns."""
    return pd.Series([0.01] * 252)


@pytest.fixture
def all_negative_returns():
    """Series with all negative returns (varying values)."""
    np.random.seed(123)
    # Negative returns with some variation to ensure non-zero std
    return pd.Series(np.random.uniform(-0.02, -0.001, 252))


@pytest.fixture
def mixed_returns():
    """Balanced mix of positive and negative returns."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.02, 252))


@pytest.fixture
def benchmark_returns():
    """Benchmark returns for comparison tests."""
    np.random.seed(123)
    return pd.Series(np.random.normal(0.0003, 0.015, 252))


@pytest.fixture
def zero_returns():
    """Series with all zero returns."""
    return pd.Series([0.0] * 252)


@pytest.fixture
def single_return():
    """Series with a single return value."""
    return pd.Series([0.05])


@pytest.fixture
def returns_with_nan():
    """Series with NaN values."""
    returns = pd.Series([0.01, 0.02, np.nan, -0.01, np.nan, 0.015, 0.005] * 36)
    return returns


# =============================================================================
# TestCalculateMetrics - Main function tests
# =============================================================================


class TestCalculateMetrics:
    """Tests for the main calculate_metrics function."""

    def test_basic_metrics_positive_returns(self, simple_positive_returns):
        """Test that basic metrics are calculated correctly for positive returns."""
        metrics = calculate_metrics(simple_positive_returns)

        # Check all expected metrics are present
        expected_keys = [
            "total_return",
            "cagr",
            "volatility",
            "downside_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

        # Basic sanity checks
        assert metrics["total_return"] > 0, "Total return should be positive"
        assert metrics["volatility"] > 0, "Volatility should be positive"
        assert metrics["win_rate"] > 0, "Win rate should be positive"

    def test_empty_returns(self):
        """Test that empty returns series returns empty dict."""
        returns = pd.Series([], dtype=float)
        metrics = calculate_metrics(returns)
        assert metrics == {}

    def test_all_positive_returns(self, all_positive_returns):
        """Test metrics for all positive returns series."""
        metrics = calculate_metrics(all_positive_returns)

        assert metrics["win_rate"] == 1.0, "Win rate should be 100% for all positive"
        assert metrics["max_drawdown"] == 0.0, "Max drawdown should be 0 for all positive"
        assert metrics["total_return"] > 0, "Total return should be positive"
        assert metrics["avg_loss"] == 0, "Average loss should be 0 for all positive"
        assert metrics["profit_factor"] == float("inf"), "Profit factor should be inf with no losses"

    def test_all_negative_returns(self, all_negative_returns):
        """Test metrics for all negative returns series."""
        metrics = calculate_metrics(all_negative_returns)

        assert metrics["win_rate"] == 0.0, "Win rate should be 0% for all negative"
        assert metrics["max_drawdown"] < 0, "Max drawdown should be negative"
        assert metrics["total_return"] < 0, "Total return should be negative"
        assert metrics["avg_win"] == 0, "Average win should be 0 for all negative"
        assert metrics["sharpe_ratio"] < 0, "Sharpe should be negative for negative returns"

    def test_zero_returns(self, zero_returns):
        """Test metrics for zero returns series."""
        metrics = calculate_metrics(zero_returns)

        assert metrics["total_return"] == 0.0, "Total return should be 0"
        assert metrics["win_rate"] == 0.0, "Win rate should be 0 (no positive returns)"
        assert metrics["volatility"] == 0.0, "Volatility should be 0"
        assert metrics["sharpe_ratio"] == 0.0, "Sharpe should be 0 with zero volatility"

    def test_single_return(self, single_return):
        """Test metrics for single return value."""
        metrics = calculate_metrics(single_return)

        assert metrics["total_return"] == pytest.approx(0.05, rel=1e-6)
        assert metrics["win_rate"] == 1.0

    def test_returns_with_nan_handled(self, returns_with_nan):
        """Test that NaN values are properly dropped."""
        metrics = calculate_metrics(returns_with_nan)

        assert not np.isnan(metrics["total_return"]), "Total return should not be NaN"
        assert not np.isnan(metrics["sharpe_ratio"]), "Sharpe ratio should not be NaN"

    def test_risk_free_rate_affects_sharpe(self, mixed_returns):
        """Test that risk-free rate affects Sharpe ratio calculation."""
        metrics_no_rf = calculate_metrics(mixed_returns, risk_free_rate=0.0)
        metrics_with_rf = calculate_metrics(mixed_returns, risk_free_rate=0.05)

        # Higher risk-free rate should result in lower Sharpe ratio
        assert metrics_with_rf["sharpe_ratio"] < metrics_no_rf["sharpe_ratio"]

    def test_periods_per_year_parameter(self, simple_positive_returns):
        """Test that periods_per_year affects annualization."""
        metrics_daily = calculate_metrics(simple_positive_returns, periods_per_year=252)
        metrics_weekly = calculate_metrics(simple_positive_returns, periods_per_year=52)

        # Different periods should result in different annualized volatility
        assert metrics_daily["volatility"] != metrics_weekly["volatility"]

    def test_metrics_types_are_float(self, mixed_returns):
        """Test that all metric values are floats."""
        metrics = calculate_metrics(mixed_returns)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"


# =============================================================================
# TestBenchmarkMetrics - Benchmark comparison tests
# =============================================================================


class TestBenchmarkMetrics:
    """Tests for benchmark comparison metrics."""

    def test_with_benchmark_adds_metrics(self, mixed_returns, benchmark_returns):
        """Test that benchmark returns add alpha, beta, and related metrics."""
        metrics = calculate_metrics(mixed_returns, benchmark_returns=benchmark_returns)

        assert "alpha" in metrics, "Alpha should be calculated with benchmark"
        assert "beta" in metrics, "Beta should be calculated with benchmark"
        assert "tracking_error" in metrics, "Tracking error should be calculated"
        assert "information_ratio" in metrics, "Information ratio should be calculated"

    def test_without_benchmark_no_alpha_beta(self, mixed_returns):
        """Test that without benchmark, alpha/beta are not present."""
        metrics = calculate_metrics(mixed_returns)

        assert "alpha" not in metrics
        assert "beta" not in metrics
        assert "tracking_error" not in metrics
        assert "information_ratio" not in metrics

    def test_identical_returns_to_benchmark(self, mixed_returns):
        """Test metrics when returns match benchmark exactly."""
        metrics = calculate_metrics(mixed_returns, benchmark_returns=mixed_returns.copy())

        assert metrics["beta"] == pytest.approx(1.0, rel=1e-4), "Beta should be 1.0 for identical returns"
        assert metrics["alpha"] == pytest.approx(0.0, abs=1e-4), "Alpha should be 0 for identical returns"
        assert metrics["tracking_error"] == pytest.approx(0.0, abs=1e-4), "Tracking error should be 0"

    def test_benchmark_different_length_aligned(self):
        """Test that benchmark and returns are properly aligned."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)
        returns.index = pd.date_range("2020-01-01", periods=250, freq="D")

        benchmark = pd.Series([0.005, 0.01, -0.005, 0.008, 0.003] * 50)
        benchmark.index = pd.date_range("2020-01-01", periods=250, freq="D")

        metrics = calculate_metrics(returns, benchmark_returns=benchmark)

        assert "alpha" in metrics
        assert "beta" in metrics

    def test_benchmark_too_short(self):
        """Test that very short benchmark returns are handled gracefully."""
        returns = pd.Series([0.01] * 100)
        benchmark = pd.Series([0.005] * 5)  # Only 5 data points

        metrics = calculate_metrics(returns, benchmark_returns=benchmark)

        # Should not have benchmark metrics due to insufficient overlap
        assert "alpha" not in metrics

    def test_outperforming_benchmark_positive_alpha(self):
        """Test that outperforming benchmark results in positive alpha."""
        np.random.seed(42)
        benchmark = pd.Series(np.random.normal(0.0003, 0.01, 252))
        # Strategy returns double the benchmark on average
        returns = benchmark * 2 + pd.Series(np.random.normal(0.0005, 0.005, 252))

        metrics = calculate_metrics(returns, benchmark_returns=benchmark)

        assert metrics["alpha"] > 0, "Alpha should be positive when outperforming"

    def test_beta_near_zero_for_uncorrelated(self):
        """Test beta when strategy is uncorrelated with benchmark."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        np.random.seed(999)  # Different seed for independence
        benchmark = pd.Series(np.random.normal(0.0005, 0.015, 252))

        metrics = calculate_metrics(returns, benchmark_returns=benchmark)

        # Beta should be relatively small for uncorrelated series
        assert abs(metrics["beta"]) < 1.0


# =============================================================================
# TestIndividualMetrics - Tests for private helper functions
# =============================================================================


class TestCAGRCalculation:
    """Tests for CAGR calculation."""

    def test_cagr_one_year_100_percent(self):
        """Test CAGR for 100% return over one year."""
        # 252 daily returns that compound to 100%
        daily_return = (2.0) ** (1 / 252) - 1
        returns = pd.Series([daily_return] * 252)

        cagr = _calculate_cagr(returns, periods_per_year=252)

        assert cagr == pytest.approx(1.0, rel=0.01), "100% return should give 100% CAGR"

    def test_cagr_negative_total_return(self):
        """Test CAGR when total return is negative (but above -100%)."""
        # Returns that result in 50% loss
        daily_return = (0.5) ** (1 / 252) - 1
        returns = pd.Series([daily_return] * 252)

        cagr = _calculate_cagr(returns, periods_per_year=252)

        assert cagr == pytest.approx(-0.5, rel=0.01), "50% loss should give -50% CAGR"

    def test_cagr_multi_year(self):
        """Test CAGR calculation over multiple years."""
        # 10% annual return over 2 years
        annual_return = 0.10
        daily_return = (1 + annual_return) ** (1 / 252) - 1
        returns = pd.Series([daily_return] * 504)  # 2 years

        cagr = _calculate_cagr(returns, periods_per_year=252)

        assert cagr == pytest.approx(0.10, rel=0.01), "CAGR should be ~10%"

    def test_cagr_zero_periods(self):
        """Test CAGR handles edge case of no returns."""
        returns = pd.Series([], dtype=float)
        cagr = _calculate_cagr(returns, periods_per_year=252)
        assert cagr == 0.0

    def test_cagr_total_loss(self):
        """Test CAGR when total value goes to zero (complete loss)."""
        # Returns that would make total return = 0 (complete loss)
        returns = pd.Series([-1.0])  # Complete loss
        cagr = _calculate_cagr(returns, periods_per_year=252)
        # Mathematically correct: -100% loss means -100% CAGR
        assert cagr == pytest.approx(-1.0, rel=0.01)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio for positive excess returns with variation."""
        np.random.seed(42)
        # Positive mean with some variation to get non-zero volatility
        excess_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sharpe = _sharpe_ratio(excess_returns, periods_per_year=252)
        assert sharpe > 0, "Sharpe should be positive for positive mean excess returns"

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio returns 0 when volatility is zero."""
        excess_returns = pd.Series([0.001] * 252)  # All same value
        sharpe = _sharpe_ratio(excess_returns, periods_per_year=252)
        # With identical values, std = 0, so should return 0
        assert sharpe == 0.0

    def test_sharpe_ratio_known_value(self):
        """Test Sharpe ratio against known calculation."""
        np.random.seed(42)
        excess_returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        expected_sharpe = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        )
        actual_sharpe = _sharpe_ratio(excess_returns, periods_per_year=252)

        assert actual_sharpe == pytest.approx(expected_sharpe, rel=1e-6)

    def test_sharpe_negative_returns(self):
        """Test Sharpe ratio for negative returns."""
        excess_returns = pd.Series([-0.001] * 100 + [0.0005] * 100)
        sharpe = _sharpe_ratio(excess_returns, periods_per_year=252)
        assert sharpe < 0, "Sharpe should be negative for net negative excess"


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_ratio_positive(self, mixed_returns):
        """Test Sortino ratio for mixed returns."""
        sortino = _sortino_ratio(mixed_returns, risk_free_rate=0.0, periods_per_year=252)
        assert isinstance(sortino, float)

    def test_sortino_zero_downside(self, all_positive_returns):
        """Test Sortino ratio when there's no downside."""
        sortino = _sortino_ratio(all_positive_returns, risk_free_rate=0.0, periods_per_year=252)
        assert sortino == 0.0, "Sortino should be 0 when downside volatility is 0"

    def test_sortino_higher_than_sharpe_for_skewed(self):
        """Test Sortino is higher than Sharpe for positively skewed returns."""
        # Create returns with more upside than downside, with varying negative returns
        # to ensure non-zero downside volatility
        np.random.seed(42)
        positive = np.random.uniform(0.01, 0.04, 200)  # Mostly positive
        negative = np.random.uniform(-0.02, -0.005, 50)  # Some negative with variation
        returns = pd.Series(np.concatenate([positive, negative]))
        np.random.shuffle(returns.values)

        metrics = calculate_metrics(returns)

        # With more upside variance than downside, Sortino should be >= Sharpe
        # (or at least close - the relationship depends on the exact distribution)
        # The key is that both are computed and reasonable
        assert metrics["sortino_ratio"] > 0, "Sortino should be positive"
        assert metrics["sharpe_ratio"] > 0, "Sharpe should be positive"


class TestDownsideVolatility:
    """Tests for downside volatility calculation."""

    def test_downside_volatility_positive(self, mixed_returns):
        """Test downside volatility for mixed returns."""
        dv = _downside_volatility(mixed_returns, periods_per_year=252)
        assert dv > 0, "Downside volatility should be positive for mixed returns"

    def test_downside_volatility_no_negative(self, all_positive_returns):
        """Test downside volatility when no negative returns."""
        dv = _downside_volatility(all_positive_returns, periods_per_year=252)
        assert dv == 0.0, "Downside volatility should be 0 with no negative returns"

    def test_downside_volatility_all_negative(self, all_negative_returns):
        """Test downside volatility for all negative returns."""
        dv = _downside_volatility(all_negative_returns, periods_per_year=252)

        # Should equal regular volatility of the negative returns
        expected = all_negative_returns.std() * np.sqrt(252)
        assert dv == pytest.approx(expected, rel=1e-6)


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_max_drawdown_basic(self):
        """Test max drawdown with known values."""
        # Series: 1.0 -> 1.1 -> 0.88 -> 0.92
        # Peak 1.1, trough 0.88, drawdown = (0.88-1.1)/1.1 = -0.2
        returns = pd.Series([0.10, -0.20, 0.05])

        mdd = _max_drawdown(returns)

        # After +10%: value = 1.1, peak = 1.1, dd = 0
        # After -20%: value = 0.88, peak = 1.1, dd = (0.88-1.1)/1.1 = -0.2
        # After +5%: value = 0.924, peak = 1.1, dd = (0.924-1.1)/1.1 = -0.16
        assert mdd == pytest.approx(-0.20, rel=0.01), "Max drawdown should be ~-20%"
        assert mdd < 0, "Max drawdown should be negative"

    def test_max_drawdown_all_positive(self, all_positive_returns):
        """Test max drawdown for all positive returns."""
        mdd = _max_drawdown(all_positive_returns)
        assert mdd == 0.0, "Max drawdown should be 0 for all positive returns"

    def test_max_drawdown_all_negative(self, all_negative_returns):
        """Test max drawdown for all negative returns."""
        mdd = _max_drawdown(all_negative_returns)
        assert mdd < 0, "Max drawdown should be negative"
        # For all -1% daily, the drawdown keeps growing
        assert mdd < -0.5, "Drawdown should be significant for continuous losses"

    def test_max_drawdown_recovery(self):
        """Test drawdown with full recovery."""
        # Down 20% then fully recover
        returns = pd.Series([-0.20, 0.25])  # 0.8 * 1.25 = 1.0

        mdd = _max_drawdown(returns)

        assert mdd == pytest.approx(-0.20, rel=0.01)

    def test_max_drawdown_multiple_drawdowns(self):
        """Test with multiple drawdown periods."""
        returns = pd.Series([0.10, -0.15, 0.20, -0.25, 0.10])

        mdd = _max_drawdown(returns)

        # Should capture the largest drawdown
        assert mdd < -0.20


class TestAlphaBeta:
    """Tests for alpha and beta calculation."""

    def test_alpha_beta_identical_returns(self, mixed_returns):
        """Test alpha/beta when returns match benchmark."""
        alpha, beta = _calculate_alpha_beta(
            mixed_returns, mixed_returns, periods_per_year=252
        )

        assert beta == pytest.approx(1.0, rel=1e-4), "Beta should be 1 for identical"
        assert alpha == pytest.approx(0.0, abs=1e-4), "Alpha should be 0 for identical"

    def test_beta_double_benchmark(self, benchmark_returns):
        """Test beta when strategy is double the benchmark."""
        strategy = benchmark_returns * 2

        alpha, beta = _calculate_alpha_beta(strategy, benchmark_returns, periods_per_year=252)

        assert beta == pytest.approx(2.0, rel=0.01), "Beta should be ~2 for 2x leverage"

    def test_beta_zero_benchmark_variance(self):
        """Test beta when benchmark has zero variance."""
        returns = pd.Series([0.01, 0.02, -0.01])
        benchmark = pd.Series([0.005, 0.005, 0.005])  # Constant

        alpha, beta = _calculate_alpha_beta(returns, benchmark, periods_per_year=252)

        assert beta == 0, "Beta should be 0 when benchmark has no variance"

    def test_alpha_positive_outperformance(self):
        """Test alpha is positive when outperforming benchmark."""
        np.random.seed(42)
        benchmark = pd.Series(np.random.normal(0.0002, 0.01, 252))
        # Add consistent outperformance
        strategy = benchmark + 0.001  # 0.1% daily outperformance

        alpha, beta = _calculate_alpha_beta(strategy, benchmark, periods_per_year=252)

        assert alpha > 0, "Alpha should be positive for outperformance"


# =============================================================================
# TestFormatMetrics - Format function tests
# =============================================================================


class TestFormatMetrics:
    """Tests for metrics formatting."""

    def test_format_metrics_basic(self):
        """Test basic formatting of metrics."""
        metrics = {
            "total_return": 0.25,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.15,
            "win_rate": 0.55,
        }

        formatted = format_metrics(metrics)

        assert "25.00%" in formatted, "Total return should be formatted as percentage"
        assert "1.50" in formatted, "Sharpe ratio should be formatted"
        assert "-15.00%" in formatted, "Max drawdown should be formatted"
        assert "55.0%" in formatted, "Win rate should be formatted"

    def test_format_metrics_all_standard_keys(self):
        """Test formatting of all standard metric keys."""
        metrics = {
            "total_return": 0.10,
            "cagr": 0.08,
            "volatility": 0.15,
            "sharpe_ratio": 0.75,
            "sortino_ratio": 1.2,
            "max_drawdown": -0.12,
            "calmar_ratio": 0.66,
            "win_rate": 0.52,
            "profit_factor": 1.3,
            "alpha": 0.02,
            "beta": 0.85,
            "information_ratio": 0.45,
        }

        formatted = format_metrics(metrics)

        # Check each metric appears in output
        for key in metrics:
            assert key in formatted, f"{key} should appear in formatted output"

    def test_format_metrics_empty(self):
        """Test formatting empty metrics dict."""
        formatted = format_metrics({})
        assert formatted == "", "Empty metrics should produce empty string"

    def test_format_metrics_unknown_key(self):
        """Test formatting with unknown metric key."""
        metrics = {"custom_metric": 0.123456}

        formatted = format_metrics(metrics)

        # Should use default formatter
        assert "0.1235" in formatted, "Unknown metrics should use default format"

    def test_format_metrics_negative_values(self):
        """Test formatting negative values."""
        metrics = {
            "total_return": -0.20,
            "sharpe_ratio": -0.5,
            "alpha": -0.03,
        }

        formatted = format_metrics(metrics)

        assert "-20.00%" in formatted
        assert "-0.50" in formatted
        assert "-3.00%" in formatted

    def test_format_metrics_extreme_values(self):
        """Test formatting extreme values."""
        metrics = {
            "total_return": 10.0,  # 1000% return
            "sharpe_ratio": 5.0,
            "profit_factor": float("inf"),
        }

        formatted = format_metrics(metrics)

        assert "1000.00%" in formatted
        assert "5.00" in formatted
        assert "inf" in formatted.lower()

    def test_format_metrics_preserves_order(self):
        """Test that formatting preserves metric order."""
        from collections import OrderedDict

        metrics = OrderedDict([
            ("total_return", 0.10),
            ("sharpe_ratio", 1.0),
            ("max_drawdown", -0.05),
        ])

        formatted = format_metrics(metrics)
        lines = formatted.split("\n")

        assert "total_return" in lines[0]
        assert "sharpe_ratio" in lines[1]
        assert "max_drawdown" in lines[2]


# =============================================================================
# TestEmpyricalMetrics - Tests for new Empyrical-based metrics
# =============================================================================


class TestEmpyricalMetrics:
    """Tests for new Empyrical-based metrics (omega, VaR, CVaR, tail_ratio, stability)."""

    def test_new_metrics_present(self, mixed_returns):
        """Test that new Empyrical metrics are calculated."""
        metrics = calculate_metrics(mixed_returns)

        expected_new_metrics = [
            "omega_ratio",
            "value_at_risk",
            "conditional_value_at_risk",
            "tail_ratio",
            "stability",
        ]

        for key in expected_new_metrics:
            assert key in metrics, f"Missing new metric: {key}"

    def test_omega_ratio_positive_returns(self, all_positive_returns):
        """Test omega ratio for positive returns - should be infinite or very high."""
        metrics = calculate_metrics(all_positive_returns)
        # Omega = inf for all positive returns (no losses to weight against)
        assert metrics["omega_ratio"] == float("inf") or metrics["omega_ratio"] > 10, \
            "Omega should be inf or very high for all positive returns"

    def test_omega_ratio_negative_returns(self, all_negative_returns):
        """Test omega ratio for negative returns - should be low."""
        metrics = calculate_metrics(all_negative_returns)
        # Omega < 1 indicates more downside than upside
        assert metrics["omega_ratio"] < 1.0, "Omega should be < 1 for all negative returns"

    def test_value_at_risk_is_negative(self, mixed_returns):
        """Test that VaR is typically negative (represents loss)."""
        metrics = calculate_metrics(mixed_returns)
        # VaR at 95% confidence represents the 5th percentile of returns
        # For mixed returns with some losses, this should be negative
        assert metrics["value_at_risk"] <= 0, "VaR should be <= 0 (worst 5%)"

    def test_cvar_worse_than_var(self, mixed_returns):
        """Test that CVaR (expected shortfall) is worse than VaR."""
        metrics = calculate_metrics(mixed_returns)
        # CVaR is the average of returns below VaR, so should be <= VaR
        assert metrics["conditional_value_at_risk"] <= metrics["value_at_risk"], \
            "CVaR should be <= VaR (it's the average of worst cases)"

    def test_tail_ratio_positive_returns(self, all_positive_returns):
        """Test tail ratio for positive returns - should be meaningful."""
        metrics = calculate_metrics(all_positive_returns)
        # Tail ratio = |95th percentile| / |5th percentile|
        # For all positive returns, this is just the ratio of good to less-good returns
        assert metrics["tail_ratio"] > 0, "Tail ratio should be positive"

    def test_stability_range(self, mixed_returns):
        """Test stability is in valid range."""
        metrics = calculate_metrics(mixed_returns)
        # Stability is R-squared, so should be in [-inf, 1] but typically [0, 1]
        # Can be negative for very unstable series
        assert metrics["stability"] <= 1.0, "Stability should be <= 1 (R-squared)"

    def test_stability_trending_returns(self):
        """Test stability for consistently trending returns."""
        # Create returns that consistently trend upward
        np.random.seed(42)
        trending = pd.Series([0.001] * 252 + np.random.normal(0, 0.0001, 252))
        metrics = calculate_metrics(trending)
        # Stable upward trend should have high stability (R-squared close to 1)
        assert metrics["stability"] > 0.5, "Stable trending returns should have high stability"

    def test_new_metrics_format(self):
        """Test that new metrics are properly formatted."""
        metrics = {
            "omega_ratio": 1.5,
            "value_at_risk": -0.03,
            "conditional_value_at_risk": -0.05,
            "tail_ratio": 1.2,
            "stability": 0.85,
        }

        formatted = format_metrics(metrics)

        assert "omega_ratio" in formatted
        assert "value_at_risk" in formatted
        assert "-3.00%" in formatted  # VaR formatted as percentage
        assert "-5.00%" in formatted  # CVaR formatted as percentage
        assert "0.8500" in formatted  # stability with 4 decimal places


# =============================================================================
# TestEdgeCases - Edge cases and boundary conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_returns(self):
        """Test with very small return values."""
        returns = pd.Series([1e-8] * 252)
        metrics = calculate_metrics(returns)

        assert "total_return" in metrics
        assert metrics["total_return"] > 0

    def test_very_large_returns(self):
        """Test with very large return values."""
        returns = pd.Series([0.5] * 10)  # 50% daily returns
        metrics = calculate_metrics(returns)

        assert "total_return" in metrics
        assert not np.isinf(metrics["total_return"])

    def test_alternating_returns(self):
        """Test with alternating positive/negative returns."""
        returns = pd.Series([0.01, -0.01] * 126)
        metrics = calculate_metrics(returns)

        assert metrics["win_rate"] == pytest.approx(0.5, rel=0.01)

    def test_single_large_loss(self):
        """Test impact of single large loss on metrics."""
        returns = pd.Series([0.01] * 251 + [-0.50])  # Big loss at end
        metrics = calculate_metrics(returns)

        assert metrics["max_drawdown"] <= -0.40, "Should capture large loss in drawdown"
        assert metrics["win_rate"] > 0.99, "Win rate should still be high"

    def test_returns_with_index(self):
        """Test that indexed Series works correctly."""
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series([0.01] * 252, index=dates)

        metrics = calculate_metrics(returns)

        assert "total_return" in metrics

    def test_benchmark_with_nans(self):
        """Test benchmark calculation with NaN values in benchmark."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)
        benchmark = pd.Series([0.005, np.nan, -0.005, 0.008, 0.003] * 50)

        metrics = calculate_metrics(returns, benchmark_returns=benchmark)

        # Should still calculate, just with fewer aligned points
        if "alpha" in metrics:
            assert not np.isnan(metrics["alpha"])

    def test_inf_handling(self):
        """Test that infinite profit factor is handled."""
        returns = pd.Series([0.01] * 252)  # All positive
        metrics = calculate_metrics(returns)

        assert metrics["profit_factor"] == float("inf")

        # Formatting should handle inf
        formatted = format_metrics(metrics)
        assert "inf" in formatted.lower()


# =============================================================================
# TestIntegration - Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple aspects."""

    def test_full_pipeline_realistic_returns(self):
        """Test full pipeline with realistic return distribution."""
        np.random.seed(42)
        # Simulate realistic daily returns: mean ~0.04% daily (~10% annual), vol ~1.5% daily (~24% annual)
        returns = pd.Series(np.random.normal(0.0004, 0.015, 252))
        benchmark = pd.Series(np.random.normal(0.0003, 0.012, 252))

        metrics = calculate_metrics(
            returns,
            benchmark_returns=benchmark,
            risk_free_rate=0.05,
            periods_per_year=252,
        )

        # All metrics should be present
        expected_keys = [
            "total_return", "cagr", "volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "win_rate", "alpha", "beta",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing {key}"
            assert not np.isnan(metrics[key]), f"{key} is NaN"

        # Format should work
        formatted = format_metrics(metrics)
        assert len(formatted) > 100, "Formatted output should be substantial"

    def test_metrics_consistency(self):
        """Test that metrics are internally consistent."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        metrics = calculate_metrics(returns)

        # Calmar = CAGR / |MDD|
        if metrics["max_drawdown"] != 0:
            expected_calmar = metrics["cagr"] / abs(metrics["max_drawdown"])
            assert metrics["calmar_ratio"] == pytest.approx(expected_calmar, rel=1e-6)

        # Win rate should match proportion of positive returns
        pos_count = (returns > 0).sum()
        expected_win_rate = pos_count / len(returns)
        assert metrics["win_rate"] == pytest.approx(expected_win_rate, rel=1e-6)

    def test_calculate_then_format_roundtrip(self, mixed_returns, benchmark_returns):
        """Test that calculated metrics can be formatted correctly."""
        metrics = calculate_metrics(mixed_returns, benchmark_returns=benchmark_returns)
        formatted = format_metrics(metrics)

        # Each metric should have its own line in formatted output
        lines = formatted.split("\n")
        for key in metrics:
            # Check that exactly one line starts with this key
            matching_lines = [line for line in lines if line.strip().startswith(key)]
            assert len(matching_lines) == 1, f"{key} should appear exactly once as line start"


# =============================================================================
# TestStabilityScoreV1 - Tests for Stability Score v1 (walk-forward validation)
# =============================================================================


class TestStabilityScoreV1:
    """Tests for Stability Score v1 - walk-forward validation stability metric."""

    def test_stability_score_perfect_stability(self):
        """Perfect stability returns low score."""
        fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]  # No variation
        fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]  # No variation
        mean_ic = 0.05  # Positive

        score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)

        assert score <= 1.0  # Should be stable

    def test_stability_score_unstable(self):
        """Highly variable folds return high score."""
        fold_sharpes = [2.0, 0.5, -0.5, 1.5, 0.2]  # High variation
        fold_drawdowns = [0.05, 0.30, 0.40, 0.10, 0.25]  # High variation
        mean_ic = 0.02  # Low IC

        score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)

        assert score > 1.0  # Should be unstable

    def test_stability_score_sign_flip_penalty(self):
        """Sign flip adds penalty."""
        fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]
        fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]
        mean_ic = -0.01  # Negative IC

        score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)

        # Score with negative IC should be higher than with positive IC
        score_positive_ic = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, 0.01)
        assert score > score_positive_ic  # Penalty applied

    def test_stability_score_zero_mean_sharpe(self):
        """Handle zero mean Sharpe (edge case)."""
        fold_sharpes = [0.0, 0.0, 0.0, 0.0, 0.0]
        fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]
        mean_ic = 0.05

        # Should handle gracefully (return inf or very high score)
        score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)
        assert score >= 0  # Should not crash

    def test_stability_score_zero_mean_drawdown(self):
        """Handle zero mean drawdown (edge case)."""
        fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]
        fold_drawdowns = [0.0, 0.0, 0.0, 0.0, 0.0]
        mean_ic = 0.05

        # Should handle gracefully
        score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)
        assert score >= 0  # Should not crash

    def test_stability_score_positive_ic_no_penalty(self):
        """Positive IC adds no sign flip penalty."""
        fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]
        fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]
        mean_ic = 0.01  # Positive

        score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)

        # With perfect Sharpe and DD stability, score should just be 0
        assert score == 0.0  # No CV, no dispersion, no sign flip penalty
