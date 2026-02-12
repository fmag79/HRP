"""Tests for factor attribution."""

import numpy as np
import pandas as pd
import pytest

from hrp.data.attribution.factor_attribution import (
    AttributionResult,
    BrinsonAttribution,
    FactorAttribution,
)


class TestAttributionResult:
    """Tests for AttributionResult dataclass."""

    def test_valid_creation(self):
        """Test creating valid AttributionResult."""
        result = AttributionResult(
            factor="Technology",
            effect_type="allocation",
            contribution_pct=1.5,
            contribution_dollar=15000.0,
            active_return=2.0,
        )
        assert result.factor == "Technology"
        assert result.effect_type == "allocation"
        assert result.contribution_pct == 1.5
        assert result.contribution_dollar == 15000.0
        assert result.active_return == 2.0

    def test_invalid_effect_type(self):
        """Test that invalid effect_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid effect_type"):
            AttributionResult(
                factor="Tech",
                effect_type="invalid",  # type: ignore
                contribution_pct=1.0,
            )

    def test_optional_fields(self):
        """Test that optional fields can be None."""
        result = AttributionResult(
            factor="Tech", effect_type="allocation", contribution_pct=1.0
        )
        assert result.contribution_dollar is None
        assert result.active_return is None


class TestBrinsonAttribution:
    """Tests for Brinson-Fachler attribution."""

    def test_basic_attribution(self):
        """Test basic Brinson attribution with 2 sectors."""
        # Create simple test case: 2 sectors, equal weight benchmark
        portfolio_weights = pd.Series({"Tech": 0.7, "Finance": 0.3})
        portfolio_returns = pd.Series({"Tech": 0.10, "Finance": 0.05})
        benchmark_weights = pd.Series({"Tech": 0.5, "Finance": 0.5})
        benchmark_returns = pd.Series({"Tech": 0.08, "Finance": 0.06})

        # Compute attribution
        brinson = BrinsonAttribution()
        results = brinson.attribute(
            portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns
        )

        # Should have 6 results (2 sectors × 3 effect types)
        assert len(results) == 6

        # Check that all results are AttributionResult
        assert all(isinstance(r, AttributionResult) for r in results)

        # Check that all effect types are present
        effect_types = {r.effect_type for r in results}
        assert effect_types == {"allocation", "selection", "interaction"}

        # Verify summation invariant
        portfolio_return = (portfolio_weights * portfolio_returns).sum()
        benchmark_return = (benchmark_weights * benchmark_returns).sum()
        active_return = portfolio_return - benchmark_return

        total_attribution = sum(r.contribution_pct for r in results) / 100
        assert abs(total_attribution - active_return) < 1e-6

    def test_zero_active_return(self):
        """Test attribution when portfolio matches benchmark."""
        # Portfolio = benchmark → zero active return
        weights = pd.Series({"A": 0.6, "B": 0.4})
        returns = pd.Series({"A": 0.10, "B": 0.05})

        brinson = BrinsonAttribution()
        results = brinson.attribute(weights, returns, weights, returns)

        # All contributions should be zero
        total_attribution = sum(r.contribution_pct for r in results)
        assert abs(total_attribution) < 1e-10

    def test_single_sector(self):
        """Test attribution with single sector."""
        portfolio_weights = pd.Series({"Tech": 1.0})
        portfolio_returns = pd.Series({"Tech": 0.10})
        benchmark_weights = pd.Series({"Tech": 1.0})
        benchmark_returns = pd.Series({"Tech": 0.08})

        brinson = BrinsonAttribution()
        results = brinson.attribute(
            portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns
        )

        # Should have 3 results (1 sector × 3 effect types)
        assert len(results) == 3

        # All allocation and interaction should be zero (same weights)
        allocation_total = sum(
            r.contribution_pct for r in results if r.effect_type == "allocation"
        )
        interaction_total = sum(
            r.contribution_pct for r in results if r.effect_type == "interaction"
        )
        assert abs(allocation_total) < 1e-10
        assert abs(interaction_total) < 1e-10

        # Selection should equal active return (2%)
        selection_total = sum(
            r.contribution_pct for r in results if r.effect_type == "selection"
        )
        expected_active_return = (0.10 - 0.08) * 100
        assert abs(selection_total - expected_active_return) < 1e-10

    def test_portfolio_value_conversion(self):
        """Test dollar attribution with portfolio value."""
        portfolio_weights = pd.Series({"Tech": 0.7, "Finance": 0.3})
        portfolio_returns = pd.Series({"Tech": 0.10, "Finance": 0.05})
        benchmark_weights = pd.Series({"Tech": 0.5, "Finance": 0.5})
        benchmark_returns = pd.Series({"Tech": 0.08, "Finance": 0.06})
        portfolio_value = 1_000_000.0

        brinson = BrinsonAttribution()
        results = brinson.attribute(
            portfolio_weights,
            portfolio_returns,
            benchmark_weights,
            benchmark_returns,
            portfolio_value=portfolio_value,
        )

        # All results should have dollar values
        assert all(r.contribution_dollar is not None for r in results)

        # Dollar total should equal active return × portfolio value
        total_dollar = sum(r.contribution_dollar for r in results)  # type: ignore
        portfolio_return = (portfolio_weights * portfolio_returns).sum()
        benchmark_return = (benchmark_weights * benchmark_returns).sum()
        expected_dollar = (portfolio_return - benchmark_return) * portfolio_value

        assert abs(total_dollar - expected_dollar) < 1e-6

    def test_misaligned_sectors(self):
        """Test attribution when portfolio and benchmark have different sectors."""
        # Portfolio only has Tech, benchmark only has Finance
        portfolio_weights = pd.Series({"Tech": 1.0})
        portfolio_returns = pd.Series({"Tech": 0.10})
        benchmark_weights = pd.Series({"Finance": 1.0})
        benchmark_returns = pd.Series({"Finance": 0.06})

        brinson = BrinsonAttribution()
        results = brinson.attribute(
            portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns
        )

        # Should have 6 results (2 sectors after union × 3 effect types)
        assert len(results) == 6

        # Factors should include both Tech and Finance
        factors = {r.factor for r in results}
        assert factors == {"Tech", "Finance"}

    def test_invalid_weights_sum(self):
        """Test that invalid weight sums raise ValueError."""
        portfolio_weights = pd.Series({"Tech": 0.7, "Finance": 0.4})  # Sum = 1.1
        portfolio_returns = pd.Series({"Tech": 0.10, "Finance": 0.05})
        benchmark_weights = pd.Series({"Tech": 0.5, "Finance": 0.5})
        benchmark_returns = pd.Series({"Tech": 0.08, "Finance": 0.06})

        brinson = BrinsonAttribution()
        with pytest.raises(ValueError, match="Portfolio weights sum"):
            brinson.attribute(
                portfolio_weights,
                portfolio_returns,
                benchmark_weights,
                benchmark_returns,
            )

    def test_nan_values(self):
        """Test that NaN values raise ValueError."""
        portfolio_weights = pd.Series({"Tech": 0.7, "Finance": 0.3})
        portfolio_returns = pd.Series({"Tech": np.nan, "Finance": 0.05})
        benchmark_weights = pd.Series({"Tech": 0.5, "Finance": 0.5})
        benchmark_returns = pd.Series({"Tech": 0.08, "Finance": 0.06})

        brinson = BrinsonAttribution()
        with pytest.raises(ValueError, match="contains NaN"):
            brinson.attribute(
                portfolio_weights,
                portfolio_returns,
                benchmark_weights,
                benchmark_returns,
            )

    def test_aggregate_by_effect(self):
        """Test aggregating results by effect type."""
        portfolio_weights = pd.Series({"Tech": 0.7, "Finance": 0.3})
        portfolio_returns = pd.Series({"Tech": 0.10, "Finance": 0.05})
        benchmark_weights = pd.Series({"Tech": 0.5, "Finance": 0.5})
        benchmark_returns = pd.Series({"Tech": 0.08, "Finance": 0.06})

        brinson = BrinsonAttribution()
        results = brinson.attribute(
            portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns
        )

        # Aggregate by effect type
        aggregated = brinson.aggregate_by_effect(results)

        # Should have 3 effect types
        assert set(aggregated.keys()) == {"allocation", "selection", "interaction"}

        # Sum should equal active return
        total = sum(aggregated.values())
        portfolio_return = (portfolio_weights * portfolio_returns).sum()
        benchmark_return = (benchmark_weights * benchmark_returns).sum()
        active_return = (portfolio_return - benchmark_return) * 100

        assert abs(total - active_return) < 1e-6


class TestFactorAttribution:
    """Tests for regression-based factor attribution."""

    def test_market_model_single_factor(self):
        """Test market model (single factor regression)."""
        # Create synthetic data: portfolio = 1.2 * market + noise
        np.random.seed(42)
        n_periods = 100
        market_returns = pd.Series(np.random.normal(0.001, 0.02, n_periods))
        portfolio_returns = 1.2 * market_returns + np.random.normal(
            0.0, 0.005, n_periods
        )

        factor_returns = pd.DataFrame({"Market": market_returns})

        # Run attribution
        factor_attr = FactorAttribution(factor_model="market")
        results = factor_attr.attribute(portfolio_returns, factor_returns)

        # Should have 2 results: Market factor + Alpha
        assert len(results) == 2

        # Check factor names
        factor_names = {r.factor for r in results}
        assert "Market" in factor_names
        assert "Alpha (residual)" in factor_names

        # Market beta should be ~1.2
        market_result = next(r for r in results if r.factor == "Market")
        assert abs(factor_attr.coefficients_["Market"] - 1.2) < 0.1  # type: ignore

        # R-squared should be high (since data is synthetic)
        assert factor_attr.r_squared_ > 0.8  # type: ignore

    def test_multi_factor_regression(self):
        """Test multi-factor regression (Fama-French style)."""
        # Create synthetic data with 3 factors
        np.random.seed(42)
        n_periods = 100
        market = pd.Series(np.random.normal(0.001, 0.02, n_periods))
        smb = pd.Series(np.random.normal(0.0, 0.01, n_periods))
        hml = pd.Series(np.random.normal(0.0, 0.01, n_periods))

        # Portfolio = 1.0 * market + 0.5 * smb + 0.3 * hml
        portfolio_returns = market + 0.5 * smb + 0.3 * hml

        factor_returns = pd.DataFrame({"Market": market, "SMB": smb, "HML": hml})

        # Run attribution
        factor_attr = FactorAttribution(factor_model="fama_french_3")
        results = factor_attr.attribute(portfolio_returns, factor_returns)

        # Should have 4 results: 3 factors + Alpha
        assert len(results) == 4

        # Check all factors present
        factor_names = {r.factor for r in results}
        assert {"Market", "SMB", "HML", "Alpha (residual)"}.issubset(factor_names)

    def test_portfolio_value_conversion(self):
        """Test dollar attribution with portfolio value."""
        np.random.seed(42)
        n_periods = 100
        market_returns = pd.Series(np.random.normal(0.001, 0.02, n_periods))
        portfolio_returns = 1.2 * market_returns

        factor_returns = pd.DataFrame({"Market": market_returns})
        portfolio_value = 1_000_000.0

        factor_attr = FactorAttribution()
        results = factor_attr.attribute(
            portfolio_returns, factor_returns, portfolio_value=portfolio_value
        )

        # All results should have dollar values
        assert all(r.contribution_dollar is not None for r in results)

    def test_invalid_inputs(self):
        """Test that invalid inputs raise ValueError."""
        factor_attr = FactorAttribution()

        # Not a Series
        with pytest.raises(ValueError, match="must be a pandas Series"):
            factor_attr.attribute(
                [0.01, 0.02],  # type: ignore
                pd.DataFrame({"Market": [0.01, 0.02]}),
            )

        # Not a DataFrame
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            factor_attr.attribute(
                pd.Series([0.01, 0.02]),
                [0.01, 0.02],  # type: ignore
            )

        # Contains NaN
        with pytest.raises(ValueError, match="contains NaN"):
            factor_attr.attribute(
                pd.Series([0.01, np.nan, 0.02]),
                pd.DataFrame({"Market": [0.01, 0.02, 0.03]}),
            )

    def test_too_few_observations(self):
        """Test that too few observations raise ValueError."""
        factor_attr = FactorAttribution()

        with pytest.raises(ValueError, match="at least 2 observations"):
            factor_attr.attribute(
                pd.Series([0.01]),
                pd.DataFrame({"Market": [0.01]}),
            )

    def test_summation_invariant(self):
        """Test that factor contributions sum to average portfolio return."""
        np.random.seed(42)
        n_periods = 100
        market_returns = pd.Series(np.random.normal(0.001, 0.02, n_periods))
        portfolio_returns = 1.2 * market_returns + 0.001  # Add constant alpha

        factor_returns = pd.DataFrame({"Market": market_returns})

        factor_attr = FactorAttribution()
        results = factor_attr.attribute(portfolio_returns, factor_returns)

        # Sum of contributions should equal average portfolio return
        total_contribution = sum(r.contribution_pct for r in results) / 100
        avg_portfolio_return = portfolio_returns.mean()

        assert abs(total_contribution - avg_portfolio_return) < 1e-6
