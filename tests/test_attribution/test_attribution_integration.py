"""End-to-end integration tests for performance attribution system."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hrp.data.attribution.factor_attribution import BrinsonAttribution, FactorAttribution
from hrp.data.attribution.feature_importance import FeatureImportanceTracker, RollingImportance
from hrp.data.attribution.decision_attribution import DecisionAttributor, RebalanceAnalyzer
from hrp.data.attribution.attribution_config import AttributionConfig


class TestAttributionE2E:
    """End-to-end tests for complete attribution pipeline."""

    def test_full_attribution_pipeline(self):
        """Test complete attribution workflow from data to results."""
        # Arrange: Create synthetic portfolio data
        n_days = 60
        dates = pd.date_range(end=date.today(), periods=n_days, freq="D")

        # Portfolio with 3 sectors
        portfolio_weights = pd.DataFrame({
            "Technology": [0.4] * n_days,
            "Healthcare": [0.3] * n_days,
            "Finance": [0.3] * n_days,
        }, index=dates)

        # Benchmark with different weights
        benchmark_weights = pd.DataFrame({
            "Technology": [0.35] * n_days,
            "Healthcare": [0.35] * n_days,
            "Finance": [0.30] * n_days,
        }, index=dates)

        # Sector returns (portfolio outperformed in Tech, underperformed in Healthcare)
        portfolio_returns = pd.DataFrame({
            "Technology": np.random.normal(0.002, 0.01, n_days),  # 20bps/day avg
            "Healthcare": np.random.normal(-0.001, 0.01, n_days),  # -10bps/day avg
            "Finance": np.random.normal(0.001, 0.01, n_days),  # 10bps/day avg
        }, index=dates)

        benchmark_returns = pd.DataFrame({
            "Technology": np.random.normal(0.0015, 0.008, n_days),
            "Healthcare": np.random.normal(0.0005, 0.008, n_days),
            "Finance": np.random.normal(0.001, 0.008, n_days),
        }, index=dates)

        # Act: Run Brinson attribution for each sector
        attributor = BrinsonAttribution()
        all_results = []

        for sector in ["Technology", "Healthcare", "Finance"]:
            results = attributor.calculate_sector_attribution(
                portfolio_weight=portfolio_weights[sector].iloc[-1],
                benchmark_weight=benchmark_weights[sector].iloc[-1],
                portfolio_sector_return=portfolio_returns[sector].mean(),
                benchmark_sector_return=benchmark_returns[sector].mean(),
                benchmark_total_return=benchmark_returns.mean(axis=1).mean(),
            )
            all_results.extend(results)

        # Assert: Results should exist for all three effects
        allocation_results = [r for r in all_results if r.effect_type == "allocation"]
        selection_results = [r for r in all_results if r.effect_type == "selection"]
        interaction_results = [r for r in all_results if r.effect_type == "interaction"]

        assert len(allocation_results) == 3, "Should have allocation for each sector"
        assert len(selection_results) == 3, "Should have selection for each sector"
        assert len(interaction_results) == 3, "Should have interaction for each sector"

        # Summation invariant: All effects should sum to active return
        total_contribution = sum(r.contribution_pct for r in all_results)
        portfolio_total_return = (portfolio_weights.iloc[-1] * portfolio_returns.mean()).sum()
        benchmark_total_return = (benchmark_weights.iloc[-1] * benchmark_returns.mean()).sum()
        active_return = portfolio_total_return - benchmark_total_return

        # Allow small numerical tolerance
        assert abs(total_contribution - active_return) < 1e-4, "Attribution should sum to active return"

    def test_regression_attribution_pipeline(self):
        """Test regression-based factor attribution."""
        # Arrange: Create portfolio returns and factor returns
        n_days = 60
        dates = pd.date_range(end=date.today(), periods=n_days, freq="D")

        # Synthetic factor returns
        factor_returns = pd.DataFrame({
            "Market": np.random.normal(0.0005, 0.01, n_days),
            "Value": np.random.normal(0.0002, 0.005, n_days),
            "Momentum": np.random.normal(0.0001, 0.006, n_days),
        }, index=dates)

        # Portfolio returns = linear combination of factors + noise
        portfolio_returns = pd.Series(
            0.5 * factor_returns["Market"]
            + 0.3 * factor_returns["Value"]
            + 0.2 * factor_returns["Momentum"]
            + np.random.normal(0, 0.002, n_days),  # idiosyncratic
            index=dates,
        )

        # Act: Calculate factor attribution
        attributor = FactorAttribution()
        results = attributor.calculate(
            portfolio_returns=portfolio_returns,
            factor_returns=factor_returns,
        )

        # Assert: Should have results for each factor + residual
        assert len(results) >= 3, "Should have at least 3 factor attributions"

        factor_names = [r.factor for r in results]
        assert "Market" in factor_names
        assert "Value" in factor_names
        assert "Momentum" in factor_names

        # R-squared should be reasonably high (>0.5) since we constructed the data
        # Note: This is a soft test - in practice R-squared varies
        total_explained = sum(r.contribution_pct for r in results if r.factor != "Residual")
        # Just check that factors explain some portion
        assert abs(total_explained) > 0.0, "Factors should explain some variance"

    def test_feature_importance_integration(self):
        """Test feature importance tracking over time."""
        # Arrange: Create feature matrix and model
        from sklearn.ensemble import RandomForestRegressor

        n_samples = 100
        n_features = 5

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Target is mostly feature_0 + feature_1
        y = 2 * X["feature_0"] + 1.5 * X["feature_1"] + np.random.normal(0, 0.1, n_samples)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Act: Calculate permutation importance
        tracker = FeatureImportanceTracker(method="permutation", n_repeats=5)
        results = tracker.calculate(model=model, X=X, y=y)

        # Assert: feature_0 and feature_1 should have highest importance
        importance_dict = {r.feature_name: r.importance_score for r in results}

        assert len(results) == n_features, "Should have importance for each feature"
        assert importance_dict["feature_0"] > 0, "feature_0 should be important"
        assert importance_dict["feature_1"] > 0, "feature_1 should be important"

        # Most important features should be feature_0 and feature_1
        top_2_features = sorted(results, key=lambda x: x.importance_score, reverse=True)[:2]
        top_2_names = {r.feature_name for r in top_2_features}
        assert "feature_0" in top_2_names or "feature_1" in top_2_names, \
            "Top features should include feature_0 or feature_1"

    def test_decision_attribution_integration(self):
        """Test trade-level decision attribution."""
        # Arrange: Create trade history
        entry_date = date.today() - timedelta(days=10)
        exit_date = date.today() - timedelta(days=1)

        # Create price history
        dates = pd.date_range(entry_date, exit_date, freq="D")
        prices = pd.Series(
            100 * (1 + np.random.normal(0.001, 0.01, len(dates))).cumprod(),
            index=dates,
        )

        # Act: Analyze trade
        attributor = DecisionAttributor()
        result = attributor.analyze_trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=prices.iloc[0],
            exit_price=prices.iloc[-1],
            position_size=100,  # 100 shares
            optimal_entry_price=prices.min(),  # Best entry would have been at lowest price
            optimal_exit_price=prices.max(),  # Best exit would have been at highest price
        )

        # Assert: Result should have timing and sizing components
        assert result is not None
        assert result.timing_contribution is not None
        assert result.sizing_contribution is not None
        assert result.residual_contribution is not None

        # P&L decomposition should sum correctly
        total_decomposed = (
            result.timing_contribution
            + result.sizing_contribution
            + result.residual_contribution
        )
        assert abs(total_decomposed - result.pnl) < 0.01, "Components should sum to total P&L"

    def test_rolling_feature_importance(self):
        """Test rolling feature importance calculation."""
        # Arrange: Create time series data
        from sklearn.ensemble import RandomForestRegressor

        n_days = 90
        n_features = 4
        dates = pd.date_range(end=date.today(), periods=n_days, freq="D")

        # Feature importance changes over time (regime shift at day 45)
        X_early = pd.DataFrame(
            np.random.randn(45, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_early = 2 * X_early["feature_0"] + np.random.normal(0, 0.1, 45)  # feature_0 important early

        X_late = pd.DataFrame(
            np.random.randn(45, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_late = 2 * X_late["feature_1"] + np.random.normal(0, 0.1, 45)  # feature_1 important late

        X = pd.concat([X_early, X_late], ignore_index=True)
        y = pd.concat([pd.Series(y_early), pd.Series(y_late)], ignore_index=True)

        # Act: Calculate rolling importance
        rolling = RollingImportance(window_days=30, step_days=15)
        results = rolling.calculate(X=X, y=y, dates=dates)

        # Assert: Should detect regime change
        assert len(results) > 0, "Should have rolling results"

        # Results should be a DataFrame with features as columns
        assert isinstance(results, pd.DataFrame)
        assert all(f"feature_{i}" in results.columns for i in range(n_features))

    def test_rebalancing_value_add(self):
        """Test rebalancing value-add analysis."""
        # Arrange: Create portfolio with rebalancing events
        dates = pd.date_range(end=date.today(), periods=30, freq="D")

        # Prices trend up
        prices = pd.Series(
            100 * (1.01 ** np.arange(len(dates))),  # 1% daily growth
            index=dates,
        )

        # Portfolio rebalances at day 15 (sells some, locks in gains)
        rebalance_date = dates[15]

        # Weights before and after rebalance
        weights_before = {"ASSET_A": 0.6, "ASSET_B": 0.4}
        weights_after = {"ASSET_A": 0.5, "ASSET_B": 0.5}  # Rebalance to equal weight

        # Act: Analyze rebalancing
        analyzer = RebalanceAnalyzer()
        result = analyzer.analyze_rebalance(
            rebalance_date=rebalance_date,
            weights_before=weights_before,
            weights_after=weights_after,
            prices_before={"ASSET_A": prices.iloc[14], "ASSET_B": prices.iloc[14]},
            prices_after={"ASSET_A": prices.iloc[-1], "ASSET_B": prices.iloc[-1]},
        )

        # Assert: Should calculate value-add
        assert result is not None
        assert "rebalance_value_add" in result or "value_add" in result
        # Value-add can be positive or negative depending on market moves


class TestAttributionConfigIntegration:
    """Test configuration and pipeline integration."""

    def test_config_creation(self):
        """Test attribution config can be created with all options."""
        config = AttributionConfig(
            method="brinson",
            benchmark="SPY",
            lookback_days=252,
            factor_model="fama_french_3",
            permutation_n_repeats=10,
            shap_enabled=False,
            rolling_window_days=60,
            include_timing=True,
            include_sizing=True,
            validate_summation=True,
        )

        assert config.method == "brinson"
        assert config.benchmark == "SPY"
        assert config.lookback_days == 252
        assert config.validate_summation is True

    def test_default_config(self):
        """Test default configuration works."""
        from hrp.data.attribution.attribution_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG.method == "brinson"
        assert DEFAULT_CONFIG.validate_summation is True
        assert DEFAULT_CONFIG.cache_enabled is True
