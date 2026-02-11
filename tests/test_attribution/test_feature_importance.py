"""Tests for feature importance tracking."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from hrp.data.attribution.feature_importance import (
    FeatureImportanceTracker,
    ImportanceResult,
    RollingImportance,
)


class TestImportanceResult:
    """Tests for ImportanceResult dataclass."""

    def test_valid_creation(self):
        """Test creating valid ImportanceResult."""
        result = ImportanceResult(
            feature_name="momentum_7d",
            importance_score=0.85,
            direction="positive",
            period="2024-01-01 to 2024-03-31",
            method="permutation",
        )
        assert result.feature_name == "momentum_7d"
        assert result.importance_score == 0.85
        assert result.direction == "positive"
        assert result.period == "2024-01-01 to 2024-03-31"
        assert result.method == "permutation"

    def test_negative_importance_raises(self):
        """Test that negative importance raises ValueError."""
        with pytest.raises(ValueError, match="importance_score must be >= 0"):
            ImportanceResult(
                feature_name="test",
                importance_score=-0.5,
                direction="positive",
            )

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            ImportanceResult(
                feature_name="test",
                importance_score=0.5,
                direction="invalid",  # type: ignore
            )

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            ImportanceResult(
                feature_name="test",
                importance_score=0.5,
                direction="positive",
                method="invalid",  # type: ignore
            )


class TestFeatureImportanceTracker:
    """Tests for FeatureImportanceTracker."""

    def test_permutation_importance_basic(self):
        """Test basic permutation importance computation."""
        # Create synthetic data: y = 2*X1 + 0.5*X2 + noise
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),  # Noise feature
            }
        )
        y = pd.Series(2 * X["feature1"] + 0.5 * X["feature2"] + np.random.randn(n_samples) * 0.1)

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Compute permutation importance
        tracker = FeatureImportanceTracker(n_repeats=10, random_state=42)
        results = tracker.compute_permutation_importance(model, X, y)

        # Should have 3 results
        assert len(results) == 3

        # All results should be ImportanceResult
        assert all(isinstance(r, ImportanceResult) for r in results)

        # Results should be sorted by importance (descending)
        importances = [r.importance_score for r in results]
        assert importances == sorted(importances, reverse=True)

        # feature1 should be most important (highest coefficient)
        assert results[0].feature_name == "feature1"

        # feature2 should be second
        assert results[1].feature_name == "feature2"

        # feature3 should be least important (noise)
        assert results[2].feature_name == "feature3"
        assert results[2].importance_score < 0.5  # Low importance

    def test_permutation_importance_normalization(self):
        """Test that importance scores are normalized to [0, 1]."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "a": np.random.randn(n_samples),
                "b": np.random.randn(n_samples),
            }
        )
        y = pd.Series(X["a"] + np.random.randn(n_samples) * 0.1)

        model = LinearRegression()
        model.fit(X, y)

        tracker = FeatureImportanceTracker()
        results = tracker.compute_permutation_importance(model, X, y)

        # All scores should be in [0, 1]
        for result in results:
            assert 0 <= result.importance_score <= 1

        # At least one feature should have score = 1.0 (most important)
        max_score = max(r.importance_score for r in results)
        assert abs(max_score - 1.0) < 1e-6

    def test_feature_directions(self):
        """Test that feature directions are computed correctly."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "positive_feature": np.random.randn(n_samples),
                "negative_feature": np.random.randn(n_samples),
            }
        )
        # positive_feature has positive coefficient, negative_feature has negative
        y = pd.Series(
            2 * X["positive_feature"] - 3 * X["negative_feature"] + np.random.randn(n_samples) * 0.1
        )

        model = LinearRegression()
        model.fit(X, y)

        tracker = FeatureImportanceTracker()
        results = tracker.compute_permutation_importance(model, X, y)

        # Find each feature's result
        pos_result = next(r for r in results if r.feature_name == "positive_feature")
        neg_result = next(r for r in results if r.feature_name == "negative_feature")

        # Check directions
        assert pos_result.direction == "positive"
        assert neg_result.direction == "negative"

    def test_invalid_inputs(self):
        """Test that invalid inputs raise ValueError."""
        tracker = FeatureImportanceTracker()
        model = LinearRegression()

        # Not a DataFrame
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            tracker.compute_permutation_importance(
                model,
                np.array([[1, 2], [3, 4]]),  # type: ignore
                pd.Series([1, 2]),
            )

        # Not a Series
        with pytest.raises(ValueError, match="must be a pandas Series"):
            tracker.compute_permutation_importance(
                model,
                pd.DataFrame({"a": [1, 2]}),
                [1, 2],  # type: ignore
            )

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            tracker.compute_permutation_importance(
                model,
                pd.DataFrame({"a": [1, 2, 3]}),
                pd.Series([1, 2]),
            )

    def test_shap_not_enabled_raises(self):
        """Test that SHAP raises error when not enabled."""
        tracker = FeatureImportanceTracker(shap_enabled=False)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(ValueError, match="SHAP is not enabled"):
            tracker.compute_shap_importance(model, X)

    def test_zero_variance_features(self):
        """Test handling of zero-variance features."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "constant": np.ones(n_samples),  # Zero variance
                "random": np.random.randn(n_samples),
            }
        )
        y = pd.Series(X["random"] + np.random.randn(n_samples) * 0.1)

        model = LinearRegression()
        model.fit(X, y)

        tracker = FeatureImportanceTracker()
        results = tracker.compute_permutation_importance(model, X, y)

        # Should handle gracefully (no exceptions)
        assert len(results) == 2

        # Constant feature should have low importance
        constant_result = next(r for r in results if r.feature_name == "constant")
        random_result = next(r for r in results if r.feature_name == "random")
        assert random_result.importance_score > constant_result.importance_score


class TestRollingImportance:
    """Tests for RollingImportance."""

    def test_rolling_importance_basic(self):
        """Test basic rolling importance computation."""
        # Create synthetic time series data
        np.random.seed(42)
        n_days = 200
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

        # Early period: feature1 important, late period: feature2 important
        feature1 = np.random.randn(n_days)
        feature2 = np.random.randn(n_days)

        # Create regime change at day 100
        y_values = np.zeros(n_days)
        y_values[:100] = 2 * feature1[:100] + np.random.randn(100) * 0.1
        y_values[100:] = 2 * feature2[100:] + np.random.randn(100) * 0.1

        X = pd.DataFrame({"feature1": feature1, "feature2": feature2})
        y = pd.Series(y_values)

        # Fit model on full data
        model = LinearRegression()
        model.fit(X, y)

        # Compute rolling importance
        rolling = RollingImportance(window_days=50, step_days=20)
        df = rolling.compute_rolling_importance(model, X, y, dates)

        # Should have multiple time periods
        assert len(df) > 1

        # Should have 2 columns (features)
        assert list(df.columns) == ["feature1", "feature2"]

        # All values should be in [0, 1]
        assert (df >= 0).all().all()
        assert (df <= 1).all().all()

    def test_rolling_importance_regime_change(self):
        """Test that rolling importance detects regime changes."""
        np.random.seed(42)
        n_days = 150
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

        # Early: feature1 important, late: feature2 important
        feature1 = np.random.randn(n_days)
        feature2 = np.random.randn(n_days)

        y_values = np.zeros(n_days)
        # First 75 days: y = 3*feature1
        y_values[:75] = 3 * feature1[:75] + np.random.randn(75) * 0.1
        # Last 75 days: y = 3*feature2
        y_values[75:] = 3 * feature2[75:] + np.random.randn(75) * 0.1

        X = pd.DataFrame({"feature1": feature1, "feature2": feature2})
        y = pd.Series(y_values)

        model = LinearRegression()
        model.fit(X, y)

        # Rolling importance with 60-day window
        rolling = RollingImportance(window_days=60, step_days=15)
        df = rolling.compute_rolling_importance(model, X, y, dates)

        # Early windows: feature1 > feature2
        early_window = df.iloc[0]
        assert early_window["feature1"] > early_window["feature2"]

        # Late windows: feature2 > feature1
        late_window = df.iloc[-1]
        assert late_window["feature2"] > late_window["feature1"]

    def test_get_top_features_by_period(self):
        """Test getting top N features for each period."""
        np.random.seed(42)
        n_days = 120
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

        X = pd.DataFrame(
            {
                "a": np.random.randn(n_days),
                "b": np.random.randn(n_days),
                "c": np.random.randn(n_days),
            }
        )
        y = pd.Series(2 * X["a"] + X["b"] + np.random.randn(n_days) * 0.1)

        model = LinearRegression()
        model.fit(X, y)

        rolling = RollingImportance(window_days=60, step_days=30)
        df = rolling.compute_rolling_importance(model, X, y, dates)

        # Get top 2 features per period
        top_features = rolling.get_top_features_by_period(n_features=2)

        # Should have results for each period
        assert len(top_features) > 0

        # Each period should have 2 features
        for period, features in top_features.items():
            assert len(features) == 2
            assert all(isinstance(f, str) for f in features)

    def test_invalid_inputs(self):
        """Test that invalid inputs raise ValueError."""
        rolling = RollingImportance()
        model = LinearRegression()

        # Not a DatetimeIndex
        with pytest.raises(ValueError, match="must be a DatetimeIndex"):
            rolling.compute_rolling_importance(
                model,
                pd.DataFrame({"a": [1, 2, 3]}),
                pd.Series([1, 2, 3]),
                pd.Index([0, 1, 2]),  # type: ignore
            )

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            rolling.compute_rolling_importance(
                model,
                pd.DataFrame({"a": [1, 2, 3]}),
                pd.Series([1, 2]),
                pd.date_range("2024-01-01", periods=3),
            )

    def test_insufficient_data_handling(self):
        """Test handling of windows with insufficient data."""
        np.random.seed(42)
        n_days = 30  # Very short series
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

        X = pd.DataFrame({"a": np.random.randn(n_days)})
        y = pd.Series(X["a"] + np.random.randn(n_days) * 0.1)

        model = LinearRegression()
        model.fit(X, y)

        # Window larger than data
        rolling = RollingImportance(window_days=50, step_days=10)

        # Should raise error if no valid windows
        with pytest.raises(ValueError, match="No valid windows"):
            rolling.compute_rolling_importance(model, X, y, dates)

    def test_get_top_features_before_compute_raises(self):
        """Test that get_top_features raises if called before compute."""
        rolling = RollingImportance()

        with pytest.raises(ValueError, match="No rolling results available"):
            rolling.get_top_features_by_period()
