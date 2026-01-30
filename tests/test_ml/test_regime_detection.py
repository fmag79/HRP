"""
Tests for HMM-based structural regime detection.

Tests cover:
- VolatilityHMM classification
- TrendHMM classification
- StructuralRegimeClassifier
- Regime combination logic
"""

import numpy as np
import pandas as pd
import pytest

from hrp.ml.regime_detection import (
    VolatilityHMM,
    TrendHMM,
    combine_regime_labels,
    StructuralRegimeClassifier,
    StructuralRegime,
)


class TestVolatilityHMM:
    """Tests for Volatility HMM."""

    def test_volatility_hmm_classification(self):
        """Volatility HMM classifies high/low vol regimes."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        low_vol = np.random.normal(0.01, 0.01, n_samples // 2)
        high_vol = np.random.normal(0.01, 0.04, n_samples // 2)
        volatility = np.concatenate([low_vol, high_vol])

        hmm = VolatilityHMM(n_regimes=2)
        hmm.fit(volatility)
        regimes = hmm.predict(volatility)

        assert len(regimes) == n_samples
        assert len(set(regimes)) <= 2  # Should classify into 2 regimes

    def test_volatility_hmm_requires_fit(self):
        """VolatilityHMM requires fit before predict."""
        hmm = VolatilityHMM(n_regimes=2)
        volatility = np.random.normal(0.01, 0.02, 100)

        with pytest.raises(RuntimeError, match="not fitted"):
            hmm.predict(volatility)

    def test_volatility_hmm_two_regimes(self):
        """VolatilityHMM with n_regimes=2."""
        np.random.seed(42)
        volatility = np.random.normal(0.02, 0.02, 200)

        hmm = VolatilityHMM(n_regimes=2)
        hmm.fit(volatility)
        regimes = hmm.predict(volatility)

        assert set(regimes).issubset({0, 1})


class TestTrendHMM:
    """Tests for Trend HMM."""

    def test_trend_hmm_classification(self):
        """Trend HMM classifies bull/bear regimes."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        bull = np.random.normal(0.001, 0.01, n_samples // 2)
        bear = np.random.normal(-0.0005, 0.01, n_samples // 2)
        returns = np.concatenate([bull, bear])

        hmm = TrendHMM(n_regimes=2)
        hmm.fit(returns)
        regimes = hmm.predict(returns)

        assert len(regimes) == n_samples
        assert len(set(regimes)) <= 2

    def test_trend_hmm_requires_fit(self):
        """TrendHMM requires fit before predict."""
        hmm = TrendHMM(n_regimes=2)
        returns = np.random.normal(0.001, 0.01, 100)

        with pytest.raises(RuntimeError, match="not fitted"):
            hmm.predict(returns)


class TestRegimeCombination:
    """Tests for regime label combination."""

    def test_structural_regime_combination(self):
        """Combine vol and trend regimes into structural regimes."""
        vol_regimes = [0, 0, 1, 1]  # low, low, high, high
        trend_regimes = [0, 1, 0, 1]  # bull, bear, bull, bear

        structural = combine_regime_labels(vol_regimes, trend_regimes)

        assert structural[0] == "low_vol_bull"
        assert structural[1] == "low_vol_bear"
        assert structural[2] == "high_vol_bull"
        assert structural[3] == "high_vol_bear"

    def test_regime_combination_length_match(self):
        """Regime arrays must have same length."""
        vol_regimes = [0, 0, 1]
        trend_regimes = [0, 1]  # Different length

        with pytest.raises(ValueError, match="same length"):
            combine_regime_labels(vol_regimes, trend_regimes)


class TestStructuralRegimeClassifier:
    """Tests for StructuralRegimeClassifier."""

    def test_classifier_initialization(self):
        """StructuralRegimeClassifier initializes with 2 HMMs."""
        classifier = StructuralRegimeClassifier()

        assert hasattr(classifier, "vol_hmm")
        assert hasattr(classifier, "trend_hmm")
        assert isinstance(classifier.vol_hmm, VolatilityHMM)
        assert isinstance(classifier.trend_hmm, TrendHMM)

    def test_classifier_fit_predict(self):
        """StructuralRegimeClassifier fits and predicts."""
        # Create synthetic price data
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range("2020-01-01", periods=n_samples)
        prices = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.normal(0.1, 1, n_samples))
        }, index=dates)

        classifier = StructuralRegimeClassifier()
        classifier.fit(prices)
        structural = classifier.predict(prices)

        # Account for data loss due to rolling windows (20-day warmup)
        # and NaN drops - predict returns fewer samples than input
        assert len(structural) < n_samples  # Some loss due to warmup
        assert len(structural) > n_samples - 50  # But not too much loss
        assert all(isinstance(r, str) for r in structural)
        assert all(r in ["low_vol_bull", "low_vol_bear", "high_vol_bull", "high_vol_bear"]
                   for r in structural)

    def test_classifier_requires_fit(self):
        """StructuralRegimeClassifier requires fit before predict."""
        classifier = StructuralRegimeClassifier()
        prices = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        with pytest.raises(RuntimeError, match="not fitted"):
            classifier.predict(prices)

    def test_get_scenario_periods(self):
        """StructuralRegimeClassifier extracts scenario periods."""
        # Create synthetic data with clear regime switches
        np.random.seed(42)
        n_samples = 300
        dates = pd.date_range("2020-01-01", periods=n_samples)

        # Create price data with regime-like behavior
        base_returns = np.concatenate([
            np.random.normal(0.001, 0.005, 100),  # Bull, low vol
            np.random.normal(-0.0005, 0.005, 100),  # Bear, low vol
            np.random.normal(0.001, 0.02, 100),  # Bull, high vol
        ])

        prices = pd.DataFrame({
            "close": 100 + np.cumsum(base_returns)
        }, index=dates)

        classifier = StructuralRegimeClassifier()
        classifier.fit(prices)
        periods = classifier.get_scenario_periods(prices, min_days=60)

        # Should return dict with 4 regime types as keys
        assert isinstance(periods, dict)
        for regime in ["low_vol_bull", "low_vol_bear", "high_vol_bull", "high_vol_bear"]:
            assert regime in periods
            assert isinstance(periods[regime], list)
