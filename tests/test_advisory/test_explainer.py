"""Tests for the recommendation explainer module."""

import pytest

from hrp.advisory.explainer import (
    RecommendationExplainer,
    _classify_feature,
    _join_english,
)


class TestClassifyFeature:
    """Tests for feature classification logic."""

    def test_momentum_bullish(self):
        desc = _classify_feature("momentum_20d", 0.05)
        assert desc.sentiment == "bullish"
        assert "momentum" in desc.description.lower()

    def test_momentum_bearish(self):
        desc = _classify_feature("momentum_20d", -0.05)
        assert desc.sentiment == "bearish"

    def test_momentum_neutral(self):
        desc = _classify_feature("momentum_20d", 0.01)
        assert desc.sentiment == "neutral"

    def test_rsi_bullish(self):
        desc = _classify_feature("rsi_14d", 55)
        assert desc.sentiment == "bullish"
        assert "RSI" in desc.description

    def test_rsi_overbought(self):
        desc = _classify_feature("rsi_14d", 75)
        assert desc.sentiment == "bearish"

    def test_volatility_low_is_bullish(self):
        desc = _classify_feature("volatility_20d", 0.15)
        assert desc.sentiment == "bullish"
        assert "low volatility" in desc.description

    def test_volatility_high_is_bearish(self):
        desc = _classify_feature("volatility_20d", 0.40)
        assert desc.sentiment == "bearish"

    def test_unknown_feature(self):
        desc = _classify_feature("custom_feature_xyz", 1.23)
        assert desc.sentiment == "neutral"
        assert "custom_feature_xyz" in desc.description

    def test_volume_ratio_bullish(self):
        desc = _classify_feature("volume_ratio", 1.5)
        assert desc.sentiment == "bullish"
        assert "above-average" in desc.description

    def test_price_to_sma_above(self):
        desc = _classify_feature("price_to_sma_50d", 0.05)
        assert desc.sentiment == "bullish"
        assert "above" in desc.description


class TestJoinEnglish:
    """Tests for English list joining."""

    def test_empty(self):
        assert _join_english([]) == ""

    def test_single(self):
        assert _join_english(["foo"]) == "foo"

    def test_two(self):
        assert _join_english(["foo", "bar"]) == "foo and bar"

    def test_three(self):
        assert _join_english(["a", "b", "c"]) == "a, b, and c"

    def test_four(self):
        result = _join_english(["a", "b", "c", "d"])
        assert result == "a, b, c, and d"


class TestRecommendationExplainer:
    """Tests for the full explainer."""

    def setup_method(self):
        self.explainer = RecommendationExplainer()

    def test_generate_thesis_with_features(self):
        features = {
            "momentum_20d": 0.08,
            "rsi_14d": 55,
            "volatility_20d": 0.18,
        }
        thesis = self.explainer.generate_thesis("AAPL", features, 0.7, "test_model")
        assert "AAPL" in thesis
        assert len(thesis) > 20

    def test_generate_thesis_empty_features(self):
        thesis = self.explainer.generate_thesis("MSFT", {}, 0.5, "test_model")
        assert "MSFT" in thesis
        assert "signal strength" in thesis.lower()

    def test_generate_risk_scenario(self):
        risk = self.explainer.generate_risk_scenario("AAPL", 150.0, 140.0, 170.0)
        assert "$140.00" in risk
        assert "$170.00" in risk
        assert "%" in risk
        assert "risk/reward" in risk.lower()

    def test_generate_confidence_high(self):
        explanation = self.explainer.generate_confidence_explanation("HIGH", 0.8)
        assert "high confidence" in explanation.lower()

    def test_generate_confidence_with_stability(self):
        explanation = self.explainer.generate_confidence_explanation(
            "MEDIUM", 0.5, stability_score=0.3
        )
        assert "moderate" in explanation.lower()
        assert "consistent" in explanation.lower()

    def test_generate_confidence_low(self):
        explanation = self.explainer.generate_confidence_explanation("LOW", 0.2)
        assert "lower confidence" in explanation.lower()
