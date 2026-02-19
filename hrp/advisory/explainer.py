"""
Plain-English explanation generator for trading recommendations.

Translates quantitative signals and model outputs into human-readable
explanations. Template-based for determinism, speed, and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureDescription:
    """Human-readable description of a feature value."""

    feature_name: str
    value: float
    description: str
    sentiment: str  # "bullish", "bearish", "neutral"


# Feature-to-English mapping templates
_FEATURE_TEMPLATES: dict[str, dict] = {
    "momentum_20d": {
        "name": "20-day momentum",
        "bullish": lambda v: f"strong upward momentum ({v:+.1%} over 20 days)",
        "bearish": lambda v: f"negative momentum ({v:+.1%} over 20 days)",
        "neutral": lambda v: f"flat momentum ({v:+.1%} over 20 days)",
        "bullish_threshold": 0.03,
        "bearish_threshold": -0.03,
    },
    "momentum_60d": {
        "name": "60-day momentum",
        "bullish": lambda v: f"solid medium-term trend ({v:+.1%} over 60 days)",
        "bearish": lambda v: f"weakening medium-term trend ({v:+.1%} over 60 days)",
        "neutral": lambda v: f"sideways over the past 60 days ({v:+.1%})",
        "bullish_threshold": 0.05,
        "bearish_threshold": -0.05,
    },
    "rsi_14d": {
        "name": "RSI",
        "bullish": lambda v: f"healthy buying pressure (RSI {v:.0f})",
        "bearish": lambda v: f"overbought territory (RSI {v:.0f})",
        "neutral": lambda v: f"neutral momentum (RSI {v:.0f})",
        "bullish_threshold": 50,
        "bearish_threshold": 70,
    },
    "volatility_20d": {
        "name": "20-day volatility",
        "bullish": lambda v: f"low volatility ({v:.1%} annualized)",
        "bearish": lambda v: f"elevated volatility ({v:.1%} annualized)",
        "neutral": lambda v: f"moderate volatility ({v:.1%} annualized)",
        "bullish_threshold": 0.20,  # below = bullish (calm)
        "bearish_threshold": 0.35,
    },
    "price_to_sma_50d": {
        "name": "price vs 50-day average",
        "bullish": lambda v: f"trading above its 50-day average ({v:.1%} above)",
        "bearish": lambda v: f"trading below its 50-day average ({v:.1%} below)",
        "neutral": lambda v: f"near its 50-day average ({v:.1%})",
        "bullish_threshold": 0.02,
        "bearish_threshold": -0.02,
    },
    "volume_ratio": {
        "name": "volume",
        "bullish": lambda v: f"above-average trading volume ({v:.1f}x normal)",
        "bearish": lambda v: f"below-average volume ({v:.1f}x normal)",
        "neutral": lambda v: f"normal trading volume ({v:.1f}x average)",
        "bullish_threshold": 1.2,
        "bearish_threshold": 0.8,
    },
    "sentiment_score_avg": {
        "name": "filing sentiment",
        "bullish": lambda v: f"positive SEC filing sentiment ({v:+.2f})",
        "bearish": lambda v: f"negative filing sentiment ({v:+.2f})",
        "neutral": lambda v: f"neutral filing sentiment ({v:+.2f})",
        "bullish_threshold": 0.1,
        "bearish_threshold": -0.1,
    },
    "macd_histogram": {
        "name": "MACD",
        "bullish": lambda v: f"bullish MACD signal (histogram {v:+.2f})",
        "bearish": lambda v: f"bearish MACD signal (histogram {v:+.2f})",
        "neutral": lambda v: f"flat MACD (histogram {v:+.2f})",
        "bullish_threshold": 0.5,
        "bearish_threshold": -0.5,
    },
}


def _classify_feature(feature_name: str, value: float) -> FeatureDescription:
    """Classify a feature value and generate a description."""
    template = _FEATURE_TEMPLATES.get(feature_name)
    if not template:
        return FeatureDescription(
            feature_name=feature_name,
            value=value,
            description=f"{feature_name} = {value:.4f}",
            sentiment="neutral",
        )

    bullish_thresh = template["bullish_threshold"]
    bearish_thresh = template["bearish_threshold"]

    # Volatility is inverted (low = bullish)
    if feature_name in ("volatility_20d",):
        if value < bullish_thresh:
            sentiment = "bullish"
        elif value > bearish_thresh:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
    # RSI: bullish if 40-65, bearish if > 70
    elif feature_name == "rsi_14d":
        if bullish_thresh <= value <= 65:
            sentiment = "bullish"
        elif value > bearish_thresh or value < 30:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
    else:
        if value > bullish_thresh:
            sentiment = "bullish"
        elif value < bearish_thresh:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

    description = template[sentiment](value)
    return FeatureDescription(
        feature_name=feature_name,
        value=value,
        description=description,
        sentiment=sentiment,
    )


class RecommendationExplainer:
    """Generates plain-English explanations from quantitative signals."""

    def generate_thesis(
        self,
        symbol: str,
        features: dict[str, float],
        signal_strength: float,
        model_name: str,
    ) -> str:
        """
        Generate a 2-3 sentence plain-English thesis.

        Args:
            symbol: Ticker symbol
            features: Feature name -> value dict
            signal_strength: Model prediction strength
            model_name: Name of the model generating the signal

        Returns:
            Plain-English thesis string
        """
        if not features:
            return (
                f"Our model identifies {symbol} as a strong candidate "
                f"with signal strength {signal_strength:.2f}."
            )

        # Classify all features
        descriptions = []
        for feat_name, value in features.items():
            desc = _classify_feature(feat_name, value)
            descriptions.append(desc)

        # Pick top 3 most informative (prefer bullish signals)
        bullish = [d for d in descriptions if d.sentiment == "bullish"]
        other = [d for d in descriptions if d.sentiment != "bullish"]
        top = (bullish + other)[:3]

        if not top:
            return (
                f"Our model identifies {symbol} as a candidate "
                f"with signal strength {signal_strength:.2f}."
            )

        # Build thesis
        drivers = [d.description for d in top]
        drivers_text = _join_english(drivers)

        confidence_word = "strong" if abs(signal_strength) > 0.6 else "moderate"
        expected_return = abs(signal_strength) * 50  # rough % estimate

        thesis = (
            f"{symbol} shows {drivers_text}. "
            f"The model sees {confidence_word} potential "
            f"with an expected return of {expected_return:.0f}% "
            f"over the recommended holding period."
        )
        return thesis

    def generate_risk_scenario(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> str:
        """
        Generate a plain-English risk scenario.

        Args:
            symbol: Ticker symbol
            entry_price: Current/entry price
            stop_price: Stop-loss price
            target_price: Target price

        Returns:
            Plain-English risk description
        """
        downside_pct = (entry_price - stop_price) / entry_price
        upside_pct = (target_price - entry_price) / entry_price

        risk = (
            f"Maximum planned loss is {downside_pct:.1%} "
            f"(stop at ${stop_price:.2f}). "
            f"Upside target is {upside_pct:.1%} (${target_price:.2f}). "
            f"A broad market selloff could trigger the stop-loss. "
            f"Risk/reward ratio: {upside_pct / downside_pct:.1f}:1."
        )
        return risk

    def generate_confidence_explanation(
        self,
        confidence: str,
        signal_strength: float,
        stability_score: float | None = None,
    ) -> str:
        """
        Explain the confidence level in plain English.

        Args:
            confidence: HIGH, MEDIUM, or LOW
            signal_strength: Raw model signal
            stability_score: Optional walk-forward stability metric

        Returns:
            Plain-English confidence explanation
        """
        parts = []
        if confidence == "HIGH":
            parts.append("High confidence — strong model signal")
        elif confidence == "MEDIUM":
            parts.append("Moderate confidence — decent signal but some uncertainty")
        else:
            parts.append("Lower confidence — signal is present but weaker")

        if stability_score is not None:
            if stability_score <= 0.5:
                parts.append("The model has been very consistent across different time periods")
            elif stability_score <= 1.0:
                parts.append("Model performance has been reasonably stable")
            else:
                parts.append("Model performance has varied across time periods")

        return ". ".join(parts) + "."


def _join_english(items: list[str]) -> str:
    """Join list items in English: 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
