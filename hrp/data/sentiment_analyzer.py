"""
Sentiment analysis for financial text using Claude API.

Analyzes SEC filings, earnings calls, and news sentiment.
"""

import os
from typing import Any, Optional

import anthropic
import pandas as pd
from loguru import logger

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class SentimentAnalyzer:
    """Sentiment analysis using Claude API.

    Returns sentiment scores on a scale:
        - Positive: 0.5 to 1.0
        - Neutral: -0.5 to 0.5
        - Negative: -1.0 to -0.5
    """

    SENTIMENT_PROMPT = """Analyze the financial sentiment of the following text. Focus on:

1. Business outlook and guidance
2. Financial performance indicators
3. Risk factors and concerns
4. Management tone and confidence

Provide a sentiment score between -1.0 (very negative) and 1.0 (very positive),
with 0.0 being neutral.

Text:
{text}

Provide your analysis in this format:
SCORE: <number between -1.0 and 1.0>
ANALYSIS: <brief explanation of key sentiment drivers>
"""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022"):
        """Initialize sentiment analyzer.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in environment or passed as api_key"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized SentimentAnalyzer with model: {model}")

    def analyze_text(self, text: str, max_length: int = 10000) -> dict[str, Any]:
        """Analyze sentiment of a text.

        Args:
            text: Text to analyze
            max_length: Max characters to analyze (Claude has context limits)

        Returns:
            Dict with keys: sentiment_score, analysis, model_used
        """
        if not text or len(text.strip()) == 0:
            return {"sentiment_score": 0.0, "analysis": "Empty text", "model_used": self.model}

        # Truncate text if too long
        text = text[:max_length]

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": self.SENTIMENT_PROMPT.format(text=text),
                    }
                ],
            )

            response_text = message.content[0].text

            # Parse response
            score = 0.0
            analysis = ""

            for line in response_text.split("\n"):
                line = line.strip()
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.split("SCORE:")[-1].strip())
                        # Clamp score to [-1.0, 1.0]
                        score = max(-1.0, min(1.0, score))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("ANALYSIS:"):
                    analysis = line.split("ANALYSIS:")[-1].strip()

            return {
                "sentiment_score": score,
                "analysis": analysis or response_text,
                "model_used": self.model,
            }

        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {
                "sentiment_score": 0.0,
                "analysis": f"Error: {str(e)}",
                "model_used": self.model,
            }

    def analyze_filing(
        self, filing_text: str, filing_type: str = None
    ) -> dict[str, Any]:
        """Analyze sentiment of an SEC filing.

        Args:
            filing_text: Full text of the filing
            filing_type: Type of filing (10-K, 10-Q, 8-K) for context

        Returns:
            Dict with sentiment score and analysis
        """
        if filing_type:
            context = f"This is a {filing_type} SEC filing. "
        else:
            context = ""

        # Add filing type context to analysis
        text_to_analyze = context + filing_text

        result = self.analyze_text(text_to_analyze)

        if filing_type:
            result["filing_type"] = filing_type

        return result

    def batch_analyze(
        self, texts: list[str], filing_types: list[str] | None = None
    ) -> pd.DataFrame:
        """Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze
            filing_types: Optional list of filing types (same length as texts)

        Returns:
            DataFrame with columns: sentiment_score, analysis, model_used, filing_type
        """
        results = []

        for i, text in enumerate(texts):
            filing_type = filing_types[i] if filing_types and i < len(filing_types) else None

            if filing_type:
                result = self.analyze_filing(text, filing_type)
            else:
                result = self.analyze_text(text)

            results.append(result)

        return pd.DataFrame(results)

    def get_sentiment_category(self, score: float) -> str:
        """Convert sentiment score to category.

        Args:
            score: Sentiment score between -1.0 and 1.0

        Returns:
            Category: "positive", "neutral", or "negative"
        """
        if score >= 0.3:
            return "positive"
        elif score <= -0.3:
            return "negative"
        else:
            return "neutral"

    def analyze_news_headlines(self, headlines: list[str]) -> pd.DataFrame:
        """Analyze sentiment of news headlines.

        Args:
            headlines: List of news headlines

        Returns:
            DataFrame with sentiment analysis for each headline
        """
        results = []

        for headline in headlines:
            result = self.analyze_text(headline)
            result["headline"] = headline
            result["category"] = self.get_sentiment_category(result["sentiment_score"])
            results.append(result)

        return pd.DataFrame(results)


def create_sentiment_analyzer(api_key: str | None = None) -> SentimentAnalyzer:
    """Factory function to create a SentimentAnalyzer.

    Args:
        api_key: Optional Anthropic API key

    Returns:
        SentimentAnalyzer instance
    """
    return SentimentAnalyzer(api_key=api_key)
