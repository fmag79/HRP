"""
Sentiment feature definitions for SEC filings and news.

Features:
    - sentiment_score_10k: Sentiment from most recent 10-K
    - sentiment_score_10q: Sentiment from most recent 10-Q
    - sentiment_score_8k: Sentiment from most recent 8-K
    - sentiment_score_avg: Average sentiment across all recent filings
    - sentiment_momentum: Change in sentiment over recent filings
    - sentiment_category: Categorical (positive/neutral/negative)
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

try:
    from hrp.data.db import get_connection
    from hrp.data.sources.edgar_source import EDGARSource
    from hrp.data.sources.factory import DataSourceFactory
except ImportError:
    logger.warning("Failed to import HRP modules - running in standalone mode")


def compute_sentiment_features(
    symbols: list[str], as_of_date: str, edgar_source: EDGARSource | None = None
) -> pd.DataFrame:
    """Compute sentiment features for given symbols.

    Args:
        symbols: List of ticker symbols
        as_of_date: As-of date (YYYY-MM-DD)
        edgar_source: Optional EDGARSource instance

    Returns:
        DataFrame with sentiment features indexed by date and symbol
    """
    if edgar_source is None:
        edgar_source = EDGARSource()

    features = []

    for symbol in symbols:
        try:
            # Get recent filings (last 6 months)
            start_date = (pd.Timestamp(as_of_date) - timedelta(days=180)).strftime(
                "%Y-%m-%d"
            )

            filings_df = edgar_source.get_filings(
                ticker=symbol,
                filing_types=["10-K", "10-Q", "8-K"],
                start_date=start_date,
                end_date=as_of_date,
                limit=20,
            )

            if filings_df.empty:
                logger.warning(f"No filings found for {symbol}")
                continue

            # Sort by filing date (most recent first)
            filings_df = filings_df.sort_values("filing_date", ascending=False)

            # Initialize feature values
            feature_row = {
                "date": as_of_date,
                "symbol": symbol,
                "sentiment_score_10k": None,
                "sentiment_score_10q": None,
                "sentiment_score_8k": None,
                "sentiment_score_avg": None,
                "sentiment_momentum": None,
                "sentiment_category": None,
            }

            # Get most recent of each filing type
            sentiment_scores = []

            for filing_type in ["10-K", "10-Q", "8-K"]:
                type_filings = filings_df[filings_df["filing_type"] == filing_type]

                if not type_filings.empty:
                    # Get most recent filing
                    latest_filing = type_filings.iloc[0]
                    document_url = latest_filing["document_url"]

                    # Fetch filing text
                    filing_text = edgar_source.get_filing_text(document_url)

                    if filing_text:
                        # Analyze sentiment (placeholder - actual analysis requires SentimentAnalyzer)
                        # For now, we'll store the document URL for later processing
                        sentiment_score = None  # Will be computed by SentimentAnalyzer

                        if filing_type == "10-K":
                            feature_row["sentiment_score_10k"] = sentiment_score
                        elif filing_type == "10-Q":
                            feature_row["sentiment_score_10q"] = sentiment_score
                        elif filing_type == "8-K":
                            feature_row["sentiment_score_8k"] = sentiment_score

                        if sentiment_score is not None:
                            sentiment_scores.append(sentiment_score)

            # Compute average sentiment
            if sentiment_scores:
                # Filter out None values before calculating average
                valid_scores = [s for s in sentiment_scores if s is not None]
                if valid_scores:
                    feature_row["sentiment_score_avg"] = sum(valid_scores) / len(valid_scores)

                # Determine category
                avg_score = feature_row["sentiment_score_avg"]
                if avg_score >= 0.3:
                    feature_row["sentiment_category"] = "positive"
                elif avg_score <= -0.3:
                    feature_row["sentiment_category"] = "negative"
                else:
                    feature_row["sentiment_category"] = "neutral"

                # Compute momentum (change from oldest to newest)
                if len(sentiment_scores) >= 2:
                    feature_row["sentiment_momentum"] = (
                        sentiment_scores[0] - sentiment_scores[-1]
                    )

            features.append(feature_row)

        except Exception as e:
            logger.error(f"Error computing sentiment features for {symbol}: {e}")
            continue

    if features:
        return pd.DataFrame(features)
    return pd.DataFrame()


def store_sentiment_features(
    features_df: pd.DataFrame, db_path: str = None
) -> None:
    """Store sentiment features to database.

    Args:
        features_df: DataFrame with sentiment features
        db_path: Optional database path
    """
    if features_df.empty:
        return

    try:
        conn = get_connection(db_path=db_path)

        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_features (
                date DATE,
                symbol VARCHAR(10),
                sentiment_score_10k DOUBLE,
                sentiment_score_10q DOUBLE,
                sentiment_score_8k DOUBLE,
                sentiment_score_avg DOUBLE,
                sentiment_momentum DOUBLE,
                sentiment_category VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, symbol)
            )
        """)

        # Upsert features
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM sentiment_features WHERE date = ?", [features_df["date"].iloc[0]])
        conn.execute("INSERT INTO sentiment_features SELECT * FROM features_df")
        conn.execute("COMMIT")

        logger.info(f"Stored {len(features_df)} sentiment feature records")

    except Exception as e:
        logger.error(f"Failed to store sentiment features: {e}")


def get_sentiment_features(
    symbols: list[str], start_date: str, end_date: str, db_path: str = None
) -> pd.DataFrame:
    """Retrieve sentiment features from database.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Optional database path

    Returns:
        DataFrame with sentiment features
    """
    try:
        conn = get_connection(db_path=db_path)

        # Use parameterized query to prevent SQL injection
        placeholders = ",".join("?" * len(symbols))
        query = f"""
            SELECT * FROM sentiment_features
            WHERE symbol IN ({placeholders})
            AND date BETWEEN ? AND ?
            ORDER BY date, symbol
        """

        params = symbols + [start_date, end_date]
        return conn.query(query, params)

    except Exception as e:
        logger.error(f"Failed to retrieve sentiment features: {e}")
        return pd.DataFrame()


# Feature registration for the feature store
SENTIMENT_FEATURES = [
    "sentiment_score_10k",
    "sentiment_score_10q",
    "sentiment_score_8k",
    "sentiment_score_avg",
    "sentiment_momentum",
    "sentiment_category",
]


def register_sentiment_features() -> None:
    """Register sentiment features in the feature registry."""
    # This would integrate with the existing feature registry
    # For now, we define the feature list
    pass
