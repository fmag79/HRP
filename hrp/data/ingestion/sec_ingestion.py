"""
SEC filings ingestion job with sentiment analysis.

Fetches SEC filings, analyzes sentiment, and stores in database.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

try:
    from hrp.data.db import get_connection
    from hrp.data.sources.edgar_source import EDGARSource
    from hrp.data.sentiment_analyzer import SentimentAnalyzer, create_sentiment_analyzer
except ImportError:
    logger.warning("Failed to import HRP modules - running in standalone mode")

# Load environment variables
load_dotenv()


class SECIngestionJob:
    """Ingests SEC filings and computes sentiment analysis."""

    def __init__(self, db_path: str = None, edgar_user_agent: str = None):
        """Initialize SEC ingestion job.

        Args:
            db_path: Database path
            edgar_user_agent: User agent for SEC requests (format: "Name Email")
        """
        self.db_path = db_path
        self.edgar_user_agent = (
            edgar_user_agent or "HRP Platform research@hrp.example.com"
        )

        self.edgar_source = EDGARSource(user_agent=self.edgar_user_agent)
        self.sentiment_analyzer = create_sentiment_analyzer()

        logger.info("Initialized SECIngestionJob")

    def get_universe_symbols(self) -> list[str]:
        """Get list of symbols from universe table.

        Returns:
            List of ticker symbols
        """
        try:
            conn = get_connection(db_path=self.db_path)
            df = conn.query("SELECT DISTINCT symbol FROM universe ORDER BY symbol")
            return df["symbol"].tolist()
        except Exception as e:
            logger.error(f"Failed to get universe symbols: {e}")
            return []

    def fetch_filings_for_symbols(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        filing_types: list[str] | None = None,
        limit_per_ticker: int = 10,
    ) -> pd.DataFrame:
        """Fetch SEC filings for given symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            filing_types: List of filing types (default: ["10-K", "10-Q", "8-K"])
            limit_per_ticker: Max filings per ticker

        Returns:
            DataFrame with filing metadata
        """
        if filing_types is None:
            filing_types = ["10-K", "10-Q", "8-K"]

        logger.info(f"Fetching SEC filings for {len(symbols)} symbols from {start_date} to {end_date}")

        filings_df = self.edgar_source.fetch_filings_data(
            tickers=symbols,
            filing_types=filing_types,
            start_date=start_date,
            end_date=end_date,
            limit_per_ticker=limit_per_ticker,
        )

        logger.info(f"Retrieved {len(filings_df)} filings")

        return filings_df

    def analyze_filing_sentiment(
        self, filings_df: pd.DataFrame, analyze_all: bool = False
    ) -> pd.DataFrame:
        """Analyze sentiment for SEC filings.

        Args:
            filings_df: DataFrame with filing metadata
            analyze_all: If True, analyze all filings. If False, analyze only recent ones.

        Returns:
            DataFrame with sentiment analysis results
        """
        if filings_df.empty:
            return pd.DataFrame()

        # Sort by filing date (most recent first)
        filings_df = filings_df.sort_values(["symbol", "filing_date"], ascending=False)

        # Keep only most recent filing per symbol per type if not analyzing all
        if not analyze_all:
            filings_df = (
                filings_df.groupby(["symbol", "filing_type"])
                .head(1)
                .reset_index(drop=True)
            )

        sentiment_results = []

        for _, row in filings_df.iterrows():
            try:
                logger.info(
                    f"Analyzing sentiment for {row['symbol']} {row['filing_type']} from {row['filing_date']}"
                )

                # Fetch filing text
                filing_text = self.edgar_source.get_filing_text(row["document_url"])

                if not filing_text:
                    logger.warning(f"Failed to fetch filing text for {row['symbol']}")
                    continue

                # Analyze sentiment
                result = self.sentiment_analyzer.analyze_filing(
                    filing_text, filing_type=row["filing_type"]
                )

                # Combine with metadata
                result["symbol"] = row["symbol"]
                result["cik"] = row["cik"]
                result["filing_type"] = row["filing_type"]
                result["filing_date"] = row["filing_date"]
                result["accession_number"] = row["accession_number"]

                sentiment_results.append(result)

            except Exception as e:
                logger.error(f"Error analyzing filing {row['symbol']}: {e}")
                continue

        if sentiment_results:
            return pd.DataFrame(sentiment_results)
        return pd.DataFrame()

    def store_filings(self, filings_df: pd.DataFrame) -> None:
        """Store filing metadata to database.

        Args:
            filings_df: DataFrame with filing metadata
        """
        if filings_df.empty:
            return

        try:
            conn = get_connection(db_path=self.db_path)

            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sec_filings (
                    symbol VARCHAR(10),
                    cik VARCHAR(10),
                    filing_type VARCHAR(10),
                    filing_date DATE,
                    accession_number VARCHAR(30),
                    document_url TEXT,
                    filing_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (accession_number)
                )
            """)

            # Upsert filings using executemany for better performance
            conn.execute("BEGIN TRANSACTION")
            conn.executemany("""
                INSERT OR REPLACE INTO sec_filings
                (symbol, cik, filing_type, filing_date, accession_number, document_url, filing_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    row["symbol"],
                    row["cik"],
                    row["filing_type"],
                    row["filing_date"],
                    row["accession_number"],
                    row["document_url"],
                    row["filing_url"],
                )
                for _, row in filings_df.iterrows()
            ])
            conn.execute("COMMIT")

            logger.info(f"Stored {len(filings_df)} SEC filings")

        except Exception as e:
            logger.error(f"Failed to store SEC filings: {e}")

    def store_sentiment_results(self, sentiment_df: pd.DataFrame) -> None:
        """Store sentiment analysis results to database.

        Args:
            sentiment_df: DataFrame with sentiment analysis results
        """
        if sentiment_df.empty:
            return

        try:
            conn = get_connection(db_path=self.db_path)

            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sec_filing_sentiment (
                    symbol VARCHAR(10),
                    cik VARCHAR(10),
                    filing_type VARCHAR(10),
                    filing_date DATE,
                    accession_number VARCHAR(30),
                    sentiment_score DOUBLE,
                    analysis TEXT,
                    model_used VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (accession_number)
                )
            """)

            # Upsert sentiment results using executemany for better performance
            conn.execute("BEGIN TRANSACTION")
            conn.executemany("""
                INSERT OR REPLACE INTO sec_filing_sentiment
                (symbol, cik, filing_type, filing_date, accession_number, sentiment_score, analysis, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    row["symbol"],
                    row["cik"],
                    row["filing_type"],
                    row["filing_date"],
                    row["accession_number"],
                    row["sentiment_score"],
                    row.get("analysis", ""),
                    row["model_used"],
                )
                for _, row in sentiment_df.iterrows()
            ])
            conn.execute("COMMIT")

            logger.info(f"Stored {len(sentiment_df)} sentiment analysis results")

        except Exception as e:
            logger.error(f"Failed to store sentiment results: {e}")

    def run(
        self,
        lookback_days: int = 180,
        analyze_all: bool = False,
        limit_per_ticker: int = 10,
    ) -> None:
        """Run the SEC ingestion job.

        Args:
            lookback_days: Number of days to look back for filings
            analyze_all: If True, analyze all filings. If False, analyze only recent ones.
            limit_per_ticker: Max filings to fetch per ticker
        """
        start_time = datetime.now()

        # Get date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        logger.info(f"Running SEC ingestion job from {start_date} to {end_date}")

        # Get universe symbols
        symbols = self.get_universe_symbols()
        if not symbols:
            logger.warning("No symbols found in universe")
            return

        # Fetch filings
        filings_df = self.fetch_filings_for_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            limit_per_ticker=limit_per_ticker,
        )

        if filings_df.empty:
            logger.warning("No filings found")
            return

        # Store filing metadata
        self.store_filings(filings_df)

        # Analyze sentiment
        sentiment_df = self.analyze_filing_sentiment(filings_df, analyze_all=analyze_all)

        if sentiment_df.empty:
            logger.warning("No sentiment analysis results")
            return

        # Store sentiment results
        self.store_sentiment_results(sentiment_df)

        # Log completion
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"SEC ingestion job completed in {elapsed:.2f}s. "
            f"Filings: {len(filings_df)}, Analyzed: {len(sentiment_df)}"
        )

    def close(self):
        """Clean up resources."""
        self.edgar_source.close()


def main():
    """Main entry point for running the SEC ingestion job."""
    import sys

    # Parse arguments
    lookback_days = 180
    analyze_all = False

    if len(sys.argv) > 1:
        try:
            lookback_days = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid lookback_days: {sys.argv[1]}, using default: 180")

    if len(sys.argv) > 2:
        analyze_all = sys.argv[2].lower() in ["true", "1", "yes"]

    # Run job
    job = SECIngestionJob()

    try:
        job.run(lookback_days=lookback_days, analyze_all=analyze_all)
    finally:
        job.close()


if __name__ == "__main__":
    main()
