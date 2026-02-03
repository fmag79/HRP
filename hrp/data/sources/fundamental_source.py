"""
Fundamental data source adapter for HRP.

Uses yfinance library to fetch fundamental metrics like P/E, P/B, market cap, etc.
"""

from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from hrp.data.sources.base import DataSourceBase


# Mapping from our feature names to yfinance info keys
FUNDAMENTAL_METRICS = {
    "market_cap": "marketCap",
    "pe_ratio": "trailingPE",
    "pb_ratio": "priceToBook",
    "dividend_yield": "dividendYield",
    "ev_ebitda": "enterpriseToEbitda",
    "shares_outstanding": "sharesOutstanding",
}


class FundamentalSource(DataSourceBase):
    """
    Fundamental data source using Yahoo Finance.

    Provides access to fundamental metrics like P/E ratio, P/B ratio,
    market cap, dividend yield, and EV/EBITDA.

    Note: Fundamentals don't change daily, so weekly updates are sufficient.
    """

    source_name = "yfinance_fundamentals"

    def __init__(self):
        """Initialize the fundamental data source."""
        super().__init__()
        logger.info("Fundamental data source initialized")

    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """
        Not implemented for fundamentals - use get_fundamentals instead.

        This method exists to satisfy the abstract base class.
        """
        raise NotImplementedError(
            "FundamentalSource does not provide daily bars. Use get_fundamentals() instead."
        )

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and has fundamental data.

        Args:
            symbol: Stock ticker to validate

        Returns:
            True if symbol exists and has fundamental info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Check if we got meaningful data (not just empty or error)
            return info.get("marketCap") is not None
        except Exception:
            return False

    def get_fundamentals(
        self,
        symbol: str,
        as_of_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Fetch fundamental metrics for a single symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            as_of_date: Date for the data (used for record-keeping, actual data
                       is always the latest available from Yahoo Finance)

        Returns:
            Dictionary with fundamental metrics:
            - symbol: Stock ticker
            - date: As-of date
            - market_cap: Market capitalization in USD
            - pe_ratio: Trailing P/E ratio
            - pb_ratio: Price-to-book ratio
            - dividend_yield: Dividend yield (decimal, e.g., 0.02 for 2%)
            - ev_ebitda: Enterprise value to EBITDA ratio
            - source: Data source name
        """
        if as_of_date is None:
            as_of_date = date.today()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            result = {
                "symbol": symbol,
                "date": as_of_date,
                "source": self.source_name,
            }

            # Extract each metric, handling missing data
            for our_name, yf_name in FUNDAMENTAL_METRICS.items():
                value = info.get(yf_name)
                # Convert to float, handling None and invalid values
                if value is not None:
                    try:
                        result[our_name] = float(value)
                    except (TypeError, ValueError):
                        result[our_name] = None
                else:
                    result[our_name] = None

            logger.debug(f"Fetched fundamentals for {symbol}: {result}")
            return result

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            raise

    def get_fundamentals_batch(
        self,
        symbols: list[str],
        as_of_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fundamental metrics for multiple symbols.

        Args:
            symbols: List of stock tickers
            as_of_date: Date for the data (used for record-keeping)

        Returns:
            DataFrame with columns:
            - symbol, date, market_cap, pe_ratio, pb_ratio, dividend_yield, ev_ebitda, source
        """
        if as_of_date is None:
            as_of_date = date.today()

        results = []
        failed_symbols = []

        for symbol in symbols:
            try:
                data = self.get_fundamentals(symbol, as_of_date)
                results.append(data)
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
                failed_symbols.append(symbol)

        if not results:
            logger.warning("No fundamental data fetched for any symbol")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Ensure column order
        columns = ["symbol", "date", "market_cap", "pe_ratio", "pb_ratio",
                   "dividend_yield", "ev_ebitda", "shares_outstanding", "source"]
        df = df[columns]

        logger.info(
            f"Fetched fundamentals for {len(results)}/{len(symbols)} symbols"
            f"{f' (failed: {len(failed_symbols)})' if failed_symbols else ''}"
        )

        return df

    def get_all_info(self, symbol: str) -> dict[str, Any]:
        """
        Get all available info for a symbol (for debugging/exploration).

        Args:
            symbol: Stock ticker

        Returns:
            Full dictionary of all available info from yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching all info for {symbol}: {e}")
            return {}
